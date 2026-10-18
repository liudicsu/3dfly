#!/usr/bin/env python3
"""Closed-loop M2–M4 acceptance.

Official flight gate: frontolateral collision-course looms (|az0| 20–70°,
both sides, several l/v). Pass if ≥80% miss the object (surface
clearance d_centre−0.58 > 0.1 m) and turn it to the far side — GF
fire is not required (Tammero & Dickinson 2002; Muijres et al. 2014).
Report GF escapes vs pre-GF saccades, DN drivers, peak yaw rate, and
the fraction of turns exceeding no-object wander (mean+2 sd of yaw
rate). Pure frontal (|az0|<20°) must fire. Receding discs must not
fire. Static 30–40° onset is report-only.

    taskset -c 0,1 python scripts/closed_loop_acceptance.py --steps 2000 --output output/m2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threedfly.control.closed_loop_metrics import (
    LOOM_RADIUS_M,
    LOOM_START_ANG_MAX_DEG,
    MISS_SURFACE_M,
    PLANT_PEAK_YAW_DEG_S,
    PR16_STEPS,
    PR16_WALLS,
    PR16_WALLS_PER_100,
    ROOM_XY,
    angular_size_deg,
    avoidance_feasibility,
    avoidance_mode,
    behavioural_response,
    closest_approach_step,
    collision_course_object,
    coverage,
    direction_correct_frac,
    dnp01_baseline_shows_loom,
    escape_turned_into,
    fire_minus_closest,
    flight_avoidance_ok,
    geometry_aware_frac,
    loom_geometry,
    max_lateral_accel_mps2,
    mean_abs_yaw_rate_rad_s,
    min_center_distance_m,
    min_surface_clearance_m,
    net_heading_delta,
    peak_yaw_rate_rad_s,
    object_behind_fly,
    onset_azimuth_band,
    start_distance_for_angular_size,
    visible_approaching_ok,
    visible_loom_placement,
    wall_ttc_s,
    walls_per_100,
    wander_yaw_threshold,
    yaw_term_away_integrals,
    wrap_heading_delta,
    xy_extent,
)
from threedfly.control.dn_controller import DNFlightController
from threedfly.control.exploration import heading_from_quaternion
from threedfly.control.flight_params import FlightParams


def beats_pr16_walls(n_contacts: int, n_steps: int) -> bool:
    """True if wall contacts per 100 steps beat PR #16 v5 (81 / 2000)."""
    return walls_per_100(n_contacts, n_steps) < PR16_WALLS_PER_100


def lateral_loom_pose(side: str) -> dict:
    """Near-wall pose for texture / FOV checks (not the official escape stimulus)."""
    speed = 0.75
    if side == "L":
        yaw = 0.70
        return {
            "pose": (-0.8, 3.8, 1.5),
            "yaw": yaw,
            "vel": (speed * float(np.cos(yaw)), speed * float(np.sin(yaw)), 0.0),
        }
    if side == "R":
        yaw = -0.70
        return {
            "pose": (-0.8, -3.8, 1.5),
            "yaw": yaw,
            "vel": (speed * float(np.cos(yaw)), speed * float(np.sin(yaw)), 0.0),
        }
    raise ValueError(f"side must be L or R, got {side!r}")


def lateral_loom_object(side: str) -> dict:
    """Compact dark object closing from one side (M1-like looming disk).

    Fly at the origin heading +X. Start ~2.8 m out (≈35°) so the GF
    window (40–60°) is during the rise, not after the object fills both
    120° eyes.
    """
    if side == "L":
        return {
            "pose": (0.0, 0.0, 1.5),
            "yaw": 0.0,
            "vel": (0.05, 0.0, 0.0),
            "loom_start": (2.00, 2.00, 1.5),
            "loom_vel": (-1.40, -1.35, 0.0),
        }
    if side == "R":
        return {
            "pose": (0.0, 0.0, 1.5),
            "yaw": 0.0,
            "vel": (0.05, 0.0, 0.0),
            "loom_start": (2.00, -2.00, 1.5),
            "loom_vel": (-1.40, 1.35, 0.0),
        }
    raise ValueError(f"side must be L or R, got {side!r}")


# Fixed RNG list for unseen frontolateral collision-course seeds.
# Chosen so the generator (not a hand-picked loom) covers L/R and 0.02/0.04/0.08.
FRONTOLAT_RANDOM_SEEDS = (14578, 11038, 11, 6527)
# Six headings at the room centre (far from walls). yaw 2.20 from (−2,0)
# is a wall-loom, not this gate.
OPEN_NOOBJ_YAWS = tuple(i * (2.0 * np.pi / 6.0) for i in range(6))
WALL_LOOM_YAW = 2.20


def _loom_start_distance_m() -> float:
    """Far enough that a 0.58 m disc starts at ≤5° (flyvis loom() start)."""
    return start_distance_for_angular_size(LOOM_START_ANG_MAX_DEG)


def frontolateral_gate_cases() -> list:
    """Collision-course looms with design |az| in 20–70° (both sides, several l/v).

    Every official loom starts ≤5° and is present from prewarm so onset
    of a 20–30° disc cannot drive the gate.
    """
    dist = _loom_start_distance_m()
    cases = []
    for az, lv in (
        (35.0, 0.04),
        (-35.0, 0.04),
        (55.0, 0.04),
        (-55.0, 0.04),
        (45.0, 0.08),
        (-45.0, 0.08),
        (60.0, 0.08),
        (-60.0, 0.08),
        (45.0, 0.02),
        (-45.0, 0.02),
        (30.0, 0.04),
        (-30.0, 0.04),
    ):
        spec = collision_course_object(az, dist, lv)
        side = spec["threat_side"]
        tag = f"fl_{side}_az{int(abs(az))}_lv{lv:g}_d{dist:g}"
        spec["tag"] = tag
        spec["band"] = "frontolat"
        spec["present_at_prewarm"] = True
        spec["stride"] = 1
        cases.append(spec)
    return cases


def random_frontolateral_cases(seeds=FRONTOLAT_RANDOM_SEEDS) -> list:
    dist = _loom_start_distance_m()
    cases = []
    for seed in seeds:
        rng = np.random.default_rng(int(seed))
        side = "L" if int(rng.integers(0, 2)) == 0 else "R"
        az = float(rng.uniform(25.0, 65.0)) * (1.0 if side == "L" else -1.0)
        lv = float(rng.choice([0.02, 0.04, 0.08]))
        spec = collision_course_object(az, dist, lv)
        spec["tag"] = f"fl_seed{int(seed)}"
        spec["band"] = "frontolat"
        spec["seed"] = int(seed)
        spec["present_at_prewarm"] = True
        spec["stride"] = 1
        cases.append(spec)
    return cases


def unseen_frontolat_cases() -> list:
    """Fixed az/l/v grid matching the box unseen audit (start ≤5°)."""
    dist = _loom_start_distance_m()
    cases = []
    for az in (25.0, 40.0, 50.0, 65.0, -25.0, -40.0, -50.0, -65.0):
        for lv in (0.05, 0.03):
            spec = collision_course_object(az, dist, lv)
            side = spec["threat_side"]
            spec["tag"] = f"unseen_{side}_az{int(abs(az))}_lv{lv:g}"
            spec["band"] = "unseen"
            spec["present_at_prewarm"] = True
            spec["stride"] = 1
            cases.append(spec)
    return cases


def static_onset_gate_cases() -> list:
    """Report-only: a 30–40° disc that appears and stays still.

    Abrupt large dark onset/dimming can drive GF; fires are reported,
    not gated.
    """
    cases = []
    for az, ang in ((40.0, 34.0), (-40.0, 34.0), (30.0, 34.0), (-30.0, 34.0)):
        dist = start_distance_for_angular_size(ang)
        spec = collision_course_object(az, dist, 0.04)
        spec["loom_vel"] = (0.0, 0.0, 0.0)
        spec["tag"] = f"static_{'L' if az > 0 else 'R'}_az{int(abs(az))}"
        spec["band"] = "static_onset"
        spec["appear_after_prewarm"] = True
        spec["present_at_prewarm"] = False
        spec["stride"] = 1
        cases.append(spec)
    return cases


def receding_gate_cases() -> list:
    """A receding disc (az 40°, start 1.5 m) must not fire.

    Object is already present during prewarm (no DC-init) so the large
    disc is not an onset. LPLC2/GF are expansion-selective.
    """
    cases = []
    for az in (40.0, -40.0):
        spec = collision_course_object(az, 1.5, 0.04)
        vx, vy, vz = spec["loom_vel"]
        spec["loom_vel"] = (-float(vx), -float(vy), -float(vz))
        spec["tag"] = f"recede_{'L' if az > 0 else 'R'}_az{int(abs(az))}"
        spec["band"] = "recede"
        spec["present_at_prewarm"] = True
        spec["appear_after_prewarm"] = False
        spec["stride"] = 1
        cases.append(spec)
    return cases


def frontal_gate_cases() -> list:
    """Pure frontal (|az| < 20°): must fire; both-GF climb counts."""
    dist = _loom_start_distance_m()
    cases = []
    for az, lv in ((0.0, 0.04), (5.0, 0.04), (12.0, 0.04)):
        spec = collision_course_object(az, dist, lv)
        spec["tag"] = f"front_az{int(az)}_lv{lv:g}"
        spec["band"] = "frontal"
        spec["present_at_prewarm"] = True
        spec["stride"] = 1
        cases.append(spec)
    return cases


def loom_horizon_steps(distance_m: float, lv_s: float, extra: int = 28, dt: float = 0.02) -> int:
    speed = 0.58 / max(float(lv_s), 1e-6)
    ttc = float(distance_m) / max(speed, 1e-6)
    return int(max(40, np.ceil(ttc / dt) + extra))


def loom_variants(side: str) -> list:
    """Offset / speed / stride robustness cases (box r6 controls)."""
    base = lateral_loom_object(side)
    sg = 1.0 if side == "L" else -1.0
    sx, sy, sz = base["loom_start"]
    vx, vy, vz = base["loom_vel"]
    return [
        {
            "tag": "far1.4x",
            "loom_start": (sx * 1.4, sy * 1.4, sz),
            "loom_vel": (vx, vy, vz),
            "stride": None,
        },
        {
            "tag": "slow0.7x",
            "loom_start": (sx, sy, sz),
            "loom_vel": (vx * 0.7, vy * 0.7, vz),
            "stride": None,
        },
        {
            "tag": "offset_x+0.8",
            "loom_start": (sx + 0.8, sg * abs(sy), sz),
            "loom_vel": (vx, vy, vz),
            "stride": None,
        },
        {
            "tag": "stride_hold",
            "loom_start": (sx, sy, sz),
            "loom_vel": (vx, vy, vz),
            "stride": 2,
        },
    ]


def headon_wall_pose() -> dict:
    """Frontal approach toward the +X wall (x = +5 m)."""
    return {
        "pose": (2.4, 0.0, 1.5),
        "yaw": 0.0,
        "vel": (0.50, 0.0, 0.0),
    }


def open_space_pose() -> dict:
    """Forward flight in open space: start west, heading +X, ≥300 steps."""
    return {
        "pose": (-3.2, 0.0, 1.5),
        "yaw": 0.0,
        "vel": (0.45, 0.0, 0.0),
        "max_steps": 320,
    }


def _eye_pack(env):
    left_img, right_img = env.get_eye_images()
    side = getattr(env, "last_side_images", (None, None))
    return left_img, right_img, side[0], side[1]


def _show_loom(env, virtual_xyz, fly_xyz=None):
    fly = env.get_fly_position() if fly_xyz is None else np.asarray(fly_xyz, dtype=float)
    vis, rad = visible_loom_placement(virtual_xyz, fly, radius=LOOM_RADIUS_M)
    env.set_loom_object(vis, radius=rad)


def _run_scenario(
    name: str,
    env,
    brain,
    ctl: DNFlightController,
    n_steps: int,
    pose=None,
    yaw: float = 0.0,
    vel=None,
    threat_side=None,
    prewarm=None,
    loom_start=None,
    loom_vel=None,
    loom_band=None,
    present_at_prewarm: bool = True,
    appear_after_prewarm: bool = False,
) -> dict:
    env.reset(seed=0)
    if pose is not None:
        env.set_pose(pose, yaw=yaw, vel=vel)
    env.hide_loom_object()
    brain.reset()
    ctl.reset()
    ctl.bind_hover(env.hover_action)

    traj = []
    headings = []
    dnp01 = []
    rises = []
    ms_brain = []
    ms_flyvis = []
    ms_flyvis_frame = []
    ms_total = []
    ms_loop = []
    wall = 0
    yaw_terms = []
    rise_geometry = []
    rise_onset = None
    geo_lock = None
    geo_fire = None
    geo_at_fire_visible = False
    n_esc_seen = 0
    distances = []
    t_wall = time.perf_counter()
    loom_pos = None if loom_start is None else np.asarray(loom_start, dtype=float).copy()
    loom_v = None if loom_vel is None else np.asarray(loom_vel, dtype=float)
    start_ang = None
    if loom_pos is not None and pose is not None:
        fly0 = np.asarray(pose, dtype=float)
        start_ang = angular_size_deg(float(np.hypot(loom_pos[0] - fly0[0], loom_pos[1] - fly0[1])))
    # Far ≤5° discs are hidden; photoreceptor rest is τ→∞-initialised to
    # that no-object frame. A large receding disc stays visible and we
    # skip DC-init so it is not grayed out and then re-appears as onset.
    show_now = loom_pos is not None and dnp01_baseline_shows_loom(
        present_at_prewarm=present_at_prewarm,
        appear_after_prewarm=appear_after_prewarm,
        start_ang_deg=start_ang,
    )
    if show_now:
        _show_loom(env, loom_pos)
    left_img, right_img, left_side, right_side = _eye_pack(env)
    n_pre = int(ctl.params.prewarm_steps if prewarm is None else prewarm)
    warmed = brain.prewarm(
        left_img,
        right_img,
        env.get_fly_angular_velocity(),
        n=n_pre,
        left_side_img=left_side,
        right_side_img=right_side,
        dc_init=not show_now,
    )
    if warmed is not None:
        ctl.capture_baseline(warmed.readout)
    if loom_pos is not None and not show_now:
        _show_loom(env, loom_pos)
    left_img, right_img, left_side, right_side = _eye_pack(env)
    for step in range(n_steps):
        t_loop = time.perf_counter()
        if loom_pos is not None and loom_v is not None:
            # Do not teleport/hide: sudden disappearance of a large dark
            # object is itself a loom. Virtual object continues past/away;
            # the in-room sphere matches its visual angle.
            loom_pos = loom_pos + loom_v * float(env.control_dt)
            _show_loom(env, loom_pos)
        res = brain.step(
            left_img,
            right_img,
            env.get_fly_angular_velocity(),
            left_side_img=left_side,
            right_side_img=right_side,
        )
        action = ctl.compute(res.readout)
        left_img, right_img, info = env.step(action, wing_strokes=ctl.last_wing_strokes)
        left_side, right_side = getattr(env, "last_side_images", (None, None))
        ms_loop.append(1000.0 * (time.perf_counter() - t_loop))
        pos = np.asarray(info["position"], dtype=float)
        heading = heading_from_quaternion(info["orientation"])
        traj.append(pos)
        headings.append(heading)
        dnp01.append(max(res.readout.dnp01_L, res.readout.dnp01_R))
        rise_now = float(ctl.last_rise)
        rises.append(rise_now)
        ms_brain.append(res.ms.get("brain", 0.0))
        ms_flyvis.append(res.ms.get("flyvis", 0.0))
        if float(res.ms.get("flyvis_ran", 1.0)) >= 0.5:
            ms_flyvis_frame.append(res.ms.get("flyvis", 0.0))
        ms_total.append(res.ms.get("total", 0.0))
        terms = dict(ctl.last_yaw_terms)
        terms["x"] = float(pos[0])
        terms["y"] = float(pos[1])
        terms["near_wall"] = 1.0 if max(abs(pos[0]), abs(pos[1])) > 3.2 else 0.0
        geo = loom_geometry(loom_pos, pos, heading) if loom_pos is not None else None
        behind = (
            object_behind_fly(loom_pos, pos, heading)
            if loom_pos is not None
            else True
        )
        visible_now = geo is not None and not behind
        distances.append(float(geo["distance_m"]) if geo is not None else float("nan"))
        if geo is not None:
            terms["azimuth_deg"] = float(geo["azimuth_deg"])
            terms["angular_size_deg"] = float(geo["angular_size_deg"])
            terms["visible"] = 1.0 if visible_now else 0.0
            if rise_onset is None and rise_now >= 0.08 and visible_now:
                rise_onset = {"step": int(step), **{k: geo[k] for k in geo}}
            if geo_lock is None and ctl.last_lock_step is not None and visible_now:
                geo_lock = {
                    "step": int(step),
                    "threat": ctl._loom_threat,
                    **{k: geo[k] for k in geo},
                }
            if ctl.brain_escape_count > n_esc_seen:
                # True geometry at fire — never last-visible (that reported
                # 85° after the object was already gone).
                geo_fire = {"step": int(step), **{k: geo[k] for k in geo}}
                geo_at_fire_visible = bool(visible_now)
                n_esc_seen = int(ctl.brain_escape_count)
            if rise_now >= 0.08:
                rise_L, rise_R = ctl._dnp01_rises(res.readout)
                rise_geometry.append(
                    {
                        "step": int(step),
                        "azimuth_deg": float(geo["azimuth_deg"]),
                        "angular_size_deg": float(geo["angular_size_deg"]),
                        "distance_m": float(geo["distance_m"]),
                        "class": geo["class"],
                        "rise": float(rise_now),
                        "rise_L": float(rise_L),
                        "rise_R": float(rise_R),
                        "lc4_L": float(res.readout.lc4_L),
                        "lc4_R": float(res.readout.lc4_R),
                        "lplc2_L": float(res.readout.lplc2_L),
                        "lplc2_R": float(res.readout.lplc2_R),
                        "dnp03_L": float(res.readout.dnp03_L),
                        "dnp03_R": float(res.readout.dnp03_R),
                        "lat": float(ctl.last_lat_integrated),
                        "threat": ctl._loom_threat or "",
                        "course_frozen": float(terms.get("course_frozen", 0.0)),
                        "bilateral_brake": float(terms.get("bilateral_brake", 0.0)),
                        "yaw": float(action[4]),
                        "forward": float(action[0]),
                        "visible": bool(visible_now),
                    }
                )
        elif ctl.brain_escape_count > n_esc_seen:
            geo_fire = None
            geo_at_fire_visible = False
            n_esc_seen = int(ctl.brain_escape_count)
        yaw_terms.append(terms)
        if env.check_collision():
            wall += 1
            env.recover_from_collision()
        if (step + 1) % 25 == 0:
            print(
                f"  {name} {step+1}/{n_steps} pos=({pos[0]:.2f},{pos[1]:.2f}) "
                f"esc={ctl.brain_escape_count} wall={wall} "
                f"ms={res.ms.get('total', 0):.1f}",
                flush=True,
            )
    wall_s = time.perf_counter() - t_wall
    traj = np.asarray(traj, dtype=float) if traj else np.zeros((0, 3))
    headings_arr = np.asarray(headings, dtype=float)
    if threat_side:
        if not any(not getattr(e, "engineered", False) for e in ctl.escape_events):
            dir_ok = 0.0
        else:
            dir_ok = direction_correct_frac(headings_arr, ctl.escape_events, threat_side)
    else:
        dir_ok = float("nan")
    brakes = [float(t.get("loom_brake", 0.0)) for t in yaw_terms]
    braked = bool(brakes) and float(np.max(brakes)) >= 0.35
    az_on = None if rise_onset is None else float(rise_onset["azimuth_deg"])
    geo_ok = (
        geometry_aware_frac(
            headings_arr,
            ctl.escape_events,
            threat_side,
            az_on,
            escaped=int(ctl.brain_escape_count) > 0,
            braked=braked,
        )
        if threat_side
        else float("nan")
    )
    ca_step = closest_approach_step(distances)
    fire_step = None if geo_fire is None else int(geo_fire["step"])
    if fire_step is None and ctl.escape_events:
        fire_step = int(ctl.escape_events[0].step)
    d_fire_ca = fire_minus_closest(fire_step, ca_step)
    va_ok = visible_approaching_ok(
        fire_step, ca_step, visible=bool(geo_at_fire_visible)
    )
    if threat_side and not va_ok:
        # After closest approach, or fire with no visible object = fail.
        dir_ok = 0.0
        geo_ok = 0.0
    band = loom_band or onset_azimuth_band(az_on)
    if band == "frontal":
        fired = int(ctl.brain_escape_count) > 0
        into = False
        if (
            fired
            and threat_side in ("L", "R")
            and az_on is not None
            and abs(float(az_on)) >= 8.0
            and headings_arr.size
        ):
            into = escape_turned_into(
                headings_arr, int(ctl.escape_events[0].step), threat_side
            )
        # Both-GF takeoff counts as fired even a few steps after geometric
        # CA (the disc already fills the field). Turning *into* still fails.
        front_ok = 1.0 if fired and not into else 0.0
        dir_ok = front_ok
        geo_ok = front_ok
    pre_toward = False
    if threat_side and rise_onset is not None and headings_arr.size:
        i0 = int(np.clip(rise_onset["step"], 0, headings_arr.size - 1))
        i1 = int(ctl.escape_events[0].step) if ctl.escape_events else min(i0 + 8, headings_arr.size - 1)
        i1 = int(np.clip(i1, 0, headings_arr.size - 1))
        dhead = wrap_heading_delta(float(headings_arr[i1]), float(headings_arr[i0]))
        if threat_side == "L":
            pre_toward = dhead > 0.02
        elif threat_side == "R":
            pre_toward = dhead < -0.02
    dx, dy = xy_extent(traj)
    min_center = min_center_distance_m(distances)
    min_surface = min_surface_clearance_m(distances)
    az_series = [t.get("azimuth_deg") for t in yaw_terms]
    avoid = flight_avoidance_ok(
        min_center_m=min_center,
        headings=headings_arr,
        threat_side=threat_side or "",
        azimuths=az_series,
        ca_step=ca_step,
    )
    yaw_away = yaw_term_away_integrals(
        yaw_terms,
        threat_side or "",
        end_step=ca_step if ca_step is not None else None,
    )
    gf_approaching = int(ctl.brain_escape_count) > 0 and bool(va_ok)
    mode = avoidance_mode(
        avoidance_ok=bool(avoid["ok"]),
        gf_fired_approaching=gf_approaching,
    )
    yaw_summary = _yaw_term_summary(yaw_terms)
    events = []
    for e in ctl.escape_events:
        rec = {
            "step": e.step,
            "dnp01": e.dnp01,
            "threat_side": e.threat_side,
            "yaw": e.yaw,
            "engineered": e.engineered,
        }
        g = next((r for r in rise_geometry if int(r["step"]) == int(e.step)), None)
        if g is not None:
            rec["azimuth_deg"] = g["azimuth_deg"]
            rec["angular_size_deg"] = g["angular_size_deg"]
            rec["loom_class"] = g["class"]
            rec["rise_L"] = g.get("rise_L")
            rec["rise_R"] = g.get("rise_R")
            rec["lc4_L"] = g.get("lc4_L")
            rec["lc4_R"] = g.get("lc4_R")
            rec["lplc2_L"] = g.get("lplc2_L")
            rec["lplc2_R"] = g.get("lplc2_R")
            rec["dnp03_L"] = g.get("dnp03_L")
            rec["dnp03_R"] = g.get("dnp03_R")
            rec["bilateral_brake"] = g.get("bilateral_brake")
            rec["yaw"] = g.get("yaw")
            rec["forward"] = g.get("forward")
        events.append(rec)
    fire_diag = None
    if events:
        fire_diag = {k: events[0].get(k) for k in (
            "threat_side", "azimuth_deg", "rise_L", "rise_R",
            "lc4_L", "lc4_R", "lplc2_L", "lplc2_R", "dnp03_L", "dnp03_R",
            "bilateral_brake", "yaw", "forward",
        )}
    wall_dist = float("nan")
    wall_ttc = float("nan")
    if traj.size and name.startswith("ctrl_noobj_"):
        last = traj[-1]
        wall_dist = float(min(5.0 - abs(last[0]), 5.0 - abs(last[1])))
        if fire_step is not None and 0 <= int(fire_step) < len(traj):
            fp = traj[int(fire_step)]
            wall_dist = float(min(5.0 - abs(fp[0]), 5.0 - abs(fp[1])))
            if vel is not None:
                speed = float(np.linalg.norm(np.asarray(vel, dtype=float)[:2]))
                wall_ttc = wall_dist / max(speed, 1e-6)
    return {
        "name": name,
        "steps": int(n_steps),
        "wall_contacts": int(wall),
        "walls_per_100": walls_per_100(wall, n_steps),
        "brain_escapes": int(ctl.brain_escape_count),
        "engineered_escapes": int(ctl.engineered_escape_count),
        "dnp01_latch_frac": float(ctl.dnp01_latch_frac),
        "escape_direction_correct": None if np.isnan(dir_ok) else float(dir_ok),
        "geometry_aware_correct": None if np.isnan(geo_ok) else float(geo_ok),
        "loom_class": None if rise_onset is None else rise_onset["class"],
        "loom_band": band,
        "azimuth_onset_deg": az_on,
        "angular_size_onset_deg": None if rise_onset is None else float(rise_onset["angular_size_deg"]),
        "azimuth_lock_deg": None if geo_lock is None else float(geo_lock["azimuth_deg"]),
        "angular_size_lock_deg": None if geo_lock is None else float(geo_lock["angular_size_deg"]),
        "azimuth_fire_deg": None if geo_fire is None else float(geo_fire["azimuth_deg"]),
        "angular_size_fire_deg": None if geo_fire is None else float(geo_fire["angular_size_deg"]),
        "closest_approach_step": ca_step,
        "fire_step": fire_step,
        "fire_minus_closest": d_fire_ca,
        "visible_at_fire": bool(geo_at_fire_visible),
        "approaching_at_fire": bool(va_ok),
        "threat_locked": None if geo_lock is None else geo_lock.get("threat"),
        "fire_diag": fire_diag,
        "wall_dist_m": None if np.isnan(wall_dist) else float(wall_dist),
        "wall_ttc_s": None if np.isnan(wall_ttc) else float(wall_ttc),
        "pre_escape_yaw_toward": bool(pre_toward),
        "braked": bool(braked),
        "false_escapes": int(ctl.brain_escape_count) if name == "open_space" else 0,
        "ms_per_step_brain": float(np.median(ms_brain)) if ms_brain else 0.0,
        "ms_per_step_flyvis": float(np.median(ms_flyvis)) if ms_flyvis else 0.0,
        "ms_per_step_brain_flyvis": float(np.median(ms_total)) if ms_total else 0.0,
        "ms_per_step_brain_flyvis_mean": float(np.mean(ms_total)) if ms_total else 0.0,
        "ms_per_step_full_loop": float(np.median(ms_loop)) if ms_loop else 0.0,
        "ms_per_step_full_loop_mean": float(np.mean(ms_loop)) if ms_loop else 0.0,
        "ms_flyvis_frame_median": float(np.median(ms_flyvis_frame)) if ms_flyvis_frame else 0.0,
        "ms_flyvis_frame_mean": float(np.mean(ms_flyvis_frame)) if ms_flyvis_frame else 0.0,
        "room_coverage": coverage(traj),
        "dx": dx,
        "dy": dy,
        "wall_seconds": float(wall_s),
        "trajectory": traj,
        "headings": headings_arr,
        "dnp01": np.asarray(dnp01, dtype=float),
        "dnp01_rise": np.asarray(rises, dtype=float),
        "dnp01_rise_max": float(np.max(rises)) if rises else 0.0,
        "min_center_m": min_center,
        "min_surface_m": min_surface,
        "peak_yaw_rate_rad_s": peak_yaw_rate_rad_s(headings_arr, float(env.control_dt)),
        "mean_abs_yaw_rate_rad_s": mean_abs_yaw_rate_rad_s(headings_arr, float(env.control_dt)),
        "control_dt_s": float(env.control_dt),
        "net_heading_rad": float(avoid["net_heading_rad"]),
        "heading_away": bool(avoid["heading_away"]),
        "passed_far_side": bool(avoid["passed_far_side"]),
        "flight_avoidance_ok": bool(avoid["ok"]),
        "avoidance_mode": mode,
        "gf_fired_approaching": bool(gf_approaching),
        "yaw_driver": yaw_away.get("driver"),
        "yaw_away_integrals": yaw_away.get("integrals"),
        "yaw_terms": yaw_summary,
        "would_fire": int(ctl.would_fire_count),
        "rise_geometry": rise_geometry,
        "escape_events": events,
        # Raw series for the post-run behavioural / feasibility scoring.
        "azimuth_series": [None if a is None else float(a) for a in az_series],
        "loom_speed_mps": None if loom_v is None else float(np.linalg.norm(loom_v[:2])),
        "gf_step": next((int(e.step) for e in ctl.escape_events if not getattr(e, "engineered", False)), None),
    }


def _escape_side_counts(sc: dict) -> dict:
    sides = {"L": 0, "R": 0, "": 0}
    for ev in sc.get("escape_events") or []:
        if ev.get("engineered"):
            continue
        key = ev.get("threat_side") or ""
        sides[key] = sides.get(key, 0) + 1
    return sides


def _escape_one_sided(sc: dict) -> bool:
    sides = _escape_side_counts(sc)
    n = int(sides.get("L", 0) + sides.get("R", 0))
    return n >= 3 and min(int(sides.get("L", 0)), int(sides.get("R", 0))) == 0


def _noobj_heading_summary(scenarios: dict) -> dict:
    per = {}
    fires = 0
    for name, sc in scenarios.items():
        if not name.startswith("ctrl_noobj_"):
            continue
        n = int(sc.get("brain_escapes") or 0)
        per[name] = {
            "escapes": n,
            "would_fire": sc.get("would_fire"),
            "rise_max": sc.get("dnp01_rise_max"),
            "steps": sc.get("steps"),
            "wall_dist_m": sc.get("wall_dist_m"),
            "wall_ttc_s": sc.get("wall_ttc_s"),
            "net_heading_rad": sc.get("net_heading_rad"),
        }
        fires += n
    return {
        "per_scenario": per,
        "n": int(len(per)),
        "fires": int(fires),
        "pass": bool(per) and fires == 0,
    }


def _yaw_control_stats(scenarios: dict) -> dict:
    """No-object yaw-rate wander vs frontolat. Control for random course."""
    noobj_h, noobj_r, noobj_signed = [], [], []
    for name, sc in scenarios.items():
        if not (name.startswith("ctrl_noobj_") or name == "open_space"):
            continue
        h = sc.get("net_heading_rad")
        r = sc.get("mean_abs_yaw_rate_rad_s")
        if h is not None:
            noobj_h.append(abs(float(h)))
            noobj_signed.append(float(h))
        if r is not None:
            noobj_r.append(abs(float(r)))
    loom_h, loom_r, n_ex = [], [], 0
    thresh = None
    if noobj_r:
        thresh = float(np.mean(noobj_r)) + 2.0 * float(np.std(noobj_r))
    for name, sc in scenarios.items():
        if not _is_frontolat(name):
            continue
        if not (sc.get("heading_away") or sc.get("flight_avoidance_ok")):
            continue
        h = sc.get("net_heading_rad")
        r = sc.get("mean_abs_yaw_rate_rad_s")
        if h is not None:
            loom_h.append(abs(float(h)))
        if r is not None:
            loom_r.append(abs(float(r)))
            if thresh is not None and abs(float(r)) > thresh:
                n_ex += 1
    no_mean = float(np.mean(noobj_h)) if noobj_h else None
    no_std = float(np.std(noobj_h)) if noobj_h else None
    rate_mean = float(np.mean(noobj_r)) if noobj_r else None
    rate_std = float(np.std(noobj_r)) if noobj_r else None
    signed_mean = float(np.mean(noobj_signed)) if noobj_signed else None
    loom_mean = float(np.mean(loom_h)) if loom_h else None
    loom_rate = float(np.mean(loom_r)) if loom_r else None
    frac = (float(n_ex) / float(len(loom_r))) if loom_r else None
    above = None if thresh is None or loom_rate is None else bool(loom_rate > thresh)
    return {
        "noobj_abs_net_heading": {
            "n": int(len(noobj_h)),
            "mean": no_mean,
            "std": no_std,
            "max": float(max(noobj_h)) if noobj_h else None,
            "signed_mean": signed_mean,
        },
        "noobj_abs_yaw_rate": {
            "n": int(len(noobj_r)),
            "mean": rate_mean,
            "std": rate_std,
            "mean_plus_2std": thresh,
        },
        "frontolat_away_abs_net_heading": {
            "n": int(len(loom_h)),
            "mean": loom_mean,
            "std": float(np.std(loom_h)) if loom_h else None,
        },
        "frontolat_away_abs_yaw_rate": {
            "n": int(len(loom_r)),
            "mean": loom_rate,
            "std": float(np.std(loom_r)) if loom_r else None,
        },
        "fraction_exceeding_noobj_rate_mean_plus_2std": frac,
        "n_exceeding_noobj_wander": int(n_ex),
        "frontolat_mean_exceeds_noobj_mean_plus_2std": above,
    }


def _is_loom_variant(name: str) -> bool:
    return name.startswith("side_loom_") or name.startswith("loomL_") or name.startswith("loomR_")


def _is_frontolat(name: str) -> bool:
    return name.startswith("fl_")


def _is_frontal(name: str) -> bool:
    return name.startswith("front_")


def _is_static_onset(name: str) -> bool:
    return name.startswith("static_")


def _is_recede(name: str) -> bool:
    return name.startswith("recede_")


def _is_unseen(name: str) -> bool:
    return name.startswith("unseen_")


def _case_ok(sc: dict, *, require_away: bool) -> float:
    """Legacy fire-while-approaching score (frontal / report-only extras)."""
    if int(sc.get("brain_escapes") or 0) <= 0:
        return 0.0
    if not sc.get("approaching_at_fire") and not sc.get("visible_at_fire"):
        return 0.0
    if sc.get("approaching_at_fire") is False:
        return 0.0
    if require_away:
        d = sc.get("escape_direction_correct")
        if d is None or float(d) < 0.8:
            return 0.0
    return 1.0


def _true_side(name: str, sc: dict) -> Optional[str]:
    if sc.get("true_side") in ("L", "R"):
        return sc["true_side"]
    if "_L_" in name or name.startswith("fl_L") or name.startswith("unseen_L"):
        return "L"
    if "_R_" in name or name.startswith("fl_R") or name.startswith("unseen_R"):
        return "R"
    az = sc.get("az_design_deg")
    if az is not None:
        if float(az) > 1e-6:
            return "L"
        if float(az) < -1e-6:
            return "R"
    return None


def _flight_ok(sc: dict) -> float:
    """1 if the fly missed and turned the object to the far side."""
    if sc.get("flight_avoidance_ok") is not None:
        return 1.0 if sc.get("flight_avoidance_ok") else 0.0
    return 0.0


def _frontolat_gate(scenarios: dict) -> dict:
    """Official flight gate: miss + far-side / heading-away, ≥80%. GF optional."""
    per = {}
    details = {}
    scores = []
    n_gf = 0
    n_saccade = 0
    drivers = {}
    for name, sc in scenarios.items():
        if not _is_frontolat(name):
            continue
        ok = _flight_ok(sc)
        mode = sc.get("avoidance_mode") or "none"
        if mode == "gf_escape":
            n_gf += 1
        elif mode == "pre_gf_saccade":
            n_saccade += 1
        drv = sc.get("yaw_driver")
        if drv:
            drivers[drv] = int(drivers.get(drv, 0)) + 1
        per[name] = ok
        details[name] = {
            "ok": ok,
            "azimuth_onset_deg": sc.get("azimuth_onset_deg"),
            "loom_band": sc.get("loom_band") or onset_azimuth_band(sc.get("azimuth_onset_deg")),
            "angular_size_fire_deg": sc.get("angular_size_fire_deg"),
            "azimuth_fire_deg": sc.get("azimuth_fire_deg"),
            "fire_minus_closest": sc.get("fire_minus_closest"),
            "visible_at_fire": sc.get("visible_at_fire"),
            "approaching_at_fire": sc.get("approaching_at_fire"),
            "escape_direction_correct": sc.get("escape_direction_correct"),
            "brain_escapes": sc.get("brain_escapes"),
            "threat_locked": sc.get("threat_locked"),
            "dnp01_rise_max": sc.get("dnp01_rise_max"),
            "true_side": _true_side(name, sc),
            "fire_diag": sc.get("fire_diag"),
            "min_center_m": sc.get("min_center_m"),
            "min_surface_m": sc.get("min_surface_m"),
            "peak_yaw_rate_rad_s": sc.get("peak_yaw_rate_rad_s"),
            "mean_abs_yaw_rate_rad_s": sc.get("mean_abs_yaw_rate_rad_s"),
            "net_heading_rad": sc.get("net_heading_rad"),
            "heading_away": sc.get("heading_away"),
            "passed_far_side": sc.get("passed_far_side"),
            "avoidance_mode": mode,
            "yaw_driver": drv,
            "yaw_away_integrals": sc.get("yaw_away_integrals"),
        }
        scores.append(ok)
    mean = float(np.mean(scores)) if scores else None
    return {
        "per_scenario": per,
        "details": details,
        "mean": mean,
        "n": int(len(scores)),
        "n_gf_escape": int(n_gf),
        "n_pre_gf_saccade": int(n_saccade),
        "yaw_drivers": drivers,
        "pass": bool(scores) and float(mean) >= 0.80,
    }


def _frontal_gate(scenarios: dict) -> dict:
    """Pure frontal: must fire, not turn into (either side / brake ok)."""
    per = {}
    details = {}
    scores = []
    for name, sc in scenarios.items():
        if not _is_frontal(name):
            continue
        # Item 5: both-GF DNp01 cross is the escape, not brake-only / va_ok.
        ok = 1.0 if int(sc.get("brain_escapes") or 0) > 0 else 0.0
        per[name] = ok
        details[name] = {
            "ok": ok,
            "azimuth_onset_deg": sc.get("azimuth_onset_deg"),
            "angular_size_fire_deg": sc.get("angular_size_fire_deg"),
            "fire_minus_closest": sc.get("fire_minus_closest"),
            "brain_escapes": sc.get("brain_escapes"),
            "braked": sc.get("braked"),
            "pre_escape_yaw_toward": sc.get("pre_escape_yaw_toward"),
            "dnp01_rise_max": sc.get("dnp01_rise_max"),
        }
        scores.append(ok)
    mean = float(np.mean(scores)) if scores else None
    return {
        "per_scenario": per,
        "details": details,
        "mean": mean,
        "n": int(len(scores)),
        "pass": bool(scores) and float(mean) >= 1.0,
    }


def _zero_fire_gate(scenarios: dict, pred) -> dict:
    """Pass only if every matching scenario has zero brain escapes."""
    per = {}
    details = {}
    scores = []
    for name, sc in scenarios.items():
        if not pred(name):
            continue
        n = int(sc.get("brain_escapes") or 0)
        ok = 1.0 if n == 0 else 0.0
        per[name] = ok
        details[name] = {
            "ok": ok,
            "brain_escapes": n,
            "dnp01_rise_max": sc.get("dnp01_rise_max"),
            "angular_size_fire_deg": sc.get("angular_size_fire_deg"),
            "azimuth_onset_deg": sc.get("azimuth_onset_deg"),
        }
        scores.append(ok)
    mean = float(np.mean(scores)) if scores else None
    return {
        "per_scenario": per,
        "details": details,
        "mean": mean,
        "n": int(len(scores)),
        "pass": bool(scores) and float(mean) >= 1.0,
    }


def _unseen_gate(scenarios: dict) -> dict:
    """Report-only unseen-like frontolat grid (same flight scoring as official)."""
    per = {}
    details = {}
    scores = []
    for name, sc in scenarios.items():
        if not _is_unseen(name):
            continue
        ok = _flight_ok(sc)
        per[name] = ok
        details[name] = {
            "ok": ok,
            "azimuth_onset_deg": sc.get("azimuth_onset_deg"),
            "angular_size_fire_deg": sc.get("angular_size_fire_deg"),
            "fire_minus_closest": sc.get("fire_minus_closest"),
            "escape_direction_correct": sc.get("escape_direction_correct"),
            "brain_escapes": sc.get("brain_escapes"),
            "dnp01_rise_max": sc.get("dnp01_rise_max"),
            "true_side": _true_side(name, sc),
            "fire_diag": sc.get("fire_diag"),
            "min_center_m": sc.get("min_center_m"),
            "net_heading_rad": sc.get("net_heading_rad"),
            "avoidance_mode": sc.get("avoidance_mode"),
            "yaw_driver": sc.get("yaw_driver"),
        }
        scores.append(ok)
    mean = float(np.mean(scores)) if scores else None
    return {
        "per_scenario": per,
        "details": details,
        "mean": mean,
        "n": int(len(scores)),
        "pass": bool(scores) and float(mean) >= 0.80,
        "gate": False,
    }


def _measure_plant(env, params) -> dict:
    """Calibrated plant numbers for the feasibility model (no vision).

    Peak yaw rate: official unit DNa02 L−R step (calibrate_saccade_plant).
    Lateral accel: forward thrust + all spare lift (upper bound). Cruise:
    drag-limited speed at forward_bias. Constant fallback if the step fails.
    """
    rec = {"source": "constant", "peak_yaw_deg_s": float(PLANT_PEAK_YAW_DEG_S), "t90_s": None}
    try:
        from threedfly.control.saccade_plant import measure_unit_dn_step

        r = measure_unit_dn_step(env, side="L", params=params)
        rec.update({"source": "measured", "peak_yaw_deg_s": float(r["peak_yaw_deg_s"]), "t90_s": r["t90_s"]})
    except Exception as exc:  # pragma: no cover - diagnostic only
        rec["error"] = repr(exc)
    mass = float(env.model.body_subtreemass[env._fly_body_id])
    g = abs(float(env.model.opt.gravity[2]))
    a_lat = max_lateral_accel_mps2(float(env.thrust_max), float(env.lift_max), mass, g)
    f = float(params.forward_bias) * float(env.thrust_max)
    b, c = float(env.drag_linear), float(env.drag_quadratic)
    v_cruise = (-b + float(np.sqrt(b * b + 4.0 * c * f))) / (2.0 * c) if c > 0 else f / max(b, 1e-9)
    rec.update({
        "mass_kg": mass,
        "a_lat_max_mps2": float(a_lat),
        "cruise_speed_mps": float(v_cruise),
    })
    return rec


def _noobj_wander(scenarios: dict) -> Optional[dict]:
    """Yaw-rate wander from the official 6-heading no-object runs."""
    runs, dts = [], []
    for name, sc in scenarios.items():
        if not name.startswith("ctrl_noobj_"):
            continue
        h = sc.get("headings")
        if h is None or len(h) < 2:
            continue
        runs.append(np.asarray(h, dtype=float))
        dts.append(float(sc.get("control_dt_s") or 0.02))
    return wander_yaw_threshold(runs, dts)


def _speed_at(traj, step: Optional[int], dt: float) -> float:
    t = np.asarray(traj, dtype=float)
    if step is None or t.ndim != 2 or t.shape[0] < 2:
        return 0.0
    i = int(np.clip(step, 1, t.shape[0] - 1))
    return float(np.hypot(*(t[i, :2] - t[i - 1, :2]))) / max(float(dt), 1e-9)


def _score_behaviour(sc: dict, name: str, wander: dict, plant: dict) -> None:
    """Attach response-window behaviour + kinematic feasibility to a loom case."""
    side = _true_side(name, sc)
    dt = float(sc.get("control_dt_s") or 0.01)
    ca = sc.get("closest_approach_step")
    resp = behavioural_response(
        headings=sc.get("headings") if sc.get("headings") is not None else [],
        dt=dt,
        ca_step=ca,
        gf_step=sc.get("gf_step"),
        threat_side=side or "",
        wander_rad_s=float(wander["threshold_rad_s"]),
    )
    resp["above_peak_wander"] = bool(
        resp["peak_yaw_rate_rad_s"] > float(wander["peak_threshold_rad_s"])
    )
    dec = resp.get("decision_step")
    az_ser = sc.get("azimuth_series") or []
    az_dec = None
    if dec is not None and az_ser:
        a = az_ser[int(np.clip(dec, 0, len(az_ser) - 1))]
        az_dec = None if a is None else float(a)
    v_meas = _speed_at(sc.get("trajectory"), dec, dt)
    v0 = max(v_meas, float(plant["cruise_speed_mps"]))
    feas = avoidance_feasibility(
        decision_step=dec,
        ca_step=ca,
        dt=dt,
        approach_speed_mps=float(sc.get("loom_speed_mps") or 0.0),
        azimuth_decision_deg=az_dec,
        v0_mps=v0,
        a_lat_mps2=float(plant["a_lat_max_mps2"]),
        yaw_rate_max_rad_s=float(np.radians(plant["peak_yaw_deg_s"])),
    )
    feas["v0_measured_mps"] = float(v_meas)
    feas["azimuth_decision_deg"] = az_dec
    surf = sc.get("min_surface_m")
    feas["cleared"] = bool(surf is not None and float(surf) > MISS_SURFACE_M)
    sc["true_side"] = side
    sc["behaviour"] = resp
    sc["feasibility"] = feas


def _behaviour_gate(scenarios: dict, pred, *, gate: bool = True) -> dict:
    """Frontolat flight gate scored in the response window.

    success = on-time (GF fire or saccade onset before closest approach)
    AND peak |yaw rate| > no-object wander (mean + 2 sd) AND heading change
    over the 100 ms window away from the TRUE side. ≥80% to pass.
    Clearance (surface > 0.1 m) is required at ≥80% only on the subset
    that is kinematically feasible given the observed decision time.
    """
    details, scores = {}, []
    feas_names, clear_scores = [], []
    modes = {"gf": 0, "saccade": 0, "none": 0}
    n_toward = 0
    for name, sc in scenarios.items():
        if not pred(name) or "behaviour" not in sc:
            continue
        b, f = sc["behaviour"], sc["feasibility"]
        ok = 1.0 if b["ok"] else 0.0
        scores.append(ok)
        modes[b.get("decision_source") or "none"] = modes.get(b.get("decision_source") or "none", 0) + 1
        if b.get("toward"):
            n_toward += 1
        if f.get("feasible"):
            feas_names.append(name)
            clear_scores.append(1.0 if f.get("cleared") else 0.0)
        details[name] = {
            "ok": ok,
            "true_side": sc.get("true_side"),
            "decision_source": b.get("decision_source"),
            "decision_step": b.get("decision_step"),
            "ca_step": sc.get("closest_approach_step"),
            "on_time": b.get("on_time"),
            "peak_yaw_rate_rad_s": b.get("peak_yaw_rate_rad_s"),
            "above_wander": b.get("above_wander"),
            "above_peak_wander": b.get("above_peak_wander"),
            "window_heading_rad": b.get("window_heading_rad"),
            "away": b.get("away"),
            "toward": b.get("toward"),
            "min_surface_m": sc.get("min_surface_m"),
            "feasible": f.get("feasible"),
            "feasible_at_gf50": f.get("feasible_at_gf50"),
            "t_avail_s": f.get("t_avail_s"),
            "max_lateral_m": f.get("max_lateral_m"),
            "latest_decision_angle_deg": f.get("latest_decision_angle_deg"),
            "azimuth_decision_deg": f.get("azimuth_decision_deg"),
            "required_lead_s": f.get("required_lead_s"),
            "feasibility_label": f.get("label"),
            "approach_speed_mps": f.get("approach_speed_mps"),
            "cleared": f.get("cleared"),
            "lv_s": sc.get("lv_s"),
            "az_design_deg": sc.get("az_design_deg"),
        }
    mean = float(np.mean(scores)) if scores else None
    clear_mean = float(np.mean(clear_scores)) if clear_scores else None
    beh_pass = bool(scores) and float(mean) >= 0.80
    clear_pass = (not clear_scores) or float(clear_mean) >= 0.80
    return {
        "details": details,
        "mean": mean,
        "n": int(len(scores)),
        "n_ok": int(sum(scores)),
        "decision_sources": modes,
        "n_toward": int(n_toward),
        "behaviour_pass": bool(beh_pass),
        "feasible_subset": feas_names,
        "n_feasible": int(len(feas_names)),
        "clearance_mean_feasible": clear_mean,
        "clearance_pass": bool(clear_pass),
        "pass": bool(beh_pass and clear_pass),
        "gate": bool(gate),
    }


def _official_pass(summary: dict, params) -> dict:
    """Gates that still decide acceptance. Eccentric L/R and yaw-2.20 do not."""
    fl = summary.get("frontolat_behaviour_gate") or {}
    fr = summary.get("frontal_gate") or {}
    no = summary.get("ctrl_noobj_headings") or {}
    st = summary.get("static_onset_gate") or {}
    rc = summary.get("recede_gate") or {}
    brain_only = (not bool(getattr(params, "use_engineered_eye_laterality", False))) and (
        not bool(getattr(params, "use_engineered_looming_residual", False))
    )
    gates = {
        "frontolat": bool(fl.get("pass")),
        "frontal": bool(fr.get("pass")),
        "noobj_open": bool(no.get("pass")),
        "recede": bool(rc.get("pass")),
        "brain_only": brain_only,
    }
    return {
        "pass": all(gates.values()),
        "gates": gates,
        "report_only": {
            "static_onset": bool(st.get("pass")),
            "frontolat_surface_away": bool((summary.get("frontolat_gate") or {}).get("pass")),
            "unseen_behaviour": bool((summary.get("unseen_behaviour") or {}).get("pass")),
        },
    }


def _variant_direction(scenarios: dict) -> dict:
    """Report-only raw away-score for eccentric official L/R + far/slow/offset."""
    dirs = []
    per = {}
    for name, sc in scenarios.items():
        if not _is_loom_variant(name):
            continue
        d = sc.get("escape_direction_correct")
        if d is None:
            d = 0.0
        per[name] = d
        dirs.append(float(d))
    return {
        "per_scenario": per,
        "mean": float(np.mean(dirs)) if dirs else None,
        "n": int(len(dirs)),
        "pass": bool(dirs) and float(np.mean(dirs)) >= 0.80,
        "gate": False,
    }


def _variant_geometry_aware(scenarios: dict) -> dict:
    """Report-only geometry-aware score for eccentric laterals. Do not gate."""
    dirs = []
    per = {}
    details = {}
    for name, sc in scenarios.items():
        if not _is_loom_variant(name):
            continue
        d = sc.get("geometry_aware_correct")
        per[name] = d
        details[name] = {
            "raw": sc.get("escape_direction_correct"),
            "geometry_aware": d,
            "loom_class": sc.get("loom_class"),
            "azimuth_onset_deg": sc.get("azimuth_onset_deg"),
            "angular_size_onset_deg": sc.get("angular_size_onset_deg"),
            "azimuth_lock_deg": sc.get("azimuth_lock_deg"),
            "azimuth_fire_deg": sc.get("azimuth_fire_deg"),
            "angular_size_fire_deg": sc.get("angular_size_fire_deg"),
            "closest_approach_step": sc.get("closest_approach_step"),
            "fire_step": sc.get("fire_step"),
            "fire_minus_closest": sc.get("fire_minus_closest"),
            "visible_at_fire": sc.get("visible_at_fire"),
            "approaching_at_fire": sc.get("approaching_at_fire"),
            "brain_escapes": sc.get("brain_escapes"),
            "threat_locked": sc.get("threat_locked"),
            "pre_escape_yaw_toward": sc.get("pre_escape_yaw_toward"),
        }
        if d is not None:
            dirs.append(float(d))
    return {
        "per_scenario": per,
        "details": details,
        "mean": float(np.mean(dirs)) if dirs else None,
        "n": int(len(dirs)),
        "pass": bool(dirs) and float(np.mean(dirs)) >= 0.80,
        "gate": False,
    }


def _yaw_term_summary(rows: list) -> dict:
    """Mean |yaw| per source, all steps and near-wall subset."""
    keys = ("dna02", "dnp15", "hse", "dnp03", "vpn_loom", "eye", "escape_kick")
    if not rows:
        return {"all": {k: 0.0 for k in keys}, "near_wall": {k: 0.0 for k in keys}, "brain_share_near_wall": 0.0}

    def _mean_abs(subset, key):
        vals = [abs(float(r.get(key, 0.0))) for r in subset]
        return float(np.mean(vals)) if vals else 0.0

    near = [r for r in rows if r.get("near_wall", 0.0) >= 1.0]
    all_m = {k: _mean_abs(rows, k) for k in keys}
    near_m = {k: _mean_abs(near, k) for k in keys} if near else {k: 0.0 for k in keys}
    brain = ("dna02", "dnp15", "hse", "dnp03", "vpn_loom")
    bsum = sum(near_m[k] for k in brain)
    engineered = near_m["eye"]
    total = bsum + engineered + near_m["escape_kick"]
    return {
        "all": all_m,
        "near_wall": near_m,
        "near_wall_steps": int(len(near)),
        "brain_share_near_wall": float(bsum / total) if total > 1e-9 else 1.0,
        "eye_share_near_wall": float(engineered / total) if total > 1e-9 else 0.0,
    }


def _plot_trajectory(scenarios: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prefer = [
        "room",
        "open_space",
        "headon_wall",
        "side_loom_left",
        "side_loom_right",
        "front_az0_lv0.04",
        "fl_L_az35_lv0.04_d3.6",
        "fl_R_az35_lv0.04_d3.6",
        "ctrl_noobj_h0",
        "ctrl_wallloom_yaw220",
    ]
    names = [n for n in prefer if n in scenarios] or list(scenarios)
    fig, ax = plt.subplots(figsize=(7, 7))
    for name in names:
        sc = scenarios[name]
        t = sc["trajectory"]
        if t.size == 0:
            continue
        ax.plot(t[:, 0], t[:, 1], label=name, lw=1.2)
        ax.scatter(t[0, 0], t[0, 1], s=20)
    ax.set_xlim(-ROOM_XY, ROOM_XY)
    ax.set_ylim(-ROOM_XY, ROOM_XY)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Closed-loop trajectories")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_escapes(scenarios: dict, path: Path, delta_hz: float) -> None:
    import math

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prefer = [n for n in scenarios if n.startswith(("fl_", "front_", "side_loom_", "ctrl_noobj_", "ctrl_wallloom_", "room"))]
    names = prefer or list(scenarios)
    ncols = 4 if len(names) > 6 else 3 if len(names) > 4 else 1
    nrows = int(math.ceil(len(names) / float(ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 2.2 * nrows), sharex=False)
    flat = np.atleast_1d(axes).ravel()
    for i, name in enumerate(names):
        ax = flat[i]
        sc = scenarios[name]
        rise = sc.get("dnp01_rise")
        if rise is None or getattr(rise, "size", 0) == 0:
            rise = sc["dnp01"]
        ax.plot(rise, color="tab:red", lw=1.0)
        ax.axhline(delta_hz, color="gray", ls="--", lw=0.8)
        for ev in sc["escape_events"]:
            ax.axvline(ev["step"], color="tab:orange", alpha=0.5)
        ang = sc.get("angular_size_fire_deg")
        extra = f" ang={ang:.0f}°" if isinstance(ang, (int, float)) and ang is not None else ""
        ax.set_title(f"{name} esc={sc['brain_escapes']}{extra}", fontsize=8)
        ax.tick_params(labelsize=7)
    for j in range(len(names), len(flat)):
        flat[j].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv=None) -> int:
    from threedfly.brain.closed_loop import ClosedLoopBrain, core_stack_available
    from threedfly.brain.constants import connectome_available, flyvis_available
    from threedfly.sim import FlyEnvironment

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default=str(ROOT / "output" / "m2"))
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--quick", action="store_true", help="Short unofficial run")
    p.add_argument("--engineered-looming-residual", action="store_true")
    p.add_argument(
        "--engineered-eye-laterality",
        action="store_true",
        help="Opt-in pixel eye-mean threat/yaw (not the official brain-only path)",
    )
    args = p.parse_args(argv)

    from threedfly.brain.core_bundle import core_bundle_available

    if not flyvis_available() or not core_stack_available():
        print(
            "ERROR: closed-loop acceptance needs flyvis and either\n"
            "  data/full_connectome.npz  or  data/core_bundle.npz\n"
            "(export the bundle on the box: python scripts/export_core_bundle.py).\n"
            "Python 3.12: pip install -e '.[brain]'.",
            file=sys.stderr,
        )
        return 2
    if not connectome_available() and core_bundle_available():
        print("Using data/core_bundle.npz (full connectome absent)", flush=True)

    room_steps = 80 if args.quick else int(args.steps)
    side_steps = 40 if args.quick else max(160, min(400, args.steps // 5))
    # ≥300 open steps; start at x=-2 so cruise does not reach the +X wall.
    open_steps = 40 if args.quick else 320
    if args.quick:
        print("--quick: unofficial short movies", flush=True)

    params = FlightParams(
        use_engineered_looming_residual=args.engineered_looming_residual,
        use_engineered_eye_laterality=args.engineered_eye_laterality,
    )
    print(
        f"GF readout (frozen): form={params.gf_form} k={params.gf_k} "
        f"threshold={params.gf_threshold} vel_window={params.gf_vel_window_s}s "
        f"(scripts/calibrate_gf_readout.py; von Reyn 2014/2017)",
        flush=True,
    )
    print("Building closed-loop brain…", flush=True)
    brain = ClosedLoopBrain.build(params=params, quick=args.quick)
    env = FlyEnvironment(render_mode="rgb_array")
    ctl = DNFlightController(params)
    ctl.bind_hover(env.hover_action)
    plant = _measure_plant(env, params)
    print(f"Plant (feasibility model): {json.dumps(plant)}", flush=True)

    print("Scenario room…", flush=True)
    room = _run_scenario(
        "room",
        env,
        brain,
        ctl,
        room_steps,
        pose=(0.0, 0.0, 1.5),
        yaw=0.0,
        vel=(0.30, 0.0, 0.0),
    )
    scenarios = {"room": room}

    def _run_loom_case(spec, n_steps, name=None):
        name = name or spec["tag"]
        prev = int(brain.params.flyvis_stride)
        if spec.get("stride"):
            brain.params.flyvis_stride = int(spec["stride"])
        # Gate looms use the brain DT (10 ms). 20 ms × stride 2 was skipping
        # the hex frames the open-loop sweep used.
        prev_dt, prev_sub = env.control_dt, env.n_substeps
        prev_vdt = float(getattr(brain, "visual_dt", 0.02))
        env.control_dt = 0.01
        env.n_substeps = max(1, int(round(0.01 / float(env.model.opt.timestep))))
        brain.visual_dt = 0.01
        print(
            f"Scenario {name} az={spec.get('az_design_deg')} lv={spec.get('lv_s')} "
            f"d={spec.get('distance_m')} dt=0.01…",
            flush=True,
        )
        threat = spec.get("threat_side")
        if threat == "F":
            threat = None
        sc = _run_scenario(
            name,
            env,
            brain,
            ctl,
            n_steps,
            pose=spec["pose"],
            yaw=spec["yaw"],
            vel=spec["vel"],
            threat_side=threat,
            loom_start=spec["loom_start"],
            loom_vel=spec["loom_vel"],
            loom_band=spec.get("band"),
            present_at_prewarm=bool(spec.get("present_at_prewarm", True)),
            appear_after_prewarm=bool(spec.get("appear_after_prewarm", False)),
        )
        sc["az_design_deg"] = spec.get("az_design_deg")
        sc["lv_s"] = spec.get("lv_s")
        sc["seed"] = spec.get("seed")
        brain.params.flyvis_stride = prev
        env.control_dt = prev_dt
        env.n_substeps = prev_sub
        brain.visual_dt = prev_vdt
        return sc

    fl_cases = frontolateral_gate_cases()
    if not args.quick:
        fl_cases = fl_cases + random_frontolateral_cases()
    else:
        fl_cases = fl_cases[:2]
    print("Gate: frontolateral collision-course (|az| 20–70°)…", flush=True)
    for spec in fl_cases:
        n = 40 if args.quick else loom_horizon_steps(spec["distance_m"], spec["lv_s"], dt=0.01)
        scenarios[spec["tag"]] = _run_loom_case(spec, n)

    print("Gate: pure frontal (|az|<20°)…", flush=True)
    for spec in frontal_gate_cases()[: 1 if args.quick else None]:
        n = 40 if args.quick else loom_horizon_steps(spec["distance_m"], spec["lv_s"], dt=0.01)
        scenarios[spec["tag"]] = _run_loom_case(spec, n)

    print("Report-only: static 30–40° onset (abrupt dimming may drive GF)…", flush=True)
    for spec in static_onset_gate_cases()[: 1 if args.quick else None]:
        scenarios[spec["tag"]] = _run_loom_case(spec, 40 if args.quick else 80)

    print("Gate: receding disc must not fire…", flush=True)
    for spec in receding_gate_cases()[: 1 if args.quick else None]:
        scenarios[spec["tag"]] = _run_loom_case(spec, 40 if args.quick else 80)

    print("Report-only: unseen-like frontolat grid (start ≤5°)…", flush=True)
    unseen_cases = unseen_frontolat_cases()[: 2 if args.quick else None]
    for spec in unseen_cases:
        n = 40 if args.quick else loom_horizon_steps(spec["distance_m"], spec["lv_s"], dt=0.01)
        scenarios[spec["tag"]] = _run_loom_case(spec, n)

    print("Report-only: eccentric official L/R + far/slow/offset (LPLC2/GF frontal bias)…", flush=True)
    for side in ("L", "R"):
        spec = lateral_loom_object(side)
        name = f"side_loom_{'left' if side == 'L' else 'right'}"
        print(f"Scenario {name} (eccentric report-only)…", flush=True)
        scenarios[name] = _run_scenario(
            name,
            env,
            brain,
            ctl,
            side_steps,
            pose=spec["pose"],
            yaw=spec["yaw"],
            vel=spec["vel"],
            threat_side=side,
            loom_start=spec["loom_start"],
            loom_vel=spec["loom_vel"],
            loom_band="eccentric",
        )
        if not args.quick:
            for var in loom_variants(side):
                vname = f"loom{side}_{var['tag']}"
                print(f"Scenario {vname} (eccentric report-only)…", flush=True)
                prev = int(brain.params.flyvis_stride)
                if var.get("stride"):
                    brain.params.flyvis_stride = int(var["stride"])
                scenarios[vname] = _run_scenario(
                    vname,
                    env,
                    brain,
                    ctl,
                    160,
                    pose=spec["pose"],
                    yaw=spec["yaw"],
                    vel=spec["vel"],
                    threat_side=side,
                    loom_start=var["loom_start"],
                    loom_vel=var["loom_vel"],
                    loom_band="eccentric",
                )
                brain.params.flyvis_stride = prev
    print("Scenario headon_wall…", flush=True)
    hw = headon_wall_pose()
    scenarios["headon_wall"] = _run_scenario(
        "headon_wall",
        env,
        brain,
        ctl,
        200 if not args.quick else 40,
        pose=hw["pose"],
        yaw=hw["yaw"],
        vel=hw["vel"],
    )
    print("Scenario open_space (forward flight, no false escape)…", flush=True)
    open_p = open_space_pose()
    open_sc = _run_scenario(
        "open_space",
        env,
        brain,
        ctl,
        min(open_steps, int(open_p["max_steps"])),
        pose=open_p["pose"],
        yaw=open_p["yaw"],
        vel=open_p["vel"],
    )
    scenarios["open_space"] = open_sc
    print("Control: centre / 16-step prewarm must NOT escape…", flush=True)
    scenarios["ctrl_center_short_prewarm"] = _run_scenario(
        "ctrl_center_short_prewarm",
        env,
        brain,
        ctl,
        120 if not args.quick else 40,
        pose=(-2.0, 0.0, 1.5),
        yaw=0.0,
        vel=(0.55, 0.0, 0.0),
        prewarm=16,
    )
    print("Control: 6-heading no-object from room centre (far from walls), 0 fires…", flush=True)
    noobj_steps = 80 if args.quick else 220
    noobj_pose = (0.0, 0.0, 1.5)
    noobj_speed = 0.35
    yaws = OPEN_NOOBJ_YAWS[:2] if args.quick else OPEN_NOOBJ_YAWS
    for i, hyaw in enumerate(yaws):
        name = f"ctrl_noobj_h{i}"
        scenarios[name] = _run_scenario(
            name,
            env,
            brain,
            ctl,
            noobj_steps,
            pose=noobj_pose,
            yaw=float(hyaw),
            vel=(noobj_speed * float(np.cos(hyaw)), noobj_speed * float(np.sin(hyaw)), 0.0),
            prewarm=16 if args.quick else 80,
        )
    print("Report-only: yaw 2.20 from (−2,0) — wall/structure loom…", flush=True)
    wall_yaw = float(WALL_LOOM_YAW)
    wall_speed = 0.40
    wall_pose = (-2.0, 0.0, 1.5)
    wall_info = wall_ttc_s(wall_pose, wall_yaw, wall_speed)
    scenarios["ctrl_wallloom_yaw220"] = _run_scenario(
        "ctrl_wallloom_yaw220",
        env,
        brain,
        ctl,
        noobj_steps,
        pose=wall_pose,
        yaw=wall_yaw,
        vel=(wall_speed * float(np.cos(wall_yaw)), wall_speed * float(np.sin(wall_yaw)), 0.0),
        prewarm=16 if args.quick else 80,
    )
    wall_sc = scenarios["ctrl_wallloom_yaw220"]
    wall_sc["wall_ray"] = wall_info
    if wall_sc["trajectory"].size and wall_sc["headings"].size:
        wall_sc["wall_ray_end"] = wall_ttc_s(
            wall_sc["trajectory"][-1], float(wall_sc["headings"][-1]), wall_speed
        )
    env.close()

    wander = _noobj_wander(scenarios) or {
        "threshold_rad_s": float("inf"), "peak_threshold_rad_s": float("inf"), "n_runs": 0,
    }
    for name, sc in scenarios.items():
        if _is_frontolat(name) or _is_unseen(name) or _is_frontal(name):
            _score_behaviour(sc, name, wander, plant)

    side_l = scenarios["side_loom_left"]
    side_r = scenarios["side_loom_right"]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    _plot_trajectory(scenarios, out / "trajectory.png")
    _plot_escapes(scenarios, out / "escapes.png", params.dnp01_escape_delta_hz)

    def _pub(sc):
        return {
            k: v
            for k, v in sc.items()
            if k not in ("trajectory", "headings", "dnp01", "dnp01_rise", "azimuth_series")
        }

    room_w100 = walls_per_100(room["wall_contacts"], room["steps"])
    metrics = {
        "params": params.as_dict(),
        "quick": bool(args.quick),
        "baseline_pr16_v5": {
            "escapes_then_latched": 1,
            "wall_contacts": PR16_WALLS,
            "steps": PR16_STEPS,
            "walls_per_100": PR16_WALLS_PER_100,
        },
        "scenarios": {k: _pub(v) for k, v in scenarios.items()},
        "summary": {
            "room_wall_contacts": room["wall_contacts"],
            "room_walls_per_100": room_w100,
            "room_brain_escapes": room["brain_escapes"],
            "room_dnp01_latch_frac": room["dnp01_latch_frac"],
            "room_coverage": room["room_coverage"],
            "side_L_escapes": side_l["brain_escapes"],
            "side_R_escapes": side_r["brain_escapes"],
            "side_L_direction_correct": side_l["escape_direction_correct"],
            "side_R_direction_correct": side_r["escape_direction_correct"],
            "side_L_geometry_aware": side_l.get("geometry_aware_correct"),
            "side_R_geometry_aware": side_r.get("geometry_aware_correct"),
            "side_L_loom_class": side_l.get("loom_class"),
            "side_R_loom_class": side_r.get("loom_class"),
            "side_L_azimuth_onset_deg": side_l.get("azimuth_onset_deg"),
            "side_R_azimuth_onset_deg": side_r.get("azimuth_onset_deg"),
            "side_L_rise_max": side_l.get("dnp01_rise_max"),
            "side_R_rise_max": side_r.get("dnp01_rise_max"),
            "open_false_escapes": open_sc["brain_escapes"],
            "open_rise_max": open_sc.get("dnp01_rise_max"),
            "open_dx": open_sc["dx"],
            "open_dy": open_sc["dy"],
            "headon_wall_contacts": scenarios["headon_wall"]["wall_contacts"],
            "headon_brain_escapes": scenarios["headon_wall"]["brain_escapes"],
            "ctrl_center_16_escapes": scenarios["ctrl_center_short_prewarm"]["brain_escapes"],
            "ctrl_noobj_headings": _noobj_heading_summary(scenarios),
            "ctrl_wallloom_yaw220": {
                "escapes": scenarios["ctrl_wallloom_yaw220"]["brain_escapes"],
                "rise_max": scenarios["ctrl_wallloom_yaw220"].get("dnp01_rise_max"),
                "wall_ray": scenarios["ctrl_wallloom_yaw220"].get("wall_ray"),
                "wall_ray_end": scenarios["ctrl_wallloom_yaw220"].get("wall_ray_end"),
                "note": (
                    "Heading 2.20 rad from (−2,0) aims at the −X wall "
                    "(start ~5.1 m, TTC ~12.7 s at 0.4 m/s, expansion ~0.08 /s). "
                    "Real expanding wall texture, not residual DC. Report-only; "
                    "the no-object gate is 6 headings from the room centre."
                ),
            },
            "room_escape_sides": _escape_side_counts(room),
            "room_escape_one_sided": _escape_one_sided(room),
            "side_L_would_fire": side_l.get("would_fire"),
            "side_R_would_fire": side_r.get("would_fire"),
            "side_L_angular_size_fire_deg": side_l.get("angular_size_fire_deg"),
            "side_R_angular_size_fire_deg": side_r.get("angular_size_fire_deg"),
            "side_L_fire_minus_closest": side_l.get("fire_minus_closest"),
            "side_R_fire_minus_closest": side_r.get("fire_minus_closest"),
            "side_L_visible_at_fire": side_l.get("visible_at_fire"),
            "side_R_visible_at_fire": side_r.get("visible_at_fire"),
            "frontolat_behaviour_gate": _behaviour_gate(scenarios, _is_frontolat),
            "unseen_behaviour": _behaviour_gate(scenarios, _is_unseen, gate=False),
            "noobj_wander": wander,
            "plant": plant,
            "frontolat_gate": _frontolat_gate(scenarios),
            "frontal_gate": _frontal_gate(scenarios),
            "static_onset_gate": _zero_fire_gate(scenarios, _is_static_onset),
            "recede_gate": _zero_fire_gate(scenarios, _is_recede),
            "yaw_control": _yaw_control_stats(scenarios),
            "unseen_frontolat": _unseen_gate(scenarios),
            "variant_direction_correct": _variant_direction(scenarios),
            "variant_geometry_aware": _variant_geometry_aware(scenarios),
            "eccentric_note": (
                "Official L/R + far/slow/offset stay report-only: |az|≈90–120° "
                "at onset. LPLC2/GF are frontally biased (Klapoetke); hex x0=60° "
                "open-loop DNp01 rise 0.20 is ~90°. Do not retune the brain for them."
            ),
            "known_limitations": (
                "No sim object pose/radius reaches stereo_to_hex or the "
                "controller. Photoreceptor rest τ=5 s, exp(−dt/τ), init "
                "at calib mean 0.5 (not frame 1). No-object prewarm "
                "initialises rest to that frame (τ→∞ equivalent of a "
                "several-τ wait) so flyvis/DNp01 settle on a DC-free "
                "field. Receding large discs stay visible and skip that "
                "initialisation so the standing disc is not an onset. "
                "GF readout is frozen (calibrate_gf_readout.py; von "
                "Reyn 2014/2017). Frontolat flight gate is behavioural, "
                "scored in a 100 ms response window: on-time decision "
                "(GF fire or saccade onset before closest approach), "
                "peak |yaw rate| above no-object wander (mean+2sd) and "
                "turn away from the true side (Tammero 2002; Muijres "
                "2014). Surface clearance >0.1 m is required only on "
                "the kinematically feasible subset (calibrated plant "
                "upper bound given the observed decision time); "
                "surface+away is report-only. Static "
                "30–40° onset is report-only. Recede must be 0 fires. "
                "Frontal both-GF takeoff. DN→yaw plant is frozen "
                "(calibrate_saccade_plant.py; Fry 2003). 1/f walls "
                "and obstacles are L/R-even so open-space drift is "
                "not a body-frame texture bias. Rear 75–165° has no "
                "flyvis cells. Eccentric looms are late or silent. "
                "No-object yaw rate is the wander control. "
                "brain+flyvis is the 25 ms bar; full-loop need not."
            ),
            "no_object_state_in_eye_or_controller": True,
            "ms_per_step_brain_flyvis": room["ms_per_step_brain_flyvis"],
            "ms_per_step_mean": room.get("ms_per_step_brain_flyvis_mean"),
            "ms_per_step_full_loop": room.get("ms_per_step_full_loop"),
            "ms_per_step_full_loop_mean": room.get("ms_per_step_full_loop_mean"),
            "ms_flyvis_frame_median": room.get("ms_flyvis_frame_median"),
            "ms_flyvis_frame_mean": room.get("ms_flyvis_frame_mean"),
            "timing_note": (
                "brain+flyvis is ClosedLoopBrain.step only (sample + "
                "flyvis + rate core). full-loop is render (2× front + "
                "2× side) + brain + controller + physics. Full-loop "
                "does not need to hit 25 ms; brain+flyvis is the 25 ms bar."
            ),
            "flyvis_path": getattr(brain, "flyvis_path", {}),
            "flyvis_stride": int(params.flyvis_stride),
            "beats_pr16_walls": beats_pr16_walls(room["wall_contacts"], room["steps"]),
            "no_latch": room["dnp01_latch_frac"] < 0.05,
            "use_engineered_eye_laterality": bool(params.use_engineered_eye_laterality),
            "use_engineered_looming_residual": bool(params.use_engineered_looming_residual),
            "room_yaw_terms": room.get("yaw_terms"),
        },
    }
    metrics["summary"]["official_pass"] = _official_pass(metrics["summary"], params)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics["summary"], indent=2))
    _print_variant_tables(metrics["summary"])
    print(f"wrote {out / 'metrics.json'}", flush=True)
    return 0


def _print_behaviour_table(summary: dict) -> None:
    bg = summary.get("frontolat_behaviour_gate") or {}
    if not bg:
        return
    w = summary.get("noobj_wander") or {}
    print(
        f"\n=== Frontolat behavioural gate (response window) {bg.get('n_ok')}/{bg.get('n')} "
        f"pass={bg.get('pass')} | wander |w|>{w.get('threshold_rad_s', float('nan')):.2f} rad/s | "
        f"feasible {bg.get('n_feasible')} clearance_pass={bg.get('clearance_pass')} ==="
    )
    print(
        f"{'case':34s} {'ok':>3s} {'src':>7s} {'dec-CA':>6s} {'az@dec':>6s} {'peak|w|':>7s} "
        f"{'away':>4s} {'lead_s':>6s} {'need_s':>6s} {'smax':>5s} {'feas':>11s} {'az_ok':>5s} {'@50':>4s} {'surf':>6s}"
    )
    for name, d in sorted((bg.get("details") or {}).items()):
        b = f = d
        dec = b.get("decision_step")
        ca = d.get("ca_step")
        az = f.get("azimuth_decision_deg")
        print(
            f"{name:34s} {('Y' if b.get('ok') else 'n'):>3s} {str(b.get('decision_source')):>7s} "
            f"{(dec - ca) if (dec is not None and ca is not None) else '-':>6} "
            f"{(f'{az:.0f}' if az is not None else '-'):>6s} {(b.get('peak_yaw_rate_rad_s') or 0.0):7.1f} "
            f"{('Y' if b.get('away') else 'n'):>4s} {f.get('t_avail_s') if d.get('t_avail_s') is not None else float('nan'):6.3f} "
            f"{f.get('required_lead_s') if d.get('required_lead_s') is not None else float('nan'):6.3f} {f.get('max_lateral_m') if d.get('max_lateral_m') is not None else float('nan'):5.2f} "
            f"{str(f.get('feasibility_label')):>11s} {f.get('latest_decision_angle_deg') if d.get('latest_decision_angle_deg') is not None else float('nan'):5.1f} "
            f"{('Y' if f.get('feasible_at_gf50') else 'n'):>4s} {d.get('min_surface_m') if d.get('min_surface_m') is not None else float('nan'):6.2f}"
        )


def _print_variant_tables(summary: dict) -> None:
    _print_behaviour_table(summary)
    fl = summary.get("frontolat_gate") or {}
    if fl.get("details"):
        print(
            "\nFrontolateral flight GATE (|az0| 20–70°, surface d−0.58 >0.1 m + far-side/heading-away; GF optional):",
            flush=True,
        )
        print(
            f"{'scenario':<28} {'ok':>4} {'dsrf':>6} {'dhd':>6} {'away':>4} {'far':>4} "
            f"{'pk°/s':>6} {'mode':>14} {'drv':>7} {'esc':>4} {'rise':>6}",
            flush=True,
        )
        for name, d in fl["details"].items():
            print(
                f"{name:<28} {_fmt(d.get('ok')):>4} {_fmt(d.get('min_surface_m')):>6} "
                f"{_fmt(d.get('net_heading_rad')):>6} "
                f"{_fmt(1 if d.get('heading_away') else 0):>4} "
                f"{_fmt(1 if d.get('passed_far_side') else 0):>4} "
                f"{_fmt(None if d.get('peak_yaw_rate_rad_s') is None else 180.0/np.pi*float(d.get('peak_yaw_rate_rad_s'))):>6} "
                f"{str(d.get('avoidance_mode') or '-'):>14} "
                f"{str(d.get('yaw_driver') or '-'):>7} "
                f"{str(d.get('brain_escapes')):>4} {_fmt(d.get('dnp01_rise_max')):>6}",
                flush=True,
            )
        print(
            f"frontolat mean={fl.get('mean')} n={fl.get('n')} pass={fl.get('pass')} "
            f"gf_escape={fl.get('n_gf_escape')} pre_gf_saccade={fl.get('n_pre_gf_saccade')} "
            f"drivers={fl.get('yaw_drivers')}",
            flush=True,
        )
        print(
            f"{'scenario':<28} {'true':>4} {'lock':>4} {'bino':>4} "
            f"{'rL':>6} {'rR':>6} {'lc4L':>6} {'lc4R':>6} {'l2L':>6} {'l2R':>6} {'yaw':>6}",
            flush=True,
        )
        for name, d in fl["details"].items():
            fd = d.get("fire_diag") or {}
            print(
                f"{name:<28} {str(d.get('true_side') or '-'):>4} "
                f"{str(d.get('threat_locked') or fd.get('threat_side') or '-'):>4} "
                f"{_fmt(fd.get('bilateral_brake')):>4} "
                f"{_fmt(fd.get('rise_L')):>6} {_fmt(fd.get('rise_R')):>6} "
                f"{_fmt(fd.get('lc4_L')):>6} {_fmt(fd.get('lc4_R')):>6} "
                f"{_fmt(fd.get('lplc2_L')):>6} {_fmt(fd.get('lplc2_R')):>6} "
                f"{_fmt(fd.get('yaw')):>6}",
                flush=True,
            )
        print("\nFrontolat yaw integrals (away-signed; driver = max):", flush=True)
        for name, d in fl["details"].items():
            integ = d.get("yaw_away_integrals") or {}
            parts = " ".join(f"{k}={_fmt(integ.get(k))}" for k in ("dnp03", "vpn_loom", "dna02", "dnp15", "hse"))
            print(
                f"  {name:<28} mode={d.get('avoidance_mode')} drv={d.get('yaw_driver')} {parts}",
                flush=True,
            )
    fr = summary.get("frontal_gate") or {}
    if fr.get("details"):
        print("\nFrontal GATE (|az0|<20°, must fire, either side / brake ok):", flush=True)
        for name, d in fr["details"].items():
            print(
                f"  {name} ok={d.get('ok')} az0={_fmt(d.get('azimuth_onset_deg'))} "
                f"angF={_fmt(d.get('angular_size_fire_deg'))} esc={d.get('brain_escapes')} "
                f"rise={_fmt(d.get('dnp01_rise_max'))} braked={d.get('braked')}",
                flush=True,
            )
        print(f"frontal mean={fr.get('mean')} pass={fr.get('pass')}", flush=True)
    st = summary.get("static_onset_gate") or {}
    if st.get("details"):
        print("\nStatic 30–40° onset REPORT-ONLY (abrupt dimming may drive GF):", flush=True)
        for name, d in st["details"].items():
            print(
                f"  {name} ok={d.get('ok')} esc={d.get('brain_escapes')} "
                f"rise={_fmt(d.get('dnp01_rise_max'))}",
                flush=True,
            )
        print(f"static_onset report-only mean={st.get('mean')} n={st.get('n')}", flush=True)
    rc = summary.get("recede_gate") or {}
    if rc.get("details"):
        print("\nReceding disc GATE (must be 0 fires):", flush=True)
        for name, d in rc["details"].items():
            print(
                f"  {name} ok={d.get('ok')} esc={d.get('brain_escapes')} "
                f"rise={_fmt(d.get('dnp01_rise_max'))}",
                flush=True,
            )
        print(f"recede pass={rc.get('pass')} n={rc.get('n')}", flush=True)
    un = summary.get("unseen_frontolat") or {}
    if un.get("details"):
        print("\nUnseen-like frontolat (report-only, start ≤5°):", flush=True)
        print(
            f"{'scenario':<28} {'ok':>4} {'az0':>7} {'angF':>6} {'dCA':>5} "
            f"{'away':>5} {'esc':>4} {'rise':>6}",
            flush=True,
        )
        for name, d in un["details"].items():
            print(
                f"{name:<28} {_fmt(d.get('ok')):>4} {_fmt(d.get('azimuth_onset_deg')):>7} "
                f"{_fmt(d.get('angular_size_fire_deg')):>6} {_fmt(d.get('fire_minus_closest')):>5} "
                f"{_fmt(d.get('escape_direction_correct')):>5} {str(d.get('brain_escapes')):>4} "
                f"{_fmt(d.get('dnp01_rise_max')):>6}",
                flush=True,
            )
        print(f"unseen mean={un.get('mean')} n={un.get('n')} pass={un.get('pass')}", flush=True)
    raw = summary.get("variant_direction_correct") or {}
    geo = summary.get("variant_geometry_aware") or {}
    details = geo.get("details") or {}
    names = list(details) or list(raw.get("per_scenario") or {})
    if not names:
        return
    print("\nEccentric REPORT-ONLY (do not gate; LPLC2/GF frontal bias):", flush=True)
    print(
        f"{'scenario':<28} {'raw':>6} {'geo':>6} {'class':>8} {'az0':>7} {'angF':>6} "
        f"{'dCA':>5} {'vis':>3} {'esc':>4} {'lock':>4} {'toward':>6}",
        flush=True,
    )
    for name in names:
        d = details.get(name, {})
        r = d.get("raw", (raw.get("per_scenario") or {}).get(name))
        g = d.get("geometry_aware")
        print(
            f"{name:<28} {_fmt(r):>6} {_fmt(g):>6} {str(d.get('loom_class') or '-'):>8} "
            f"{_fmt(d.get('azimuth_onset_deg')):>7} {_fmt(d.get('angular_size_fire_deg')):>6} "
            f"{_fmt(d.get('fire_minus_closest')):>5} "
            f"{('Y' if d.get('visible_at_fire') else 'n'):>3} "
            f"{str(d.get('brain_escapes')):>4} "
            f"{str(d.get('threat_locked') or '-'):>4} {str(d.get('pre_escape_yaw_toward')):>6}",
            flush=True,
        )
    print(
        f"raw mean={raw.get('mean')} pass={raw.get('pass')} | "
        f"geometry-aware mean={geo.get('mean')} pass={geo.get('pass')}",
        flush=True,
    )
    noobj = summary.get("ctrl_noobj_headings") or {}
    if noobj.get("per_scenario"):
        print("\nOpen-space no-object (wall TTC at fire; <1 s is a real wall):", flush=True)
        for name, d in noobj["per_scenario"].items():
            print(
                f"  {name} esc={d.get('escapes')} rise={_fmt(d.get('rise_max'))} "
                f"yaw={_fmt(d.get('net_heading_rad'))} "
                f"wall_m={_fmt(d.get('wall_dist_m'))} ttc={_fmt(d.get('wall_ttc_s'))}s",
                flush=True,
            )
    yc = summary.get("yaw_control") or {}
    if yc:
        no = yc.get("noobj_abs_net_heading") or {}
        lm = yc.get("frontolat_away_abs_net_heading") or {}
        nr = yc.get("noobj_abs_yaw_rate") or {}
        lr = yc.get("frontolat_away_abs_yaw_rate") or {}
        print(
            f"\nYaw control (no-object wander vs frontolat): "
            f"noobj |Δh| n={no.get('n')} mean={_fmt(no.get('mean'))} "
            f"signed={_fmt(no.get('signed_mean'))} | "
            f"|ω| mean={_fmt(nr.get('mean'))} +2sd={_fmt(nr.get('mean_plus_2std'))} | "
            f"frontolat n={lm.get('n')} |ω|={_fmt(lr.get('mean'))} "
            f"frac>wander={_fmt(yc.get('fraction_exceeding_noobj_rate_mean_plus_2std'))} "
            f"n_ex={yc.get('n_exceeding_noobj_wander')}",
            flush=True,
        )
    print(
        f"OFFICIAL pass={(summary.get('official_pass') or {}).get('pass')} gates={(summary.get('official_pass') or {}).get('gates')} | "
        f"frontolat gate mean={summary.get('frontolat_gate', {}).get('mean')} pass={summary.get('frontolat_gate', {}).get('pass')} "
        f"gf={summary.get('frontolat_gate', {}).get('n_gf_escape')} saccade={summary.get('frontolat_gate', {}).get('n_pre_gf_saccade')} | "
        f"frontal pass={summary.get('frontal_gate', {}).get('pass')} | "
        f"static_onset report-only={(summary.get('static_onset_gate') or {}).get('pass')} | "
        f"recede pass={(summary.get('recede_gate') or {}).get('pass')} | "
        f"unseen mean={(summary.get('unseen_frontolat') or {}).get('mean')} | "
        f"no-object (open) fires={noobj.get('fires')} n={noobj.get('n')} pass={noobj.get('pass')} | "
        f"wallloom yaw2.20 esc={(summary.get('ctrl_wallloom_yaw220') or {}).get('escapes')} | "
        f"room sides={summary.get('room_escape_sides')} one_sided={summary.get('room_escape_one_sided')} | "
        f"ang_fire L/R={summary.get('side_L_angular_size_fire_deg')}/{summary.get('side_R_angular_size_fire_deg')} | "
        f"dCA L/R={summary.get('side_L_fire_minus_closest')}/{summary.get('side_R_fire_minus_closest')} | "
        f"brain+flyvis={summary.get('ms_per_step_brain_flyvis')} "
        f"full-loop={summary.get('ms_per_step_full_loop')} "
        f"flyvis_frame={summary.get('ms_flyvis_frame_mean')} path={summary.get('flyvis_path')}",
        flush=True,
    )


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "Y" if v else "n"
    try:
        return f"{float(v):.2f}"
    except (TypeError, ValueError):
        return str(v)


if __name__ == "__main__":
    sys.exit(main())
