#!/usr/bin/env python3
"""Closed-loop M2–M4 acceptance.

Official escape gate: frontolateral collision-course looms (|az0| 20–70°,
both sides, several l/v and start distances, plus fixed-seed randoms).
Pass if ≥80% fire while visible and approaching (dCA ≤ 0) and turn away.
Pure frontal (|az0|<20°) must fire (brake/climb or either side ok).
Eccentric official L/R (|az|≈90–120°) are report-only.

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
    PR16_STEPS,
    PR16_WALLS,
    PR16_WALLS_PER_100,
    ROOM_XY,
    closest_approach_step,
    collision_course_object,
    coverage,
    direction_correct_frac,
    escape_turned_into,
    fire_minus_closest,
    geometry_aware_frac,
    loom_geometry,
    object_behind_fly,
    onset_azimuth_band,
    visible_approaching_ok,
    wall_ttc_s,
    walls_per_100,
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


def frontolateral_gate_cases() -> list:
    """Collision-course looms with design |az| in 20–70° (both sides, several l/v)."""
    cases = []
    for az, lv, dist in (
        (35.0, 0.04, 3.6),
        (-35.0, 0.04, 3.6),
        (55.0, 0.04, 3.2),
        (-55.0, 0.04, 3.2),
        (45.0, 0.08, 3.4),
        (-45.0, 0.08, 3.4),
        (60.0, 0.08, 3.0),
        (-60.0, 0.08, 3.0),
        (45.0, 0.02, 4.0),
        (-45.0, 0.02, 4.0),
        (30.0, 0.04, 4.0),
        (-30.0, 0.04, 4.0),
    ):
        spec = collision_course_object(az, dist, lv)
        side = spec["threat_side"]
        tag = f"fl_{side}_az{int(abs(az))}_lv{lv:g}_d{dist:g}"
        spec["tag"] = tag
        spec["band"] = "frontolat"
        # Fast l/v looms last a few control steps; do not skip flyvis frames.
        spec["stride"] = 1
        cases.append(spec)
    return cases


def random_frontolateral_cases(seeds=FRONTOLAT_RANDOM_SEEDS) -> list:
    cases = []
    for seed in seeds:
        rng = np.random.default_rng(int(seed))
        side = "L" if int(rng.integers(0, 2)) == 0 else "R"
        az = float(rng.uniform(25.0, 65.0)) * (1.0 if side == "L" else -1.0)
        lv = float(rng.choice([0.02, 0.04, 0.08]))
        dist = float(rng.uniform(2.2, 3.2))
        spec = collision_course_object(az, dist, lv)
        spec["tag"] = f"fl_seed{int(seed)}"
        spec["band"] = "frontolat"
        spec["seed"] = int(seed)
        spec["stride"] = 1
        cases.append(spec)
    return cases


def frontal_gate_cases() -> list:
    """Pure frontal (|az| < 20°): must fire; brake/climb or either side ok."""
    cases = []
    for az, lv, dist in ((0.0, 0.04, 3.6), (12.0, 0.04, 3.6)):
        spec = collision_course_object(az, dist, lv)
        spec["tag"] = f"front_az{int(az)}_lv{lv:g}"
        spec["band"] = "frontal"
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
    left_img, right_img = env.get_eye_images()
    n_pre = int(ctl.params.prewarm_steps if prewarm is None else prewarm)
    warmed = brain.prewarm(left_img, right_img, env.get_fly_angular_velocity(), n=n_pre)
    if warmed is not None:
        ctl.capture_baseline(warmed.readout)
    loom_pos = None if loom_start is None else np.asarray(loom_start, dtype=float).copy()
    loom_v = None if loom_vel is None else np.asarray(loom_vel, dtype=float)
    if loom_pos is not None:
        env.set_loom_object(loom_pos)
    left_img, right_img = env.get_eye_images()
    for step in range(n_steps):
        if loom_pos is not None and loom_v is not None:
            # Do not teleport/hide: sudden disappearance of a large dark
            # object is itself a loom (every official R fire was 1–12
            # steps after hide). The object continues past/away.
            loom_pos = loom_pos + loom_v * float(env.control_dt)
            env.set_loom_object(loom_pos)
        res = brain.step(
            left_img,
            right_img,
            env.get_fly_angular_velocity(),
            loom_body=env.loom_centers_body(),
            loom_radius=env.loom_radius(),
        )
        action = ctl.compute(res.readout)
        left_img, right_img, info = env.step(action, wing_strokes=ctl.last_wing_strokes)
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
                rise_geometry.append(
                    {
                        "step": int(step),
                        "azimuth_deg": float(geo["azimuth_deg"]),
                        "angular_size_deg": float(geo["angular_size_deg"]),
                        "distance_m": float(geo["distance_m"]),
                        "class": geo["class"],
                        "rise": float(rise_now),
                        "lat": float(ctl.last_lat_integrated),
                        "threat": ctl._loom_threat or "",
                        "course_frozen": float(terms.get("course_frozen", 0.0)),
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
        # Must fire. Brake/climb or either-side turn is ok; turning *into* fails.
        front_ok = 1.0 if fired and not into and (va_ok or not fired) else 0.0
        if fired and not va_ok:
            front_ok = 0.0
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
        events.append(rec)
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
        "pre_escape_yaw_toward": bool(pre_toward),
        "braked": bool(braked),
        "false_escapes": int(ctl.brain_escape_count) if name == "open_space" else 0,
        "ms_per_step_brain": float(np.median(ms_brain)) if ms_brain else 0.0,
        "ms_per_step_flyvis": float(np.median(ms_flyvis)) if ms_flyvis else 0.0,
        "ms_per_step_brain_flyvis": float(np.median(ms_total)) if ms_total else 0.0,
        "ms_per_step_brain_flyvis_mean": float(np.mean(ms_total)) if ms_total else 0.0,
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
        "yaw_terms": yaw_summary,
        "would_fire": int(ctl.would_fire_count),
        "rise_geometry": rise_geometry,
        "escape_events": events,
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
        }
        fires += n
    return {
        "per_scenario": per,
        "n": int(len(per)),
        "fires": int(fires),
        "pass": bool(per) and fires == 0,
    }


def _is_loom_variant(name: str) -> bool:
    return name.startswith("side_loom_") or name.startswith("loomL_") or name.startswith("loomR_")


def _is_frontolat(name: str) -> bool:
    return name.startswith("fl_")


def _is_frontal(name: str) -> bool:
    return name.startswith("front_")


def _case_ok(sc: dict, *, require_away: bool) -> float:
    """1 if fire while visible+approaching (and turn away when required)."""
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


def _frontolat_gate(scenarios: dict) -> dict:
    """Official escape gate: frontolateral collision-course, ≥80%."""
    per = {}
    details = {}
    scores = []
    for name, sc in scenarios.items():
        if not _is_frontolat(name):
            continue
        ok = _case_ok(sc, require_away=True)
        per[name] = ok
        details[name] = {
            "ok": ok,
            "azimuth_onset_deg": sc.get("azimuth_onset_deg"),
            "loom_band": sc.get("loom_band") or onset_azimuth_band(sc.get("azimuth_onset_deg")),
            "angular_size_fire_deg": sc.get("angular_size_fire_deg"),
            "fire_minus_closest": sc.get("fire_minus_closest"),
            "visible_at_fire": sc.get("visible_at_fire"),
            "approaching_at_fire": sc.get("approaching_at_fire"),
            "escape_direction_correct": sc.get("escape_direction_correct"),
            "brain_escapes": sc.get("brain_escapes"),
            "threat_locked": sc.get("threat_locked"),
            "dnp01_rise_max": sc.get("dnp01_rise_max"),
        }
        scores.append(ok)
    mean = float(np.mean(scores)) if scores else None
    return {
        "per_scenario": per,
        "details": details,
        "mean": mean,
        "n": int(len(scores)),
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
        ok = _case_ok(sc, require_away=False)
        # Turning into a slightly-offset frontal object fails (geo already applied).
        g = sc.get("geometry_aware_correct")
        if g is not None and float(g) < 0.5:
            ok = 0.0
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


def _official_pass(summary: dict, params) -> dict:
    """Gates that still decide acceptance. Eccentric L/R and yaw-2.20 do not."""
    fl = summary.get("frontolat_gate") or {}
    fr = summary.get("frontal_gate") or {}
    no = summary.get("ctrl_noobj_headings") or {}
    brain_only = (not bool(getattr(params, "use_engineered_eye_laterality", False))) and (
        not bool(getattr(params, "use_engineered_looming_residual", False))
    )
    gates = {
        "frontolat": bool(fl.get("pass")),
        "frontal": bool(fr.get("pass")),
        "noobj_open": bool(no.get("pass")),
        "brain_only": brain_only,
    }
    return {"pass": all(gates.values()), "gates": gates}


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
    print("Building closed-loop brain…", flush=True)
    brain = ClosedLoopBrain.build(params=params, quick=args.quick)
    env = FlyEnvironment(render_mode="rgb_array", camera_width=160, camera_height=160)
    ctl = DNFlightController(params)
    ctl.bind_hover(env.hover_action)

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
        env.control_dt = 0.01
        env.n_substeps = max(1, int(round(0.01 / float(env.model.opt.timestep))))
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
        )
        sc["az_design_deg"] = spec.get("az_design_deg")
        sc["lv_s"] = spec.get("lv_s")
        sc["seed"] = spec.get("seed")
        brain.params.flyvis_stride = prev
        env.control_dt = prev_dt
        env.n_substeps = prev_sub
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
            if k not in ("trajectory", "headings", "dnp01", "dnp01_rise")
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
            "frontolat_gate": _frontolat_gate(scenarios),
            "frontal_gate": _frontal_gate(scenarios),
            "variant_direction_correct": _variant_direction(scenarios),
            "variant_geometry_aware": _variant_geometry_aware(scenarios),
            "eccentric_note": (
                "Official L/R + far/slow/offset stay report-only: |az|≈90–120° "
                "at onset. LPLC2/GF are frontally biased (Klapoetke); hex x0=60° "
                "open-loop DNp01 rise 0.20 is ~90°. Do not retune the brain for them."
            ),
            "known_limitations": (
                "Hex-native open-loop centered discs at l/v=0.04 fire DNp01 rise "
                "0.20 at 38–49°. The same l/v as a 3D camera disc on a collision "
                "course (20 ms tick) peaks at ~0.06–0.11 while approaching — too "
                "few hex-dark frames for GF. Slow 3D collision-course (l/v~0.3) "
                "can fire on the way in but at 75–120°, matching the late "
                "eccentric GF. Room coverage stays ~0.06 after VPN rest (false "
                "left kicks gone). Do not retune the controller to chase either."
            ),
            "ms_per_step_brain_flyvis": room["ms_per_step_brain_flyvis"],
            "ms_per_step_mean": room.get("ms_per_step_brain_flyvis_mean"),
            "ms_flyvis_frame_median": room.get("ms_flyvis_frame_median"),
            "ms_flyvis_frame_mean": room.get("ms_flyvis_frame_mean"),
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


def _print_variant_tables(summary: dict) -> None:
    fl = summary.get("frontolat_gate") or {}
    if fl.get("details"):
        print("\nFrontolateral collision-course GATE (|az0| 20–70°, visible+approaching, turn away):", flush=True)
        print(
            f"{'scenario':<28} {'ok':>4} {'az0':>7} {'angF':>6} {'dCA':>5} {'away':>5} {'esc':>4} {'rise':>6}",
            flush=True,
        )
        for name, d in fl["details"].items():
            print(
                f"{name:<28} {_fmt(d.get('ok')):>4} {_fmt(d.get('azimuth_onset_deg')):>7} "
                f"{_fmt(d.get('angular_size_fire_deg')):>6} {_fmt(d.get('fire_minus_closest')):>5} "
                f"{_fmt(d.get('escape_direction_correct')):>5} {str(d.get('brain_escapes')):>4} "
                f"{_fmt(d.get('dnp01_rise_max')):>6}",
                flush=True,
            )
        print(f"frontolat mean={fl.get('mean')} n={fl.get('n')} pass={fl.get('pass')}", flush=True)
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
    print(
        f"OFFICIAL pass={(summary.get('official_pass') or {}).get('pass')} gates={(summary.get('official_pass') or {}).get('gates')} | "
        f"frontolat gate mean={summary.get('frontolat_gate', {}).get('mean')} pass={summary.get('frontolat_gate', {}).get('pass')} | "
        f"frontal pass={summary.get('frontal_gate', {}).get('pass')} | "
        f"no-object (open) fires={noobj.get('fires')} n={noobj.get('n')} pass={noobj.get('pass')} | "
        f"wallloom yaw2.20 esc={(summary.get('ctrl_wallloom_yaw220') or {}).get('escapes')} | "
        f"room sides={summary.get('room_escape_sides')} one_sided={summary.get('room_escape_one_sided')} | "
        f"ang_fire L/R={summary.get('side_L_angular_size_fire_deg')}/{summary.get('side_R_angular_size_fire_deg')} | "
        f"dCA L/R={summary.get('side_L_fire_minus_closest')}/{summary.get('side_R_fire_minus_closest')} | "
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
