#!/usr/bin/env python3
"""Closed-loop M2–M4 acceptance: room flight + lateral loom L/R + open-space.

Single command on the verification box (connectome + flyvis + 3.12):

    python scripts/closed_loop_acceptance.py
    python scripts/closed_loop_acceptance.py --quick
    python scripts/closed_loop_acceptance.py --steps 2000 --output output/m2

Writes ``output/<dir>/metrics.json``, ``trajectory.png``, ``escapes.png``.
Missing data/flyvis is an error (unit tests skip instead).

Metrics live in ``threedfly.control.closed_loop_metrics`` so they can be
imported without MuJoCo (NumPy 2: use ``np.ptp``, never ``ndarray.ptp``).
Wall contacts are scored per 100 steps against PR #16's 81/2000 = 4.05.
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
    coverage,
    direction_correct_frac,
    geometry_aware_frac,
    loom_geometry,
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
    ms_total = []
    wall = 0
    yaw_terms = []
    rise_geometry = []
    rise_onset = None
    geo_lock = None
    geo_fire = None
    n_esc_seen = 0
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
            loom_pos = loom_pos + loom_v * float(env.control_dt)
            # Once the object has passed behind the fly it is not a loom.
            if loom_pos[0] < -0.4:
                env.hide_loom_object()
                loom_pos = None
            else:
                env.set_loom_object(loom_pos)
        res = brain.step(left_img, right_img, env.get_fly_angular_velocity())
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
        ms_total.append(res.ms.get("total", 0.0))
        terms = dict(ctl.last_yaw_terms)
        terms["x"] = float(pos[0])
        terms["y"] = float(pos[1])
        terms["near_wall"] = 1.0 if max(abs(pos[0]), abs(pos[1])) > 3.2 else 0.0
        geo = loom_geometry(loom_pos, pos, heading) if loom_pos is not None else None
        if geo is not None:
            terms["azimuth_deg"] = float(geo["azimuth_deg"])
            terms["angular_size_deg"] = float(geo["angular_size_deg"])
            if rise_onset is None and rise_now >= 0.08:
                rise_onset = {"step": int(step), **{k: geo[k] for k in geo}}
            if geo_lock is None and ctl.last_lock_step is not None:
                geo_lock = {
                    "step": int(step),
                    "threat": ctl._loom_threat,
                    **{k: geo[k] for k in geo},
                }
            if ctl.brain_escape_count > n_esc_seen:
                geo_fire = {"step": int(step), **{k: geo[k] for k in geo}}
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
                    }
                )
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
    dir_ok = (
        direction_correct_frac(headings_arr, ctl.escape_events, threat_side)
        if threat_side
        else float("nan")
    )
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
        "azimuth_onset_deg": az_on,
        "angular_size_onset_deg": None if rise_onset is None else float(rise_onset["angular_size_deg"]),
        "azimuth_lock_deg": None if geo_lock is None else float(geo_lock["azimuth_deg"]),
        "angular_size_lock_deg": None if geo_lock is None else float(geo_lock["angular_size_deg"]),
        "azimuth_fire_deg": None if geo_fire is None else float(geo_fire["azimuth_deg"]),
        "angular_size_fire_deg": None if geo_fire is None else float(geo_fire["angular_size_deg"]),
        "threat_locked": None if geo_lock is None else geo_lock.get("threat"),
        "pre_escape_yaw_toward": bool(pre_toward),
        "braked": bool(braked),
        "false_escapes": int(ctl.brain_escape_count) if name == "open_space" else 0,
        "ms_per_step_brain": float(np.median(ms_brain)) if ms_brain else 0.0,
        "ms_per_step_flyvis": float(np.median(ms_flyvis)) if ms_flyvis else 0.0,
        "ms_per_step_brain_flyvis": float(np.median(ms_total)) if ms_total else 0.0,
        "ms_per_step_brain_flyvis_mean": float(np.mean(ms_total)) if ms_total else 0.0,
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


def _is_loom_variant(name: str) -> bool:
    return name.startswith("side_loom_") or name.startswith("loomL_") or name.startswith("loomR_")


def _variant_direction(scenarios: dict) -> dict:
    """Raw ≥80% away across official + offset/speed/stride L/R variants."""
    dirs = []
    per = {}
    for name, sc in scenarios.items():
        if not _is_loom_variant(name):
            continue
        d = sc.get("escape_direction_correct")
        per[name] = d
        if d is not None:
            dirs.append(float(d))
    return {
        "per_scenario": per,
        "mean": float(np.mean(dirs)) if dirs else None,
        "n": int(len(dirs)),
        "pass": bool(dirs) and float(np.mean(dirs)) >= 0.80,
    }


def _variant_geometry_aware(scenarios: dict) -> dict:
    """Geometry-aware variant score. Lateral no-escape is 0; do not relabel."""
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

    fig, ax = plt.subplots(figsize=(7, 7))
    for name, sc in scenarios.items():
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
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(scenarios)
    fig, axes = plt.subplots(len(names), 1, figsize=(8, 2.4 * len(names)), sharex=False)
    if len(names) == 1:
        axes = [axes]
    for ax, name in zip(axes, names):
        sc = scenarios[name]
        rise = sc.get("dnp01_rise")
        if rise is None or getattr(rise, "size", 0) == 0:
            rise = sc["dnp01"]
        ax.plot(rise, color="tab:red", lw=1.0, label="DNp01 rise")
        ax.axhline(delta_hz, color="gray", ls="--", lw=0.8, label=f"rise thr {delta_hz:.2f}")
        for ev in sc["escape_events"]:
            ax.axvline(ev["step"], color="tab:orange", alpha=0.5)
        ax.set_ylabel("ΔHz")
        ax.set_title(f"{name}: escapes={sc['brain_escapes']} latch={sc['dnp01_latch_frac']:.3f}")
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("step")
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
    env = FlyEnvironment(render_mode="rgb_array", camera_width=160, camera_height=120)
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
    for side in ("L", "R"):
        spec = lateral_loom_object(side)
        name = f"side_loom_{'left' if side == 'L' else 'right'}"
        print(f"Scenario {name} (dark object from {side}, stride={brain.params.flyvis_stride})…", flush=True)
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
        )
        if not args.quick:
            for var in loom_variants(side):
                vname = f"loom{side}_{var['tag']}"
                print(f"Scenario {vname}…", flush=True)
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
            "side_L_would_fire": side_l.get("would_fire"),
            "side_R_would_fire": side_r.get("would_fire"),
            "variant_direction_correct": _variant_direction(scenarios),
            "variant_geometry_aware": _variant_geometry_aware(scenarios),
            "ms_per_step_brain_flyvis": room["ms_per_step_brain_flyvis"],
            "ms_per_step_mean": room.get("ms_per_step_brain_flyvis_mean"),
            "flyvis_stride": int(params.flyvis_stride),
            "beats_pr16_walls": beats_pr16_walls(room["wall_contacts"], room["steps"]),
            "no_latch": room["dnp01_latch_frac"] < 0.05,
            "use_engineered_eye_laterality": bool(params.use_engineered_eye_laterality),
            "use_engineered_looming_residual": bool(params.use_engineered_looming_residual),
            "room_yaw_terms": room.get("yaw_terms"),
        },
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics["summary"], indent=2))
    _print_variant_tables(metrics["summary"])
    print(f"wrote {out / 'metrics.json'}", flush=True)
    return 0


def _print_variant_tables(summary: dict) -> None:
    raw = summary.get("variant_direction_correct") or {}
    geo = summary.get("variant_geometry_aware") or {}
    details = geo.get("details") or {}
    names = list(details) or list(raw.get("per_scenario") or {})
    if not names:
        return
    print("\nRaw direction (must turn away from scenario side):", flush=True)
    print(f"{'scenario':<28} {'raw':>6} {'geo':>6} {'class':>8} {'az0':>7} {'esc':>4} {'lock':>4} {'toward':>6}", flush=True)
    for name in names:
        d = details.get(name, {})
        r = d.get("raw", (raw.get("per_scenario") or {}).get(name))
        g = d.get("geometry_aware")
        print(
            f"{name:<28} {_fmt(r):>6} {_fmt(g):>6} {str(d.get('loom_class') or '-'):>8} "
            f"{_fmt(d.get('azimuth_onset_deg')):>7} {str(d.get('brain_escapes')):>4} "
            f"{str(d.get('threat_locked') or '-'):>4} {str(d.get('pre_escape_yaw_toward')):>6}",
            flush=True,
        )
    print(
        f"raw mean={raw.get('mean')} pass={raw.get('pass')} | "
        f"geometry-aware mean={geo.get('mean')} pass={geo.get('pass')}",
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
