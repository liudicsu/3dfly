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
    walls_per_100,
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
            "stride": 3,
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
        traj.append(pos)
        headings.append(heading_from_quaternion(info["orientation"]))
        dnp01.append(max(res.readout.dnp01_L, res.readout.dnp01_R))
        rises.append(float(ctl.last_rise))
        ms_brain.append(res.ms.get("brain", 0.0))
        ms_flyvis.append(res.ms.get("flyvis", 0.0))
        ms_total.append(res.ms.get("total", 0.0))
        terms = dict(ctl.last_yaw_terms)
        terms["x"] = float(pos[0])
        terms["y"] = float(pos[1])
        terms["near_wall"] = 1.0 if max(abs(pos[0]), abs(pos[1])) > 3.2 else 0.0
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
    dx, dy = xy_extent(traj)
    yaw_summary = _yaw_term_summary(yaw_terms)
    return {
        "name": name,
        "steps": int(n_steps),
        "wall_contacts": int(wall),
        "walls_per_100": walls_per_100(wall, n_steps),
        "brain_escapes": int(ctl.brain_escape_count),
        "engineered_escapes": int(ctl.engineered_escape_count),
        "dnp01_latch_frac": float(ctl.dnp01_latch_frac),
        "escape_direction_correct": None if np.isnan(dir_ok) else float(dir_ok),
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
        "escape_events": [
            {
                "step": e.step,
                "dnp01": e.dnp01,
                "threat_side": e.threat_side,
                "yaw": e.yaw,
                "engineered": e.engineered,
            }
            for e in ctl.escape_events
        ],
    }


def _variant_direction(scenarios: dict) -> dict:
    """≥80% away across official + offset/speed/stride L/R variants."""
    dirs = []
    per = {}
    for name, sc in scenarios.items():
        if not (name.startswith("side_loom_") or name.startswith("loomL_") or name.startswith("loomR_")):
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
    print(f"wrote {out / 'metrics.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
