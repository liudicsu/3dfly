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
    """True lateral loom: mostly +X heading, closing on the left or right wall.

    Body frame: left = +Y when heading +X. A small yaw toward that wall makes
    the distance decrease (classic loom) so DNp01 can rise over the EMA rest.
    The previous scenario faced +Y (yaw=π/2) and drove into the frontal wall.
    """
    speed = 0.55
    if side == "L":
        yaw = 0.55  # ~31° toward +Y (left)
        return {
            "pose": (-1.2, 3.1, 1.5),
            "yaw": yaw,
            "vel": (speed * float(np.cos(yaw)), speed * float(np.sin(yaw)), 0.0),
        }
    if side == "R":
        yaw = -0.55
        return {
            "pose": (-1.2, -3.1, 1.5),
            "yaw": yaw,
            "vel": (speed * float(np.cos(yaw)), speed * float(np.sin(yaw)), 0.0),
        }
    raise ValueError(f"side must be L or R, got {side!r}")


def open_space_pose() -> dict:
    """Centre of the room, heading +X. Keep the run short so no wall is reached."""
    return {
        "pose": (0.0, 0.0, 1.5),
        "yaw": 0.0,
        "vel": (0.30, 0.0, 0.0),
        "max_steps": 120,
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
) -> dict:
    env.reset(seed=0)
    if pose is not None:
        env.set_pose(pose, yaw=yaw, vel=vel)
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
    t_wall = time.perf_counter()
    left_img, right_img = env.get_eye_images()
    warmed = brain.prewarm(left_img, right_img, env.get_fly_angular_velocity())
    if warmed is not None:
        ctl.capture_baseline(warmed.readout)
        left_img, right_img = env.get_eye_images()
    for step in range(n_steps):
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
        "room_coverage": coverage(traj),
        "dx": dx,
        "dy": dy,
        "wall_seconds": float(wall_s),
        "trajectory": traj,
        "headings": headings_arr,
        "dnp01": np.asarray(dnp01, dtype=float),
        "dnp01_rise": np.asarray(rises, dtype=float),
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
    # 120 steps × 0.02 s × 0.35 m/s ≈ 0.8 m from the centre — stays off the walls.
    open_steps = 40 if args.quick else 120
    if args.quick:
        print("--quick: unofficial short movies", flush=True)

    params = FlightParams(use_engineered_looming_residual=args.engineered_looming_residual)
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
        pose = lateral_loom_pose(side)
        name = f"side_loom_{'left' if side == 'L' else 'right'}"
        print(f"Scenario {name} (heading +X, {side} wall)…", flush=True)
        scenarios[name] = _run_scenario(
            name,
            env,
            brain,
            ctl,
            side_steps,
            pose=pose["pose"],
            yaw=pose["yaw"],
            vel=pose["vel"],
            threat_side=side,
        )
    print("Scenario open_space (no false escape)…", flush=True)
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
            "open_false_escapes": open_sc["brain_escapes"],
            "ms_per_step_brain_flyvis": room["ms_per_step_brain_flyvis"],
            "beats_pr16_walls": beats_pr16_walls(room["wall_contacts"], room["steps"]),
            "no_latch": room["dnp01_latch_frac"] < 0.05,
        },
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics["summary"], indent=2))
    print(f"wrote {out / 'metrics.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
