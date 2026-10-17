#!/usr/bin/env python3
"""Closed-loop M2–M4 acceptance: room flight + side loom + open-space.

Single command on the verification box (connectome + flyvis + 3.12):

    python scripts/closed_loop_acceptance.py
    python scripts/closed_loop_acceptance.py --quick
    python scripts/closed_loop_acceptance.py --steps 800 --output output/m2

Writes ``output/<dir>/metrics.json``, ``trajectory.png``, ``escapes.png``.
Missing data/flyvis is an error (unit tests skip instead).
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

from threedfly.brain.closed_loop import ClosedLoopBrain, core_stack_available
from threedfly.brain.constants import connectome_available, flyvis_available
from threedfly.control.dn_controller import DNFlightController
from threedfly.control.exploration import heading_from_quaternion
from threedfly.control.flight_params import FlightParams
from threedfly.sim import FlyEnvironment


ROOM_XY = 5.0
BIN_M = 0.5


def _coverage(traj: np.ndarray) -> float:
    if traj.size == 0:
        return 0.0
    gx = np.clip(((traj[:, 0] + ROOM_XY) / BIN_M).astype(int), 0, int(2 * ROOM_XY / BIN_M) - 1)
    gy = np.clip(((traj[:, 1] + ROOM_XY) / BIN_M).astype(int), 0, int(2 * ROOM_XY / BIN_M) - 1)
    visited = len(set(zip(gx.tolist(), gy.tolist())))
    total = int((2 * ROOM_XY / BIN_M) ** 2)
    return float(visited) / float(total)


def _direction_correct(events, expected_threat: str) -> float:
    if not events:
        return float("nan")
    ok = sum(1 for e in events if e.threat_side == expected_threat and not e.engineered)
    return float(ok) / float(len(events))


def _run_scenario(
    name: str,
    env: FlyEnvironment,
    brain: ClosedLoopBrain,
    ctl: DNFlightController,
    n_steps: int,
    pose=None,
    yaw: float = 0.0,
    vel=None,
    expected_threat=None,
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
    ms_brain = []
    ms_flyvis = []
    ms_total = []
    wall = 0
    t_wall = time.perf_counter()
    left_img, right_img = env.get_eye_images()
    for step in range(n_steps):
        res = brain.step(left_img, right_img, env.get_fly_angular_velocity())
        action = ctl.compute(res.readout)
        left_img, right_img, info = env.step(action, wing_strokes=ctl.last_wing_strokes)
        pos = np.asarray(info["position"], dtype=float)
        traj.append(pos)
        headings.append(heading_from_quaternion(info["orientation"]))
        dnp01.append(max(res.readout.dnp01_L, res.readout.dnp01_R))
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
    dir_ok = _direction_correct(ctl.escape_events, expected_threat) if expected_threat else float("nan")
    return {
        "name": name,
        "steps": int(n_steps),
        "wall_contacts": int(wall),
        "brain_escapes": int(ctl.brain_escape_count),
        "engineered_escapes": int(ctl.engineered_escape_count),
        "dnp01_latch_frac": float(ctl.dnp01_latch_frac),
        "escape_direction_correct": None if np.isnan(dir_ok) else float(dir_ok),
        "false_escapes": int(ctl.brain_escape_count) if name == "open_space" else 0,
        "ms_per_step_brain": float(np.median(ms_brain)) if ms_brain else 0.0,
        "ms_per_step_flyvis": float(np.median(ms_flyvis)) if ms_flyvis else 0.0,
        "ms_per_step_brain_flyvis": float(np.median(ms_total)) if ms_total else 0.0,
        "room_coverage": _coverage(traj),
        "dx": float(traj[:, 0].ptp()) if traj.size else 0.0,
        "dy": float(traj[:, 1].ptp()) if traj.size else 0.0,
        "wall_seconds": float(wall_s),
        "trajectory": traj,
        "headings": np.asarray(headings, dtype=float),
        "dnp01": np.asarray(dnp01, dtype=float),
        "escape_events": [
            {"step": e.step, "dnp01": e.dnp01, "threat_side": e.threat_side, "yaw": e.yaw, "engineered": e.engineered}
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


def _plot_escapes(scenarios: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(scenarios)
    fig, axes = plt.subplots(len(names), 1, figsize=(8, 2.4 * len(names)), sharex=False)
    if len(names) == 1:
        axes = [axes]
    for ax, name in zip(axes, names):
        sc = scenarios[name]
        ax.plot(sc["dnp01"], color="tab:red", lw=1.0, label="DNp01")
        ax.axhline(12.0, color="gray", ls="--", lw=0.8, label="escape thr")
        for ev in sc["escape_events"]:
            ax.axvline(ev["step"], color="tab:orange", alpha=0.5)
        ax.set_ylabel("Hz")
        ax.set_title(f"{name}: escapes={sc['brain_escapes']} latch={sc['dnp01_latch_frac']:.3f}")
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("step")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default=str(ROOT / "output" / "m2"))
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--quick", action="store_true", help="Short unofficial run")
    p.add_argument("--engineered-looming-residual", action="store_true")
    args = p.parse_args(argv)

    if not connectome_available() or not flyvis_available() or not core_stack_available():
        print(
            "ERROR: closed-loop acceptance needs data/full_connectome.npz and flyvis.\n"
            "Python 3.12: pip install -e '.[brain]'. "
            "Local editable flyvis + pretrained data: pip install -e /path/to/flyvis --no-deps.",
            file=sys.stderr,
        )
        return 2

    room_steps = 80 if args.quick else int(args.steps)
    side_steps = 40 if args.quick else max(120, args.steps // 3)
    open_steps = 40 if args.quick else max(80, args.steps // 4)
    if args.quick:
        print("--quick: unofficial short movies", flush=True)

    params = FlightParams(use_engineered_looming_residual=args.engineered_looming_residual)
    print("Building closed-loop brain…", flush=True)
    brain = ClosedLoopBrain.build(params=params, quick=args.quick)
    env = FlyEnvironment(render_mode="rgb_array", camera_width=160, camera_height=120)
    ctl = DNFlightController(params)
    ctl.bind_hover(env.hover_action)

    print("Scenario room…", flush=True)
    room = _run_scenario("room", env, brain, ctl, room_steps, pose=(0.0, 0.0, 1.5), yaw=0.0)
    print("Scenario side_loom_left (approach +y wall from the left)…", flush=True)
    # Face +Y (left of +X). Wall at y=+5 is ahead-left of a slight +X offset.
    side = _run_scenario(
        "side_loom_left",
        env,
        brain,
        ctl,
        side_steps,
        pose=(1.2, 3.6, 1.5),
        yaw=np.pi / 2,
        vel=(0.0, 0.6, 0.0),
        expected_threat="L",
    )
    print("Scenario open_space (no false escape)…", flush=True)
    open_sc = _run_scenario(
        "open_space",
        env,
        brain,
        ctl,
        open_steps,
        pose=(0.0, 0.0, 1.5),
        yaw=0.0,
        vel=(0.4, 0.0, 0.0),
    )
    env.close()

    scenarios = {"room": room, "side_loom_left": side, "open_space": open_sc}
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    _plot_trajectory(scenarios, out / "trajectory.png")
    _plot_escapes(scenarios, out / "escapes.png")

    def _pub(sc):
        d = {k: v for k, v in sc.items() if k not in ("trajectory", "headings", "dnp01")}
        return d

    metrics = {
        "params": params.as_dict(),
        "quick": bool(args.quick),
        "baseline_pr16_v5": {"escapes_then_latched": 1, "wall_contacts": 81},
        "scenarios": {k: _pub(v) for k, v in scenarios.items()},
        "summary": {
            "room_wall_contacts": room["wall_contacts"],
            "room_brain_escapes": room["brain_escapes"],
            "room_dnp01_latch_frac": room["dnp01_latch_frac"],
            "side_escapes": side["brain_escapes"],
            "side_direction_correct": side["escape_direction_correct"],
            "open_false_escapes": open_sc["brain_escapes"],
            "ms_per_step_brain_flyvis": room["ms_per_step_brain_flyvis"],
            "beats_pr16_walls": room["wall_contacts"] < 81,
            "no_latch": room["dnp01_latch_frac"] < 0.05,
        },
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics["summary"], indent=2))
    print(f"wrote {out / 'metrics.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
