#!/usr/bin/env python3
"""Open-loop looming-disc sweep: LPLC2 / LC4 / DNp01 vs angular size.

Fly held still. Hex-native constant-l/v discs (Gabbiani / von Reyn /
Klapoetke) on both sides at several l/v. Writes a figure and JSON.

    python scripts/loom_openloop_sweep.py --output output/m2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threedfly.brain.constants import DT
from threedfly.brain.stimuli import gray, loom, loom_theta_deg
from threedfly.control.dn_controller import DNFlightController
from threedfly.control.flight_params import FlightParams


LV_VALUES = (0.02, 0.04, 0.08, 0.16)


def _ipsi(readout, side: str, name: str) -> float:
    return float(getattr(readout, f"{name}_{side}"))


def run_hex_condition(brain, ctl, side: str, lv: float, pre: float = 0.4, stim: float = 0.8):
    """One still-fly hex loom. Returns per-frame rates + angular size."""
    movie = np.concatenate([gray(pre, dt=DT), loom(stim, lv=lv, dt=DT)])
    theta = np.concatenate([
        np.full(int(round(pre / DT)), 0.0, dtype=np.float32),
        loom_theta_deg(stim, lv=lv, dt=DT),
    ])
    blank = gray(pre + stim, dt=DT)
    left = movie if side == "L" else blank
    right = movie if side == "R" else blank
    brain.reset()
    ctl.reset()
    # Prewarm on gray hex (do not call prewarm() — that stereo-samples images).
    g = gray(0.2, dt=DT)[0]
    last = None
    for _ in range(20):
        last = brain.step(g, g, None, hex_L=g, hex_R=g)
    if last is not None:
        ctl.capture_baseline(last.readout)
    rows = []
    dummy = np.zeros((8, 8), dtype=np.float32)
    for i in range(left.shape[0]):
        res = brain.step(dummy, dummy, None, hex_L=left[i], hex_R=right[i])
        ctl.compute(res.readout)
        ro = res.readout
        rows.append(
            {
                "frame": i,
                "angular_size_deg": float(theta[i]),
                "dnp01": _ipsi(ro, side, "dnp01"),
                "dnp01_rise": float(ctl.last_rise),
                "lc4": _ipsi(ro, side, "lc4"),
                "lplc2": _ipsi(ro, side, "lplc2"),
                "lc4_p90": float(getattr(ro, f"lc4_p90_{side}")),
                "lplc2_p90": float(getattr(ro, f"lplc2_p90_{side}")),
            }
        )
    return rows


def onset_angle(rows, key: str, thresh: float) -> float | None:
    for r in rows:
        if float(r.get(key, 0.0)) >= float(thresh) and float(r["angular_size_deg"]) > 5.0:
            return float(r["angular_size_deg"])
    return None


def plot_sweep(results: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sides = ("L", "R")
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.0), sharex=True)
    keys = (("lplc2", "LPLC2"), ("lc4", "LC4"), ("dnp01", "DNp01"), ("dnp01_rise", "DNp01 rise"))
    for ax, (key, title) in zip(axes.ravel(), keys):
        for side in sides:
            for lv in LV_VALUES:
                rows = results[f"{side}_lv{lv}"]
                # Drop the 90° saturation spike so the 40–60° window is readable.
                xs, ys = [], []
                for r in rows:
                    ang = float(r["angular_size_deg"])
                    if ang >= 88.0:
                        continue
                    xs.append(ang)
                    ys.append(r[key])
                ax.plot(xs, ys, lw=1.1, label=f"{side} l/v={lv:g}")
        ax.axvspan(40.0, 60.0, color="tab:green", alpha=0.12, label="bio GF 40–60°")
        ax.set_title(title)
        ax.set_xlim(8, 86)
        ax.grid(True, alpha=0.3)
    axes[1, 0].set_xlabel("angular size (deg)")
    axes[1, 1].set_xlabel("angular size (deg)")
    axes[0, 1].legend(fontsize=7, loc="upper left")
    fig.suptitle("Open-loop looming discs (fly still): flyvis → handoff → RateCore")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default=str(ROOT / "output" / "m2"))
    args = p.parse_args(argv)

    from threedfly.brain.closed_loop import ClosedLoopBrain, core_stack_available
    from threedfly.brain.constants import flyvis_available

    if not flyvis_available() or not core_stack_available():
        print("ERROR: sweep needs flyvis + core bundle / connectome", file=sys.stderr)
        return 2

    params = FlightParams()
    print("Building closed-loop brain for open-loop sweep…", flush=True)
    brain = ClosedLoopBrain.build(params=params)
    # Honest neural curve: every frame.
    brain.params.flyvis_stride = 1
    ctl = DNFlightController(params)
    results = {}
    onsets = {}
    for side in ("L", "R"):
        for lv in LV_VALUES:
            name = f"{side}_lv{lv}"
            print(f"  {name}…", flush=True)
            rows = run_hex_condition(brain, ctl, side, lv)
            results[name] = rows
            onsets[name] = {
                "dnp01_rise_0.20": onset_angle(rows, "dnp01_rise", 0.20),
                "lc4_0.10": onset_angle(rows, "lc4", 0.10),
                "lplc2_0.10": onset_angle(rows, "lplc2", 0.10),
                "peak_dnp01_rise": max(r["dnp01_rise"] for r in rows),
                "peak_lc4": max(r["lc4"] for r in rows),
                "peak_lplc2": max(r["lplc2"] for r in rows),
            }
            print(f"    onsets {onsets[name]}", flush=True)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    fig_path = out / "loom_sweep.png"
    plot_sweep(results, fig_path)
    payload = {
        "lv": list(LV_VALUES),
        "dt": DT,
        "onsets": onsets,
        "note": (
            "Hex-native dark discs (flyvis loom, gray 0.5). "
            "Centered l/v=0.04: DNp01 rise 0.20 at 38–49° (bio GF window). "
            "Eccentric x0=40/60°: 70° / 90° — LPLC2/GF are frontally biased "
            "(Klapoetke). Official laterals at |az|~90° match x0=60. "
            "g_inh=8 and p99.9 are not the late-fire cause for frontal discs. "
            "Sub-hex checkers + hide-as-loom were the closed-loop artifacts."
        ),
        "traces": {k: v[::2] for k, v in results.items()},
    }
    (out / "loom_sweep.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(onsets, indent=2))
    print(f"wrote {fig_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
