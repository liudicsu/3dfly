#!/usr/bin/env python3
"""Calibrate the GF readout ONCE on the canonical flyvis loom set.

Canonical stimulus (do not add more conditions; do not retune after):
    flyvis ``loom()``, l/v = 40 ms, fly still, gray background
        * both eyes, centred (x0 = 0)
        * left eye only at az +30°
        * right eye only at az −30°

Target: first GF crossing at 40–60° angular size (aim 50°), matching
von Reyn et al. 2014 Nat Neurosci / 2017 Curr Biol (GF encodes angular
size + angular velocity). Forms considered: rise + k·d(rise)/dt, or
integrate-to-threshold of the same instantaneous score.

Writes:
    output/m2/gf_readout_calibration.json
    threedfly/control/gf_readout_traces.json   (recompute-test fixture)

Prints the frozen (form, k, threshold) and the three fire angles.
Those numbers are copied into ``FlightParams``; this script is the
citation trail, not a live retuner.

    python scripts/calibrate_gf_readout.py --output output/m2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threedfly.brain.constants import DT
from threedfly.control.dn_controller import DNFlightController
from threedfly.control.flight_params import FlightParams
from threedfly.control.gf_readout import (
    GF_CITATION,
    GF_LV_S,
    GF_SCRIPT,
    GF_TARGET_DEG,
    GF_WINDOW_HI_DEG,
    GF_WINDOW_LO_DEG,
    TRACES_PATH,
    collect_canonical_traces,
    first_fire_angle,
    fit_gf_readout,
    traces_to_jsonable,
)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default=str(ROOT / "output" / "m2"))
    p.add_argument("--lv", type=float, default=GF_LV_S)
    args = p.parse_args(argv)

    from threedfly.brain.closed_loop import ClosedLoopBrain, core_stack_available
    from threedfly.brain.constants import flyvis_available

    if not flyvis_available() or not core_stack_available():
        print("ERROR: calibration needs flyvis + core bundle / connectome", file=sys.stderr)
        return 2

    params = FlightParams()
    print("Building closed-loop brain for GF readout calibration…", flush=True)
    brain = ClosedLoopBrain.build(params=params)
    # Honest neural curve: every hex frame, flyvis-native 10 ms.
    brain.params.flyvis_stride = 1
    ctl = DNFlightController(params)

    print(
        f"Collecting canonical traces (loom() l/v={args.lv*1000:.0f} ms, "
        f"both / L+30 / R−30)…",
        flush=True,
    )
    traces = collect_canonical_traces(brain, ctl, lv=float(args.lv), dt=DT)
    fit = fit_gf_readout(traces, dt=DT)

    rise_only = {}
    for name, tr in traces.items():
        rise_only[name] = first_fire_angle(
            tr["theta_deg"], tr["rise"], DT, "pd", 0.0, 0.20
        )

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "citation": GF_CITATION,
        "script": GF_SCRIPT,
        "target_deg": GF_TARGET_DEG,
        "window_deg": [GF_WINDOW_LO_DEG, GF_WINDOW_HI_DEG],
        "lv_s": float(args.lv),
        "dt": DT,
        "fit": fit.as_dict(),
        "rise_0.20_fire_deg": rise_only,
        "flight_params": {
            "gf_form": fit.form,
            "gf_k": fit.k,
            "gf_threshold": fit.threshold,
            "gf_vel_window_s": fit.vel_window_s,
        },
        "note": (
            "Freeze these in FlightParams. Do not retune on closed-loop "
            "gates. If static/recede/no-object false-fire, the calibration "
            "is wrong — report, do not patch."
        ),
    }
    (out / "gf_readout_calibration.json").write_text(json.dumps(payload, indent=2))
    TRACES_PATH.write_text(json.dumps(traces_to_jsonable(traces), indent=2))

    print("\n=== GF readout calibration (canonical hex loom only) ===")
    print(f"citation: {GF_CITATION}")
    print(f"form={fit.form}  k={fit.k:.6g}  threshold={fit.threshold:.6g}")
    print(f"vel_window={fit.vel_window_s}s  RMS to {GF_TARGET_DEG}°={fit.rms_to_target:.2f}")
    print(f"n in {GF_WINDOW_LO_DEG}–{GF_WINDOW_HI_DEG}°: {fit.n_in_window}/3")
    for name, ang in fit.fire_angles_deg.items():
        r50 = fit.rise_at_target_hz.get(name)
        r20 = rise_only.get(name)
        print(
            f"  {name:12} fire={ang:6.2f}°  rise@50°={r50:.3f} Hz  "
            f"old 0.20-bar={r20 if r20 is None else f'{r20:.1f}°'}"
        )
    print(f"wrote {out / 'gf_readout_calibration.json'}")
    print(f"wrote {TRACES_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
