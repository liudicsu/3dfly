#!/usr/bin/env python3
"""Calibrate the DN → yaw/roll plant ONCE on a stimulus-free unit step.

Canonical input (do not add more conditions; do not retune after):
    unit DNa02 L−R = ``dna02_scale_hz`` through ``DNFlightController``
    (tanh-gain, action_alpha, L−R stroke + residual torque). No images.

Target: 90° in 50–80 ms, peak yaw 1000–5000 °/s
    (Fry, Sayaman & Dickinson 2003 J Exp Biol; Tammero & Dickinson 2002;
    Muijres et al. 2014 Science).

Also reports body Izz and safety-damping time constant so an oversized
inertia is not mistaken for a weak mapping.

Writes ``output/m2/saccade_plant_calibration.json``. Frozen scale is
copied into ``threedfly/control/saccade_plant.py`` and the environment
gains; this script is the citation trail, not a live retuner.

    python scripts/calibrate_saccade_plant.py --output output/m2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threedfly.control.saccade_plant import (
    SACCADE_CITATION,
    SACCADE_PEAK_HI_DEG_S,
    SACCADE_PEAK_LO_DEG_S,
    SACCADE_PLANT_SCALE,
    SACCADE_SCRIPT,
    SACCADE_T90_HI_S,
    SACCADE_T90_LO_S,
    apply_plant_scale,
    measure_unit_dn_step,
    plant_in_window,
)


def _inertia_report(env) -> dict:
    bid = env._fly_body_id
    izz = float(env.model.body_inertia[bid][2])
    kd = float(env.haltere_kd)
    return {
        "mass_kg": float(env.model.body_subtreemass[bid]),
        "body_izz_kgm2": izz,
        "haltere_kd": kd,
        "mech_tau_s": izz / max(kd, 1e-12),
        "note": (
            "Izz ~1e-8 and τ=I/kd ~5 ms: the pre-calibration 323 °/s / "
            "320 ms saccade was torque-limited, not an oversized inertia."
        ),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default=str(ROOT / "output" / "m2"))
    p.add_argument("--search", action="store_true", help="Sweep scale; default is the frozen value")
    args = p.parse_args(argv)

    from threedfly.sim import FlyEnvironment

    env = FlyEnvironment(render_mode=None)
    inertia = _inertia_report(env)
    print(f"inertia Izz={inertia['body_izz_kgm2']:.4g} τ={inertia['mech_tau_s']*1000:.2f} ms", flush=True)

    scales = [SACCADE_PLANT_SCALE]
    if args.search:
        scales = [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

    rows = []
    chosen = None
    for s in scales:
        apply_plant_scale(env, s)
        left = measure_unit_dn_step(env, side="L", scale=s)
        right = measure_unit_dn_step(env, side="R", scale=s)
        rec = {
            "scale": s,
            "L": left,
            "R": right,
            "pass": plant_in_window(left) and plant_in_window(right),
        }
        rows.append(rec)
        print(
            f"  scale={s:.1f} L t90={left['t90_s']} peak={left['peak_yaw_deg_s']:.0f} °/s "
            f"R t90={right['t90_s']} peak={right['peak_yaw_deg_s']:.0f} °/s pass={rec['pass']}",
            flush=True,
        )
        if rec["pass"] and chosen is None:
            chosen = rec

    env.close()
    if chosen is None:
        chosen = next((r for r in rows if r["scale"] == SACCADE_PLANT_SCALE), rows[-1])

    out = {
        "citation": SACCADE_CITATION,
        "script": SACCADE_SCRIPT,
        "target": {
            "t90_s": [SACCADE_T90_LO_S, SACCADE_T90_HI_S],
            "peak_yaw_deg_s": [SACCADE_PEAK_LO_DEG_S, SACCADE_PEAK_HI_DEG_S],
            "input": "unit DNa02 L-R = dna02_scale_hz through DNFlightController",
        },
        "inertia": inertia,
        "frozen_scale": SACCADE_PLANT_SCALE,
        "chosen": chosen,
        "sweep": rows if args.search else None,
        "note": (
            "Scale multiplies WING_YAW_GAIN, WING_ROLL_GAIN, and torque_max. "
            "GF readout and brain weights are not touched."
        ),
    }
    dest = Path(args.output)
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / "saccade_plant_calibration.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"wrote {path} frozen_scale={SACCADE_PLANT_SCALE} pass={chosen['pass']}", flush=True)
    return 0 if chosen["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
