#!/usr/bin/env python3
"""Still-fly camera disc vs hex_ref at az 35°, l/v 0.04 after stimulus fix."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import importlib.util

_spec = importlib.util.spec_from_file_location("eye_factorial", ROOT / "scripts" / "eye_factorial.py")
_fac = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_fac)
AZ_DEG = _fac.AZ_DEG
DIST0 = _fac.DIST0
DT = _fac.DT
PRE_S = _fac.PRE_S
STIM_S = _fac.STIM_S
camera_movie = _fac.camera_movie
hex_ref_movie = _fac.hex_ref_movie
hex_stats = _fac.hex_stats
run_trace = _fac.run_trace
summarise = _fac.summarise
from threedfly.control.flight_params import FlightParams


def main() -> int:
    from threedfly.brain.closed_loop import ClosedLoopBrain, core_stack_available
    from threedfly.brain.constants import flyvis_available
    from threedfly.sim import FlyEnvironment

    if not flyvis_available() or not core_stack_available():
        return 2
    params = FlightParams()
    brain = ClosedLoopBrain.build(params=params)
    brain.params.flyvis_stride = 1
    env = FlyEnvironment(render_mode="rgb_array")
    env.reset(seed=0)
    env.set_pose((0.0, 0.0, 1.5), yaw=0.0, vel=(0.0, 0.0, 0.0))
    hL, hR, th, room = camera_movie(env, 0.04, DIST0, DT, PRE_S, STIM_S, translate=False)
    env.close()
    mid = int(round(PRE_S / DT)) + 20
    cam = {**summarise(run_trace(brain, hL, hR, pre_hex=room), th), "frame": hex_stats(hL[mid])}
    hL_r, hR_r, th_r = hex_ref_movie(0.04, DT, PRE_S, STIM_S)
    href = {**summarise(run_trace(brain, hL_r, hR_r), th_r), "frame": hex_stats(hL_r[mid])}
    out = {"az_deg": AZ_DEG, "lv": 0.04, "camera": cam, "hex_ref": href}
    path = ROOT / "output" / "m2" / "eye_disc_onset.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
