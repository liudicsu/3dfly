"""Frozen DN → yaw/roll plant calibration (stimulus-free unit step).

Literature (free-flight *Drosophila* body saccades):
  Fry, Sayaman & Dickinson 2003 J Exp Biol — ~90° in ~50–80 ms,
    peak yaw ~1000–5000 °/s.
  Tammero & Dickinson 2002 J Exp Biol — banked saccades away from
    expansion.
  Muijres et al. 2014 Science — evasive banked turns, same timescale.

Canonical input: a unit DNa02 L−R step (L−R = ``dna02_scale_hz``) through
the official controller mapping (tanh-gain, action smoothing, L−R stroke
+ residual torque). No visual stimulus. Fit is ``scripts/calibrate_saccade_plant.py``;
do not retune on closed-loop gates. GF params and brain weights stay frozen.

Inertia check (same script): fly-body Izz ~ 1e-8 kg·m², safety yaw
damping 2e-6 N·m/(rad/s) → mechanical τ ≈ 5 ms. The old 320 ms / 323 °/s
saccade was torque-limited, not inertia-limited.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Optional

import numpy as np

from threedfly.brain.readouts import DNReadout
from threedfly.control.closed_loop_metrics import (  # noqa: F401 — re-export
    mean_abs_yaw_rate_rad_s,
    peak_yaw_rate_rad_s,
    yaw_rates,
)
from threedfly.control.dn_controller import DNFlightController
from threedfly.control.exploration import heading_from_quaternion
from threedfly.control.flight_params import FlightParams

# Fry / Tammero / Muijres window.
SACCADE_T90_LO_S = 0.050
SACCADE_T90_HI_S = 0.080
SACCADE_PEAK_LO_DEG_S = 1000.0
SACCADE_PEAK_HI_DEG_S = 5000.0
SACCADE_ANGLE_DEG = 90.0
# Frozen plant scale vs the pre-calibration gains (8e-6 / 3e-6 / 5e-6).
# scripts/calibrate_saccade_plant.py; do not retune on gates.
SACCADE_PLANT_SCALE = 12.0
SACCADE_WING_YAW_GAIN0 = 8e-6
SACCADE_WING_ROLL_GAIN0 = 3e-6
SACCADE_TORQUE_MAX0 = 5e-6
SACCADE_CITATION = (
    "Fry, Sayaman & Dickinson 2003 J Exp Biol; "
    "Tammero & Dickinson 2002 J Exp Biol; "
    "Muijres et al. 2014 Science"
)
SACCADE_SCRIPT = "scripts/calibrate_saccade_plant.py"


def unit_dna02_readout(params: Optional[FlightParams] = None, side: str = "L") -> DNReadout:
    """Stimulus-free unit DNa02 L−R (one M1 scale). Other DNs stay at 0."""
    p = params or FlightParams()
    ro = DNReadout()
    if side == "L":
        ro.dna02_L = float(p.dna02_scale_hz)
    elif side == "R":
        ro.dna02_R = float(p.dna02_scale_hz)
    else:
        raise ValueError(f"side must be L or R, got {side!r}")
    return ro


def apply_plant_scale(env, scale: float) -> None:
    """Scale stroke-difference yaw/roll and residual body torque together."""
    s = float(scale)
    env.wing_yaw_gain = SACCADE_WING_YAW_GAIN0 * s
    env.wing_roll_gain = SACCADE_WING_ROLL_GAIN0 * s
    env.torque_max = SACCADE_TORQUE_MAX0 * s


def measure_unit_dn_step(
    env,
    *,
    side: str = "L",
    params: Optional[FlightParams] = None,
    n_steps: int = 40,
    scale: Optional[float] = None,
) -> dict:
    """Official controller + plant: unit DNa02 L−R, no vision."""
    p = params or FlightParams()
    # Held unit step is DC; course-rest is a separate open-space
    # subtraction and must not eat the calibration input.
    p = replace(p, course_rest_tau_steps=0.0, course_deadzone=0.0)
    if scale is not None:
        apply_plant_scale(env, scale)
    ctl = DNFlightController(p)
    ctl.bind_hover(env.hover_action)
    ctl.reset()
    ctl.capture_baseline(DNReadout())
    ro = unit_dna02_readout(p, side=side)
    env.reset(seed=0)
    env.set_pose((0.0, 0.0, 1.5), yaw=0.0)
    hover = env.hover_action
    headings = [heading_from_quaternion(env.get_fly_orientation())]
    wz = [float(env.get_fly_angular_velocity()[2])]
    h0 = headings[0]
    dt = float(env.control_dt)
    for _ in range(int(n_steps)):
        action = ctl.compute(ro)
        env.step(action, wing_strokes=ctl.last_wing_strokes)
        headings.append(heading_from_quaternion(env.get_fly_orientation()))
        wz.append(float(env.get_fly_angular_velocity()[2]))
    headings = np.asarray(headings, dtype=float)
    rel = np.arctan2(np.sin(headings - h0), np.cos(headings - h0))
    wz = np.asarray(wz, dtype=float)
    t = np.arange(rel.size) * dt
    target = np.radians(SACCADE_ANGLE_DEG) * (1.0 if side == "L" else -1.0)
    hit = np.where(np.sign(target) * rel >= abs(target))[0]
    t90 = float(t[int(hit[0])]) if hit.size else None
    peak = float(np.max(np.abs(wz)))
    return {
        "side": side,
        "t90_s": t90,
        "peak_yaw_rad_s": peak,
        "peak_yaw_deg_s": peak * 180.0 / np.pi,
        "heading_end_rad": float(rel[-1]),
        "dt_s": dt,
        "n_steps": int(n_steps),
        "scale": None if scale is None else float(scale),
        "in_t90_window": bool(t90 is not None and SACCADE_T90_LO_S <= t90 <= SACCADE_T90_HI_S),
        "in_peak_window": bool(SACCADE_PEAK_LO_DEG_S <= peak * 180.0 / np.pi <= SACCADE_PEAK_HI_DEG_S),
        "haltere_kd": float(env.haltere_kd),
        "wing_yaw_gain": float(env.wing_yaw_gain),
        "torque_max": float(env.torque_max),
        "command_yaw": float(ctl.prev_action[4]),
    }


def plant_in_window(rec: dict) -> bool:
    return bool(rec.get("in_t90_window") and rec.get("in_peak_window"))
