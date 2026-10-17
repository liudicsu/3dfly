"""Stroke-averaged quasi-steady wing aerodynamics.

A full ~200 Hz wingbeat at 2 ms MuJoCo substeps is expensive and stiff.
Lift and yaw/roll are therefore the cycle-averaged forces from left/right
stroke amplitude, while wing joints are animated visually only.

Convention (body frame, +X forward, +Y left, +Z dorsal):
  - sum(L, R) stroke → upward lift
  - L − R stroke → positive yaw (and a smaller roll)
  - hover stroke = weight / lift_max so rest lift balances weight
  - lift_max is ~3× body weight (set by the environment)
"""

from typing import Tuple

import numpy as np

# Default lever arm (m) at the 10× life-size visual scale (~2.5 cm body).
DEFAULT_WING_SPAN = 0.016


def stroke_averaged_forces(
    stroke_left: float,
    stroke_right: float,
    lift_max: float,
    wing_span: float = DEFAULT_WING_SPAN,
    yaw_gain: float = None,
    roll_gain: float = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Map left/right stroke amplitudes in [0, 1] to body-frame force and torque.

    Args:
        stroke_left: Left wing stroke amplitude in [0, 1]
        stroke_right: Right wing stroke amplitude in [0, 1]
        lift_max: Peak combined lift (N), typically ~3× body weight
        wing_span: Effective aerodynamic lever (m)
        yaw_gain: Yaw torque per unit (L−R). Default 0.5 * lift_max * wing_span
        roll_gain: Roll torque per unit (L−R). Default 0.35 * lift_max * wing_span

    Returns:
        force (3,) body-frame Newtons, torque (3,) body-frame N·m
    """
    s_l = float(np.clip(stroke_left, 0.0, 1.0))
    s_r = float(np.clip(stroke_right, 0.0, 1.0))
    lift = 0.5 * float(lift_max) * (s_l + s_r)
    force = np.array([0.0, 0.0, lift], dtype=np.float64)

    delta = s_l - s_r
    if yaw_gain is None:
        yaw_gain = 0.5 * float(lift_max) * float(wing_span)
    if roll_gain is None:
        roll_gain = 0.35 * float(lift_max) * float(wing_span)
    torque = np.array(
        [float(roll_gain) * delta, 0.0, float(yaw_gain) * delta],
        dtype=np.float64,
    )
    return force, torque


def visual_wing_angles(
    phase: float,
    stroke_left: float,
    stroke_right: float,
    max_angle: float = 0.7,
) -> Tuple[float, float]:
    """Visual-only flapping:  opposite-phase sinusoid scaled by stroke amplitude."""
    s_l = float(np.clip(stroke_left, 0.0, 1.0))
    s_r = float(np.clip(stroke_right, 0.0, 1.0))
    left = float(max_angle) * s_l * np.sin(phase)
    right = float(max_angle) * s_r * np.sin(phase + np.pi)
    return left, right
