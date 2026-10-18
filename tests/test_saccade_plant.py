"""Frozen DN → yaw plant: unit DNa02 step matches literature saccades."""

from __future__ import annotations

import numpy as np
import pytest

from threedfly.control.saccade_plant import (
    SACCADE_PLANT_SCALE,
    SACCADE_TORQUE_MAX0,
    SACCADE_WING_ROLL_GAIN0,
    SACCADE_WING_YAW_GAIN0,
    measure_unit_dn_step,
    plant_in_window,
)
from threedfly.sim.environment import WING_ROLL_GAIN, WING_YAW_GAIN


def test_frozen_plant_gains_are_the_calibrated_scale():
    assert SACCADE_PLANT_SCALE == 12.0
    assert WING_YAW_GAIN == pytest.approx(SACCADE_WING_YAW_GAIN0 * SACCADE_PLANT_SCALE)
    assert WING_ROLL_GAIN == pytest.approx(SACCADE_WING_ROLL_GAIN0 * SACCADE_PLANT_SCALE)


def test_unit_dna02_step_is_in_saccade_window():
    from threedfly.sim import FlyEnvironment

    env = FlyEnvironment(render_mode=None)
    assert env.torque_max == pytest.approx(SACCADE_TORQUE_MAX0 * SACCADE_PLANT_SCALE)
    left = measure_unit_dn_step(env, side="L")
    right = measure_unit_dn_step(env, side="R")
    env.close()
    assert plant_in_window(left), left
    assert plant_in_window(right), right
    assert left["heading_end_rad"] > 0
    assert right["heading_end_rad"] < 0
    assert abs(left["peak_yaw_deg_s"] - right["peak_yaw_deg_s"]) < 50.0


def test_yaw_inertia_is_not_the_rate_limiter():
    from threedfly.sim import FlyEnvironment

    env = FlyEnvironment(render_mode=None)
    bid = env._fly_body_id
    izz = float(env.model.body_inertia[bid][2])
    tau = izz / max(float(env.haltere_kd), 1e-12)
    env.close()
    assert izz < 5e-8
    assert tau < 0.02
