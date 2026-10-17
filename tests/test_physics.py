"""Tests that flight actuators can actually lift the fly and move it horizontally."""

import numpy as np
import pytest

from threedfly.sim import FlyEnvironment


def _fly_mass(env: FlyEnvironment) -> float:
    fly_id = env.model.body("fly").id
    return float(env.model.body_subtreemass[fly_id])


def _actuator_gain(env: FlyEnvironment, name: str) -> float:
    actuator_id = env.model.actuator(name).id
    return float(env.model.actuator_gainprm[actuator_id, 0])


def test_max_up_thrust_exceeds_weight():
    """Max upward thrust must exceed fly weight so gravity does not always win."""
    env = FlyEnvironment(render_mode=None)
    env.reset(seed=0)

    mass = _fly_mass(env)
    weight = mass * 9.81
    gain = _actuator_gain(env, "thrust_up")
    max_force = abs(env.thrust_max * gain)

    assert mass > 0
    assert max_force > weight, (
        f"Max up thrust {max_force:.6g} N must exceed weight {weight:.6g} N "
        f"(mass={mass:.6g} kg)"
    )
    env.close()


def test_constant_forward_command_displaces_horizontally():
    """A constant forward + hover command must produce horizontal displacement."""
    env = FlyEnvironment(render_mode=None)
    env.reset(seed=1)

    # Identity orientation so body +x is world +x
    env.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    env.data.qpos[0:3] = [0.0, 0.0, 1.5]
    env.data.qvel[:] = 0.0

    mass = _fly_mass(env)
    gain = _actuator_gain(env, "thrust_up")
    hover = float(np.clip((mass * 9.81) / max(abs(env.thrust_max * gain), 1e-12), 0.0, 1.0))
    action = np.array([0.8, hover, 0.0, 0.0, 0.0])

    start = env.get_fly_position().copy()
    for _ in range(200):
        env.step(action)
    end = env.get_fly_position()

    horizontal = np.linalg.norm(end[:2] - start[:2])
    assert horizontal > 0.05, (
        f"Expected horizontal motion from forward thrust, got {horizontal:.4f} m "
        f"(start={start}, end={end})"
    )
    env.close()
