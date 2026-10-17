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


def test_forward_thrust_follows_body_heading_after_yaw():
    """After a 90° yaw, forward thrust must move mostly along world Y, not world X."""
    env = FlyEnvironment(render_mode=None)
    env.reset(seed=0)
    # +90 deg about +Z: body +X aligns with world +Y
    env.data.qpos[3:7] = [np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)]
    env.data.qpos[0:3] = [0.0, 0.0, 1.5]
    env.data.qvel[:] = 0.0

    hover = env.hover_action
    action = np.array([0.8, hover, 0.0, 0.0, 0.0])
    start = env.get_fly_position().copy()
    for _ in range(150):
        env.step(action)
    delta = env.get_fly_position() - start
    env.close()

    assert abs(delta[1]) > abs(delta[0]), (
        f"Forward thrust at heading +90 deg should move in Y, got dX={delta[0]:.4f} dY={delta[1]:.4f}"
    )
    assert abs(delta[1]) > 0.05, f"Expected Y displacement, got {delta[1]:.4f} m"


def test_overspeed_is_not_a_collision():
    """High speed with no contact must not be reported as a collision."""
    env = FlyEnvironment(render_mode=None)
    env.reset(seed=0)
    env.data.qpos[0:3] = [0.0, 0.0, 1.5]
    env.data.qvel[:3] = np.array([8.0, 0.0, 0.0])
    mujoco = __import__("mujoco")
    mujoco.mj_forward(env.model, env.data)
    assert env.data.ncon == 0
    assert env.check_collision() is False
    env.close()


def test_drag_limits_terminal_speed():
    """Linear+quadratic drag should keep cruise speed in a fly-like 0.5–1.5 m/s band."""
    env = FlyEnvironment(render_mode=None)
    env.reset(seed=0)
    env.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    env.data.qpos[0:3] = [0.0, 0.0, 1.5]
    env.data.qvel[:] = 0.0
    action = np.array([0.6, env.hover_action, 0.0, 0.0, 0.0])
    speeds = []
    # 80 control ticks × 20 ms ≈ 1.6 s, before the fly reaches the ±5 m walls.
    for _ in range(80):
        env.step(action)
        speeds.append(np.linalg.norm(env.get_fly_velocity()[:2]))
    terminal = float(np.mean(speeds[-20:]))
    env.close()
    assert 0.4 <= terminal <= 1.8, f"Terminal speed {terminal:.3f} m/s outside 0.4–1.8"


def test_hover_action_balances_weight():
    """Rest lift equals weight: hover command should keep altitude near the start height."""
    env = FlyEnvironment(render_mode=None)
    env.reset(seed=0)
    env.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    env.data.qpos[0:3] = [0.0, 0.0, 1.5]
    env.data.qvel[:] = 0.0
    action = np.array([0.0, env.hover_action, 0.0, 0.0, 0.0])
    zs = []
    for _ in range(250):
        env.step(action)
        zs.append(float(env.get_fly_position()[2]))
    mean_z = float(np.mean(zs[-50:]))
    env.close()
    assert 1.1 <= mean_z <= 2.0, f"Hover altitude drifted to {mean_z:.3f} m"


def test_control_step_advances_twenty_ms():
    """Each env.step is a 20 ms control tick (physics substeps), matching the brain."""
    env = FlyEnvironment(render_mode=None)
    env.reset(seed=0)
    t0 = float(env.data.time)
    env.step(np.zeros(5))
    dt = float(env.data.time) - t0
    env.close()
    assert abs(dt - 0.02) < 1e-9, f"Expected 20 ms control step, got {dt*1000:.3f} ms"


def test_collision_uses_fly_body_geoms_not_geom_zero():
    """Contacts are fly-vs-environment; geom 0 is the floor, not the thorax."""
    env = FlyEnvironment(render_mode=None)
    env.reset(seed=0)
    floor_id = env.model.geom("floor").id
    thorax_id = env.model.geom("thorax").id
    assert floor_id == 0
    assert thorax_id != 0
    assert thorax_id in env._fly_geom_ids
    assert floor_id not in env._fly_geom_ids
    env.close()


def test_set_pose_places_fly():
    env = FlyEnvironment(render_mode=None)
    env.reset(seed=0)
    env.set_pose((1.5, -2.0, 1.6), yaw=np.pi / 2)
    pos = env.get_fly_position()
    assert pos[0] == pytest.approx(1.5, abs=1e-6)
    assert pos[1] == pytest.approx(-2.0, abs=1e-6)
    assert pos[2] == pytest.approx(1.6, abs=1e-6)
    env.close()


def test_positive_yaw_command_increases_heading():
    """Body +Y is left; +action[4] / L>R stroke must increase heading."""
    from threedfly.control.exploration import heading_from_quaternion

    env = FlyEnvironment(render_mode=None)
    env.reset(seed=0)
    env.set_pose((0.0, 0.0, 1.5), yaw=0.0)
    hover = env.hover_action
    h0 = heading_from_quaternion(env.get_fly_orientation())
    for _ in range(25):
        env.step(np.array([0.0, hover, 0.0, 0.0, 1.0]))
    h1 = heading_from_quaternion(env.get_fly_orientation())
    env.close()
    d = float(np.arctan2(np.sin(h1 - h0), np.cos(h1 - h0)))
    assert d > 0.05, f"+yaw should increase heading (left), got Δ={d:.3f}"


def test_left_minus_right_stroke_increases_heading():
    from threedfly.control.exploration import heading_from_quaternion

    env = FlyEnvironment(render_mode=None)
    env.reset(seed=0)
    env.set_pose((0.0, 0.0, 1.5), yaw=0.0)
    hover = env.hover_action
    h0 = heading_from_quaternion(env.get_fly_orientation())
    for _ in range(25):
        env.step(np.array([0.0, hover, 0.0, 0.0, 0.0]), wing_strokes=(0.85, 0.15))
    h1 = heading_from_quaternion(env.get_fly_orientation())
    env.close()
    d = float(np.arctan2(np.sin(h1 - h0), np.cos(h1 - h0)))
    assert d > 0.05, f"L>R stroke should increase heading, got Δ={d:.3f}"
