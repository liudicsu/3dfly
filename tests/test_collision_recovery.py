"""Tests for collision recovery functionality."""

import numpy as np
import pytest
from threedfly.sim import FlyEnvironment


def test_collision_recovery_moves_fly():
    """Test that collision recovery moves fly away from collision point."""
    env = FlyEnvironment(render_mode=None)
    env.reset(seed=42)
    
    # Move fly above the actual room ceiling (walls span z≈0–4 m)
    env.data.qpos[2] = 3.9
    initial_z = env.data.qpos[2]
    
    # Trigger recovery
    env.recover_from_collision()
    
    # Check that fly moved down from ceiling
    assert env.data.qpos[2] < initial_z, "Fly should move down after ceiling collision"
    assert env.data.qpos[2] >= 0.3, "Fly should stay above ground"
    
    env.close()


def test_collision_recovery_dampens_velocity():
    """Test that collision recovery dampens velocity to prevent immediate re-collision."""
    env = FlyEnvironment(render_mode=None)
    env.reset(seed=42)
    
    # Set high velocity
    env.data.qvel[:3] = np.array([2.0, 2.0, 2.0])
    initial_vel = np.linalg.norm(env.data.qvel[:3])
    
    # Trigger recovery
    env.recover_from_collision()
    
    # Check that velocity was dampened (now gentler: 0.3 factor instead of 0.05)
    final_vel = np.linalg.norm(env.data.qvel[:3])
    assert final_vel < initial_vel * 0.4, "Velocity should be dampened"
    
    env.close()


def test_collision_recovery_does_not_teleport_randomly():
    """Recovery is a small deterministic nudge, not a random teleport."""
    env = FlyEnvironment(render_mode=None)
    env.reset(seed=0)
    env.data.qpos[:3] = np.array([0.4, -0.3, 1.4])
    env.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    start = env.data.qpos[:3].copy()

    env.recover_from_collision()
    first = env.data.qpos[:3].copy()
    env.data.qpos[:3] = start
    env.recover_from_collision()
    second = env.data.qpos[:3].copy()

    assert np.allclose(first, second), "Recovery must be deterministic"
    assert np.linalg.norm(first - start) <= 0.35, "Recovery must not teleport far"
    env.close()


def test_collision_recovery_keeps_orientation():
    """Test that minimal collision recovery keeps orientation (no randomization)."""
    env = FlyEnvironment(render_mode=None)
    env.reset(seed=42)
    
    initial_quat = env.data.qpos[3:7].copy()
    
    # Trigger recovery - orientation should NOT change (minimal recovery)
    env.recover_from_collision()
    final_quat = env.data.qpos[3:7]
    
    # Check that orientation stayed the same
    diff = np.linalg.norm(initial_quat - final_quat)
    assert diff < 0.01, "Minimal recovery should not randomize orientation"
    
    env.close()


def test_collision_recovery_stays_in_bounds():
    """Test that collision recovery keeps fly within room bounds."""
    env = FlyEnvironment(render_mode=None)
    env.reset(seed=42)
    
    # Test multiple random positions
    np.random.seed(42)
    for _ in range(20):
        # Random position
        env.data.qpos[0] = np.random.uniform(-3, 3)
        env.data.qpos[1] = np.random.uniform(-3, 3)
        env.data.qpos[2] = np.random.uniform(0, 3.5)
        
        # Trigger recovery
        env.recover_from_collision()
        
        # Check bounds: room walls at ±5 m / z≈4 m, not an inner ±1.9 box
        pos = env.get_fly_position()
        assert -4.9 <= pos[0] <= 4.9, f"X position {pos[0]} out of bounds"
        assert -4.9 <= pos[1] <= 4.9, f"Y position {pos[1]} out of bounds"
        assert 0.2 <= pos[2] <= 3.85, f"Z position {pos[2]} out of bounds"
    
    env.close()


def test_collision_recovery_does_not_clamp_xy_to_inner_box():
    """Open-room positions must not be snapped to ±1.9 m (walls are at ±5 m)."""
    env = FlyEnvironment(render_mode=None)
    env.reset(seed=0)
    env.data.qpos[:3] = np.array([2.4, -2.3, 1.5])
    env.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    env.data.qvel[:] = 0.0
    env.recover_from_collision()
    pos = env.get_fly_position()
    assert abs(pos[0]) > 1.95, f"X was clamped inward to {pos[0]:.3f}"
    assert abs(pos[1]) > 1.95, f"Y was clamped inward to {pos[1]:.3f}"
    assert abs(pos[0]) < 4.9 and abs(pos[1]) < 4.9
    env.close()


def test_soft_recovery_extends_flight():
    """Test that soft recovery allows fly to continue flying longer."""
    from threedfly.connectome import ConnectomeLoader
    from threedfly.connectome.simulator import RateSimulator
    pytest.importorskip("open3d")
    from threedfly.vision import StereoVision, PointCloudMapper
    from threedfly.control import FlightController, ExplorationPolicy
    
    np.random.seed(42)
    
    # Load minimal subgraph
    subgraph = ConnectomeLoader.load_subgraph("data/demo_subgraph.npz")
    
    def run_simulation(enable_recovery, max_steps=500):
        """Helper to run simulation with or without recovery."""
        brain = RateSimulator(subgraph["adjacency"], dt=0.010)
        env = FlyEnvironment(render_mode=None, camera_width=64, camera_height=48)
        stereo = StereoVision(baseline=0.012, focal_length=0.01)
        mapper = PointCloudMapper(voxel_size=0.05)
        controller = FlightController(subgraph["n_neurons"], use_rate_model=True)
        explorer = ExplorationPolicy(exploration_weight=0.8)
        
        left_img, right_img = env.reset(seed=42)
        
        collision_count = 0
        steps_completed = 0
        
        for step in range(max_steps):
            visual_features = stereo.get_visual_features(left_img, right_img)
            brain_input = np.concatenate([
                visual_features["left_ommatidia"],
                visual_features["right_ommatidia"]
            ])
            
            brain.set_input(brain_input)
            brain_activity = brain.step()
            
            fly_pos = env.get_fly_position()
            exploration_signal = explorer.compute_exploration_signal(fly_pos, mapper)
            
            action = controller.compute_action(brain_activity, exploration_signal)
            left_img, right_img, info = env.step(action)
            
            if env.check_collision():
                collision_count += 1
                if enable_recovery:
                    env.recover_from_collision()
                    if collision_count >= 10:  # Limit for test
                        break
                else:
                    break  # Stop on first collision (old behavior)
            
            steps_completed = step + 1
        
        env.close()
        return steps_completed, collision_count
    
    # Test without recovery (stops on first real contact)
    steps_no_recovery, collisions_no_recovery = run_simulation(enable_recovery=False)
    
    # Test with recovery (continues after contacts)
    steps_with_recovery, collisions_with_recovery = run_simulation(enable_recovery=True)
    
    # Overspeed is no longer a fake collision, so the demo subgraph may finish
    # both runs. Recovery must never shorten the flight; if contacts occur it
    # must extend it.
    assert steps_with_recovery >= steps_no_recovery, \
        f"Soft recovery shortened flight: {steps_with_recovery} vs {steps_no_recovery} steps"
    if collisions_no_recovery > 0:
        assert steps_with_recovery > steps_no_recovery, \
            f"Soft recovery should extend flight: {steps_with_recovery} vs {steps_no_recovery} steps"
    
    print(f"\n✅ Flight steps {steps_no_recovery} → {steps_with_recovery}")
    print(f"   Collisions {collisions_no_recovery} without recovery vs {collisions_with_recovery} with recovery")


if __name__ == "__main__":
    # Run tests
    test_collision_recovery_moves_fly()
    print("✓ test_collision_recovery_moves_fly")
    
    test_collision_recovery_dampens_velocity()
    print("✓ test_collision_recovery_dampens_velocity")
    
    test_collision_recovery_keeps_orientation()
    print("✓ test_collision_recovery_keeps_orientation")
    
    test_collision_recovery_stays_in_bounds()
    print("✓ test_collision_recovery_stays_in_bounds")
    
    test_soft_recovery_extends_flight()
    print("✓ test_soft_recovery_extends_flight")
    
    print("\n✅ All collision recovery tests passed!")
