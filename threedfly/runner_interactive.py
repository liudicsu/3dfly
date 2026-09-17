"""Interactive simulation runner with unified 3D + dashboard web interface."""

import numpy as np
from pathlib import Path
from tqdm import tqdm
import time
import sys

from threedfly.connectome import ConnectomeLoader, BrainSimulator
from threedfly.connectome.simulator import RateSimulator
from threedfly.sim import FlyEnvironment
from threedfly.vision import StereoVision, PointCloudMapper, BrainDepthEstimator
from threedfly.control import FlightController, ExplorationPolicy
from threedfly.viz import InteractiveVisualizer


def run_interactive_simulation(
    subgraph_path: str,
    max_steps: int = 5000,
    save_output_dir: str = "output",
    simulator_type: str = "rate",
    seed: int = 42,
    host: str = "0.0.0.0",
    port: int = 8080,
    max_collisions: int = 100,
    enable_soft_recovery: bool = True,
):
    """
    Run interactive 3dfly simulation with unified web interface.
    
    Single-page web interface featuring:
    - Interactive 3D god's-eye view: room, fly, trajectory, point cloud
    - Dashboard panels: left/right eye views, trajectory plot, brain activity, 
      flight commands, and system status
    - Play/pause/step/reset controls
    - Adjustable playback speed
    
    Args:
        subgraph_path: Path to subgraph npz file
        max_steps: Maximum simulation steps
        save_output_dir: Output directory
        simulator_type: "rate" or "lif"
        seed: Random seed
        host: Visualization server host
        port: Visualization server port
        max_collisions: Maximum collisions before stopping (0 = unlimited)
        enable_soft_recovery: Enable soft collision recovery (back off and continue)
    """
    np.random.seed(seed)
    
    # Output directory
    output_dir = Path(save_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("3dfly: Interactive Connectome-Driven Flight Simulator")
    print("=" * 70)
    
    print("\n[1/7] Loading connectome subgraph...")
    subgraph = ConnectomeLoader.load_subgraph(Path(subgraph_path))
    n_neurons = subgraph["n_neurons"]
    adjacency = subgraph["adjacency"]
    print(f"  ✓ Loaded {n_neurons} neurons")
    
    print("\n[2/7] Initializing brain simulator ({})...".format(simulator_type))
    if simulator_type == "lif":
        brain = BrainSimulator(adjacency, dt=0.001)
        brain_dt = 0.001
    else:
        brain = RateSimulator(adjacency, dt=0.010)
        brain_dt = 0.010
    print(f"  ✓ Brain ready (dt={brain_dt}s)")
    
    print("\n[3/7] Creating MuJoCo environment...")
    env = FlyEnvironment(
        render_mode="rgb_array",
        camera_width=160,
        camera_height=120
    )
    print("  ✓ Environment ready")
    
    print("\n[4/7] Initializing vision and mapping...")
    stereo = StereoVision(baseline=0.012, focal_length=0.01)
    mapper = PointCloudMapper(voxel_size=0.05)
    print("  ✓ Vision system ready")
    
    print("\n[5/8] Setting up brain depth estimation...")
    brain_depth = BrainDepthEstimator(
        n_neurons=n_neurons,
        adjacency=adjacency,
        use_rate_model=(simulator_type == "rate")
    )
    print("  ✓ Brain depth estimator ready")
    
    print("\n[6/8] Setting up control and exploration...")
    controller = FlightController(n_neurons, use_rate_model=(simulator_type == "rate"))
    explorer = ExplorationPolicy(exploration_weight=0.8)  # Increased from 0.5 for stronger exploration
    print("  ✓ Control system ready")
    
    print("\n[7/8] Initializing interactive web interface...")
    viz = InteractiveVisualizer(host=host, port=port)
    print("  ✓ Unified web interface ready (3D view + dashboard panels)")
    
    print("\n[8/8] Starting interactive simulation...")
    print("=" * 70)
    print("\n💡 Using BRAIN-BASED DEPTH for primary reconstruction")
    print("   StereoBM available as comparison baseline")
    print("=" * 70)
    print("\n📋 Controls:")
    print("  - Open the web browser to interact with the 3D scene")
    print("  - Use mouse to orbit, pan, and zoom")
    print("  - Use the UI panel to play/pause, step, and reset")
    print("  - Press Ctrl+C to exit")
    print("\n" + "=" * 70 + "\n")
    
    # Simulation parameters
    sim_dt = 0.02  # 20ms per step
    brain_steps_per_sim = int(sim_dt / brain_dt)
    viz_update_interval = 3  # Update viz every N steps (more frequent for smoother point cloud)
    
    # Initial state
    def reset_simulation():
        """Reset simulation to initial state."""
        left_img, right_img = env.reset(seed=seed)
        mapper.__init__(voxel_size=0.05)  # Reset mapper
        brain_depth.reset()  # Reset brain depth estimator
        viz.reset_trajectory()
        return left_img, right_img, 0, 0  # step, collision_count
    
    left_img, right_img, step, collision_count = reset_simulation()
    
    # Main simulation loop
    try:
        while step < max_steps:
            # Check control state
            controls = viz.check_controls()
            
            # Handle reset
            if controls["should_reset"]:
                print("\n🔄 Resetting simulation...")
                left_img, right_img, step, collision_count = reset_simulation()
                continue
            
            # Handle pause (unless stepping)
            if controls["paused"] and not controls["should_step"]:
                time.sleep(0.05)  # Small delay to reduce CPU usage
                continue
            
            # Adjust simulation speed
            speed = controls["speed"]
            
            # === Simulation Step ===
            
            # Get visual input
            visual_features = stereo.get_visual_features(left_img, right_img)
            
            # *** NEW: Get fly velocity for motion parallax ***
            fly_velocity = env.get_fly_velocity()
            
            # *** NEW: Prepare optical flow features for brain (motion parallax) ***
            flow_features, flow_info = brain_depth.prepare_flow_features_for_brain(
                left_img,
                right_img,
                fly_velocity
            )
            
            # Feed to brain: ommatidial samples + looming + FLOW FEATURES (motion parallax)
            brain_input = np.concatenate([
                visual_features["left_ommatidia"],
                visual_features["right_ommatidia"],
                visual_features["looming_features"],  # Obstacle proximity signals
                flow_features  # *** NEW: Motion parallax/optic flow features ***
            ])
            
            # Run brain for multiple timesteps (connectome processes flow features)
            for _ in range(brain_steps_per_sim):
                brain.set_input(brain_input)
                if simulator_type == "lif":
                    brain.step()
                    brain_activity = brain.get_firing_rates(window=50)
                else:
                    brain_activity = brain.step()
            
            # *** BRAIN-BASED DEPTH ESTIMATION (now using motion parallax) ***
            # Extract depth from brain activity after connectome processing
            visual_features_with_flow = {
                **visual_features,
                "flow_magnitude": flow_info["flow_magnitude"],
                "left_img": left_img,
                "right_img": right_img,
            }
            brain_depth_result = brain_depth.estimate_depth_from_activity(
                brain_activity,
                stereo_features=visual_features_with_flow,
                velocity=fly_velocity
            )
            
            # Upsample brain depth map to match camera resolution
            h, w = visual_features["depth_map"].shape
            brain_depth_map_upsampled = brain_depth.upsample_depth_map(
                brain_depth_result["depth_map"],
                target_height=h,
                target_width=w
            )
            brain_confidence_upsampled = brain_depth.upsample_depth_map(
                brain_depth_result["confidence"],
                target_height=h,
                target_width=w
            )
            
            # Get exploration signal
            fly_pos = env.get_fly_position()
            exploration_signal = explorer.compute_exploration_signal(fly_pos, mapper)
            
            # Compute control action
            action = controller.compute_action(brain_activity, exploration_signal)
            
            # Step environment
            left_img, right_img, info = env.step(action)
            
            # Update point cloud map
            left_pose, right_pose = env.get_camera_poses()
            
            # PRIMARY: Add brain-depth observation (every 3 steps for frequent updates)
            if step % 3 == 0:
                n_added = mapper.add_depth_observation(
                    brain_depth_map_upsampled,
                    visual_features["rgb_for_cloud"],
                    left_pose,
                    brain_confidence_upsampled,
                    min_confidence=0.15,  # Lower threshold for brain estimates
                    source="brain"
                )
                
                if step % 30 == 0 and n_added > 0:
                    # Periodic logging of point cloud growth
                    print(f"  Added {n_added} brain-depth points to cloud")
            
            # COMPARISON: Add StereoBM observation (every 2 steps, denser for informative comparison)
            if step % 2 == 0:
                mapper.add_depth_observation(
                    visual_features["depth_map"],
                    visual_features["rgb_for_cloud"],
                    left_pose,
                    visual_features["confidence"],
                    min_confidence=0.08,  # Lower threshold for denser comparison
                    source="stereo"
                )
            
            # Update visualization
            if step % viz_update_interval == 0:
                fly_orientation = env.get_fly_orientation()
                brain_stats = brain.get_population_activity()
                control_stats = controller.get_status()
                
                viz.update(
                    fly_position=fly_pos,
                    fly_orientation=fly_orientation,
                    left_img=left_img,
                    right_img=right_img,
                    brain_activity=brain_activity,
                    control_action=action,
                    point_cloud_mapper=mapper,
                    brain_stats=brain_stats,
                    control_stats=control_stats,
                )
            
            # Check collision
            if env.check_collision():
                collision_count += 1
                if enable_soft_recovery:
                    print(f"\n⚠️  Collision {collision_count} at step {step} - recovering...")
                    env.recover_from_collision()
                    # Check if we've hit max collisions
                    if max_collisions > 0 and collision_count >= max_collisions:
                        print(f"\n🛑 Reached maximum collisions ({max_collisions}).")
                        print("Press reset to continue or Ctrl+C to exit.")
                        # Pause and wait for reset
                        while not controls["should_reset"]:
                            controls = viz.check_controls()
                            time.sleep(0.1)
                        continue
                else:
                    print(f"\n⚠️  Collision detected at step {step}!")
                    print("Press reset to continue or Ctrl+C to exit.")
                    # Pause and wait for reset
                    while not controls["should_reset"]:
                        controls = viz.check_controls()
                        time.sleep(0.1)
                    continue
            
            step += 1
            
            # Adjust frame rate based on speed
            time.sleep(sim_dt / speed)
            
            # Print progress occasionally
            if step % 100 == 0:
                map_stats = mapper.get_statistics()
                print(f"Step {step}/{max_steps} | "
                      f"Pos: ({fly_pos[0]:.1f}, {fly_pos[1]:.1f}, {fly_pos[2]:.1f}) | "
                      f"Points: {map_stats['total_points']} | "
                      f"Exploration: {100*map_stats['exploration_ratio']:.1f}%")
    
    except KeyboardInterrupt:
        print("\n\n⏹️  Simulation interrupted by user.")
    
    # Save outputs
    print("\n" + "=" * 70)
    print("Saving outputs...")
    print("=" * 70)
    
    # Point clouds (both brain and stereo)
    brain_pc_path = output_dir / "point_cloud_brain.ply"
    stereo_pc_path = output_dir / "point_cloud_stereo.ply"
    
    # Export with very dense voxel size for inspection (1cm instead of 5cm/1.25cm)
    mapper.save_point_cloud(str(brain_pc_path), voxel_size=0.01, source="brain")
    mapper.save_point_cloud(str(stereo_pc_path), voxel_size=0.01, source="stereo")
    
    # Statistics
    map_stats = mapper.get_statistics()
    brain_depth_stats = brain_depth.get_statistics()
    
    print("\n" + "=" * 70)
    print("Final Statistics:")
    print("=" * 70)
    print(f"Total simulation steps: {step}")
    print(f"Total collisions: {collision_count}")
    print(f"Brain neurons: {n_neurons}")
    print(f"\nPrimary (Brain Depth):")
    print(f"  Point cloud points: {map_stats['total_points']}")
    print(f"  Mean depth: {brain_depth_stats.get('mean_depth_history', 0):.2f}m")
    print(f"\nComparison (StereoBM):")
    print(f"  Point cloud points: {map_stats['stereo_points']}")
    print(f"\nExploration:")
    print(f"  Occupied voxels: {map_stats['occupied_voxels']}")
    print(f"  Exploration ratio: {100 * map_stats['exploration_ratio']:.2f}%")
    print("=" * 70)
    
    # Keep visualization open
    print("\n✨ Simulation complete!")
    print("The visualization server is still running.")
    print("You can continue to explore the 3D scene in your browser.")
    print("Press Ctrl+C to exit.\n")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    
    # Cleanup
    viz.close()
    env.close()
    
    print("\n👋 Goodbye!\n")


if __name__ == "__main__":
    # Quick test
    run_interactive_simulation(
        subgraph_path="data/demo_subgraph.npz",
        max_steps=2000,
        save_output_dir="output",
        simulator_type="rate",
        seed=42,
        host="0.0.0.0",
        port=8080,
    )
