"""Interactive simulation runner with god's-eye 3D visualization."""

import numpy as np
from pathlib import Path
from tqdm import tqdm
import time
import sys

from threedfly.connectome import ConnectomeLoader, BrainSimulator
from threedfly.connectome.simulator import RateSimulator
from threedfly.sim import FlyEnvironment
from threedfly.vision import StereoVision, PointCloudMapper
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
):
    """
    Run interactive 3dfly simulation with god's-eye 3D visualization.
    
    Features:
    - Interactive 3D view with orbit/pan/zoom
    - Play/pause/step/reset controls
    - Real-time visualization of fly, trajectory, and point cloud
    - Adjustable playback speed
    
    Args:
        subgraph_path: Path to subgraph npz file
        max_steps: Maximum simulation steps
        save_output_dir: Output directory
        simulator_type: "rate" or "lif"
        seed: Random seed
        host: Visualization server host
        port: Visualization server port
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
        camera_width=64,
        camera_height=48
    )
    print("  ✓ Environment ready")
    
    print("\n[4/7] Initializing vision and mapping...")
    stereo = StereoVision(baseline=0.012, focal_length=0.01)
    mapper = PointCloudMapper(voxel_size=0.05)
    print("  ✓ Vision system ready")
    
    print("\n[5/7] Setting up control and exploration...")
    controller = FlightController(n_neurons, use_rate_model=(simulator_type == "rate"))
    explorer = ExplorationPolicy(exploration_weight=0.5)
    print("  ✓ Control system ready")
    
    print("\n[6/7] Initializing interactive 3D visualization...")
    viz = InteractiveVisualizer(host=host, port=port)
    print("  ✓ Visualization ready")
    
    print("\n[7/7] Starting interactive simulation...")
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
    viz_update_interval = 5  # Update viz every N steps
    
    # Initial state
    def reset_simulation():
        """Reset simulation to initial state."""
        left_img, right_img = env.reset(seed=seed)
        mapper.__init__(voxel_size=0.05)  # Reset mapper
        viz.reset_trajectory()
        return left_img, right_img, 0
    
    left_img, right_img, step = reset_simulation()
    
    # Main simulation loop
    try:
        while step < max_steps:
            # Check control state
            controls = viz.check_controls()
            
            # Handle reset
            if controls["should_reset"]:
                print("\n🔄 Resetting simulation...")
                left_img, right_img, step = reset_simulation()
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
            
            # Feed to brain (ommatidial samples as input)
            brain_input = np.concatenate([
                visual_features["left_ommatidia"],
                visual_features["right_ommatidia"]
            ])
            
            # Run brain for multiple timesteps
            for _ in range(brain_steps_per_sim):
                brain.set_input(brain_input)
                if simulator_type == "lif":
                    brain.step()
                    brain_activity = brain.get_firing_rates(window=50)
                else:
                    brain_activity = brain.step()
            
            # Get exploration signal
            fly_pos = env.get_fly_position()
            exploration_signal = explorer.compute_exploration_signal(fly_pos, mapper)
            
            # Compute control action
            action = controller.compute_action(brain_activity, exploration_signal)
            
            # Step environment
            left_img, right_img, info = env.step(action)
            
            # Update point cloud map
            if step % 5 == 0:
                left_pose, right_pose = env.get_camera_poses()
                mapper.add_depth_observation(
                    visual_features["depth_map"],
                    visual_features["rgb_for_cloud"],
                    left_pose,
                    visual_features["confidence"],
                    min_confidence=0.3
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
    
    # Point cloud
    pc_path = output_dir / "point_cloud.ply"
    mapper.save_point_cloud(str(pc_path))
    
    # Statistics
    map_stats = mapper.get_statistics()
    print("\n" + "=" * 70)
    print("Final Statistics:")
    print("=" * 70)
    print(f"Total simulation steps: {step}")
    print(f"Brain neurons: {n_neurons}")
    print(f"Point cloud points: {map_stats['total_points']}")
    print(f"Occupied voxels: {map_stats['occupied_voxels']}")
    print(f"Exploration ratio: {100 * map_stats['exploration_ratio']:.1f}%")
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
