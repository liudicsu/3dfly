"""Main simulation runner."""

import numpy as np
from pathlib import Path
from tqdm import tqdm
import time

from threedfly.connectome import ConnectomeLoader, BrainSimulator
from threedfly.connectome.simulator import RateSimulator
from threedfly.sim import FlyEnvironment
from threedfly.vision import StereoVision, PointCloudMapper
from threedfly.control import FlightController, ExplorationPolicy
from threedfly.viz import Visualizer, WebVisualizer


def run_simulation(
    subgraph_path: str,
    n_steps: int = 2000,
    viz_mode: str = "matplotlib",
    save_output_dir: str = "output",
    simulator_type: str = "rate",
    seed: int = 42,
    web_host: str = "127.0.0.1",
    web_port: int = 8050,
):
    """
    Run complete 3dfly simulation.
    
    Args:
        subgraph_path: Path to subgraph npz file
        n_steps: Number of simulation steps
        viz_mode: Visualization mode (matplotlib|open3d|both|web|none)
        save_output_dir: Output directory
        simulator_type: "rate" or "lif"
        seed: Random seed
        web_host: Host for web UI (when viz_mode="web")
        web_port: Port for web UI (when viz_mode="web")
    """
    np.random.seed(seed)
    
    # Output directory
    output_dir = Path(save_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n[1/7] Loading connectome subgraph...")
    subgraph = ConnectomeLoader.load_subgraph(Path(subgraph_path))
    n_neurons = subgraph["n_neurons"]
    adjacency = subgraph["adjacency"]
    
    print(f"\n[2/7] Initializing brain simulator ({simulator_type})...")
    if simulator_type == "lif":
        brain = BrainSimulator(adjacency, dt=0.001)
        brain_dt = 0.001
    else:
        brain = RateSimulator(adjacency, dt=0.010)
        brain_dt = 0.010
    
    print("\n[3/7] Creating MuJoCo environment...")
    env = FlyEnvironment(
        render_mode="rgb_array" if viz_mode != "none" else None,
        camera_width=64,
        camera_height=48
    )
    
    print("\n[4/7] Initializing vision and mapping...")
    stereo = StereoVision(baseline=0.012, focal_length=0.01)
    mapper = PointCloudMapper(voxel_size=0.05)
    
    print("\n[5/7] Setting up control and exploration...")
    controller = FlightController(n_neurons, use_rate_model=(simulator_type == "rate"))
    explorer = ExplorationPolicy(exploration_weight=0.5)
    
    print("\n[6/7] Initializing visualization...")
    if viz_mode == "web":
        viz = WebVisualizer(host=web_host, port=web_port)
        viz.run_async()
        print(f"\n🌐 Web UI running at http://{web_host}:{web_port}")
        print("Open your browser to interact with the simulation")
        print("-" * 70)
    elif viz_mode != "none":
        viz = Visualizer(mode=viz_mode, window_size=(12, 8))
        viz.show(block=False)
    else:
        viz = None
    
    print("\n[7/7] Running simulation...")
    print("=" * 70)
    
    # Reset environment
    left_img, right_img = env.reset(seed=seed)
    
    # Simulation loop
    sim_dt = 0.02  # 20ms per step
    brain_steps_per_sim = int(sim_dt / brain_dt)
    
    viz_update_interval = 10  # Update viz every N steps
    
    for step in tqdm(range(n_steps), desc="Simulation"):
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
        if step % 5 == 0:  # Update map every 5 steps
            left_pose, right_pose = env.get_camera_poses()
            mapper.add_depth_observation(
                visual_features["depth_map"],
                left_img,
                left_pose,
                visual_features["confidence"],
                min_confidence=0.3
            )
        
        # Update visualization
        if viz is not None and step % viz_update_interval == 0:
            brain_stats = brain.get_population_activity()
            control_stats = controller.get_status()
            
            if viz_mode == "web":
                viz.update(
                    step=step,
                    left_img=left_img,
                    right_img=right_img,
                    fly_position=fly_pos,
                    brain_activity=brain_activity,
                    control_action=action,
                    point_cloud_mapper=mapper,
                    brain_stats=brain_stats,
                    control_stats=control_stats,
                )
            else:
                viz.update(
                    left_img=left_img,
                    right_img=right_img,
                    fly_position=fly_pos,
                    brain_activity=brain_activity,
                    control_action=action,
                    point_cloud_mapper=mapper,
                    brain_stats=brain_stats,
                    control_stats=control_stats,
                )
        
        # Check collision
        if env.check_collision():
            print(f"\nCollision detected at step {step}!")
            break
    
    print("\n" + "=" * 70)
    print("Simulation complete!")
    print("=" * 70)
    
    # Save outputs
    print("\nSaving outputs...")
    
    # Point cloud
    pc_path = output_dir / "point_cloud.ply"
    mapper.save_point_cloud(str(pc_path))
    
    # Final visualization
    if viz is not None:
        if viz_mode == "web":
            print("\n" + "=" * 70)
            print(f"Web UI still running at http://{web_host}:{web_port}")
            print("Press Ctrl+C to stop the server")
            print("=" * 70)
            try:
                # Keep server running
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nShutting down web server...")
                viz.close()
        else:
            fig_path = output_dir / "final_state.png"
            viz.save_figure(str(fig_path))
            print(f"✓ Saved figure: {fig_path}")
            
            # Show final visualization
            print("\nClose the visualization window to exit.")
            viz.show(block=True)
            viz.close()
    
    # Statistics
    map_stats = mapper.get_statistics()
    print("\n" + "=" * 70)
    print("Final Statistics:")
    print("=" * 70)
    print(f"Total simulation steps: {step + 1}")
    print(f"Brain neurons: {n_neurons}")
    print(f"Point cloud points: {map_stats['total_points']}")
    print(f"Occupied voxels: {map_stats['occupied_voxels']}")
    print(f"Exploration ratio: {100 * map_stats['exploration_ratio']:.1f}%")
    print("=" * 70)
    
    env.close()


if __name__ == "__main__":
    # Quick test
    run_simulation(
        subgraph_path="data/demo_subgraph.npz",
        n_steps=500,
        viz_mode="matplotlib",
        save_output_dir="output",
        simulator_type="rate",
        seed=42
    )
