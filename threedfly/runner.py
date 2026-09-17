"""Main simulation runner."""

import numpy as np
from pathlib import Path
from tqdm import tqdm
import time

from threedfly.connectome import ConnectomeLoader, BrainSimulator
from threedfly.connectome.simulator import RateSimulator
from threedfly.sim import FlyEnvironment
from threedfly.vision import StereoVision, PointCloudMapper, BrainDepthEstimator
from threedfly.control import FlightController, ExplorationPolicy
from threedfly.viz import Visualizer


def run_simulation(
    subgraph_path: str,
    n_steps: int = 2000,
    viz_mode: str = "matplotlib",
    save_output_dir: str = "output",
    simulator_type: str = "rate",
    seed: int = 42,
    max_collisions: int = 50,
    enable_soft_recovery: bool = True,
):
    """
    Run complete 3dfly simulation.
    
    Args:
        subgraph_path: Path to subgraph npz file
        n_steps: Number of simulation steps
        viz_mode: Visualization mode
        save_output_dir: Output directory
        simulator_type: "rate" or "lif"
        seed: Random seed
        max_collisions: Maximum collisions before stopping (0 = unlimited)
        enable_soft_recovery: Enable soft collision recovery (back off and continue)
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
    
    print("\n[5/7] Setting up brain depth estimation...")
    brain_depth = BrainDepthEstimator(
        n_neurons=n_neurons,
        adjacency=adjacency,
        use_rate_model=(simulator_type == "rate")
    )
    
    print("\n[6/7] Setting up control and exploration...")
    controller = FlightController(n_neurons, use_rate_model=(simulator_type == "rate"))
    explorer = ExplorationPolicy(exploration_weight=0.8)  # Increased from 0.5 for stronger exploration
    
    print("\n[7/8] Initializing visualization...")
    if viz_mode != "none":
        viz = Visualizer(mode=viz_mode, window_size=(12, 8))
        viz.show(block=False)
    else:
        viz = None
    
    print("\n[8/8] Running simulation...")
    print("=" * 70)
    print("\n💡 Using BRAIN-BASED DEPTH for primary reconstruction")
    print("   StereoBM available as comparison baseline")
    print("=" * 70)
    
    # Reset environment
    left_img, right_img = env.reset(seed=seed)
    
    # Simulation loop
    sim_dt = 0.02  # 20ms per step
    brain_steps_per_sim = int(sim_dt / brain_dt)
    
    viz_update_interval = 10  # Update viz every N steps
    
    # Collision tracking
    collision_count = 0
    
    for step in tqdm(range(n_steps), desc="Simulation"):
        # Get visual input (includes StereoBM for comparison + looming features)
        visual_features = stereo.get_visual_features(left_img, right_img)
        
        # Feed to brain (ommatidial samples + looming/proximity for avoidance)
        brain_input = np.concatenate([
            visual_features["left_ommatidia"],
            visual_features["right_ommatidia"],
            visual_features["looming_features"]  # Obstacle proximity signals
        ])
        
        # Run brain for multiple timesteps
        for _ in range(brain_steps_per_sim):
            brain.set_input(brain_input)
            if simulator_type == "lif":
                brain.step()
                brain_activity = brain.get_firing_rates(window=50)
            else:
                brain_activity = brain.step()
        
        # *** BRAIN-BASED DEPTH ESTIMATION ***
        # Extract depth from brain activity (primary reconstruction)
        brain_depth_result = brain_depth.estimate_depth_from_activity(
            brain_activity,
            stereo_features=visual_features
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
        if step % 5 == 0:  # Update map every 5 steps
            left_pose, right_pose = env.get_camera_poses()
            
            # PRIMARY: Add brain-depth observation
            mapper.add_depth_observation(
                brain_depth_map_upsampled,
                visual_features["rgb_for_cloud"],
                left_pose,
                brain_confidence_upsampled,
                min_confidence=0.2,  # Lower threshold for brain estimates
                source="brain"
            )
            
            # COMPARISON: Add StereoBM observation (less frequently)
            if step % 20 == 0:  # StereoBM every 20 steps (for comparison only)
                mapper.add_depth_observation(
                    visual_features["depth_map"],
                    visual_features["rgb_for_cloud"],
                    left_pose,
                    visual_features["confidence"],
                    min_confidence=0.3,
                    source="stereo"
                )
        
        # Update visualization
        if viz is not None and step % viz_update_interval == 0:
            brain_stats = brain.get_population_activity()
            control_stats = controller.get_status()
            
            viz.update(
                left_img=left_img,
                right_img=right_img,
                fly_position=fly_pos,
                brain_activity=brain_activity,
                control_action=action,
                point_cloud_mapper=mapper,
                brain_stats=brain_stats,
                control_stats=control_stats,
                looming_features=visual_features["looming_features"],
            )
        
        # Check collision
        if env.check_collision():
            collision_count += 1
            if enable_soft_recovery:
                print(f"\nCollision {collision_count} at step {step} - recovering...")
                env.recover_from_collision()
                # Check if we've hit max collisions
                if max_collisions > 0 and collision_count >= max_collisions:
                    print(f"Reached maximum collisions ({max_collisions}), stopping.")
                    break
            else:
                print(f"\nCollision detected at step {step}!")
                break
    
    print("\n" + "=" * 70)
    print("Simulation complete!")
    print("=" * 70)
    
    # Save outputs
    print("\nSaving outputs...")
    
    # Point clouds (both brain and stereo)
    brain_pc_path = output_dir / "point_cloud_brain.ply"
    stereo_pc_path = output_dir / "point_cloud_stereo.ply"
    
    # Export with very dense voxel size for inspection (1cm instead of 5cm)
    mapper.save_point_cloud(str(brain_pc_path), voxel_size=0.01, source="brain")
    mapper.save_point_cloud(str(stereo_pc_path), voxel_size=0.01, source="stereo")
    
    # Final visualization
    if viz is not None:
        fig_path = output_dir / "final_state.png"
        viz.save_figure(str(fig_path))
        print(f"✓ Saved figure: {fig_path}")
        
        # Show final visualization
        print("\nClose the visualization window to exit.")
        viz.show(block=True)
        viz.close()
    
    # Statistics
    map_stats = mapper.get_statistics()
    brain_depth_stats = brain_depth.get_statistics()
    
    print("\n" + "=" * 70)
    print("Final Statistics:")
    print("=" * 70)
    print(f"Total simulation steps: {step + 1}")
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
