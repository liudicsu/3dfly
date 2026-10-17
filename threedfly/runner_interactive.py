"""Interactive simulation runner with unified 3D + dashboard web interface."""

import numpy as np
from pathlib import Path
from tqdm import tqdm
import time
import sys

from threedfly.connectome import ConnectomeLoader, BrainSimulator
from threedfly.connectome.simulator import RateSimulator
from threedfly.connectome.neuron_selection import (
    load_neurotransmitter_data,
    build_transmitter_sign_vector,
)
from threedfly.connectome.simulator import (
    calibrate_looming_to_dnp01,
    count_directed_hops,
)
from threedfly.sim import FlyEnvironment
from threedfly.vision import StereoVision, PointCloudMapper, BrainDepthEstimator
from threedfly.vision.brain_depth import extract_t4_t5_flow_features
from threedfly.control import FlightController, ExplorationPolicy
from threedfly.control.controller import BiologicalFlightController
from threedfly.control.dn_controller import DNFlightController
from threedfly.control.exploration import heading_from_quaternion
from threedfly.control.flight_params import FlightParams
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
    warmup_steps: int = 800,
    apply_ei_signs: bool = True,
    use_core_tplus=None,
    engineered_looming_residual: bool = False,
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
    
    from threedfly.brain.closed_loop import ClosedLoopBrain, core_stack_available

    flight_params = FlightParams(use_engineered_looming_residual=engineered_looming_residual)
    core_brain = None
    dn_controller = None
    if use_core_tplus is False:
        use_core = False
    elif use_core_tplus is True or core_stack_available():
        use_core = True
    else:
        use_core = False
    if use_core and not core_stack_available():
        raise RuntimeError(
            "coreTplus closed loop requested but connectome/flyvis is missing."
        )

    use_biological = False
    brain_depth = None
    n_neurons = 0
    brain = None
    controller = None
    brain_dt = 0.010
    if use_core:
        print("\n[1/7] Building coreTplus closed-loop brain...")
        core_brain = ClosedLoopBrain.build(params=flight_params)
        dn_controller = DNFlightController(flight_params)
        controller = dn_controller
        brain = core_brain
        n_neurons = core_brain.n
        print(f"  ✓ coreTplus n={n_neurons}")
    else:
        print("\n[1/7] Loading connectome subgraph...")
        subgraph = ConnectomeLoader.load_subgraph(Path(subgraph_path))
        n_neurons = subgraph["n_neurons"]
        adjacency = subgraph["adjacency"]
        annotations = subgraph.get("annotations")
        body_ids = subgraph.get("body_ids")
        body_id_to_idx = subgraph.get("body_id_to_idx")
        use_biological = annotations is not None and body_ids is not None and body_id_to_idx is not None
        print(f"  ✓ Loaded {n_neurons} neurons")

        nt_data = None
        transmitter_signs = None
        if use_biological:
            nt_path = "data/connectome-raw/body-neurotransmitters-male-cns-v1.0.feather"
            nt_data = load_neurotransmitter_data(nt_path)
            if nt_data is not None:
                print(f"  ✓ Neurotransmitter table ({len(nt_data)} entries)")
                transmitter_signs = build_transmitter_sign_vector(
                    n_neurons, body_ids, body_id_to_idx, nt_data
                )

        print("\n[2/7] Initializing brain simulator ({})...".format(simulator_type))
        ei_kwargs = {}
        if apply_ei_signs and transmitter_signs is not None:
            ei_kwargs = {"apply_ei_signs": True, "transmitter_signs": transmitter_signs}
            print("  ✓ Applying E/I signs (sparse.diags @ adj)")
        if simulator_type == "lif":
            brain = BrainSimulator(adjacency, dt=0.001, **ei_kwargs)
            brain_dt = 0.001
        else:
            brain = RateSimulator(adjacency, dt=0.010, **ei_kwargs)
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
    if not use_core:
        brain_depth = BrainDepthEstimator(
            n_neurons=n_neurons,
            adjacency=adjacency,
            use_rate_model=(simulator_type == "rate")
        )
        print("  ✓ Brain depth estimator ready")
    else:
        print("  coreTplus path: stereo depth for the point-cloud panel")
    
    print("\n[6/8] Setting up control and exploration...")
    explorer = ExplorationPolicy(exploration_weight=0.8, random_exploration_prob=0.05)
    if use_core:
        dn_controller.bind_hover(env.hover_action)
        print("  ✓ DNFlightController")
    elif use_biological:
        print("  Using biological neuron mappings from annotations")
        controller = BiologicalFlightController(
            n_neurons=n_neurons,
            annotations=annotations,
            body_ids=body_ids,
            body_id_to_idx=body_id_to_idx,
            nt_data=nt_data,
            warmup_steps=warmup_steps,
        )
        controller.hover_action = env.hover_action
        loom_idx = np.concatenate([
            controller.input_mapping["looming_left"],
            controller.input_mapping["looming_right"],
        ])
        dnp_idx = controller.output_mapping.get("DNp01", np.array([], dtype=np.int32))
        hops = count_directed_hops(brain.adj, loom_idx, dnp_idx)
        print(
            f"  LPLC2/LC4→DNp01 hops: 1-hop={hops['n_one_hop']}, 2-hop={hops['n_two_hop']}"
        )
        brain.adj = calibrate_looming_to_dnp01(brain.adj, loom_idx, dnp_idx)
    else:
        controller = FlightController(n_neurons, use_rate_model=(simulator_type == "rate"))
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
    brain_steps_per_sim = max(1, int(round(sim_dt / brain_dt)))
    viz_update_interval = 3  # Update viz every N steps (more frequent for smoother point cloud)

    def run_warmup(left_img, right_img):
        if use_core:
            core_brain.reset()
            dn_controller.reset()
            dn_controller.bind_hover(env.hover_action)
            return left_img, right_img
        if not use_biological or controller.warmup_steps <= 0:
            return left_img, right_img
        print(f"\n[Warmup] Up to {controller.warmup_steps} grounded brain steps...")
        brain.reset()
        controller.warmup_complete = False
        controller._warmup_count = 0
        controller.warmup_history = {}
        controller.escape_timer = 0
        controller.brain_escape_count = 0
        controller.engineered_escape_count = 0
        controller.dnp01_prev_z = 0.0
        controller.resting_pair_mean = {}
        controller.resting_pair_scale = {}
        for warmup_step in range(controller.warmup_steps):
            visual_features = stereo.get_visual_features(left_img, right_img)
            fly_velocity = env.get_fly_velocity()
            _, flow_info = brain_depth.prepare_flow_features_for_brain(
                left_img, right_img, fly_velocity
            )
            controller.set_visual_input(
                left_luminance=visual_features["left_ommatidia"],
                right_luminance=visual_features["right_ommatidia"],
                optic_flow_features=extract_t4_t5_flow_features(
                    flow_info.get("flow_left"), flow_info.get("flow_right")
                ),
                looming_features={
                    "left": visual_features["looming_features"][:4].mean(),
                    "right": visual_features["looming_features"][4:].mean(),
                },
                brain=brain,
                angular_velocity=env.get_fly_angular_velocity(),
            )
            for _ in range(brain_steps_per_sim):
                if simulator_type == "lif":
                    brain.step()
                    brain_activity = brain.get_firing_rates(window=50)
                else:
                    brain_activity = brain.step()
            controller.record_resting_rates(brain_activity)
            if controller.warmup_converged():
                print(f"  Warmup converged at step {warmup_step + 1}")
                break
        controller.finalize_warmup()
        return left_img, right_img
    
    # Initial state
    def reset_simulation():
        """Reset simulation to initial state."""
        left_img, right_img = env.reset(seed=seed)
        mapper.__init__(voxel_size=0.05)  # Reset mapper
        if brain_depth is not None:
            brain_depth.reset()
        viz.reset_trajectory()
        left_img, right_img = run_warmup(left_img, right_img)
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
            
            fly_pos = env.get_fly_position()
            heading = heading_from_quaternion(env.get_fly_orientation())
            exploration_signal = explorer.compute_exploration_signal(
                fly_pos, mapper, current_heading=heading
            )

            flow_features = np.zeros(8, dtype=np.float32)
            flow_info = {"flow_left": None, "flow_right": None, "flow_magnitude": 0.0}
            if use_core:
                core_step = core_brain.step(
                    left_img, right_img, env.get_fly_angular_velocity()
                )
                brain_activity = core_step.rates
                action = dn_controller.compute(
                    core_step.readout,
                    looming_left=float(visual_features["looming_features"][:4].mean()),
                    looming_right=float(visual_features["looming_features"][4:].mean()),
                )
                brain_depth_map_upsampled = visual_features["depth_map"]
                brain_confidence_upsampled = visual_features["confidence"]
            else:
                flow_features, flow_info = brain_depth.prepare_flow_features_for_brain(
                    left_img,
                    right_img,
                    fly_velocity
                )
                if use_biological:
                    left_lum, right_lum = explorer.apply_visual_bias_pattern(
                        visual_features["left_ommatidia"],
                        visual_features["right_ommatidia"],
                        exploration_signal,
                    )
                    controller.set_visual_input(
                        left_luminance=left_lum,
                        right_luminance=right_lum,
                        optic_flow_features=extract_t4_t5_flow_features(
                            flow_info.get("flow_left"), flow_info.get("flow_right")
                        ),
                        looming_features={
                            "left": visual_features["looming_features"][:4].mean(),
                            "right": visual_features["looming_features"][4:].mean(),
                        },
                        brain=brain,
                        angular_velocity=env.get_fly_angular_velocity(),
                    )
                else:
                    brain_input = np.concatenate([
                        visual_features["left_ommatidia"],
                        visual_features["right_ommatidia"],
                        visual_features["looming_features"],
                        flow_features
                    ])
                    brain.set_input(brain_input)
                for _ in range(brain_steps_per_sim):
                    if simulator_type == "lif":
                        brain.step()
                        brain_activity = brain.get_firing_rates(window=50)
                    else:
                        brain_activity = brain.step()
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
                action = controller.compute_action(brain_activity)
            
            # Step environment (DNg02 L/R → wing strokes when the controller provides them)
            wing_strokes = getattr(controller, "last_wing_strokes", None)
            left_img, right_img, info = env.step(action, wing_strokes=wing_strokes)
            
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
            
            mapper.add_depth_observation(
                visual_features["depth_map"],
                visual_features["rgb_for_cloud"],
                left_pose,
                visual_features["confidence"],
                min_confidence=0.05,
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
                    wing_angles=env.get_wing_angles(),
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
    mapper.save_point_cloud(str(brain_pc_path), voxel_size=0.005, source="brain")
    mapper.save_point_cloud(str(stereo_pc_path), voxel_size=0.005, source="stereo")
    
    # Statistics
    map_stats = mapper.get_statistics()
    brain_depth_stats = brain_depth.get_statistics() if brain_depth is not None else {}
    
    print("\n" + "=" * 70)
    print("Final Statistics:")
    print("=" * 70)
    print(f"Total simulation steps: {step}")
    print(f"Simulated time: {step * sim_dt:.1f} s")
    print(f"Total collisions: {collision_count}")
    print(f"Brain neurons: {n_neurons}")
    if use_core:
        print(
            f"DNp01 escapes: brain-triggered={dn_controller.brain_escape_count}, "
            f"engineered-residual={dn_controller.engineered_escape_count}"
        )
    elif use_biological:
        print(f"DNp01 escapes: brain-triggered={controller.brain_escape_count}, "
              f"engineered-residual={controller.engineered_escape_count}")
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
