"""Main simulation runner."""

import numpy as np
from pathlib import Path
from tqdm import tqdm
import time

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
from threedfly.vision import StereoVision, BrainDepthEstimator
from threedfly.vision.brain_depth import extract_t4_t5_flow_features
from threedfly.control import FlightController, ExplorationPolicy
from threedfly.control.controller import BiologicalFlightController
from threedfly.control.dn_controller import DNFlightController
from threedfly.control.exploration import heading_from_quaternion
from threedfly.control.flight_params import FlightParams
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
    warmup_steps: int = 800,
    apply_ei_signs: bool = True,
    use_core_tplus=None,
    engineered_looming_residual: bool = False,
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
        warmup_steps: Max grounded brain steps before flight (biological mode)
        apply_ei_signs: Apply neurotransmitter E/I signs when NT data is available
    """
    np.random.seed(seed)
    
    # Output directory
    output_dir = Path(save_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
            "coreTplus closed loop requested but connectome/flyvis is missing. "
            "On Python 3.12: pip install -e '.[brain]' (flyvis --no-deps if local editable)."
        )
    
    use_biological = False
    brain_depth = None
    subgraph = None
    adjacency = None
    n_neurons = 0
    brain = None
    controller = None
    sim_dt = 0.02  # 20ms per physics/control step
    brain_steps_per_sim = 2

    if use_core:
        print("\n[1/7] Building coreTplus closed-loop brain (flyvis → RateCore)...")
        core_brain = ClosedLoopBrain.build(params=flight_params)
        dn_controller = DNFlightController(flight_params)
        controller = dn_controller
        brain = core_brain
        n_neurons = core_brain.n
        brain_dt = 0.010
        brain_steps_per_sim = 1
        print(f"  n={n_neurons}  engineered_residual={flight_params.use_engineered_looming_residual}")
    else:
        print("\n[1/7] Loading connectome subgraph...")
        subgraph = ConnectomeLoader.load_subgraph(Path(subgraph_path))
        n_neurons = subgraph["n_neurons"]
        adjacency = subgraph["adjacency"]
        annotations = subgraph.get("annotations")
        body_ids = subgraph.get("body_ids")
        body_id_to_idx = subgraph.get("body_id_to_idx")
        use_biological = annotations is not None and body_ids is not None and body_id_to_idx is not None

        nt_data = None
        transmitter_signs = None
        if use_biological:
            nt_path = "data/connectome-raw/body-neurotransmitters-male-cns-v1.0.feather"
            nt_data = load_neurotransmitter_data(nt_path)
            if nt_data is not None:
                print(f"  Loaded neurotransmitter data ({len(nt_data)} entries)")
                transmitter_signs = build_transmitter_sign_vector(
                    n_neurons, body_ids, body_id_to_idx, nt_data
                )

        print(f"\n[2/7] Initializing brain simulator ({simulator_type})...")
        ei_kwargs = {}
        if apply_ei_signs and transmitter_signs is not None:
            ei_kwargs = {"apply_ei_signs": True, "transmitter_signs": transmitter_signs}
            print("  Applying E/I signs from neurotransmitter table (sparse.diags @ adj)")
        if simulator_type == "lif":
            brain = BrainSimulator(adjacency, dt=0.001, **ei_kwargs)
            brain_dt = 0.001
        else:
            brain = RateSimulator(adjacency, dt=0.010, **ei_kwargs)
            brain_dt = 0.010
        brain_steps_per_sim = max(1, int(round(sim_dt / brain_dt)))
    
    print("\n[3/7] Creating MuJoCo environment...")
    # Always render camera images for vision system (even in headless mode)
    env = FlyEnvironment(
        render_mode="rgb_array",
        camera_width=160,
        camera_height=120
    )
    
    print("\n[4/7] Initializing vision and mapping...")
    from threedfly.vision.pointcloud import PointCloudMapper

    stereo = StereoVision(baseline=0.012, focal_length=0.01)
    mapper = PointCloudMapper(voxel_size=0.05)
    
    print("\n[5/7] Setting up brain depth estimation...")
    if not use_core:
        brain_depth = BrainDepthEstimator(
            n_neurons=n_neurons,
            adjacency=adjacency,
            use_rate_model=(simulator_type == "rate")
        )
    else:
        print("  coreTplus path: stereo depth for mapping (no 184k brain-depth router)")
    
    print("\n[6/7] Setting up control and exploration...")
    explorer = ExplorationPolicy(exploration_weight=0.8, random_exploration_prob=0.05)
    if use_core:
        dn_controller.bind_hover(env.hover_action)
        print("  DNFlightController (DNp01 escape, DNp03 dodge, DNa02 yaw, DNg02 thrust)")
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
            f"  LPLC2/LC4→DNp01 hops: 1-hop={hops['n_one_hop']}, "
            f"2-hop={hops['n_two_hop']} (src={hops['n_sources']}, tgt={hops['n_targets']})"
        )
        brain.adj = calibrate_looming_to_dnp01(brain.adj, loom_idx, dnp_idx)
        print("  Calibrated looming→DNp01 incoming gain (engineered path assist if 1-hop missing)")
    else:
        print("  No annotations found, using legacy controller")
        controller = FlightController(n_neurons, use_rate_model=(simulator_type == "rate"))
    
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
    if use_core:
        if hasattr(env, "hide_loom_object"):
            env.hide_loom_object()
            left_img, right_img = env.get_eye_images()
        print("\n[Warmup] coreTplus prewarm + DNp01 baseline (same as acceptance)…")
        warmed = core_brain.prewarm(left_img, right_img, env.get_fly_angular_velocity())
        if warmed is not None:
            dn_controller.capture_baseline(warmed.readout)
        left_img, right_img = env.get_eye_images()
    
    # Warmup period for biological controller (brain_steps_per_sim is defined above)
    if use_biological and controller.warmup_steps > 0:
        print(f"\n[Warmup] Up to {controller.warmup_steps} grounded brain steps "
              f"(min {controller.min_warmup_steps}, stop on convergence)...")
        for warmup_step in range(controller.warmup_steps):
            visual_features = stereo.get_visual_features(left_img, right_img)
            fly_velocity = env.get_fly_velocity()
            _, flow_info = brain_depth.prepare_flow_features_for_brain(
                left_img, right_img, fly_velocity
            )
            optic_flow_features = extract_t4_t5_flow_features(
                flow_info.get("flow_left"), flow_info.get("flow_right")
            )
            controller.set_visual_input(
                left_luminance=visual_features["left_ommatidia"],
                right_luminance=visual_features["right_ommatidia"],
                optic_flow_features=optic_flow_features,
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
    
    # Simulation loop
    
    viz_update_interval = 10  # Update viz every N steps
    
    # Collision tracking
    collision_count = 0
    
    for step in tqdm(range(n_steps), desc="Simulation"):
        # Get visual input (includes StereoBM for comparison + looming features)
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
            if flight_params.use_engineered_looming_residual:
                loom_l = float(visual_features["looming_features"][:4].mean())
                loom_r = float(visual_features["looming_features"][4:].mean())
                action = dn_controller.compute(
                    core_step.readout, looming_left=loom_l, looming_right=loom_r
                )
            else:
                action = dn_controller.compute(core_step.readout)
            brain_depth_map_upsampled = visual_features["depth_map"]
            brain_confidence_upsampled = visual_features["confidence"]
        else:
            flow_features, flow_info = brain_depth.prepare_flow_features_for_brain(
                left_img,
                right_img,
                fly_velocity
            )
            if use_biological:
                optic_flow_features = extract_t4_t5_flow_features(
                    flow_info.get("flow_left"), flow_info.get("flow_right")
                )
                left_lum, right_lum = explorer.apply_visual_bias_pattern(
                    visual_features["left_ommatidia"],
                    visual_features["right_ommatidia"],
                    exploration_signal,
                )
                controller.set_visual_input(
                    left_luminance=left_lum,
                    right_luminance=right_lum,
                    optic_flow_features=optic_flow_features,
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
        
        # PRIMARY: Add brain-depth observation (every 5 steps)
        if step % 5 == 0:
            mapper.add_depth_observation(
                brain_depth_map_upsampled,
                visual_features["rgb_for_cloud"],
                left_pose,
                brain_confidence_upsampled,
                min_confidence=0.2,  # Lower threshold for brain estimates
                source="brain"
            )
        
        # COMPARISON: StereoBM every control step (20 ms); keep density after slower/drag physics
        mapper.add_depth_observation(
            visual_features["depth_map"],
            visual_features["rgb_for_cloud"],
            left_pose,
            visual_features["confidence"],
            min_confidence=0.05,
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
    mapper.save_point_cloud(str(brain_pc_path), voxel_size=0.005, source="brain")
    mapper.save_point_cloud(str(stereo_pc_path), voxel_size=0.005, source="stereo")
    
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
    brain_depth_stats = brain_depth.get_statistics() if brain_depth is not None else {}
    
    print("\n" + "=" * 70)
    print("Final Statistics:")
    print("=" * 70)
    print(f"Total simulation steps: {step + 1}")
    print(f"Simulated time: {(step + 1) * sim_dt:.1f} s (control dt={sim_dt*1000:.0f} ms)")
    print(f"Total collisions: {collision_count}")
    print(f"Brain neurons: {n_neurons}")
    if use_core:
        print(
            f"DNp01 escapes: brain-triggered={dn_controller.brain_escape_count}, "
            f"engineered-residual={dn_controller.engineered_escape_count}, "
            f"latch_frac={dn_controller.dnp01_latch_frac:.3f}"
        )
        if core_brain.last_ms:
            print(
                f"Last step ms: flyvis={core_brain.last_ms.get('flyvis', 0):.2f} "
                f"brain={core_brain.last_ms.get('brain', 0):.2f} "
                f"total={core_brain.last_ms.get('total', 0):.2f}"
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
