"""Command-line interface for 3dfly."""

import click
from pathlib import Path
import sys


@click.group()
@click.version_option(version="0.1.0")
def main():
    """
    3dfly: Fruit fly connectome-driven 3D exploration.
    
    A simulator where a male fruit fly's brain (MaleCNS v1.0 connectome)
    controls flight in a MuJoCo environment, building a 3D point cloud map.
    """
    pass


@main.command()
@click.option(
    "--output-dir",
    type=click.Path(),
    default=None,
    help="Output directory for downloaded files (default: data/connectome-raw)"
)
@click.option(
    "--files",
    type=str,
    default="annotations,weights",
    help="Comma-separated list of files to download (annotations, weights, neurotransmitters)"
)
def download(output_dir, files):
    """Download MaleCNS v1.0 connectome data."""
    from threedfly.connectome import download_connectome_data
    
    files_list = [f.strip() for f in files.split(",")]
    
    try:
        output_path = download_connectome_data(
            output_dir=output_dir,
            files_to_download=files_list
        )
        click.echo(f"\n✓ Data downloaded to: {output_path}")
    except Exception as e:
        click.echo(f"Error downloading data: {e}", err=True)
        sys.exit(1)


@main.command()
@click.option(
    "--data-dir",
    type=click.Path(exists=True),
    default=None,
    help="Directory with downloaded connectome data"
)
@click.option(
    "--output",
    type=click.Path(),
    default="data/subgraph.npz",
    help="Output file for subgraph"
)
@click.option(
    "--max-neurons",
    type=int,
    default=None,
    help="Maximum neurons to include (for demo/testing)"
)
@click.option(
    "--demo",
    is_flag=True,
    help="Create small demo subgraph (1000 neurons)"
)
def extract(data_dir, output, max_neurons, demo):
    """Extract visual-flight subgraph from full connectome."""
    from threedfly.connectome import ConnectomeLoader
    
    if demo:
        max_neurons = 1000
        output = "data/demo_subgraph.npz"
    
    try:
        loader = ConnectomeLoader(data_dir=data_dir)
        loader.load_annotations()
        loader.load_weights()
        
        loader.extract_subgraph(max_neurons=max_neurons)
        loader.save_subgraph(Path(output))
        
        click.echo(f"\n✓ Subgraph saved to: {output}")
    except Exception as e:
        click.echo(f"Error extracting subgraph: {e}", err=True)
        sys.exit(1)


@main.command("build-full")
@click.option(
    "--data-dir",
    type=click.Path(exists=True),
    default=None,
    help="Directory with downloaded connectome data"
)
@click.option(
    "--output",
    type=click.Path(),
    default="data/full_connectome.npz",
    help="Output file for full connectome"
)
@click.option(
    "--min-synapses",
    type=int,
    default=3,
    help="Minimum synaptic weight to include (default: 3, recommended for memory efficiency)"
)
def build_full(data_dir, output, min_synapses):
    """Build full-brain connectome with all neurons.
    
    This creates a sparse adjacency matrix for the entire MaleCNS v1.0
    connectome (~130k-185k neurons depending on filtering).
    
    The default min-synapses=3 filter reduces the dataset from 150M to ~10M
    connections while keeping biologically significant synapses, making it
    memory-efficient (~4-8GB RAM during build, final file ~35 MB).
    
    For the complete unfiltered connectome, use --min-synapses 1 (requires
    more memory and produces a larger file).
    
    Annotations are automatically saved alongside the connectome for
    neuron type-based I/O selection at runtime.
    """
    from threedfly.connectome import ConnectomeLoader
    
    click.echo("=" * 70)
    click.echo("Building Full MaleCNS Connectome")
    click.echo("=" * 70)
    click.echo(f"Data directory: {data_dir or 'data/connectome-raw'}")
    click.echo(f"Output: {output}")
    click.echo(f"Min synapses: {min_synapses}")
    click.echo("=" * 70)
    
    try:
        loader = ConnectomeLoader(data_dir=data_dir)
        loader.load_annotations()
        
        loader.build_full_connectome(min_synapses=min_synapses)
        loader.save_subgraph(Path(output))
        
        click.echo(f"\n✓ Full connectome saved to: {output}")
        click.echo("\nYou can now use this with:")
        click.echo(f"  threedfly run --subgraph {output}")
        click.echo(f"  threedfly run-interactive --subgraph {output}")
        click.echo(f"\nTo inspect neuron types:")
        click.echo(f"  threedfly inspect-types {output}")
    except Exception as e:
        click.echo(f"Error building full connectome: {e}", err=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


@main.command("inspect-types")
@click.argument("subgraph_path", type=click.Path(exists=True))
def inspect_types(subgraph_path):
    """Inspect neuron types in a connectome subgraph.
    
    Shows breakdown of visual, descending, and central complex neurons
    identified by type annotations. Useful for understanding the full-brain
    connectome composition.
    """
    from threedfly.connectome import ConnectomeLoader, get_neuron_selection_summary
    
    try:
        subgraph = ConnectomeLoader.load_subgraph(Path(subgraph_path))
        
        if subgraph.get("annotations") is None:
            click.echo("✗ No annotations found in this subgraph.", err=True)
            click.echo("Rebuild with: threedfly build-full", err=True)
            sys.exit(1)
        
        annotations = subgraph["annotations"]
        body_ids = subgraph["body_ids"]
        body_id_to_idx = subgraph["body_id_to_idx"]
        
        selection = get_neuron_selection_summary(annotations, body_ids, body_id_to_idx)
        
        click.echo("\n" + "=" * 70)
        click.echo("Neuron Type Summary")
        click.echo("=" * 70)
        
        for category, subcategories in selection.items():
            click.echo(f"\n{category.upper()}:")
            for name, indices in subcategories.items():
                pct = 100 * len(indices) / subgraph['n_neurons']
                click.echo(f"  {name:25s}: {len(indices):7,} neurons ({pct:5.1f}%)")
        
        total_classified = sum(
            len(indices) 
            for subcats in selection.values() 
            for name, indices in subcats.items() 
            if not name.startswith("all_")
        )
        
        click.echo("\n" + "=" * 70)
        click.echo(f"Total classified: {total_classified:,} / {subgraph['n_neurons']:,} neurons ({100 * total_classified / subgraph['n_neurons']:.1f}%)")
        click.echo("=" * 70)
        
    except Exception as e:
        click.echo(f"Error inspecting types: {e}", err=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


@main.command("diagnose-looming")
@click.argument("subgraph_path", type=click.Path(exists=True))
@click.option("--no-ei", is_flag=True, help="Skip neurotransmitter E/I signs")
@click.option("--looming", type=float, default=1.0, help="Injected looming current")
@click.option(
    "--looming-inhibition-scale",
    type=float,
    default=None,
    help="Scale I→LPLC2/LC4 synapses only (default 0.5).",
)
@click.option(
    "--inhibition-json",
    type=click.Path(),
    default="lplc2_inhibition.json",
    help="Write top inhibitory presynaptic types onto LPLC2/LC4 as JSON.",
)
def diagnose_looming(subgraph_path, no_ei, looming, looming_inhibition_scale, inhibition_json):
    """Inject strong synthetic looming into LPLC2/LC4 and print DNp01/DNp03 vs rest."""
    import json
    import numpy as np
    from threedfly.connectome import ConnectomeLoader
    from threedfly.connectome.neuron_selection import (
        load_neurotransmitter_data,
        build_transmitter_sign_vector,
    )
    from threedfly.connectome.simulator import (
        RateSimulator,
        apply_ei_signs,
        calibrate_looming_to_dnp01,
        diagnose_lplc2_inhibition,
        diagnose_looming_to_dnp01,
        count_directed_hops,
        DEFAULT_LOOMING_INHIBITION_SCALE,
    )
    from threedfly.control.controller import BiologicalFlightController

    subgraph = ConnectomeLoader.load_subgraph(Path(subgraph_path))
    annotations = subgraph.get("annotations")
    if annotations is None:
        click.echo("✗ Subgraph has no annotations.", err=True)
        sys.exit(1)
    body_ids = subgraph["body_ids"]
    body_id_to_idx = subgraph["body_id_to_idx"]
    adj = subgraph["adjacency"]
    n_neurons = subgraph["n_neurons"]

    if not no_ei:
        nt_data = load_neurotransmitter_data(
            "data/connectome-raw/body-neurotransmitters-male-cns-v1.0.feather"
        )
        if nt_data is not None:
            signs = build_transmitter_sign_vector(
                n_neurons, body_ids, body_id_to_idx, nt_data
            )
            adj = apply_ei_signs(adj, signs)
            click.echo("Applied E/I signs from neurotransmitter table.")

    controller = BiologicalFlightController(
        n_neurons=n_neurons,
        annotations=annotations,
        body_ids=body_ids,
        body_id_to_idx=body_id_to_idx,
        warmup_steps=0,
    )
    loom = np.concatenate([
        controller.input_mapping["looming_left"],
        controller.input_mapping["looming_right"],
    ])
    dnp = controller.output_mapping.get("DNp01")
    dnp03 = controller.output_mapping.get("DNp03", np.array([], dtype=int))
    hops = count_directed_hops(adj, loom, dnp)
    click.echo(
        f"LPLC2/LC4 n={hops['n_sources']}  DNp01 n={hops['n_targets']}  "
        f"DNp03 n={len(dnp03)}  1-hop={hops['n_one_hop']}  2-hop={hops['n_two_hop']}"
    )
    other = []
    for key, idx in controller.input_mapping.items():
        if key.startswith("looming"):
            continue
        if idx is not None and len(idx) > 0:
            other.append(np.asarray(idx, dtype=np.int32))
    other_idx = np.unique(np.concatenate(other)) if other else np.array([], dtype=np.int32)
    inhib_scale = (
        DEFAULT_LOOMING_INHIBITION_SCALE
        if looming_inhibition_scale is None
        else float(looming_inhibition_scale)
    )
    click.echo(
        f"Flight-like other sensory n={other_idx.size} at 0.8; "
        f"looming baseline 0.3 + pulse; I→LPLC2/LC4 scale={inhib_scale}"
    )
    report = diagnose_looming_to_dnp01(
        adjacency=adj,
        looming_indices=loom,
        dnp01_indices=dnp,
        dnp03_indices=dnp03,
        looming_left=controller.input_mapping["looming_left"],
        looming_right=controller.input_mapping["looming_right"],
        looming_current=float(looming),
        other_sensory_indices=other_idx,
        other_sensory_current=0.8,
        baseline_current=0.3,
        looming_inhibition_scale=inhib_scale,
    )
    calibrated = calibrate_looming_to_dnp01(
        adj, loom, dnp, dnp03_indices=dnp03, looming_inhibition_scale=inhib_scale
    )
    sim = RateSimulator(calibrated, dt=0.010, syn_weight_scale=0.1, max_rate=50.0)
    idx, val = [], []
    if other_idx.size:
        idx.extend(other_idx.tolist())
        val.extend([0.8] * int(other_idx.size))
    if loom.size:
        idx.extend(loom.tolist())
        val.extend([0.3] * int(loom.size))
    for _ in range(24):
        if idx:
            sim.set_input(np.asarray(val, dtype=np.float32), neuron_indices=np.asarray(idx, dtype=np.int32))
        sim.step()
    inhib = diagnose_lplc2_inhibition(
        calibrated,
        loom,
        sim.rates,
        annotations=annotations,
        body_ids=body_ids,
    )
    inhib["looming_inhibition_scale"] = inhib_scale
    json_path = Path(inhibition_json)
    json_path.write_text(json.dumps(inhib, indent=2))
    click.echo(f"Wrote LPLC2/LC4 inhibitory sources to {json_path}")
    for row in inhib["top_inhibitory_types"][:8]:
        click.echo(
            f"  I {row['cell_type']}: Σ(w·r)={row['sum_weight_x_rate']:.4g} "
            f"n={int(row['n_synapses'])}"
        )
    click.echo(f"Looming cells driven rate: {report['looming_driven']:.4g}")
    click.echo(
        f"DNp01 rest: {report['dnp01_rest']:.4g}  driven: {report['dnp01_driven']:.4g}  "
        f"latency: {report['dnp01_latency_steps']} steps  "
        f"L/R sel: {report['dnp01_lr_selectivity']:.3f}"
    )
    click.echo(
        f"DNp03 rest: {report['dnp03_rest']:.4g}  driven: {report['dnp03_driven']:.4g}  "
        f"latency: {report['dnp03_latency_steps']} steps  "
        f"L/R sel: {report['dnp03_lr_selectivity']:.3f}"
    )
    click.echo(
        f"DNp01 left-loom driven: {report['dnp01_left_driven']:.4g}  "
        f"right-loom driven: {report['dnp01_right_driven']:.4g}"
    )
    click.echo("DNp01 rose above rest." if report["rose"] else "DNp01 did NOT rise above rest.")
    click.echo(
        f"Closed-loop flight-like tonic-then-pulse: rest={report['dnp01_closed_loop_rest']:.4g}  "
        f"driven={report['dnp01_closed_loop_driven']:.4g}  "
        f"late={report['dnp01_closed_loop_late_driven']:.4g}  "
        f"crossed={int(report['dnp01_closed_loop_crossed'])}  "
        f"inhib_scale={report.get('looming_inhibition_scale', inhib_scale)}"
    )
    if report["dnp01_closed_loop_crossed"]:
        click.echo("DNp01 crossed rest+margin after tonic warm-up.")
    else:
        click.echo("DNp01 did NOT cross rest+margin after tonic warm-up.")


@main.command()
@click.option(
    "--subgraph",
    type=click.Path(exists=True),
    default="data/demo_subgraph.npz",
    help="Path to subgraph file"
)
@click.option(
    "--steps",
    type=int,
    default=2000,
    help="Number of simulation steps"
)
@click.option(
    "--viz-mode",
    type=click.Choice(["matplotlib", "open3d", "both", "none"]),
    default="matplotlib",
    help="Visualization mode"
)
@click.option(
    "--headless",
    is_flag=True,
    help="Run in headless mode (no visualization)"
)
@click.option(
    "--save-output",
    type=click.Path(),
    default="output",
    help="Directory to save outputs (point cloud, plots)"
)
@click.option(
    "--simulator-type",
    type=click.Choice(["rate", "lif"]),
    default="rate",
    help="Neural simulator type (rate or LIF)"
)
@click.option(
    "--seed",
    type=int,
    default=42,
    help="Random seed"
)
@click.option(
    "--max-collisions",
    type=int,
    default=50,
    help="Maximum collisions before stopping (0 = unlimited)"
)
@click.option(
    "--no-soft-recovery",
    is_flag=True,
    help="Disable soft collision recovery (stop on first collision)"
)
@click.option(
    "--engineered-looming-residual",
    is_flag=True,
    default=False,
    help="Ablation only: enable the engineered looming-residual escape trigger (default OFF).",
)
@click.option(
    "--looming-path-gain",
    type=float,
    default=None,
    help="Engineered LPLC2/LC4→DNp01/DNp03 one-hop gain (default 2.0).",
)
@click.option(
    "--looming-rest-incoming-scale",
    type=float,
    default=None,
    help="Scale other excitatory incoming onto DNp01/DNp03 (default 1.0 = no attenuation).",
)
@click.option(
    "--looming-inhibition-scale",
    type=float,
    default=None,
    help="Scale inhibitory synapses onto LPLC2/LC4 only (default 0.5).",
)
def run(subgraph, steps, viz_mode, headless, save_output, simulator_type, seed, max_collisions, no_soft_recovery, engineered_looming_residual, looming_path_gain, looming_rest_incoming_scale, looming_inhibition_scale):
    """Run 3dfly exploration simulation (static visualization)."""
    from threedfly.runner import run_simulation
    
    if headless:
        viz_mode = "none"
    
    click.echo("=" * 70)
    click.echo("3dfly: Connectome-Driven Flight Simulator")
    click.echo("=" * 70)
    click.echo(f"Subgraph: {subgraph}")
    click.echo(f"Steps: {steps}")
    click.echo(f"Simulator: {simulator_type}")
    click.echo(f"Visualization: {viz_mode}")
    click.echo(f"Output: {save_output}")
    click.echo("=" * 70)
    
    try:
        run_simulation(
            subgraph_path=subgraph,
            n_steps=steps,
            viz_mode=viz_mode,
            save_output_dir=save_output,
            simulator_type=simulator_type,
            seed=seed,
            max_collisions=max_collisions,
            enable_soft_recovery=not no_soft_recovery,
            use_engineered_looming_residual=engineered_looming_residual,
            looming_path_gain=looming_path_gain,
            looming_rest_incoming_scale=looming_rest_incoming_scale,
            looming_inhibition_scale=looming_inhibition_scale,
        )
        click.echo("\n✓ Simulation complete!")
    except Exception as e:
        click.echo(f"\nError running simulation: {e}", err=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


@main.command("run-interactive")
@click.option(
    "--subgraph",
    type=click.Path(exists=True),
    default="data/demo_subgraph.npz",
    help="Path to subgraph file"
)
@click.option(
    "--steps",
    type=int,
    default=5000,
    help="Maximum simulation steps"
)
@click.option(
    "--save-output",
    type=click.Path(),
    default="output",
    help="Directory to save outputs"
)
@click.option(
    "--simulator-type",
    type=click.Choice(["rate", "lif"]),
    default="rate",
    help="Neural simulator type"
)
@click.option(
    "--seed",
    type=int,
    default=42,
    help="Random seed"
)
@click.option(
    "--host",
    type=str,
    default="0.0.0.0",
    help="Visualization server host"
)
@click.option(
    "--port",
    type=int,
    default=8080,
    help="Visualization server port"
)
@click.option(
    "--max-collisions",
    type=int,
    default=100,
    help="Maximum collisions before stopping (0 = unlimited)"
)
@click.option(
    "--no-soft-recovery",
    is_flag=True,
    help="Disable soft collision recovery (pause on collision)"
)
@click.option(
    "--engineered-looming-residual",
    is_flag=True,
    default=False,
    help="Ablation only: enable the engineered looming-residual escape trigger (default OFF).",
)
@click.option(
    "--looming-path-gain",
    type=float,
    default=None,
    help="Engineered LPLC2/LC4→DNp01/DNp03 one-hop gain (default 2.0).",
)
@click.option(
    "--looming-rest-incoming-scale",
    type=float,
    default=None,
    help="Scale other excitatory incoming onto DNp01/DNp03 (default 1.0 = no attenuation).",
)
@click.option(
    "--looming-inhibition-scale",
    type=float,
    default=None,
    help="Scale inhibitory synapses onto LPLC2/LC4 only (default 0.5).",
)
def run_interactive(subgraph, steps, save_output, simulator_type, seed, host, port, max_collisions, no_soft_recovery, engineered_looming_residual, looming_path_gain, looming_rest_incoming_scale, looming_inhibition_scale):
    """Run 3dfly with unified interactive web interface.
    
    Single-page interface with:
    - Interactive 3D god's-eye view (orbit/pan/zoom with mouse)
    - Dashboard panels (stereo views, trajectory, brain, commands, status)
    - Play/pause/step/reset controls
    - Adjustable playback speed
    
    Open your browser to http://localhost:8080 after starting.
    """
    from threedfly.runner_interactive import run_interactive_simulation
    
    try:
        run_interactive_simulation(
            subgraph_path=subgraph,
            max_steps=steps,
            save_output_dir=save_output,
            simulator_type=simulator_type,
            seed=seed,
            host=host,
            port=port,
            max_collisions=max_collisions,
            enable_soft_recovery=not no_soft_recovery,
            use_engineered_looming_residual=engineered_looming_residual,
            looming_path_gain=looming_path_gain,
            looming_rest_incoming_scale=looming_rest_incoming_scale,
            looming_inhibition_scale=looming_inhibition_scale,
        )
    except Exception as e:
        click.echo(f"\nError running interactive simulation: {e}", err=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


@main.command()
def demo():
    """Run quick demo with bundled demo subgraph."""
    from pathlib import Path
    import subprocess
    
    demo_subgraph = Path("data/demo_subgraph.npz")
    
    if not demo_subgraph.exists():
        click.echo("Demo subgraph not found. Creating it now...")
        click.echo("This requires downloading the full connectome data (~1.1GB).")
        click.echo("Press Ctrl+C to cancel, or wait to continue...")
        
        import time
        time.sleep(3)
        
        # Download
        ctx = click.get_current_context()
        ctx.invoke(download, output_dir=None, files="annotations,weights")
        
        # Extract demo
        ctx.invoke(extract, data_dir=None, output=None, max_neurons=None, demo=True)
    
    # Run demo
    click.echo("\nRunning demo simulation...")
    ctx = click.get_current_context()
    ctx.invoke(
        run,
        subgraph=str(demo_subgraph),
        steps=1000,
        viz_mode="matplotlib",
        headless=False,
        save_output="output",
        simulator_type="rate",
        seed=42
    )


if __name__ == "__main__":
    main()
