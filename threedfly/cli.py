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
def run(subgraph, steps, viz_mode, headless, save_output, simulator_type, seed, max_collisions, no_soft_recovery):
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
def run_interactive(subgraph, steps, save_output, simulator_type, seed, host, port, max_collisions, no_soft_recovery):
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
