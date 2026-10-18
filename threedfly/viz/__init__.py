"""Interactive visualization.

``InteractiveVisualizer`` needs viser; ``Visualizer`` needs matplotlib
(and open3d only for viz_mode='open3d'). Import interactively so
``from threedfly.runner import run_simulation`` works without viser.
"""

from threedfly.viz.visualizer import Visualizer

__all__ = [
    "Visualizer",
    "InteractiveVisualizer",
]


def __getattr__(name: str):
    if name == "InteractiveVisualizer":
        from threedfly.viz.interactive import InteractiveVisualizer

        return InteractiveVisualizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
