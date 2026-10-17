"""
3dfly: Fruit fly connectome-driven 3D exploration and mapping.

A simulator where a male fruit fly's brain (MaleCNS v1.0 connectome)
controls flight in a MuJoCo environment, building a 3D point cloud map
through stereo vision.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = [
    "ConnectomeLoader",
    "BrainSimulator",
    "StereoVision",
    "PointCloudMapper",
    "FlyEnvironment",
    "FlightController",
    "ExplorationPolicy",
    "Visualizer",
]

# Lazy exports so `threedfly.brain` can be imported without mujoco/open3d.
_EXPORTS = {
    "ConnectomeLoader": ("threedfly.connectome", "ConnectomeLoader"),
    "BrainSimulator": ("threedfly.connectome", "BrainSimulator"),
    "StereoVision": ("threedfly.vision", "StereoVision"),
    "PointCloudMapper": ("threedfly.vision", "PointCloudMapper"),
    "FlyEnvironment": ("threedfly.sim", "FlyEnvironment"),
    "FlightController": ("threedfly.control", "FlightController"),
    "ExplorationPolicy": ("threedfly.control", "ExplorationPolicy"),
    "Visualizer": ("threedfly.viz", "Visualizer"),
}


def __getattr__(name: str):
    if name in _EXPORTS:
        import importlib

        module_name, attr = _EXPORTS[name]
        return getattr(importlib.import_module(module_name), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
