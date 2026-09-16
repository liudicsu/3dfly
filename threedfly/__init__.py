"""
3dfly: Fruit fly connectome-driven 3D exploration and mapping.

A simulator where a male fruit fly's brain (MaleCNS v1.0 connectome)
controls flight in a MuJoCo environment, building a 3D point cloud map
through stereo vision.
"""

__version__ = "0.1.0"

from threedfly.connectome import ConnectomeLoader, BrainSimulator
from threedfly.vision import StereoVision, PointCloudMapper
from threedfly.sim import FlyEnvironment
from threedfly.control import FlightController, ExplorationPolicy
from threedfly.viz import Visualizer

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
