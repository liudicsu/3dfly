"""Vision processing: stereo depth, brain depth, and point cloud mapping."""

from threedfly.vision.stereo import (
    StereoVision,
    compute_expansion_looming,
    looming_approach_correlation,
    forward_looming_and_approach,
    LOOMING_FORWARD_SECTORS,
)
from threedfly.vision.pointcloud import PointCloudMapper
from threedfly.vision.brain_depth import BrainDepthEstimator
from threedfly.vision.hex_eye import hex_lattice_uv, sample_image_hex, N_OMMATIDIA_PER_EYE

__all__ = [
    "StereoVision",
    "compute_expansion_looming",
    "looming_approach_correlation",
    "forward_looming_and_approach",
    "LOOMING_FORWARD_SECTORS",
    "PointCloudMapper",
    "BrainDepthEstimator",
    "hex_lattice_uv",
    "sample_image_hex",
    "N_OMMATIDIA_PER_EYE",
]
