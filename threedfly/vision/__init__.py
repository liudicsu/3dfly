"""Vision processing: stereo depth, brain depth, and point cloud mapping.

``PointCloudMapper`` needs open3d. Import it lazily so ``StereoVision`` /
hex-eye tests collect on boxes that do not have open3d (the box with
open3d installed still runs the mapper tests).
"""

from threedfly.vision.stereo import StereoVision
from threedfly.vision.brain_depth import BrainDepthEstimator
from threedfly.vision.hex_eye import hex_lattice_uv, sample_image_hex, N_OMMATIDIA_PER_EYE

__all__ = [
    "StereoVision",
    "PointCloudMapper",
    "BrainDepthEstimator",
    "hex_lattice_uv",
    "sample_image_hex",
    "N_OMMATIDIA_PER_EYE",
]


def __getattr__(name: str):
    if name == "PointCloudMapper":
        from threedfly.vision.pointcloud import PointCloudMapper

        return PointCloudMapper
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
