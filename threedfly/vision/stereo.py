"""Stereo vision processing for depth estimation."""

import numpy as np
import cv2
from typing import Tuple, Optional, List

from threedfly.vision.hex_eye import (
    N_OMMATIDIA_PER_EYE,
    hex_lattice_uv,
    sample_image_hex,
)

LOOMING_N_SECTORS = 8
# Ignore sub-pixel Farneback jitter; expansion is scored with a percentile.
_LOOMING_NOISE_FLOOR = 0.05
_LOOMING_PERCENTILE = 75.0
_LOOMING_SCALE = 2.0


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image


def compute_farneback_flow(prev_gray: np.ndarray, curr_gray: np.ndarray) -> np.ndarray:
    """Dense optical flow (prev → curr), pixels per frame."""
    return cv2.calcOpticalFlowFarneback(
        prev_gray,
        curr_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )


def sector_slices(h: int, w: int) -> List[tuple]:
    """Image slices for the 8 looming sectors (same layout as the LPLC2 RFs)."""
    return [
        (slice(h // 3, 2 * h // 3), slice(w // 3, 2 * w // 3)),  # 0 forward
        (slice(h // 4, 3 * h // 4), slice(0, w // 3)),  # 1 forward-left
        (slice(None, None), slice(0, w // 4)),  # 2 left
        (slice(0, h // 4), slice(0, w // 3)),  # 3 back-left
        None,  # 4 back: top+bottom edges
        (slice(0, h // 4), slice(2 * w // 3, None)),  # 5 back-right
        (slice(None, None), slice(3 * w // 4, None)),  # 6 right
        (slice(h // 4, 3 * h // 4), slice(2 * w // 3, None)),  # 7 forward-right
    ]


def expansion_map(flow: np.ndarray) -> np.ndarray:
    """Positive divergence of the flow field (uniform translation → 0)."""
    fx = flow[..., 0]
    fy = flow[..., 1]
    div = np.gradient(fx, axis=1) + np.gradient(fy, axis=0)
    return np.maximum(div, 0.0).astype(np.float32)


def compute_expansion_looming(flow: Optional[np.ndarray]) -> np.ndarray:
    """Per-sector expansion in [0, 1]. 0 when nothing is approaching."""
    out = np.zeros(LOOMING_N_SECTORS, dtype=np.float32)
    if flow is None:
        return out
    exp = expansion_map(np.asarray(flow, dtype=np.float32))
    h, w = exp.shape
    slices = sector_slices(h, w)
    for i, sl in enumerate(slices):
        if i == 4:
            region = np.concatenate([exp[: h // 4, :].ravel(), exp[3 * h // 4 :, :].ravel()])
        else:
            ys, xs = sl
            region = exp[ys, xs].ravel()
        valid = region[region >= _LOOMING_NOISE_FLOOR]
        if valid.size < 8:
            continue
        p = float(np.percentile(valid, _LOOMING_PERCENTILE))
        if p < _LOOMING_NOISE_FLOOR * 1.5:
            continue
        out[i] = float(np.clip(p / _LOOMING_SCALE, 0.0, 1.0))
    return out


def looming_approach_correlation(looming_strength, approach_rate) -> float:
    """Pearson r between flow-based looming and GT closing speed (diagnostic)."""
    a = np.asarray(looming_strength, dtype=np.float64).ravel()
    b = np.asarray(approach_rate, dtype=np.float64).ravel()
    n = min(a.size, b.size)
    if n < 3:
        return 0.0
    a = a[:n]
    b = b[:n]
    if float(np.std(a)) < 1e-12 or float(np.std(b)) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


class StereoVision:
    """
    Stereo depth estimation from left/right eye images.
    
    Uses block matching for disparity and converts to depth map.
    """
    
    def __init__(
        self,
        baseline: float = 0.012,  # Distance between eyes in meters (~12mm for fly)
        focal_length: float = 0.01,  # Approximate focal length
        block_size: int = 5,
        num_disparities: int = 64,  # Must be divisible by 16 (64 for denser comparison)
        downsample_factor: int = 1,  # No downsampling for denser clouds
    ):
        """
        Initialize stereo processor.
        
        Args:
            baseline: Stereo baseline (eye separation)
            focal_length: Camera focal length
            block_size: Block size for matching (odd number)
            num_disparities: Max disparity search range
            downsample_factor: Downsample images for speed
        """
        self.baseline = baseline
        self.focal_length = focal_length
        self.downsample_factor = downsample_factor
        
        # Create stereo matcher with better parameters for denser matching
        self.stereo = cv2.StereoBM_create(
            numDisparities=num_disparities,
            blockSize=block_size
        )
        # Tune parameters for better matching
        self.stereo.setPreFilterCap(31)
        self.stereo.setMinDisparity(0)
        self.stereo.setTextureThreshold(10)
        self.stereo.setUniquenessRatio(5)
        self.stereo.setSpeckleWindowSize(100)
        self.stereo.setSpeckleRange(32)
        
        # Hex-lattice ommatidial sampling (~750/eye, ~5 deg). Replaces the old 10×10 grid.
        self.n_ommatidia = N_OMMATIDIA_PER_EYE
        self.ommatidia_uv = hex_lattice_uv(n_ommatidia=self.n_ommatidia)
        self.prev_left_gray = None
        self.prev_right_gray = None
        
    def estimate_depth(
        self,
        left_img: np.ndarray,
        right_img: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate depth from stereo pair.
        
        Args:
            left_img: Left eye image (H, W, 3) uint8
            right_img: Right eye image (H, W, 3) uint8
        
        Returns:
            (depth_map, confidence_map) in meters
        """
        # Convert to grayscale
        left_gray = cv2.cvtColor(left_img, cv2.COLOR_RGB2GRAY)
        right_gray = cv2.cvtColor(right_img, cv2.COLOR_RGB2GRAY)
        
        # Downsample for efficiency
        if self.downsample_factor > 1:
            h, w = left_gray.shape
            new_h = h // self.downsample_factor
            new_w = w // self.downsample_factor
            left_gray = cv2.resize(left_gray, (new_w, new_h))
            right_gray = cv2.resize(right_gray, (new_w, new_h))
        
        # Compute disparity
        disparity = self.stereo.compute(left_gray, right_gray)
        
        # Convert to float and handle invalid values
        disparity = disparity.astype(np.float32) / 16.0  # StereoBM returns fixed-point
        
        # Compute depth: depth = (baseline * focal_length_pixels) / disparity
        # focal_length in pixels approximated as image_width / 2
        h, w = left_gray.shape
        focal_length_pixels = w / 2.0
        
        # Avoid division by zero
        valid_mask = disparity > 0.5  # Require minimum disparity
        depth = np.zeros_like(disparity)
        depth[valid_mask] = (self.baseline * focal_length_pixels) / disparity[valid_mask]
        
        # Clip unreasonable depths
        depth = np.clip(depth, 0, 5.0)  # Max 5 meters (more appropriate for fly scale)
        
        # Confidence based on disparity magnitude (higher disparity = closer = more confident)
        confidence = np.zeros_like(depth)
        confidence[valid_mask] = np.clip(disparity[valid_mask] / 32.0, 0, 1)
        
        return depth, confidence
    
    def ommatidial_sample(self, image: np.ndarray) -> np.ndarray:
        """
        Sample image in ommatidial pattern (fly compound eye).
        
        Downsamples image to N representative intensity values.
        
        Args:
            image: Input image (H, W, 3) uint8
        
        Returns:
            Sampled intensities (n_ommatidia,) in [0, 1]
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        return sample_image_hex(gray, self.ommatidia_uv)
    
    def _update_eye_flow(self, image: np.ndarray, side: str) -> Optional[np.ndarray]:
        gray = _to_gray(image)
        prev_attr = "prev_left_gray" if side == "left" else "prev_right_gray"
        prev = getattr(self, prev_attr)
        if prev is None or prev.shape != gray.shape:
            setattr(self, prev_attr, gray)
            return None
        flow = compute_farneback_flow(prev, gray)
        setattr(self, prev_attr, gray)
        return flow

    def compute_looming_features(
        self,
        flow: Optional[np.ndarray],
        _confidence: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Per-sector expansion from one eye's optic flow (not StereoBM depth)."""
        return compute_expansion_looming(flow)
    
    def get_visual_features(
        self,
        left_img: np.ndarray,
        right_img: np.ndarray,
    ) -> dict:
        """
        Extract visual features for brain input.
        
        Args:
            left_img: Left eye image
            right_img: Right eye image
        
        Returns:
            Dictionary with features:
                - left_ommatidia: sampled left eye
                - right_ommatidia: sampled right eye
                - looming_features: max of per-eye expansion (8 sectors, viz)
                - looming_left / looming_right: per-eye expansion from that eye's flow
                - depth_estimate: mean depth
                - depth_map: depth map (downsampled if downsample_factor > 1)
                - confidence: confidence map (same size as depth_map)
                - rgb_for_cloud: RGB image matching depth_map size
        """
        # Ommatidial sampling
        left_samples = self.ommatidial_sample(left_img)
        right_samples = self.ommatidial_sample(right_img)
        
        # Depth estimate (comparison baseline only — not used for looming)
        depth_map, confidence = self.estimate_depth(left_img, right_img)

        flow_left = self._update_eye_flow(left_img, "left")
        flow_right = self._update_eye_flow(right_img, "right")
        looming_left = self.compute_looming_features(flow_left)
        looming_right = self.compute_looming_features(flow_right)
        looming = np.maximum(looming_left, looming_right)
        
        # Downsample RGB to match depth map if needed
        if self.downsample_factor > 1:
            h, w = left_img.shape[:2]
            new_h = h // self.downsample_factor
            new_w = w // self.downsample_factor
            rgb_for_cloud = cv2.resize(left_img, (new_w, new_h))
        else:
            rgb_for_cloud = left_img
        
        # Mean depth weighted by confidence
        if confidence.sum() > 0:
            mean_depth = (depth_map * confidence).sum() / confidence.sum()
        else:
            mean_depth = 1.0  # Default
        
        return {
            "left_ommatidia": left_samples,
            "right_ommatidia": right_samples,
            "ommatidia_uv": self.ommatidia_uv,
            "looming_features": looming,
            "looming_left": looming_left,
            "looming_right": looming_right,
            "depth_estimate": mean_depth,
            "depth_map": depth_map,
            "confidence": confidence,
            "rgb_for_cloud": rgb_for_cloud,
        }
