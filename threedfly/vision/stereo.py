"""Stereo vision processing for depth estimation."""

import numpy as np
import cv2
from typing import Tuple, Optional


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
        num_disparities: int = 16,  # Must be divisible by 16
        downsample_factor: int = 2,  # Downsample for efficiency
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
        
        # Create stereo matcher
        self.stereo = cv2.StereoBM_create(
            numDisparities=num_disparities,
            blockSize=block_size
        )
        
        # For ommatidial-like sampling
        self.n_ommatidia = 100  # Simplified fly eye model
        
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
        
        # Compute depth: depth = (baseline * focal_length) / disparity
        # Avoid division by zero
        valid_mask = disparity > 0
        depth = np.zeros_like(disparity)
        depth[valid_mask] = (self.baseline * self.focal_length) / disparity[valid_mask]
        
        # Clip unreasonable depths
        depth = np.clip(depth, 0, 10.0)  # Max 10 meters
        
        # Confidence based on disparity magnitude
        confidence = np.zeros_like(depth)
        confidence[valid_mask] = np.clip(disparity[valid_mask] / 16.0, 0, 1)
        
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
        
        # Grid sampling
        h, w = gray.shape
        n_rows = int(np.sqrt(self.n_ommatidia))
        n_cols = n_rows
        
        samples = []
        for i in range(n_rows):
            for j in range(n_cols):
                y = int((i + 0.5) * h / n_rows)
                x = int((j + 0.5) * w / n_cols)
                samples.append(gray[y, x])
        
        samples = np.array(samples[:self.n_ommatidia], dtype=np.float32) / 255.0
        
        return samples
    
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
                - depth_estimate: mean depth
                - motion_cue: frame difference (if tracking previous)
        """
        # Ommatidial sampling
        left_samples = self.ommatidial_sample(left_img)
        right_samples = self.ommatidial_sample(right_img)
        
        # Depth estimate
        depth_map, confidence = self.estimate_depth(left_img, right_img)
        
        # Mean depth weighted by confidence
        if confidence.sum() > 0:
            mean_depth = (depth_map * confidence).sum() / confidence.sum()
        else:
            mean_depth = 1.0  # Default
        
        return {
            "left_ommatidia": left_samples,
            "right_ommatidia": right_samples,
            "depth_estimate": mean_depth,
            "depth_map": depth_map,
            "confidence": confidence,
        }
