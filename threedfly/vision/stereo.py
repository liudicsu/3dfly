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
    
    def compute_looming_features(
        self,
        depth_map: np.ndarray,
        confidence: np.ndarray,
    ) -> np.ndarray:
        """
        Compute looming/proximity features from depth map.
        
        Divides the visual field into sectors and computes proximity signals
        for each sector. These features feed into the brain for obstacle avoidance.
        
        Args:
            depth_map: Depth map (H, W)
            confidence: Confidence map (H, W)
        
        Returns:
            Looming features (n_sectors,) where higher values = closer obstacles
        """
        h, w = depth_map.shape
        
        # Divide into 8 sectors: forward, forward-left, left, back-left, 
        # back, back-right, right, forward-right
        n_sectors = 8
        looming = np.zeros(n_sectors)
        
        # Define sector boundaries (roughly)
        # Sector 0: forward (center)
        # Sectors 1-3: left side
        # Sector 4: back (edges)
        # Sectors 5-7: right side
        
        for i in range(n_sectors):
            angle = i * 2 * np.pi / n_sectors
            
            # Define sector region in image
            if i == 0:  # Forward center
                region = depth_map[h//3:2*h//3, w//3:2*w//3]
                region_conf = confidence[h//3:2*h//3, w//3:2*w//3]
            elif i == 1:  # Forward-left
                region = depth_map[h//4:3*h//4, :w//3]
                region_conf = confidence[h//4:3*h//4, :w//3]
            elif i == 2:  # Left
                region = depth_map[:, :w//4]
                region_conf = confidence[:, :w//4]
            elif i == 3:  # Back-left
                region = depth_map[:h//4, :w//3]
                region_conf = confidence[:h//4, :w//3]
            elif i == 4:  # Back (top and bottom edges)
                top = depth_map[:h//4, :]
                bottom = depth_map[3*h//4:, :]
                region = np.concatenate([top.flatten(), bottom.flatten()])
                top_conf = confidence[:h//4, :]
                bottom_conf = confidence[3*h//4:, :]
                region_conf = np.concatenate([top_conf.flatten(), bottom_conf.flatten()])
            elif i == 5:  # Back-right
                region = depth_map[:h//4, 2*w//3:]
                region_conf = confidence[:h//4, 2*w//3:]
            elif i == 6:  # Right
                region = depth_map[:, 3*w//4:]
                region_conf = confidence[:, 3*w//4:]
            else:  # i == 7, Forward-right
                region = depth_map[h//4:3*h//4, 2*w//3:]
                region_conf = confidence[h//4:3*h//4, 2*w//3:]
            
            # Compute proximity signal: inverse depth weighted by confidence
            # Closer objects = higher signal
            valid = (region_conf > 0.1) & (region > 0.01)
            if valid.sum() > 0:
                # Use minimum depth in sector (closest obstacle)
                min_depth = np.min(region[valid])
                # Convert to proximity (0 at far, 1 at very close)
                proximity = np.clip(1.0 / (min_depth + 0.1), 0, 10) / 10
                looming[i] = proximity
            else:
                looming[i] = 0.0
        
        return looming
    
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
                - looming_features: proximity/looming signals (8 sectors)
                - depth_estimate: mean depth
                - depth_map: depth map (downsampled if downsample_factor > 1)
                - confidence: confidence map (same size as depth_map)
                - rgb_for_cloud: RGB image matching depth_map size
        """
        # Ommatidial sampling
        left_samples = self.ommatidial_sample(left_img)
        right_samples = self.ommatidial_sample(right_img)
        
        # Depth estimate
        depth_map, confidence = self.estimate_depth(left_img, right_img)
        
        # Looming/proximity features for obstacle avoidance
        looming = self.compute_looming_features(depth_map, confidence)
        
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
            "looming_features": looming,
            "depth_estimate": mean_depth,
            "depth_map": depth_map,
            "confidence": confidence,
            "rgb_for_cloud": rgb_for_cloud,
        }
