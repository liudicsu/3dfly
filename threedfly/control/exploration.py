"""Exploration policy for curiosity-driven navigation."""

import numpy as np
from typing import Optional, Tuple


def heading_from_quaternion(quat: np.ndarray) -> float:
    """Yaw about +Z from a MuJoCo (w, x, y, z) quaternion."""
    w, x, y, z = [float(v) for v in quat]
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


class ExplorationPolicy:
    """
    Curiosity-driven exploration using point cloud occupancy.
    
    Biases flight toward under-mapped regions.
    """
    
    def __init__(
        self,
        exploration_weight: float = 0.5,
        random_exploration_prob: float = 0.2,
    ):
        """
        Initialize exploration policy.
        
        Args:
            exploration_weight: Weight of exploration drive [0, 1]
            random_exploration_prob: Probability of random exploration
        """
        self.exploration_weight = exploration_weight
        self.random_prob = random_exploration_prob
        
        # Sample directions for frontier search
        n_samples = 8
        self.sample_angles = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
        self.sample_distance = 1.0  # Look ahead distance
        
    def compute_exploration_signal(
        self,
        current_position: np.ndarray,
        point_cloud_mapper,
        current_heading: Optional[float] = None,
    ) -> np.ndarray:
        """
        Compute exploration bias for *sensory input*, not motor override.
        
        Args:
            current_position: Current fly position (x, y, z)
            point_cloud_mapper: PointCloudMapper instance
            current_heading: Body yaw (rad). Turn is heading error in [-1, 1],
                never a world-frame heading used as a yaw command.
        
        Returns:
            Exploration signal [forward, turn, up]
        """
        # Random exploration (still an input bias, clipped)
        if np.random.rand() < self.random_prob:
            rnd = np.random.randn(3) * 0.35
            rnd[1] = float(np.clip(rnd[1], -1.0, 1.0))
            rnd[2] = 0.0
            return rnd
        
        # Sample exploration scores around current position
        scores = []
        positions = []
        
        for angle in self.sample_angles:
            # Sample position in this direction
            sample_pos = current_position + self.sample_distance * np.array([
                np.cos(angle),
                np.sin(angle),
                0
            ])
            
            # Get exploration score (higher = less explored)
            score = point_cloud_mapper.get_exploration_score(sample_pos, radius=0.5)
            scores.append(score)
            positions.append(sample_pos)
        
        scores = np.array(scores)
        
        # Find most interesting direction
        best_idx = np.argmax(scores)
        best_pos = positions[best_idx]
        
        # Compute direction to best position
        direction = best_pos - current_position
        direction_norm = np.linalg.norm(direction[:2])  # Horizontal distance
        
        if direction_norm > 0.1:
            desired = float(np.arctan2(direction[1], direction[0]))
        else:
            desired = float(current_heading or 0.0)
        
        heading = 0.0 if current_heading is None else float(current_heading)
        error = (desired - heading + np.pi) % (2.0 * np.pi) - np.pi
        turn_signal = float(np.clip(error / (np.pi / 2.0), -1.0, 1.0))
        
        forward_signal = 0.3 * self.exploration_weight
        signal = np.array([forward_signal, turn_signal, 0.0])
        return signal

    def apply_visual_bias(
        self,
        left_luminance: float,
        right_luminance: float,
        signal: np.ndarray,
        scale: float = 0.4,
    ) -> Tuple[float, float]:
        """Bias left vs right photoreceptor drive toward the unexplored side."""
        turn = float(np.clip(signal[1], -1.0, 1.0))
        left = float(left_luminance) + scale * max(turn, 0.0)
        right = float(right_luminance) + scale * max(-turn, 0.0)
        return left, right

    def apply_visual_bias_pattern(
        self,
        left_samples: np.ndarray,
        right_samples: np.ndarray,
        signal: np.ndarray,
        scale: float = 0.4,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Bias hex-lattice ommatidia by scaling each eye toward the unexplored side."""
        left_m, right_m = self.apply_visual_bias(
            float(np.mean(left_samples)), float(np.mean(right_samples)), signal, scale=scale
        )

        def _scale(arr: np.ndarray, target: float) -> np.ndarray:
            m = float(np.mean(arr))
            if m > 1e-8:
                return arr * (target / m)
            return np.full_like(arr, target, dtype=arr.dtype)

        return _scale(left_samples, left_m), _scale(right_samples, right_m)
    
    def get_frontier_directions(
        self,
        current_position: np.ndarray,
        point_cloud_mapper,
        top_k: int = 3,
    ) -> list:
        """
        Get top-K frontier directions for visualization.
        
        Returns:
            List of (position, score) tuples
        """
        scores = []
        positions = []
        
        for angle in self.sample_angles:
            sample_pos = current_position + self.sample_distance * np.array([
                np.cos(angle),
                np.sin(angle),
                0
            ])
            
            score = point_cloud_mapper.get_exploration_score(sample_pos, radius=0.5)
            scores.append(score)
            positions.append(sample_pos)
        
        # Sort by score
        sorted_indices = np.argsort(scores)[::-1]
        
        frontiers = [
            (positions[i], scores[i])
            for i in sorted_indices[:top_k]
        ]
        
        return frontiers
