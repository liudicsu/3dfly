"""Exploration policy for curiosity-driven navigation."""

import numpy as np
from typing import Optional, Tuple


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
    ) -> np.ndarray:
        """
        Compute exploration bias signal.
        
        Args:
            current_position: Current fly position (x, y, z)
            point_cloud_mapper: PointCloudMapper instance
        
        Returns:
            Exploration signal [forward, turn, up]
        """
        # Random exploration
        if np.random.rand() < self.random_prob:
            return np.random.randn(3) * 0.5
        
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
            direction_unit = direction / direction_norm
        else:
            # No strong preference, continue forward
            direction_unit = np.array([1, 0, 0])
        
        # Exploration signal
        forward_signal = 1.2  # Strongly encourage forward motion
        turn_signal = np.arctan2(direction_unit[1], direction_unit[0])  # Desired heading
        up_signal = direction[2] * 0.3  # Reduced vertical adjustment
        
        signal = np.array([forward_signal, turn_signal, up_signal])
        signal *= self.exploration_weight
        
        return signal
    
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
