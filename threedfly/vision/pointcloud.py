"""3D point cloud accumulation and mapping."""

import numpy as np
import open3d as o3d
from typing import Optional, List, Tuple


class PointCloudMapper:
    """
    Accumulate stereo depth estimates into a global 3D point cloud map.
    
    Tracks explored space for curiosity-driven exploration.
    """
    
    def __init__(
        self,
        voxel_size: float = 0.05,  # 5cm voxel grid
        max_depth: float = 5.0,
        map_bounds: Tuple[float, float, float] = (10.0, 10.0, 5.0),  # x, y, z limits
    ):
        """
        Initialize mapper.
        
        Args:
            voxel_size: Voxel grid resolution for downsampling
            max_depth: Maximum depth to consider
            map_bounds: (x_max, y_max, z_max) map boundaries
        """
        self.voxel_size = voxel_size
        self.max_depth = max_depth
        self.map_bounds = np.array(map_bounds)
        
        # Global point clouds (separate for brain and stereo sources)
        self.global_cloud = o3d.geometry.PointCloud()  # Primary (brain depth)
        self.stereo_cloud = o3d.geometry.PointCloud()  # Comparison (StereoBM)
        
        # Voxel occupancy grid for curiosity
        self.voxel_grid_size = (map_bounds[0] / voxel_size, 
                                map_bounds[1] / voxel_size,
                                map_bounds[2] / voxel_size)
        self.occupancy_grid = np.zeros(
            (int(self.voxel_grid_size[0] * 2),  # -x to +x
             int(self.voxel_grid_size[1] * 2),  # -y to +y
             int(self.voxel_grid_size[2])),     # 0 to +z
            dtype=np.int32
        )
        
        self.total_points_added = 0
        self.stereo_points_added = 0
    
    def add_depth_observation(
        self,
        depth_map: np.ndarray,
        rgb_image: np.ndarray,
        camera_pose: np.ndarray,
        confidence: Optional[np.ndarray] = None,
        min_confidence: float = 0.3,
        source: str = "brain",  # "brain" or "stereo"
    ) -> int:
        """
        Add depth observation to global map.
        
        Args:
            depth_map: Depth map (H, W) in meters
            rgb_image: RGB image (H, W, 3)
            camera_pose: 4x4 camera pose matrix (world coordinates)
            confidence: Confidence map (H, W)
            min_confidence: Minimum confidence threshold
            source: "brain" for brain-based depth, "stereo" for StereoBM comparison
        
        Returns:
            Number of points added
        """
        h, w = depth_map.shape
        
        # Filter by confidence
        if confidence is None:
            valid_mask = (depth_map > 0) & (depth_map < self.max_depth)
        else:
            valid_mask = (depth_map > 0) & (depth_map < self.max_depth) & (confidence > min_confidence)
        
        if not valid_mask.any():
            return 0
        
        # Generate pixel coordinates
        v, u = np.mgrid[0:h, 0:w]
        
        # Intrinsic parameters (simplified pinhole)
        fx = fy = w / 2.0  # Approximate focal length in pixels
        cx, cy = w / 2.0, h / 2.0
        
        # Unproject to 3D camera coordinates
        # Note: camera frame is typically z forward, x right, y down
        z = depth_map[valid_mask]
        x = (u[valid_mask] - cx) * z / fx
        y = (v[valid_mask] - cy) * z / fy
        
        # Stack as [x, y, z] where z is depth (forward direction)
        points_camera = np.stack([x, y, z], axis=-1)
        
        # Transform to world coordinates
        points_homogeneous = np.hstack([points_camera, np.ones((len(points_camera), 1))])
        points_world = (camera_pose @ points_homogeneous.T).T[:, :3]
        
        # Filter by map bounds
        in_bounds = (
            (np.abs(points_world[:, 0]) < self.map_bounds[0]) &
            (np.abs(points_world[:, 1]) < self.map_bounds[1]) &
            (points_world[:, 2] >= 0) &
            (points_world[:, 2] < self.map_bounds[2])
        )
        points_world = points_world[in_bounds]
        
        if len(points_world) == 0:
            return 0
        
        # Get colors - need to flatten and filter properly
        rgb_flat = rgb_image.reshape(-1, 3)  # Flatten to (H*W, 3)
        rgb = rgb_flat[valid_mask.flatten()][in_bounds] / 255.0
        
        # Create point cloud
        new_cloud = o3d.geometry.PointCloud()
        new_cloud.points = o3d.utility.Vector3dVector(points_world)
        new_cloud.colors = o3d.utility.Vector3dVector(rgb)
        
        # Add to appropriate cloud based on source
        if source == "brain":
            self.global_cloud += new_cloud
            # Update occupancy grid (only from primary brain cloud)
            self._update_occupancy(points_world)
            self.total_points_added += len(points_world)
        else:  # stereo comparison
            self.stereo_cloud += new_cloud
            self.stereo_points_added += len(points_world)
        
        return len(points_world)
    
    def _update_occupancy(self, points: np.ndarray):
        """Update voxel occupancy grid."""
        # Convert points to grid indices
        grid_origin = np.array([
            -self.map_bounds[0],
            -self.map_bounds[1],
            0
        ])
        
        grid_indices = ((points - grid_origin) / self.voxel_size).astype(int)
        
        # Clip to grid bounds
        grid_indices = np.clip(
            grid_indices,
            [0, 0, 0],
            np.array(self.occupancy_grid.shape) - 1
        )
        
        # Increment occupancy
        for idx in grid_indices:
            self.occupancy_grid[idx[0], idx[1], idx[2]] += 1
    
    def get_occupancy_at_position(self, position: np.ndarray) -> int:
        """
        Get occupancy count at a world position.
        
        Args:
            position: World position (x, y, z)
        
        Returns:
            Occupancy count (0 = unexplored)
        """
        grid_origin = np.array([
            -self.map_bounds[0],
            -self.map_bounds[1],
            0
        ])
        
        grid_idx = ((position - grid_origin) / self.voxel_size).astype(int)
        
        # Check bounds
        if np.any(grid_idx < 0) or np.any(grid_idx >= self.occupancy_grid.shape):
            return 0
        
        return int(self.occupancy_grid[grid_idx[0], grid_idx[1], grid_idx[2]])
    
    def get_exploration_score(self, position: np.ndarray, radius: float = 0.5) -> float:
        """
        Get exploration score around a position (higher = less explored).
        
        Args:
            position: World position
            radius: Search radius
        
        Returns:
            Exploration score [0, 1] (1 = completely unexplored)
        """
        # Sample points in radius
        n_samples = 20
        angles = np.linspace(0, 2 * np.pi, n_samples)
        
        total_occupancy = 0
        for angle in angles:
            sample_pos = position + radius * np.array([np.cos(angle), np.sin(angle), 0])
            total_occupancy += self.get_occupancy_at_position(sample_pos)
        
        # Normalize (higher occupancy = lower score)
        max_occupancy = 50  # Saturation threshold
        avg_occupancy = total_occupancy / n_samples
        score = np.exp(-avg_occupancy / max_occupancy)
        
        return float(score)
    
    def downsample(self, voxel_size: Optional[float] = None, source: str = "brain") -> o3d.geometry.PointCloud:
        """
        Downsample cloud for visualization.
        
        Args:
            voxel_size: Voxel size for downsampling (default: use mapper's voxel_size)
            source: "brain" for primary cloud, "stereo" for comparison cloud
        
        Returns:
            Downsampled point cloud
        """
        cloud = self.global_cloud if source == "brain" else self.stereo_cloud
        
        if len(cloud.points) == 0:
            return cloud
        
        if voxel_size is None:
            voxel_size = self.voxel_size
        
        downsampled = cloud.voxel_down_sample(voxel_size=voxel_size)
        return downsampled
    
    def save_point_cloud(self, filename: str, voxel_size: Optional[float] = None, source: str = "brain"):
        """
        Save point cloud to file (PLY format).
        
        Args:
            filename: Output filename
            voxel_size: Voxel size for downsampling (default: use 1/2 of mapper's voxel_size for denser export)
            source: "brain" for primary cloud, "stereo" for comparison cloud
        """
        if voxel_size is None:
            # Use smaller voxel size for export (denser than default downsample)
            voxel_size = self.voxel_size / 2.0
        
        downsampled = self.downsample(voxel_size=voxel_size, source=source)
        o3d.io.write_point_cloud(filename, downsampled)
        source_label = "brain-depth" if source == "brain" else "StereoBM"
        print(f"✓ Saved {source_label} point cloud: {filename} ({len(downsampled.points)} points)")
    
    def get_statistics(self) -> dict:
        """Get mapping statistics."""
        occupied_voxels = int((self.occupancy_grid > 0).sum())
        total_voxels = int(self.occupancy_grid.size)
        
        # Calculate exploration ratio as percentage of reachable space
        # Use a more meaningful denominator: voxels within flying height
        flying_height_voxels = int(self.map_bounds[2] / self.voxel_size)  # Height dimension
        reachable_voxels = self.occupancy_grid.shape[0] * self.occupancy_grid.shape[1] * flying_height_voxels
        exploration_ratio = occupied_voxels / max(1, reachable_voxels)
        
        return {
            "total_points": len(self.global_cloud.points),  # Brain depth (primary)
            "points_added": self.total_points_added,
            "stereo_points": len(self.stereo_cloud.points),  # StereoBM (comparison)
            "stereo_points_added": self.stereo_points_added,
            "occupied_voxels": occupied_voxels,
            "total_voxels": total_voxels,
            "exploration_ratio": float(exploration_ratio),
        }
