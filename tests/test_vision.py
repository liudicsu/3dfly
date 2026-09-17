"""Tests for vision processing."""

import numpy as np
import pytest

from scipy import sparse
from threedfly.vision import StereoVision, PointCloudMapper, BrainDepthEstimator


def test_stereo_vision_init():
    """Test StereoVision initialization."""
    stereo = StereoVision(baseline=0.012, focal_length=0.01)
    
    assert stereo.baseline == 0.012
    assert stereo.focal_length == 0.01


def test_stereo_depth_estimation():
    """Test depth estimation from stereo pair."""
    stereo = StereoVision()
    
    # Create synthetic stereo pair
    h, w = 48, 64
    left_img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    right_img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    
    depth, confidence = stereo.estimate_depth(left_img, right_img)
    
    assert depth.shape[0] <= h  # May be downsampled
    assert depth.shape[1] <= w
    assert confidence.shape == depth.shape
    assert np.all(depth >= 0)
    assert np.all(confidence >= 0)
    assert np.all(confidence <= 1)


def test_ommatidial_sampling():
    """Test ommatidial sampling."""
    stereo = StereoVision()
    
    img = np.random.randint(0, 256, (48, 64, 3), dtype=np.uint8)
    
    samples = stereo.ommatidial_sample(img)
    
    assert samples.shape == (stereo.n_ommatidia,)
    assert np.all(samples >= 0)
    assert np.all(samples <= 1)


def test_visual_features():
    """Test visual feature extraction."""
    stereo = StereoVision()
    
    left_img = np.random.randint(0, 256, (48, 64, 3), dtype=np.uint8)
    right_img = np.random.randint(0, 256, (48, 64, 3), dtype=np.uint8)
    
    features = stereo.get_visual_features(left_img, right_img)
    
    assert "left_ommatidia" in features
    assert "right_ommatidia" in features
    assert "depth_estimate" in features
    assert "depth_map" in features
    assert features["left_ommatidia"].shape == (stereo.n_ommatidia,)


def test_point_cloud_mapper_init():
    """Test PointCloudMapper initialization."""
    mapper = PointCloudMapper(voxel_size=0.05)
    
    assert mapper.voxel_size == 0.05
    assert mapper.total_points_added == 0


def test_point_cloud_add_observation():
    """Test adding depth observation to map."""
    mapper = PointCloudMapper(voxel_size=0.1)
    
    # Create synthetic depth observation
    h, w = 24, 32
    depth_map = np.random.uniform(0.5, 2.0, (h, w))
    rgb_image = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    camera_pose = np.eye(4)
    camera_pose[:3, 3] = [0, 0, 1.5]  # Camera at height 1.5m
    
    n_added = mapper.add_depth_observation(depth_map, rgb_image, camera_pose, source="brain")
    
    assert n_added > 0
    assert mapper.total_points_added == n_added


def test_exploration_score():
    """Test exploration score computation."""
    mapper = PointCloudMapper(voxel_size=0.05)  # Smaller voxels for better sensitivity
    
    position = np.array([1.0, 1.0, 1.5])
    
    # Initially unexplored
    score1 = mapper.get_exploration_score(position, radius=0.5)
    assert score1 > 0.5  # Should be high (unexplored)
    
    # Add many observations at this position
    depth_map = np.ones((30, 30)) * 0.5
    rgb_image = np.ones((30, 30, 3), dtype=np.uint8) * 128
    camera_pose = np.eye(4)
    camera_pose[:3, 3] = position
    
    # Add observations multiple times to increase occupancy
    for _ in range(20):
        mapper.add_depth_observation(depth_map, rgb_image, camera_pose, source="brain")
    
    # Now explored
    score2 = mapper.get_exploration_score(position, radius=0.5)
    assert score2 <= score1  # Should be lower or equal (explored)


def test_mapping_statistics():
    """Test mapping statistics."""
    mapper = PointCloudMapper()
    
    stats = mapper.get_statistics()
    
    assert "total_points" in stats
    assert "occupied_voxels" in stats
    assert "exploration_ratio" in stats
    assert stats["total_points"] == 0  # Initially empty


def test_brain_depth_estimator():
    """Test brain-based depth estimation."""
    # Create small test connectome
    n_neurons = 100
    adjacency = sparse.random(n_neurons, n_neurons, density=0.1, format='csr')
    
    estimator = BrainDepthEstimator(
        n_neurons=n_neurons,
        adjacency=adjacency,
        use_rate_model=True
    )
    
    # Create synthetic brain activity
    brain_activity = np.random.rand(n_neurons) * 5.0
    
    # Test depth estimation
    result = estimator.estimate_depth_from_activity(brain_activity)
    
    assert "depth_map" in result
    assert "confidence" in result
    assert "mean_depth" in result
    assert "depth_distribution" in result
    
    depth_map = result["depth_map"]
    confidence = result["confidence"]
    mean_depth = result["mean_depth"]
    
    # Check shapes
    assert depth_map.shape == (8, 6)  # Coarse depth map
    assert confidence.shape == (8, 6)
    
    # Check value ranges
    assert np.all(depth_map >= estimator.depth_min)
    assert np.all(depth_map <= estimator.depth_max)
    assert np.all(confidence >= 0) and np.all(confidence <= 1)
    assert estimator.depth_min <= mean_depth <= estimator.depth_max


def test_brain_depth_upsample():
    """Test brain depth map upsampling."""
    n_neurons = 100
    adjacency = sparse.random(n_neurons, n_neurons, density=0.1, format='csr')
    
    estimator = BrainDepthEstimator(
        n_neurons=n_neurons,
        adjacency=adjacency,
        use_rate_model=True
    )
    
    # Create coarse depth map
    coarse_depth = np.ones((8, 6)) * 1.5
    
    # Upsample
    upsampled = estimator.upsample_depth_map(coarse_depth, target_height=48, target_width=64)
    
    assert upsampled.shape == (48, 64)
    assert np.allclose(upsampled.mean(), coarse_depth.mean(), atol=0.1)


def test_brain_depth_statistics():
    """Test brain depth statistics tracking."""
    n_neurons = 100
    adjacency = sparse.random(n_neurons, n_neurons, density=0.1, format='csr')
    
    estimator = BrainDepthEstimator(
        n_neurons=n_neurons,
        adjacency=adjacency,
        use_rate_model=True
    )
    
    # Run several estimates
    for _ in range(10):
        brain_activity = np.random.rand(n_neurons) * 5.0
        estimator.estimate_depth_from_activity(brain_activity)
    
    # Check statistics
    stats = estimator.get_statistics()
    
    assert "last_depth" in stats
    assert "n_visual_neurons" in stats
    assert "mean_depth_history" in stats
    assert "std_depth_history" in stats
    
    assert stats["n_visual_neurons"] == estimator.n_visual
    assert len(estimator.depth_history) == 10


def test_brain_depth_reset():
    """Test brain depth estimator reset."""
    n_neurons = 100
    adjacency = sparse.random(n_neurons, n_neurons, density=0.1, format='csr')
    
    estimator = BrainDepthEstimator(
        n_neurons=n_neurons,
        adjacency=adjacency,
        use_rate_model=True
    )
    
    # Run some estimates
    for _ in range(5):
        brain_activity = np.random.rand(n_neurons) * 5.0
        estimator.estimate_depth_from_activity(brain_activity)
    
    # Reset
    estimator.reset()
    
    assert estimator.last_depth_estimate == 1.0
    assert len(estimator.depth_history) == 0


def test_point_cloud_mapper_sources():
    """Test point cloud mapper with different sources."""
    mapper = PointCloudMapper()
    
    depth = np.ones((24, 32)) * 1.0
    rgb = np.random.randint(0, 256, (24, 32, 3), dtype=np.uint8)
    pose = np.eye(4)
    
    # Add to brain cloud
    n_brain = mapper.add_depth_observation(depth, rgb, pose, source="brain")
    
    # Add to stereo cloud
    n_stereo = mapper.add_depth_observation(depth, rgb, pose, source="stereo")
    
    # Verify separate clouds
    assert len(mapper.global_cloud.points) == n_brain
    assert len(mapper.stereo_cloud.points) == n_stereo
    
    # Test statistics
    stats = mapper.get_statistics()
    assert stats["total_points"] == n_brain  # Brain points
    assert stats["stereo_points"] == n_stereo  # Stereo points
    
    # Test downsampling both sources
    brain_downsampled = mapper.downsample(source="brain")
    stereo_downsampled = mapper.downsample(source="stereo")
    
    assert len(brain_downsampled.points) > 0
    assert len(stereo_downsampled.points) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
