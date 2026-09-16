"""Tests for vision processing."""

import numpy as np
import pytest

from threedfly.vision import StereoVision, PointCloudMapper


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
    
    n_added = mapper.add_depth_observation(depth_map, rgb_image, camera_pose)
    
    assert n_added > 0
    assert mapper.total_points_added == n_added


def test_exploration_score():
    """Test exploration score computation."""
    mapper = PointCloudMapper(voxel_size=0.1)
    
    position = np.array([0, 0, 1.5])
    
    # Initially unexplored
    score1 = mapper.get_exploration_score(position, radius=0.5)
    assert score1 > 0.5  # Should be high (unexplored)
    
    # Add some observations
    depth_map = np.ones((20, 20)) * 1.0
    rgb_image = np.ones((20, 20, 3), dtype=np.uint8) * 128
    camera_pose = np.eye(4)
    camera_pose[:3, 3] = position
    
    for _ in range(5):
        mapper.add_depth_observation(depth_map, rgb_image, camera_pose)
    
    # Now explored
    score2 = mapper.get_exploration_score(position, radius=0.5)
    assert score2 < score1  # Should be lower (explored)


def test_mapping_statistics():
    """Test mapping statistics."""
    mapper = PointCloudMapper()
    
    stats = mapper.get_statistics()
    
    assert "total_points" in stats
    assert "occupied_voxels" in stats
    assert "exploration_ratio" in stats
    assert stats["total_points"] == 0  # Initially empty


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
