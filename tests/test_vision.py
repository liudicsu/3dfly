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
    
    # Create synthetic stereo pair (using higher resolution)
    h, w = 120, 160
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
    
    img = np.random.randint(0, 256, (120, 160, 3), dtype=np.uint8)
    
    samples = stereo.ommatidial_sample(img)
    
    assert samples.shape == (stereo.n_ommatidia,)
    assert np.all(samples >= 0)
    assert np.all(samples <= 1)


def test_visual_features():
    """Test visual feature extraction."""
    stereo = StereoVision()
    
    left_img = np.random.randint(0, 256, (120, 160, 3), dtype=np.uint8)
    right_img = np.random.randint(0, 256, (120, 160, 3), dtype=np.uint8)
    
    features = stereo.get_visual_features(left_img, right_img)
    
    assert "left_ommatidia" in features
    assert "right_ommatidia" in features
    assert "depth_estimate" in features
    assert "depth_map" in features
    assert features["left_ommatidia"].shape == (stereo.n_ommatidia,)
    assert "looming_left" in features and "looming_right" in features
    assert features["looming_left"].shape == (8,)
    assert features["looming_right"].shape == (8,)


def _ring_frame(h, w, radius, width=7):
    img = np.full((h, w, 3), 28, dtype=np.uint8)
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((yy - h / 2.0) ** 2 + (xx - w / 2.0) ** 2)
    img[np.abs(r - radius) < width] = 230
    rng = np.random.RandomState(0)
    noise = rng.randint(0, 12, (h, w, 1), dtype=np.uint8)
    noise = np.maximum(noise, np.fliplr(noise))
    img ^= np.repeat(noise, 3, axis=2)
    return img


def test_static_scene_looming_is_near_zero():
    """No expansion → looming ~0, even if StereoBM would invent close depth."""
    stereo = StereoVision()
    img = _ring_frame(120, 160, radius=40)
    stereo.get_visual_features(img, img)
    feat = stereo.get_visual_features(img, img)
    assert float(np.max(feat["looming_features"])) < 0.2
    assert float(np.max(feat["looming_left"])) < 0.2
    assert float(np.max(feat["looming_right"])) < 0.2


def test_expansion_looming_not_saturated_and_covers_left_sectors():
    """Per-eye expansion covers the left hemifield and does not pin at ~0.85."""
    from threedfly.vision.stereo import compute_expansion_looming

    h, w = 60, 80
    yy, xx = np.indices((h, w))
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    flow = np.zeros((h, w, 2), dtype=np.float32)
    flow[..., 0] = (xx - cx) * 0.08
    flow[..., 1] = (yy - cy) * 0.08
    loom = compute_expansion_looming(flow)
    assert loom.shape == (8,)
    assert float(loom.max()) <= 1.0 + 1e-6
    assert float(loom.max()) < 0.95, "percentile expansion must not saturate like 1/(d+0.1)/10"
    assert float(loom[0]) > 0.05
    assert float(loom[1]) > 0.02 or float(loom[2]) > 0.02

    trans = np.zeros((h, w, 2), dtype=np.float32)
    trans[..., 0] = 2.0
    loom_t = compute_expansion_looming(trans)
    assert float(loom_t.max()) < 0.15


def test_mirrored_scene_left_right_looming_symmetric():
    """Left and right eyes must match on a left-right mirrored expanding scene."""
    stereo = StereoVision()
    h, w = 120, 160
    f1 = _ring_frame(h, w, radius=28)
    f2 = _ring_frame(h, w, radius=48)
    stereo.get_visual_features(f1, np.fliplr(f1))
    feat = stereo.get_visual_features(f2, np.fliplr(f2))
    left, right = feat["looming_left"], feat["looming_right"]
    # Mirror sectors: 0↔0, 1↔7, 2↔6, 3↔5, 4↔4
    mirrored_right = np.array(
        [right[0], right[7], right[6], right[5], right[4], right[3], right[2], right[1]],
        dtype=np.float64,
    )
    assert float(np.max(np.abs(left - mirrored_right))) < 0.12, (
        f"L/R looming not symmetric: left={left} mirrored_right={mirrored_right}"
    )
    assert float(left.max()) > 0.05 and float(right.max()) > 0.05


def test_left_eye_expansion_drives_left_looming_without_stereo():
    """Looming comes from each eye's own image, not StereoBM (left columns often invalid)."""
    stereo = StereoVision()
    h, w = 120, 160
    blank = np.full((h, w, 3), 40, dtype=np.uint8)
    blank ^= np.random.RandomState(1).randint(0, 8, blank.shape, dtype=np.uint8)
    left1 = blank.copy()
    left2 = blank.copy()
    x = np.arange(w)
    left1[:, np.abs(x - 25) < 8] = 220
    left2[:, np.abs(x - 25) < 18] = 220
    stereo.get_visual_features(left1, blank)
    feat = stereo.get_visual_features(left2, blank)
    # Left-of-image expansion should light left sectors, not only 0/4-7.
    assert float(feat["looming_left"][1] + feat["looming_left"][2]) > 0.04
    assert float(feat["looming_right"].max()) < float(feat["looming_left"].max()) + 0.05


def test_approach_correlation_helper():
    from threedfly.vision.stereo import looming_approach_correlation

    r = looming_approach_correlation(
        np.array([0.0, 0.2, 0.8, 0.9]),
        np.array([0.0, 0.25, 0.7, 1.0]),
    )
    assert r > 0.9
    assert looming_approach_correlation(np.array([1.0]), np.array([1.0])) == 0.0


def test_forward_sector_approach_scores_closing_speed():
    """Diagnostic Pearson uses forward sectors (0, 1, 7), not the max of all 8."""
    from threedfly.vision.stereo import (
        LOOMING_FORWARD_SECTORS,
        forward_looming_and_approach,
        looming_approach_correlation,
    )

    assert tuple(LOOMING_FORWARD_SECTORS) == (0, 1, 7)
    loom_l = np.array([0.8, 0.6, 0.0, 0.0, 0.9, 0.0, 0.0, 0.5], dtype=np.float32)
    loom_r = loom_l.copy()
    prev = np.array([2.0, 1.8, 4.0, 4.0, 4.0, 4.0, 4.0, 1.5], dtype=np.float32)
    curr = np.array([1.6, 1.5, 4.0, 4.0, 0.05, 4.0, 4.0, 1.2], dtype=np.float32)
    loom_v, approach_v = forward_looming_and_approach(loom_l, loom_r, prev, curr, dt=0.02)
    assert approach_v > 0
    # Back-sector closing (index 4) must not dominate the diagnostic.
    back_close = float(max(prev[4] - curr[4], 0.0) / 0.02)
    assert approach_v < back_close, (
        f"forward approach {approach_v:.1f} should ignore back closing {back_close:.1f}"
    )
    series_loom, series_app = [], []
    distances = np.linspace(2.0, 0.4, 8)
    for i in range(1, len(distances)):
        p = np.full(8, 4.0, dtype=np.float32)
        c = np.full(8, 4.0, dtype=np.float32)
        p[list(LOOMING_FORWARD_SECTORS)] = distances[i - 1]
        c[list(LOOMING_FORWARD_SECTORS)] = distances[i]
        lv = np.zeros(8, dtype=np.float32)
        lv[list(LOOMING_FORWARD_SECTORS)] = 1.0 / distances[i]
        loom_i, app_i = forward_looming_and_approach(lv, lv, p, c, dt=0.02)
        series_loom.append(loom_i)
        series_app.append(app_i)
    assert looming_approach_correlation(series_loom, series_app) >= 0.4


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
    
    # Check shapes (NEW: 12×16 motion parallax grid, upgraded from 8×6)
    assert depth_map.shape == (12, 16)  # Motion parallax flow grid
    assert confidence.shape == (12, 16)
    
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
    
    # Create coarse depth map (NEW: 12×16 resolution)
    coarse_depth = np.ones((12, 16)) * 1.5
    
    # Upsample to higher resolution
    upsampled = estimator.upsample_depth_map(coarse_depth, target_height=120, target_width=160)
    
    assert upsampled.shape == (120, 160)
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
