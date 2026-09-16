"""Tests for flight control."""

import numpy as np
import pytest

from threedfly.control import FlightController, ExplorationPolicy


def test_flight_controller_init():
    """Test FlightController initialization."""
    n_neurons = 1000
    controller = FlightController(n_neurons)
    
    assert controller.n_neurons == n_neurons
    assert controller.n_dn > 0


def test_compute_action():
    """Test action computation from brain activity."""
    n_neurons = 1000
    controller = FlightController(n_neurons)
    
    # Random brain activity
    brain_activity = np.random.rand(n_neurons)
    
    action = controller.compute_action(brain_activity)
    
    assert action.shape == (5,)  # [forward, up, roll, pitch, yaw]
    assert np.all(action >= -1.0)
    assert np.all(action <= 1.0)


def test_action_with_exploration():
    """Test action with exploration signal."""
    n_neurons = 1000
    controller = FlightController(n_neurons)
    
    brain_activity = np.random.rand(n_neurons)
    exploration_signal = np.array([0.5, 0.2, 0.1])  # [forward, turn, up]
    
    action = controller.compute_action(brain_activity, exploration_signal)
    
    assert action.shape == (5,)
    assert np.all(action >= -1.0)
    assert np.all(action <= 1.0)


def test_exploration_policy_init():
    """Test ExplorationPolicy initialization."""
    policy = ExplorationPolicy(exploration_weight=0.5)
    
    assert policy.exploration_weight == 0.5
    assert len(policy.sample_angles) > 0


def test_exploration_signal():
    """Test exploration signal computation."""
    from threedfly.vision import PointCloudMapper
    
    policy = ExplorationPolicy(exploration_weight=0.5, random_exploration_prob=0.0)
    mapper = PointCloudMapper(voxel_size=0.1)
    
    position = np.array([0, 0, 1.5])
    
    signal = policy.compute_exploration_signal(position, mapper)
    
    assert signal.shape == (3,)  # [forward, turn, up]
    assert np.all(np.isfinite(signal))


def test_frontier_directions():
    """Test frontier direction identification."""
    from threedfly.vision import PointCloudMapper
    
    policy = ExplorationPolicy()
    mapper = PointCloudMapper()
    
    position = np.array([0, 0, 1.5])
    
    frontiers = policy.get_frontier_directions(position, mapper, top_k=3)
    
    assert len(frontiers) == 3
    for pos, score in frontiers:
        assert len(pos) == 3
        assert 0 <= score <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
