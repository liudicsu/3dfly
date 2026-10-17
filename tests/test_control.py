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


def _synthetic_bio_annotations(n=80):
    import pandas as pd

    types = (
        ["R1"] * 8
        + ["T4a"] * 8
        + ["T5a"] * 8
        + ["LPLC2"] * 4
        + ["LC4"] * 4
        + ["DNp01"] * 2
        + ["DNg02_a"] * 8
        + ["DNa01"] * 2
        + ["DNa02"] * 2
        + ["DNb01"] * 8
        + ["other"] * (n - 54)
    )
    types = types[:n]
    sides = ["L" if i % 2 == 0 else "R" for i in range(n)]
    body_ids = np.arange(1, n + 1)
    annotations = pd.DataFrame({
        "bodyId": body_ids,
        "type": types,
        "celltype": types,
        "somaSide": sides,
        "rootSide": sides,
    })
    body_id_to_idx = {int(bid): idx for idx, bid in enumerate(body_ids)}
    return n, annotations, body_ids, body_id_to_idx


def test_biological_controller_normalizes_and_does_not_saturate():
    """Readouts use tanh((r-rest)/scale); actions should not pin at ±1."""
    from threedfly.control.controller import BiologicalFlightController

    n, annotations, body_ids, body_id_to_idx = _synthetic_bio_annotations()
    controller = BiologicalFlightController(
        n_neurons=n,
        annotations=annotations,
        body_ids=body_ids,
        body_id_to_idx=body_id_to_idx,
        warmup_steps=20,
    )
    rest = np.full(n, 2.0)
    for _ in range(8):
        controller.record_resting_rates(rest + np.random.randn(n) * 0.05)
    controller.finalize_warmup()

    high = rest.copy()
    high[controller.output_mapping["DNp01"]] = 10.0
    high[controller.output_mapping["DNg02"]] = 10.0
    high[controller.output_mapping["DNa02"]] = 10.0

    actions = [controller.compute_action(high) for _ in range(30)]
    stacked = np.stack(actions)
    sat_frac = np.mean(np.abs(stacked) >= 0.999)
    assert sat_frac < 0.5, f"Too many saturated actions: {sat_frac:.2%}"


def test_dnp01_is_event_trigger_not_continuous_pitch():
    """DNp01 should pulse pitch on threshold crossing, not hold pitch at the rail."""
    from threedfly.control.controller import BiologicalFlightController

    n, annotations, body_ids, body_id_to_idx = _synthetic_bio_annotations()
    controller = BiologicalFlightController(
        n_neurons=n,
        annotations=annotations,
        body_ids=body_ids,
        body_id_to_idx=body_id_to_idx,
        warmup_steps=10,
    )
    rest = np.full(n, 1.0)
    for _ in range(5):
        controller.record_resting_rates(rest)
    controller.finalize_warmup()

    baseline = rest.copy()
    burst = rest.copy()
    burst[controller.output_mapping["DNp01"]] = 8.0

    pitches = []
    pitches.append(float(controller.compute_action(baseline)[3]))
    for _ in range(3):
        pitches.append(float(controller.compute_action(burst)[3]))
    for _ in range(20):
        pitches.append(float(controller.compute_action(burst)[3]))

    assert max(np.abs(pitches[1:4])) > abs(pitches[0]) + 0.05
    assert np.mean(np.abs(pitches[-10:])) < 0.9


def test_yaw_falls_back_when_dna01_silent():
    """If DNa01 is dead, yaw should use DNa02 or DNg02 left-right difference."""
    from threedfly.control.controller import BiologicalFlightController

    n, annotations, body_ids, body_id_to_idx = _synthetic_bio_annotations()
    controller = BiologicalFlightController(
        n_neurons=n,
        annotations=annotations,
        body_ids=body_ids,
        body_id_to_idx=body_id_to_idx,
        warmup_steps=10,
    )
    rest = np.zeros(n)
    rest[controller.output_mapping["DNg02"]] = 1.0
    rest[controller.output_mapping["DNa02"]] = 1.0
    for _ in range(5):
        controller.record_resting_rates(rest)
    controller.finalize_warmup()

    activity = rest.copy()
    dng_r = [i for i in controller.output_mapping.get("DNg02_R", [])]
    if len(dng_r) == 0:
        dng_r = list(controller.output_mapping["DNg02"][::2])
    activity[np.array(dng_r, dtype=int)] = 4.0
    activity[controller.output_mapping["DNa02"]] = 3.0

    action = controller.compute_action(activity)
    assert abs(action[4]) > 1e-3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
