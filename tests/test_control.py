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
    pytest.importorskip("open3d")
    from threedfly.vision import PointCloudMapper
    
    policy = ExplorationPolicy(exploration_weight=0.5, random_exploration_prob=0.0)
    mapper = PointCloudMapper(voxel_size=0.1)
    
    position = np.array([0, 0, 1.5])
    
    signal = policy.compute_exploration_signal(position, mapper)
    
    assert signal.shape == (3,)  # [forward, turn, up]
    assert np.all(np.isfinite(signal))


def test_frontier_directions():
    """Test frontier direction identification."""
    pytest.importorskip("open3d")
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


def test_yaw_uses_bilateral_dna02_and_dng02():
    """Yaw is L−R of DNa02 and DNg02, so left vs right drive opposite turns."""
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
    for _ in range(6):
        controller.record_resting_rates(rest)
    controller.finalize_warmup()
    controller.prev_action[:] = 0.0
    controller.alpha = 1.0

    left_high = rest.copy()
    left_high[controller.output_mapping["DNa02_L"]] = 6.0
    left_high[controller.output_mapping["DNg02_L"]] = 6.0
    yaw_left = float(controller.compute_action(left_high)[4])

    controller.prev_action[:] = 0.0
    right_high = rest.copy()
    right_high[controller.output_mapping["DNa02_R"]] = 6.0
    right_high[controller.output_mapping["DNg02_R"]] = 6.0
    yaw_right = float(controller.compute_action(right_high)[4])

    assert yaw_left * yaw_right < 0, f"Expected opposite yaw, got {yaw_left:.3f} vs {yaw_right:.3f}"
    assert abs(yaw_left) > 0.02 and abs(yaw_right) > 0.02


def test_biological_controller_does_not_add_exploration_to_action():
    """Exploration must bias sensory input, not override the DN action."""
    from threedfly.control.controller import BiologicalFlightController

    n, annotations, body_ids, body_id_to_idx = _synthetic_bio_annotations()
    controller = BiologicalFlightController(
        n_neurons=n,
        annotations=annotations,
        body_ids=body_ids,
        body_id_to_idx=body_id_to_idx,
        warmup_steps=8,
    )
    rest = np.full(n, 1.0)
    for _ in range(5):
        controller.record_resting_rates(rest)
    controller.finalize_warmup()
    controller.prev_action[:] = 0.0
    controller.alpha = 1.0
    bio = controller.compute_action(rest)
    controller.prev_action[:] = 0.0
    mixed = controller.compute_action(rest, exploration_signal=np.array([1.0, 1.0, 0.0]))
    assert np.allclose(bio, mixed), "Biological action must ignore exploration_signal"


def test_exploration_policy_biases_luminance_not_world_heading():
    """Turn signal is a heading error in [-1, 1]; apply_visual_bias shifts L vs R."""
    pytest.importorskip("open3d")
    from threedfly.vision import PointCloudMapper

    policy = ExplorationPolicy(exploration_weight=1.0, random_exploration_prob=0.0)
    mapper = PointCloudMapper(voxel_size=0.1)
    pos = np.array([0.0, 0.0, 1.5])
    signal = policy.compute_exploration_signal(pos, mapper, current_heading=0.0)
    assert signal.shape == (3,)
    assert abs(signal[1]) <= 1.0 + 1e-6
    left, right = policy.apply_visual_bias(0.2, 0.2, np.array([0.0, 0.8, 0.0]))
    assert left > right
    left2, right2 = policy.apply_visual_bias(0.2, 0.2, np.array([0.0, -0.8, 0.0]))
    assert right2 > left2


def test_strong_looming_raises_dnp01_above_rest():
    """Strong synthetic looming into LPLC2/LC4 must raise DNp01 on a real-style pipeline."""
    from scipy import sparse
    from threedfly.connectome.simulator import RateSimulator
    from threedfly.control.controller import BiologicalFlightController

    n, annotations, body_ids, body_id_to_idx = _synthetic_bio_annotations()
    controller = BiologicalFlightController(
        n_neurons=n,
        annotations=annotations,
        body_ids=body_ids,
        body_id_to_idx=body_id_to_idx,
        warmup_steps=5,
    )
    adj = sparse.csr_matrix((n, n), dtype=np.float32).tolil()
    dnp = controller.output_mapping["DNp01"]
    loom = np.concatenate([
        controller.input_mapping["looming_left"],
        controller.input_mapping["looming_right"],
    ])
    assert len(dnp) > 0 and len(loom) > 0
    for pre in loom:
        for post in dnp:
            adj[int(pre), int(post)] = 8.0
    brain = RateSimulator(adj.tocsr(), dt=0.010, syn_weight_scale=0.5)

    rest_rates = []
    for _ in range(8):
        controller.set_visual_input(
            left_luminance=0.0,
            right_luminance=0.0,
            optic_flow_features={},
            looming_features={"left": 0.0, "right": 0.0},
            brain=brain,
        )
        rates = brain.step()
        controller.record_resting_rates(rates)
        rest_rates.append(float(rates[dnp].mean()))
    controller.finalize_warmup()
    rest = float(np.mean(rest_rates[-4:]))

    driven = []
    for _ in range(20):
        controller.set_visual_input(
            left_luminance=0.0,
            right_luminance=0.0,
            optic_flow_features={},
            looming_features={"left": 4.0, "right": 4.0},
            brain=brain,
        )
        rates = brain.step()
        driven.append(float(rates[dnp].mean()))
    peak = float(np.max(driven))
    assert peak > rest + 0.15, f"DNp01 did not rise under looming (rest={rest:.4f} peak={peak:.4f})"


def test_yaw_is_zero_mean_at_asymmetric_rest():
    """L/R rest mismatch must not produce a constant yaw offset."""
    from threedfly.control.controller import BiologicalFlightController

    n, annotations, body_ids, body_id_to_idx = _synthetic_bio_annotations()
    controller = BiologicalFlightController(
        n_neurons=n,
        annotations=annotations,
        body_ids=body_ids,
        body_id_to_idx=body_id_to_idx,
        warmup_steps=20,
    )
    rest = np.full(n, 1.0)
    rest[controller.output_mapping["DNa02_L"]] = 9.0
    rest[controller.output_mapping["DNa02_R"]] = 1.6
    rest[controller.output_mapping["DNg02_L"]] = 4.0
    rest[controller.output_mapping["DNg02_R"]] = 0.5
    for _ in range(12):
        controller.record_resting_rates(rest)
    controller.finalize_warmup()
    controller.prev_action[:] = 0.0
    controller.alpha = 1.0
    yaw = float(controller.compute_action(rest)[4])
    assert abs(yaw) < 0.08, f"Yaw at rest should be ~0, got {yaw:.3f}"


def test_yaw_drops_silent_side_pair():
    """If one side of DNg02 is silent, that pair must not contribute a one-sided yaw."""
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
    rest[controller.output_mapping["DNg02_R"]] = 0.0
    for _ in range(6):
        controller.record_resting_rates(rest)
    controller.finalize_warmup()
    controller.prev_action[:] = 0.0
    controller.alpha = 1.0

    driven = rest.copy()
    driven[controller.output_mapping["DNg02_L"]] = 8.0
    yaw = float(controller.compute_action(driven)[4])
    assert abs(yaw) < 0.08, f"Silent-side pair leaked into yaw: {yaw:.3f}"


def test_warmup_uses_robust_median_mad():
    """Rest stats use median/MAD so a few pinned samples do not dominate."""
    from threedfly.control.controller import BiologicalFlightController

    n, annotations, body_ids, body_id_to_idx = _synthetic_bio_annotations()
    controller = BiologicalFlightController(
        n_neurons=n,
        annotations=annotations,
        body_ids=body_ids,
        body_id_to_idx=body_id_to_idx,
        warmup_steps=40,
    )
    base = np.full(n, 2.0)
    for i in range(30):
        sample = base.copy()
        if i >= 25:
            sample[controller.output_mapping["DNa02"]] = 50.0
        controller.record_resting_rates(sample)
    controller.finalize_warmup()
    rest = controller.resting_rates["DNa02"]
    assert abs(rest - 2.0) < 1.0, f"Median rest should ignore pinned tail, got {rest:.3f}"


def test_warmup_converges_when_median_stable():
    """Spread relative to itself is OK; stable median should count as converged."""
    from threedfly.control.controller import BiologicalFlightController

    n, annotations, body_ids, body_id_to_idx = _synthetic_bio_annotations()
    controller = BiologicalFlightController(
        n_neurons=n,
        annotations=annotations,
        body_ids=body_ids,
        body_id_to_idx=body_id_to_idx,
        warmup_steps=80,
        min_warmup_steps=20,
    )
    rng = np.random.RandomState(0)
    for _ in range(40):
        sample = np.full(n, 3.0) + rng.randn(n) * 1.5
        controller.record_resting_rates(sample)
    assert controller.warmup_converged() is True


def test_visual_input_is_population_normalized():
    """Injected current per population must stay well below the rate ceiling."""
    from threedfly.connectome.simulator import RateSimulator
    from threedfly.control.controller import BiologicalFlightController
    from scipy import sparse

    n, annotations, body_ids, body_id_to_idx = _synthetic_bio_annotations()
    controller = BiologicalFlightController(
        n_neurons=n,
        annotations=annotations,
        body_ids=body_ids,
        body_id_to_idx=body_id_to_idx,
        warmup_steps=5,
    )
    brain = RateSimulator(sparse.csr_matrix((n, n), dtype=np.float32), dt=0.010, max_rate=50.0)
    controller.set_visual_input(
        left_luminance=12.0,
        right_luminance=12.0,
        optic_flow_features={"t4a_left": np.array([9.0])},
        looming_features={"left": 8.0, "right": 8.0},
        brain=brain,
    )
    driven = brain.external_input[brain.external_input != 0]
    assert driven.size > 0
    assert float(np.max(np.abs(driven))) <= 2.0


def test_brain_triggered_escape_is_counted_separately():
    """DNp01 threshold crossings are brain escapes; looming residual is engineered."""
    from threedfly.control.controller import BiologicalFlightController

    n, annotations, body_ids, body_id_to_idx = _synthetic_bio_annotations()
    controller = BiologicalFlightController(
        n_neurons=n,
        annotations=annotations,
        body_ids=body_ids,
        body_id_to_idx=body_id_to_idx,
        warmup_steps=8,
    )
    rest = np.full(n, 1.0)
    for _ in range(5):
        controller.record_resting_rates(rest)
    controller.finalize_warmup()
    controller.prev_action[:] = 0.0
    controller.alpha = 1.0
    controller._last_looming = 0.0
    burst = rest.copy()
    burst[controller.output_mapping["DNp01"]] = 20.0
    controller.compute_action(rest)
    controller.compute_action(burst)
    assert controller.brain_escape_count >= 1
    assert controller.engineered_escape_count == 0


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
