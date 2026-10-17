"""Brain-driven escape (no engineered residual by default) and fly-model polish."""

import numpy as np
import pytest
from scipy import sparse

from tests.test_control import _synthetic_bio_annotations


def _bio_controller(**kwargs):
    from threedfly.control.controller import BiologicalFlightController

    n, annotations, body_ids, body_id_to_idx = _synthetic_bio_annotations()
    defaults = dict(
        n_neurons=n,
        annotations=annotations,
        body_ids=body_ids,
        body_id_to_idx=body_id_to_idx,
        warmup_steps=8,
    )
    defaults.update(kwargs)
    return n, BiologicalFlightController(**defaults)


def _looming_connectome(controller, n, weight=8.0):
    adj = sparse.csr_matrix((n, n), dtype=np.float32).tolil()
    dnp = controller.output_mapping["DNp01"]
    dnp03 = controller.output_mapping.get("DNp03", np.array([], dtype=np.int32))
    left = controller.input_mapping["looming_left"]
    right = controller.input_mapping["looming_right"]
    for pre in left:
        for post in dnp:
            adj[int(pre), int(post)] = weight
        for post in dnp03:
            adj[int(pre), int(post)] = weight * 0.5
    for pre in right:
        for post in dnp:
            adj[int(pre), int(post)] = weight
        for post in dnp03:
            adj[int(pre), int(post)] = weight * 0.5
    return adj.tocsr(), np.concatenate([left, right]), dnp


def test_engineered_residual_escape_off_by_default():
    """Raw looming must not trigger escape when DNp01 stays at rest (flag default OFF)."""
    n, controller = _bio_controller()
    assert controller.use_engineered_looming_residual is False
    rest = np.full(n, 1.0)
    for _ in range(5):
        controller.record_resting_rates(rest)
    controller.finalize_warmup()
    controller.prev_action[:] = 0.0
    controller.alpha = 1.0
    controller._last_looming = 5.0
    controller.compute_action(rest)
    controller.compute_action(rest)
    assert controller.engineered_escape_count == 0
    assert controller.brain_escape_count == 0
    assert controller.escape_timer == 0


def test_engineered_residual_escape_opt_in_for_ablation():
    """The engineered looming-residual trigger exists only as an opt-in ablation."""
    n, controller = _bio_controller(use_engineered_looming_residual=True)
    rest = np.full(n, 1.0)
    for _ in range(5):
        controller.record_resting_rates(rest)
    controller.finalize_warmup()
    controller.prev_action[:] = 0.0
    controller.alpha = 1.0
    controller._last_looming = 5.0
    controller.compute_action(rest)
    controller.compute_action(rest)
    assert controller.engineered_escape_count >= 1
    assert controller.brain_escape_count == 0


def test_looming_on_lplc2_raises_dnp01_and_causes_escape():
    """Synthetic LPLC2/LC4 → DNp01 path: looming raises DNp01 and fires a brain escape."""
    from threedfly.connectome.simulator import RateSimulator, calibrate_looming_to_dnp01

    n, controller = _bio_controller(warmup_steps=6)
    adj, loom, dnp = _looming_connectome(controller, n)
    adj = calibrate_looming_to_dnp01(adj, loom, dnp)
    brain = RateSimulator(adj, dt=0.010, syn_weight_scale=0.1, max_rate=50.0)

    rest_hist = []
    for _ in range(10):
        controller.set_visual_input(
            left_luminance=0.0,
            right_luminance=0.0,
            optic_flow_features={},
            looming_features={"left": 0.0, "right": 0.0},
            brain=brain,
        )
        rates = brain.step()
        controller.record_resting_rates(rates)
        rest_hist.append(float(rates[dnp].mean()))
    controller.finalize_warmup()
    controller.apply_giant_fiber_calibration(brain)
    rest = float(np.mean(rest_hist[-4:]))

    controller.prev_action[:] = 0.0
    controller.alpha = 1.0
    driven = []
    for _ in range(25):
        controller.set_visual_input(
            left_luminance=0.0,
            right_luminance=0.0,
            optic_flow_features={},
            looming_features={"left": 4.0, "right": 4.0},
            brain=brain,
        )
        rates = brain.step()
        driven.append(float(rates[dnp].mean()))
        controller.compute_action(rates)

    peak = float(np.max(driven))
    sat_frac = float(np.mean(np.asarray(driven) >= 49.0))
    assert peak > rest + 0.15, f"DNp01 did not rise (rest={rest:.4f} peak={peak:.4f})"
    assert controller.brain_escape_count >= 1
    assert controller.engineered_escape_count == 0
    assert sat_frac < 0.5, f"DNp01 pinned at ceiling {sat_frac:.0%} of driven steps"


def test_left_looming_turns_right_and_vice_versa():
    """Escape yaw comes from LPLC2/LC4 (or DNp03) side, away from the threat."""
    from threedfly.connectome.simulator import RateSimulator, calibrate_looming_to_dnp01

    n, controller = _bio_controller(warmup_steps=6)
    adj, loom, dnp = _looming_connectome(controller, n)
    adj = calibrate_looming_to_dnp01(adj, loom, dnp)
    brain = RateSimulator(adj, dt=0.010, syn_weight_scale=0.5, max_rate=50.0)

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
    controller.finalize_warmup()
    controller.apply_giant_fiber_calibration(brain)
    controller.alpha = 1.0

    def _drive(left, right):
        brain.reset()
        controller.escape_timer = 0
        controller.dnp01_prev_z = 0.0
        controller.prev_action[:] = 0.0
        yaws = []
        for _ in range(20):
            controller.set_visual_input(
                left_luminance=0.0,
                right_luminance=0.0,
                optic_flow_features={},
                looming_features={"left": left, "right": right},
                brain=brain,
            )
            rates = brain.step()
            action = controller.compute_action(rates)
            if controller.brain_escape_count > 0 or controller.escape_timer > 0:
                yaws.append(float(action[4]))
        return yaws

    yaws_left = _drive(4.0, 0.0)
    escapes_left = controller.brain_escape_count
    controller.brain_escape_count = 0
    yaws_right = _drive(0.0, 4.0)

    assert escapes_left >= 1, "Left looming should trigger a brain escape"
    assert controller.brain_escape_count >= 1, "Right looming should trigger a brain escape"
    assert len(yaws_left) > 0 and len(yaws_right) > 0
    mean_left = float(np.mean(yaws_left))
    mean_right = float(np.mean(yaws_right))
    # Threat on the left → turn right (negative yaw); threat on the right → turn left.
    assert mean_left < -0.02, f"Left looming should yaw right (neg), got {mean_left:.3f}"
    assert mean_right > 0.02, f"Right looming should yaw left (pos), got {mean_right:.3f}"
    assert mean_left * mean_right < 0


def _loom_drive(controller, brain, left, right):
    controller.set_visual_input(
        left_luminance=0.0,
        right_luminance=0.0,
        optic_flow_features={},
        looming_features={"left": left, "right": right},
        brain=brain,
    )


def test_looming_encoding_is_absolute_not_mean_normalized():
    """Looming is an absolute rate code: 0 stays 0; 0.1 vs 0.8 must not collapse to the same drive."""
    from threedfly.connectome.simulator import RateSimulator

    n, controller = _bio_controller(warmup_steps=3)
    brain = RateSimulator(sparse.csr_matrix((n, n), dtype=np.float32), dt=0.010, max_rate=50.0)
    loom = np.concatenate([
        controller.input_mapping["looming_left"],
        controller.input_mapping["looming_right"],
    ])

    _loom_drive(controller, brain, 0.0, 0.0)
    assert float(np.max(np.abs(brain.external_input[loom]))) < 1e-6

    _loom_drive(controller, brain, 0.1, 0.1)
    weak = brain.external_input[loom].copy()
    _loom_drive(controller, brain, 0.8, 0.8)
    strong = brain.external_input[loom].copy()
    assert float(np.max(np.abs(weak))) <= 2.0
    assert float(np.max(np.abs(strong))) <= 2.0
    assert not np.allclose(weak, strong, atol=1e-3)
    assert float(np.max(np.abs(strong))) > float(np.max(np.abs(weak))) * 3.0


def test_looming_encoding_is_retinotopic_across_sectors():
    """Each LPLC2/LC4 cell has a seeded RF in its hemifield; local sectors drive different cells."""
    from threedfly.connectome.simulator import RateSimulator

    n, controller = _bio_controller(warmup_steps=3)
    brain = RateSimulator(sparse.csr_matrix((n, n), dtype=np.float32), dt=0.010, max_rate=50.0)
    left_idx = controller.input_mapping["looming_left"]
    right_idx = controller.input_mapping["looming_right"]

    left_sector = np.zeros(8, dtype=np.float32)
    left_sector[1] = 0.8  # forward-left
    right_sector = np.zeros(8, dtype=np.float32)
    right_sector[6] = 0.8  # right

    _loom_drive(controller, brain, left_sector, np.zeros(8))
    assert float(brain.external_input[left_idx].max()) > float(brain.external_input[right_idx].max()) + 0.05

    _loom_drive(controller, brain, np.zeros(8), right_sector)
    assert float(brain.external_input[right_idx].max()) > float(brain.external_input[left_idx].max()) + 0.05

    mixed = np.array([0.1, 0.8, 0.0, 0.4, 0.05, 0.2, 0.9, 0.3], dtype=np.float32)
    _loom_drive(controller, brain, mixed, mixed)
    assert float(np.std(brain.external_input[left_idx])) > 0.01
    assert float(np.std(brain.external_input[right_idx])) > 0.01


def test_looming_onset_boosts_rising_expansion():
    """Rising looming adds an onset kick; holding the same level does not."""
    from threedfly.connectome.simulator import RateSimulator

    n, controller = _bio_controller(warmup_steps=3)
    brain = RateSimulator(sparse.csr_matrix((n, n), dtype=np.float32), dt=0.010, max_rate=50.0)
    loom = np.concatenate([
        controller.input_mapping["looming_left"],
        controller.input_mapping["looming_right"],
    ])
    for _ in range(4):
        _loom_drive(controller, brain, 0.2, 0.2)
    tonic = float(np.mean(np.abs(brain.external_input[loom])))
    _loom_drive(controller, brain, 0.9, 0.9)
    onset = float(np.mean(np.abs(brain.external_input[loom])))
    _loom_drive(controller, brain, 0.9, 0.9)
    held = float(np.mean(np.abs(brain.external_input[loom])))
    assert onset > tonic + 0.05
    assert held < onset - 1e-4


def test_tonic_cruise_then_ramp_still_raises_dnp01():
    """Closed-loop-like: tonic cruise must not collapse the path; score the pulse after rise, not the late hold."""
    from threedfly.connectome.simulator import RateSimulator, calibrate_looming_to_dnp01

    n, controller = _bio_controller(warmup_steps=6)
    adj, loom, dnp = _looming_connectome(controller, n)
    adj = calibrate_looming_to_dnp01(adj, loom, dnp)
    brain = RateSimulator(adj, dt=0.010, syn_weight_scale=0.1, max_rate=50.0)

    tonic_hist = []
    tonic_in = []
    for _ in range(80):
        _loom_drive(controller, brain, 0.05, 0.05)
        tonic_in.append(float(np.max(np.abs(brain.external_input[loom]))))
        rates = brain.step()
        controller.record_resting_rates(rates)
        tonic_hist.append(float(rates[dnp].mean()))
    controller.finalize_warmup()
    tonic = float(np.mean(tonic_hist[-8:]))
    tonic_drive = float(np.mean(tonic_in[-8:]))

    driven = []
    ramp_in = []
    for _ in range(25):
        _loom_drive(controller, brain, 0.8, 0.8)
        ramp_in.append(float(np.max(np.abs(brain.external_input[loom]))))
        rates = brain.step()
        driven.append(float(rates[dnp].mean()))
    pulse = float(np.max(driven[:8]))
    onset_drive = float(np.max(ramp_in[:4]))
    late_drive = float(np.mean(ramp_in[-8:]))
    assert onset_drive > tonic_drive + 0.2, (
        f"Onset of a ramp must exceed decayed cruise drive "
        f"(tonic_in={tonic_drive:.3f} onset_in={onset_drive:.3f})"
    )
    assert late_drive < onset_drive - 1e-4, (
        f"Held expansion must decay, not stay as a tonic level "
        f"(onset={onset_drive:.3f} late={late_drive:.3f})"
    )
    assert pulse > tonic + 0.15, (
        f"DNp01 did not rise on the looming pulse (tonic={tonic:.4f} pulse={pulse:.4f})"
    )


def test_haltere_and_flow_use_fixed_reference_not_mean():
    """Halteres and T4/T5 scale by a fixed reference so zeros stay zero and strength is preserved."""
    from threedfly.connectome.simulator import RateSimulator

    n, controller = _bio_controller(warmup_steps=3)
    brain = RateSimulator(sparse.csr_matrix((n, n), dtype=np.float32), dt=0.010, max_rate=50.0)
    t4 = controller.input_mapping["t4a_left"]

    controller.set_visual_input(
        left_luminance=0.0,
        right_luminance=0.0,
        optic_flow_features={"t4a_left": np.array([0.0])},
        looming_features={"left": 0.0, "right": 0.0},
        brain=brain,
        angular_velocity=np.array([0.0, 0.0, 0.0]),
    )
    assert float(np.max(np.abs(brain.external_input[t4]))) < 1e-6

    controller.set_visual_input(
        left_luminance=0.0,
        right_luminance=0.0,
        optic_flow_features={"t4a_left": np.array([0.1])},
        looming_features={"left": 0.0, "right": 0.0},
        brain=brain,
    )
    weak = float(np.max(np.abs(brain.external_input[t4])))
    controller.set_visual_input(
        left_luminance=0.0,
        right_luminance=0.0,
        optic_flow_features={"t4a_left": np.array([0.8])},
        looming_features={"left": 0.0, "right": 0.0},
        brain=brain,
    )
    strong = float(np.max(np.abs(brain.external_input[t4])))
    assert strong > weak * 3.0, f"T4/T5 should scale with a fixed ref (0.1→{weak:.3f} vs 0.8→{strong:.3f})"

    if controller.input_mapping["haltere_left"].size == 0:
        return
    controller.set_haltere_input(np.array([0.0, 0.0, 0.4]), brain)
    h_weak = float(np.max(np.abs(brain.external_input[controller.input_mapping["haltere_left"]])))
    controller.set_haltere_input(np.array([0.0, 0.0, 2.0]), brain)
    h_strong = float(np.max(np.abs(brain.external_input[controller.input_mapping["haltere_left"]])))
    assert h_weak < h_strong
    assert h_strong <= 2.0


def test_yaw_drops_near_silent_dna02_side():
    """A near-silent DNa02 side (0.26 Hz vs 33.5 Hz) must not drive a one-sided yaw."""
    n, controller = _bio_controller(warmup_steps=12)
    rest = np.full(n, 1.0)
    rest[controller.output_mapping["DNa02_L"]] = 33.5
    rest[controller.output_mapping["DNa02_R"]] = 0.26
    rest[controller.output_mapping["DNg02_L"]] = 1.0
    rest[controller.output_mapping["DNg02_R"]] = 1.0
    for _ in range(8):
        controller.record_resting_rates(rest)
    controller.finalize_warmup()
    controller.prev_action[:] = 0.0
    controller.alpha = 1.0
    driven = rest.copy()
    driven[controller.output_mapping["DNa02_L"]] = 50.0
    yaw = float(controller.compute_action(driven)[4])
    assert abs(yaw) < 0.08, f"Near-silent DNa02_R leaked into yaw: {yaw:.3f}"


def test_dng02_readout_averages_per_side_zscores():
    """Pooled DNg02 must not saturate from mixing a hot side with a quiet side."""
    n, controller = _bio_controller(warmup_steps=12)
    rest = np.full(n, 1.0)
    rest[controller.output_mapping["DNg02_L"]] = 10.0
    rest[controller.output_mapping["DNg02_R"]] = 0.5
    for _ in range(8):
        controller.record_resting_rates(rest)
    controller.finalize_warmup()
    at_rest = float(controller._normalized_readout("DNg02", rest))
    assert abs(at_rest) < 0.2, f"DNg02 rest readout should be ~0, got {at_rest:.3f}"
    spiked = rest.copy()
    spiked[controller.output_mapping["DNg02_L"]] = 50.0
    z = float(controller._normalized_readout("DNg02", spiked))
    assert abs(z) < 0.85, f"One-sided DNg02 spike should not pin the pooled readout, got {z:.3f}"


def test_looming_path_calibration_defaults_are_minimal():
    """After absolute encoding, boost ≤ 2× and do not attenuate other incoming onto DNp01."""
    from threedfly.connectome.simulator import (
        DEFAULT_LOOMING_PATH_GAIN,
        DEFAULT_LOOMING_REST_INCOMING_SCALE,
        DEFAULT_LOOMING_INHIBITION_SCALE,
        calibrate_looming_to_dnp01,
    )

    assert DEFAULT_LOOMING_PATH_GAIN <= 2.0 + 1e-9
    assert DEFAULT_LOOMING_REST_INCOMING_SCALE == pytest.approx(1.0)
    assert 0.2 - 1e-9 <= DEFAULT_LOOMING_INHIBITION_SCALE <= 0.5 + 1e-9

    n, controller = _bio_controller(warmup_steps=2)
    loom = np.concatenate([
        controller.input_mapping["looming_left"],
        controller.input_mapping["looming_right"],
    ])
    dnp = controller.output_mapping["DNp01"]
    other = int(controller.output_mapping["DNb"][0])
    adj = sparse.csr_matrix((n, n), dtype=np.float32).tolil()
    for pre in loom:
        for post in dnp:
            adj[int(pre), int(post)] = 2.0
    for post in dnp:
        adj[other, int(post)] = 4.0
    out = calibrate_looming_to_dnp01(adj.tocsr(), loom, dnp)
    assert float(out[int(loom[0]), int(dnp[0])]) == pytest.approx(
        2.0 * DEFAULT_LOOMING_PATH_GAIN
    )
    assert float(out[other, int(dnp[0])]) == pytest.approx(4.0)


def test_diagnose_looming_reports_dnp01_dnp03_latency_and_selectivity():
    """diagnose-looming reports rest/driven/latency and L/R selectivity for DNp01 and DNp03."""
    from threedfly.connectome.simulator import diagnose_looming_to_dnp01

    n, controller = _bio_controller(warmup_steps=3)
    adj, loom, dnp = _looming_connectome(controller, n, weight=2.0)
    left = controller.input_mapping["looming_left"]
    right = controller.input_mapping["looming_right"]
    dnp03 = controller.output_mapping["DNp03"]
    report = diagnose_looming_to_dnp01(
        adjacency=adj,
        looming_indices=loom,
        dnp01_indices=dnp,
        dnp03_indices=dnp03,
        looming_left=left,
        looming_right=right,
        n_rest=6,
        n_drive=12,
        looming_current=1.0,
    )
    for key in (
        "dnp01_rest",
        "dnp01_driven",
        "dnp01_latency_steps",
        "dnp03_rest",
        "dnp03_driven",
        "dnp03_latency_steps",
        "dnp01_left_driven",
        "dnp01_right_driven",
        "dnp01_lr_selectivity",
        "dnp03_lr_selectivity",
    ):
        assert key in report, f"missing diagnostic key {key}"
    assert report["dnp01_driven"] > report["dnp01_rest"] + 0.05
    assert np.isfinite(report["dnp01_latency_steps"])


def test_yaw_running_baseline_when_pair_dropped():
    """Engineered slow L−R baseline: if a DN pair is dropped, remaining yaw is zero-mean at rest."""
    n, controller = _bio_controller(warmup_steps=12)
    rest = np.full(n, 1.0)
    rest[controller.output_mapping["DNa02_L"]] = 33.5
    rest[controller.output_mapping["DNa02_R"]] = 0.26
    rest[controller.output_mapping["DNg02_L"]] = 4.0
    rest[controller.output_mapping["DNg02_R"]] = 1.0
    for _ in range(8):
        controller.record_resting_rates(rest)
    controller.finalize_warmup()
    controller.prev_action[:] = 0.0
    controller.alpha = 1.0
    held = rest.copy()
    held[controller.output_mapping["DNg02_L"]] = 8.0
    yaws = []
    for _ in range(80):
        yaws.append(float(controller.compute_action(held)[4]))
    assert abs(yaws[0]) > 0.05, "Remaining DNg02 L−R should initially turn"
    assert abs(yaws[-1]) < abs(yaws[0]) * 0.5, (
        f"Slow yaw baseline should cancel DC bias ({yaws[0]:.3f} → {yaws[-1]:.3f})"
    )


def test_scale_inhibition_only_negative_incoming_onto_lplc2_lc4():
    """looming_inhibition_scale multiplies I→LPLC2/LC4 only; excitatory and other cells stay put."""
    from threedfly.connectome.simulator import (
        DEFAULT_LOOMING_PATH_GAIN,
        scale_inhibition_onto_targets,
        calibrate_looming_to_dnp01,
    )

    adj = sparse.csr_matrix(np.array([
        [0.0, -20.0, -3.0],
        [0.0, 4.0, 10.0],
        [0.0, -8.0, 0.0],
    ], dtype=np.float32))
    loom = np.array([1], dtype=np.int32)
    dnp = np.array([2], dtype=np.int32)
    scaled = scale_inhibition_onto_targets(adj, loom, scale=0.5)
    assert float(scaled[0, 1]) == pytest.approx(-10.0)
    assert float(scaled[1, 1]) == pytest.approx(4.0)
    assert float(scaled[2, 1]) == pytest.approx(-4.0)
    assert float(scaled[0, 2]) == pytest.approx(-3.0)
    assert float(scaled[1, 2]) == pytest.approx(10.0)

    calibrated = calibrate_looming_to_dnp01(
        adj, loom, dnp, looming_inhibition_scale=0.5
    )
    assert float(calibrated[0, 1]) == pytest.approx(-10.0)
    assert float(calibrated[1, 2]) == pytest.approx(10.0 * DEFAULT_LOOMING_PATH_GAIN)
    assert float(calibrated[0, 2]) == pytest.approx(-3.0)


def test_diagnose_lplc2_inhibition_lists_top_inhibitory_types(tmp_path):
    """Top inhibitory presynaptic types onto LPLC2/LC4 by summed weight×rate, written as JSON."""
    import json
    import pandas as pd
    from threedfly.connectome.simulator import diagnose_lplc2_inhibition

    adj = sparse.csr_matrix(np.array([
        [0.0, -20.0, 0.0, 0.0],
        [0.0, 0.0, 8.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, -5.0, 0.0, 0.0],
    ], dtype=np.float32))
    annotations = pd.DataFrame({
        "bodyId": [10, 11, 12, 13],
        "type": ["R1", "LPLC2", "DNp01", "LPi04"],
        "celltype": ["R1", "LPLC2", "DNp01", "LPi04"],
    })
    body_ids = np.array([10, 11, 12, 13])
    rates = np.array([0.8, 0.3, 0.0, 0.8], dtype=np.float32)
    report = diagnose_lplc2_inhibition(
        adj,
        looming_indices=np.array([1], dtype=np.int32),
        rates=rates,
        annotations=annotations,
        body_ids=body_ids,
        top_k=5,
    )
    types = [row["cell_type"] for row in report["top_inhibitory_types"]]
    assert types[0] == "R1"
    assert "LPi04" in types
    assert report["top_inhibitory_types"][0]["sum_weight_x_rate"] < 0.0
    out = tmp_path / "lplc2_inhibition.json"
    out.write_text(json.dumps(report, indent=2))
    loaded = json.loads(out.read_text())
    assert loaded["n_looming_cells"] == 1
    assert loaded["top_inhibitory_types"][0]["cell_type"] == "R1"


def test_flight_like_diagnose_crossed_with_inhibition_scale():
    """Under flight-like other-sensory drive, unscaled I→LPLC2 blocks DNp01; scaled I lets it cross."""
    from threedfly.connectome.simulator import diagnose_looming_to_dnp01

    adj = sparse.csr_matrix(np.array([
        [0.0, -20.0, 0.0],
        [0.0, 0.0, 20.0],
        [0.0, 0.0, 0.0],
    ], dtype=np.float32))
    loom = np.array([1], dtype=np.int32)
    dnp = np.array([2], dtype=np.int32)
    other = np.array([0], dtype=np.int32)
    kwargs = dict(
        adjacency=adj,
        looming_indices=loom,
        dnp01_indices=dnp,
        other_sensory_indices=other,
        other_sensory_current=0.8,
        n_rest=6,
        n_drive=16,
        n_baseline=20,
        baseline_current=0.3,
        looming_current=1.0,
        n_pulse=8,
        closed_loop_margin=0.05,
    )
    blocked = diagnose_looming_to_dnp01(**kwargs, looming_inhibition_scale=1.0)
    assert float(blocked["dnp01_closed_loop_crossed"]) == 0.0
    released = diagnose_looming_to_dnp01(**kwargs, looming_inhibition_scale=0.5)
    assert float(released["dnp01_closed_loop_crossed"]) >= 1.0


def test_diagnose_looming_closed_loop_tonic_then_ramp():
    """diagnose-looming must report whether DNp01 still crosses rest+margin after tonic warmup."""
    from threedfly.connectome.simulator import diagnose_looming_to_dnp01

    n, controller = _bio_controller(warmup_steps=3)
    adj, loom, dnp = _looming_connectome(controller, n, weight=2.0)
    report = diagnose_looming_to_dnp01(
        adjacency=adj,
        looming_indices=loom,
        dnp01_indices=dnp,
        dnp03_indices=controller.output_mapping["DNp03"],
        n_rest=6,
        n_drive=16,
        n_baseline=20,
        baseline_current=0.05,
        looming_current=1.0,
    )
    for key in (
        "dnp01_closed_loop_rest",
        "dnp01_closed_loop_driven",
        "dnp01_closed_loop_late_driven",
        "dnp01_closed_loop_crossed",
    ):
        assert key in report, f"missing diagnostic key {key}"
    assert report["dnp01_closed_loop_late_driven"] > report["dnp01_closed_loop_rest"] + 0.05
    assert float(report["dnp01_closed_loop_crossed"]) >= 1.0


def test_dnp03_selected_when_annotated():
    from threedfly.connectome.neuron_selection import select_output_neurons_for_motor
    import pandas as pd

    annotations = pd.DataFrame({
        "bodyId": [1, 2, 3, 4],
        "type": ["DNp03", "DNp03", "DNp01", "other"],
        "celltype": ["DNp03", "DNp03", "DNp01", "other"],
        "somaSide": ["L", "R", "L", "L"],
        "rootSide": ["L", "R", "L", "L"],
    })
    body_ids = np.array([1, 2, 3, 4])
    mapping = {int(b): i for i, b in enumerate(body_ids)}
    motor = select_output_neurons_for_motor(annotations, body_ids, mapping)
    assert len(motor["DNp03"]) == 2
    assert len(motor["DNp03_L"]) == 1
    assert len(motor["DNp03_R"]) == 1


def test_dnp03_and_dna02_sides_from_aliases_suffixes_and_unlabeled_split():
    """Left/right DN channels must both exist: aliases, type suffix, or unlabeled split."""
    from threedfly.connectome.neuron_selection import select_output_neurons_for_motor
    import pandas as pd

    aliased = pd.DataFrame({
        "bodyId": [1, 2, 3, 4],
        "type": ["DNp03", "DNp03", "DNa02", "DNa02"],
        "celltype": ["DNp03", "DNp03", "DNa02", "DNa02"],
        "somaSide": ["left", "RIGHT", "LHS", "rhs"],
        "rootSide": ["?", "?", "?", "?"],
    })
    body_ids = np.array([1, 2, 3, 4])
    mapping = {int(b): i for i, b in enumerate(body_ids)}
    motor = select_output_neurons_for_motor(aliased, body_ids, mapping)
    assert len(motor["DNp03_L"]) == 1 and len(motor["DNp03_R"]) == 1
    assert len(motor["DNa02_L"]) == 1 and len(motor["DNa02_R"]) == 1

    suffixed = pd.DataFrame({
        "bodyId": [1, 2, 3, 4],
        "type": ["DNp03_L", "DNp03_R", "DNa02_L", "DNa02_R"],
        "celltype": ["DNp03_L", "DNp03_R", "DNa02_L", "DNa02_R"],
        "somaSide": ["?", "?", "?", "?"],
        "rootSide": ["?", "?", "?", "?"],
    })
    motor = select_output_neurons_for_motor(suffixed, body_ids, mapping)
    assert set(motor["DNp03_L"]) == {0} and set(motor["DNp03_R"]) == {1}
    assert set(motor["DNa02_L"]) == {2} and set(motor["DNa02_R"]) == {3}

    unlabeled = pd.DataFrame({
        "bodyId": [1, 2, 3, 4],
        "type": ["DNp03", "DNp03", "DNa02", "DNa02"],
        "celltype": ["DNp03", "DNp03", "DNa02", "DNa02"],
        "somaSide": ["?", "?", "?", "?"],
        "rootSide": ["?", "?", "?", "?"],
    })
    motor = select_output_neurons_for_motor(unlabeled, body_ids, mapping)
    assert len(motor["DNp03_L"]) == 1 and len(motor["DNp03_R"]) == 1
    assert len(motor["DNa02_L"]) == 1 and len(motor["DNa02_R"]) == 1
    assert set(motor["DNp03_L"]).isdisjoint(set(motor["DNp03_R"]))
    assert set(motor["DNa02_L"]).isdisjoint(set(motor["DNa02_R"]))


def test_legs_attach_at_thorax_coxa():
    """Welded legs sit at thorax coxa positions, not ~1 cm below the thorax."""
    from threedfly.sim.environment import FlyEnvironment
    from threedfly.sim.mjcf import COXA_T2_LEFT, COXA_T2_RIGHT

    env = FlyEnvironment(render_mode=None)
    env.reset(seed=0)
    env.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    env.data.qpos[0:3] = [0.0, 0.0, 1.5]
    import mujoco
    mujoco.mj_forward(env.model, env.data)
    fly_id = env._fly_body_id
    t2l = int(env.model.body("left_leg_T2").id)
    t2r = int(env.model.body("right_leg_T2").id)
    rel_l = env.data.xpos[t2l] - env.data.xpos[fly_id]
    rel_r = env.data.xpos[t2r] - env.data.xpos[fly_id]
    env.close()
    assert np.linalg.norm(rel_l - np.asarray(COXA_T2_LEFT)) < 0.003
    assert np.linalg.norm(rel_r - np.asarray(COXA_T2_RIGHT)) < 0.003
    # Not floating a centimetre below the thorax (thorax ventral ~ +7 mm in fly frame).
    assert rel_l[2] > 0.004
    assert rel_r[2] > 0.004


def test_wing_hinge_at_thorax_wing_base():
    """Wing stroke joints sit at the membrane base on the thorax, mirrored L/R."""
    from threedfly.sim.environment import FlyEnvironment
    from threedfly.sim.mjcf import WING_LEFT_HINGE, WING_RIGHT_HINGE

    env = FlyEnvironment(render_mode=None)
    env.reset(seed=0)
    env.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    env.data.qpos[0:3] = [0.0, 0.0, 1.5]
    import mujoco
    mujoco.mj_forward(env.model, env.data)
    fly_id = env._fly_body_id
    left = env.data.xpos[int(env.model.body("left_wing").id)] - env.data.xpos[fly_id]
    right = env.data.xpos[int(env.model.body("right_wing").id)] - env.data.xpos[fly_id]
    left_axis = np.array(env.model.jnt_axis[env._wing_left_jnt])
    right_axis = np.array(env.model.jnt_axis[env._wing_right_jnt])
    env.close()
    assert np.linalg.norm(left - np.asarray(WING_LEFT_HINGE)) < 0.002
    assert np.linalg.norm(right - np.asarray(WING_RIGHT_HINGE)) < 0.002
    # Left and right hinges are mirrored across the midline; joint axes too.
    assert left[1] * right[1] < 0
    assert np.dot(left_axis, right_axis) < 0


def test_mjcf_znear_and_closeup_camera():
    """Near plane must not clip a 2.5 cm fly; a close-up camera follows the body."""
    from threedfly.sim.environment import FlyEnvironment

    env = FlyEnvironment(render_mode=None)
    env.reset(seed=0)
    near_m = float(env.model.vis.map.znear) * float(env.model.stat.extent)
    names = [env.model.camera(i).name for i in range(env.model.ncam)]
    env.close()
    assert near_m < 0.005, f"near plane {near_m:.4f} m would clip a 2.5 cm fly"
    assert "fly_closeup" in names
    assert "left_eye" in names and "right_eye" in names


def test_visual_wings_flap_in_phase_with_mirrored_hinges():
    """200 Hz visual flap: equal strokes share phase; L/R hinges are mirrored in MJCF."""
    from threedfly.sim.wings import visual_wing_angles

    left, right = visual_wing_angles(0.3, 1.0, 1.0)
    assert left * right > 0
    left0, right0 = visual_wing_angles(0.0, 1.0, 1.0)
    assert abs(left0) < 1e-9 and abs(right0) < 1e-9


def test_sector_ray_distances_are_diagnostic_only():
    """MuJoCo sector rays exist for sanity checks and are not the brain's looming input."""
    from threedfly.sim.environment import FlyEnvironment
    from threedfly.vision import StereoVision

    env = FlyEnvironment(render_mode=None)
    env.reset(seed=0)
    env.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    env.data.qpos[0:3] = [0.0, 0.0, 1.5]
    import mujoco
    mujoco.mj_forward(env.model, env.data)
    left_d = env.sector_ray_distances("left")
    right_d = env.sector_ray_distances("right")
    env.close()
    assert left_d.shape == (8,) and right_d.shape == (8,)
    assert np.all(left_d > 0) and np.all(right_d > 0)
    stereo = StereoVision()
    assert not hasattr(stereo, "gt_ray_distances") or stereo.get_visual_features is not None
