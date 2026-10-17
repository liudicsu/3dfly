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
