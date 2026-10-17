"""Tests for the flybody-based fly: model, wing lift, yaw, haltere, hex eyes."""

import numpy as np
import pandas as pd
import pytest

from threedfly.sim import FlyEnvironment
from threedfly.sim.wings import stroke_averaged_forces
from threedfly.vision.hex_eye import (
    N_OMMATIDIA_PER_EYE,
    HEX_SPACING_DEG,
    hex_lattice_uv,
    sample_image_hex,
)
from threedfly.connectome.neuron_selection import (
    select_haltere_afferent_neurons,
    find_optic_lobe_hex_columns,
)


def test_flybody_model_loads():
    """Flybody-style MJCF must load with head, thorax, abdomen, wings, and legs."""
    env = FlyEnvironment(render_mode=None)
    env.reset(seed=0)

    def _body(name: str) -> int:
        return int(env.model.body(name).id)

    thorax = _body("thorax") if "thorax" in _body_names(env) else _body("fly")
    assert thorax >= 0
    for name in ("head", "abdomen", "left_wing", "right_wing"):
        bid = _body(name)
        assert bid > 0, f"Missing body {name}"

    leg_names = [env.model.body(i).name for i in range(env.model.nbody)]
    n_legs = sum(1 for n in leg_names if "leg" in n)
    assert n_legs >= 6, f"Expected retracted flight legs, found {n_legs}: {leg_names}"

    # Jointed wings (stroke hinges) plus freejoint
    joint_names = [env.model.joint(i).name for i in range(env.model.njnt)]
    assert any("wing" in n and "left" in n for n in joint_names)
    assert any("wing" in n and "right" in n for n in joint_names)

    mass = float(env.model.body_subtreemass[env._fly_body_id])
    assert mass > 0
    env.close()


def _body_names(env) -> set:
    return {env.model.body(i).name for i in range(env.model.nbody)}


def test_lift_balances_weight_at_rest():
    """Equal hover-level wing strokes must keep altitude near the start height."""
    env = FlyEnvironment(render_mode=None)
    env.reset(seed=0)
    env.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    env.data.qpos[0:3] = [0.0, 0.0, 1.5]
    env.data.qvel[:] = 0.0

    hover = env.hover_action
    zs = []
    for _ in range(250):
        env.step(
            np.array([0.0, hover, 0.0, 0.0, 0.0]),
            wing_strokes=(hover, hover),
        )
        zs.append(float(env.get_fly_position()[2]))
    mean_z = float(np.mean(zs[-50:]))
    env.close()
    assert 1.1 <= mean_z <= 2.0, f"Hover altitude drifted to {mean_z:.3f} m"


def test_left_greater_right_stroke_yields_yaw():
    """Higher left than right stroke amplitude must produce yaw (L−R → torque)."""
    env = FlyEnvironment(render_mode=None)
    env.reset(seed=0)
    env.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    env.data.qpos[0:3] = [0.0, 0.0, 1.5]
    env.data.qvel[:] = 0.0

    hover = env.hover_action
    yaw_rates = []
    headings = []
    for _ in range(80):
        env.step(
            np.array([0.0, hover, 0.0, 0.0, 0.0]),
            wing_strokes=(min(1.0, hover + 0.35), max(0.0, hover - 0.35)),
        )
        yaw_rates.append(float(env.get_fly_angular_velocity()[2]))
        headings.append(_yaw_from_quat(env.get_fly_orientation()))
    env.close()

    mean_yaw_rate = float(np.mean(yaw_rates[10:]))
    heading_change = abs(headings[-1] - headings[10])
    heading_change = min(heading_change, 2 * np.pi - heading_change)
    assert mean_yaw_rate > 0.05 or heading_change > 0.05, (
        f"L>R stroke should yaw; mean_wz={mean_yaw_rate:.4f} dpsi={heading_change:.4f}"
    )
    assert mean_yaw_rate > 0.0, f"L>R should produce positive yaw, got {mean_yaw_rate:.4f}"


def test_stroke_averaged_forces_sum_and_difference():
    """Quasi-steady model: sum(L,R) is lift, difference is yaw/roll torque."""
    weight = 0.015
    lift_max = 3.0 * weight
    hover = weight / lift_max
    f_h, t_h = stroke_averaged_forces(hover, hover, lift_max=lift_max, wing_span=0.02)
    assert abs(f_h[2] - weight) < 1e-9
    assert np.allclose(t_h, 0.0, atol=1e-12)

    f_max, _ = stroke_averaged_forces(1.0, 1.0, lift_max=lift_max, wing_span=0.02)
    assert abs(f_max[2] - lift_max) < 1e-9
    assert f_max[2] <= 3.01 * weight

    _, t_lr = stroke_averaged_forces(0.8, 0.2, lift_max=lift_max, wing_span=0.02)
    _, t_rl = stroke_averaged_forces(0.2, 0.8, lift_max=lift_max, wing_span=0.02)
    assert t_lr[2] > 0, "L>R must yield positive yaw torque"
    assert t_rl[2] < 0, "R>L must yield negative yaw torque"
    assert t_lr[2] * t_rl[2] < 0


def _yaw_from_quat(quat: np.ndarray) -> float:
    w, x, y, z = [float(v) for v in quat]
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _haltere_annotations(n=40):
    types = (
        ["haltere_campaniform"] * 8
        + ["R1"] * 4
        + ["DNg02_a"] * 4
        + ["DNa02"] * 4
        + ["other"] * (n - 20)
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


def test_haltere_input_mapping():
    """Body angular velocity is mapped onto annotated haltere afferents."""
    from scipy import sparse
    from threedfly.connectome.simulator import RateSimulator
    from threedfly.control.controller import BiologicalFlightController

    n, annotations, body_ids, body_id_to_idx = _haltere_annotations()
    mapping = select_haltere_afferent_neurons(annotations, body_ids, body_id_to_idx)
    assert mapping["left"].size > 0, "Synthetic 'haltere' types should be found"
    assert mapping["right"].size > 0
    assert mapping["all"].size == mapping["left"].size + mapping["right"].size

    controller = BiologicalFlightController(
        n_neurons=n,
        annotations=annotations,
        body_ids=body_ids,
        body_id_to_idx=body_id_to_idx,
        warmup_steps=2,
    )
    assert controller.input_mapping["haltere_left"].size > 0
    assert controller.input_mapping["haltere_right"].size > 0

    brain = RateSimulator(sparse.csr_matrix((n, n), dtype=np.float32), dt=0.010, max_rate=50.0)
    controller.set_haltere_input(np.array([0.0, 0.0, 2.5]), brain)
    driven = brain.external_input[controller.input_mapping["haltere_left"]]
    assert driven.size > 0
    assert float(np.max(np.abs(driven))) > 0.0
    assert float(np.max(np.abs(driven))) <= 2.0


def test_haltere_fallback_when_unannotated():
    """No 'haltere' types → empty mapping (engineered damping is the documented fallback)."""
    n = 12
    body_ids = np.arange(1, n + 1)
    annotations = pd.DataFrame({
        "bodyId": body_ids,
        "type": ["R1"] * 6 + ["DNp01"] * 6,
        "celltype": ["R1"] * 6 + ["DNp01"] * 6,
        "somaSide": ["L", "R"] * 6,
        "rootSide": ["L", "R"] * 6,
    })
    body_id_to_idx = {int(bid): i for i, bid in enumerate(body_ids)}
    mapping = select_haltere_afferent_neurons(annotations, body_ids, body_id_to_idx)
    assert mapping["all"].size == 0
    assert mapping["fallback"] == "engineered_damping"


def test_hex_eye_sampling():
    """R1–R6 sampling is a hex lattice (~750 ommatidia/eye, ~5 deg), not a 10×10 grid."""
    uv = hex_lattice_uv(n_ommatidia=N_OMMATIDIA_PER_EYE, spacing_deg=HEX_SPACING_DEG)
    assert uv.shape == (N_OMMATIDIA_PER_EYE, 2)
    assert N_OMMATIDIA_PER_EYE >= 700
    assert uv.min() >= 0.0 and uv.max() <= 1.0
    assert HEX_SPACING_DEG == pytest.approx(5.0)

    # Hex packing: nearest-neighbour distances should be relatively uniform
    d = np.sqrt(((uv[None, :, :] - uv[:, None, :]) ** 2).sum(axis=-1))
    np.fill_diagonal(d, np.inf)
    nn = d.min(axis=1)
    nn = nn[np.isfinite(nn)]
    cv = float(nn.std() / (nn.mean() + 1e-12))
    assert cv < 0.35, f"Hex lattice neighbour spacing too irregular (cv={cv:.3f})"

    img = np.zeros((120, 160), dtype=np.uint8)
    img[:, :80] = 40
    img[:, 80:] = 200
    samples = sample_image_hex(img, uv)
    assert samples.shape == (N_OMMATIDIA_PER_EYE,)
    assert samples.max() > samples.min()

    from threedfly.vision import StereoVision

    stereo = StereoVision()
    assert stereo.n_ommatidia == N_OMMATIDIA_PER_EYE
    color = np.random.randint(0, 256, (120, 160, 3), dtype=np.uint8)
    omma = stereo.ommatidial_sample(color)
    assert omma.shape == (N_OMMATIDIA_PER_EYE,)
    features = stereo.get_visual_features(color, color)
    assert features["left_ommatidia"].shape == (N_OMMATIDIA_PER_EYE,)
    assert features["ommatidia_uv"].shape == (N_OMMATIDIA_PER_EYE, 2)


def test_optic_lobe_hex_columns_absent_on_synthetic_tables():
    """MaleCNS-style synthetic tables have no hex columns; document the lattice fallback."""
    annotations = pd.DataFrame({
        "bodyId": [1, 2],
        "type": ["R1", "R1"],
        "celltype": ["R1", "R1"],
    })
    cols = find_optic_lobe_hex_columns(annotations)
    assert cols is None


def test_dng02_sets_left_right_wing_strokes():
    """DNg02 L/R readouts set left/right wing stroke amplitudes around hover."""
    from threedfly.control.controller import BiologicalFlightController
    from test_control import _synthetic_bio_annotations

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
    controller.hover_action = 0.32

    left_high = rest.copy()
    left_high[controller.output_mapping["DNg02_L"]] = 6.0
    controller.compute_action(left_high)
    sL, sR = controller.last_wing_strokes
    assert sL > sR, f"High DNg02_L should raise left stroke ({sL:.3f} vs {sR:.3f})"
    assert 0.0 <= sR <= 1.0 and 0.0 <= sL <= 1.0
