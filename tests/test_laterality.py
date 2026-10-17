"""Mirror-scene laterality: swapped L/R input must swap DNp03 / DNa02 / HSE."""

from __future__ import annotations

import numpy as np
import pytest

from threedfly.brain.closed_loop import ClosedLoopBrain, core_stack_available
from threedfly.brain.constants import N_HEX, flyvis_available
from threedfly.brain.core_bundle import core_bundle_available
from threedfly.brain.stimuli import loom
from threedfly.control.flight_params import FlightParams


needs_stack = pytest.mark.skipif(
    not (flyvis_available() and core_bundle_available()),
    reason="mirror-scene laterality needs flyvis + data/core_bundle.npz",
)


def _mean_dns(brain: ClosedLoopBrain, hex_L, hex_R, n: int = 8):
    last = None
    blank = np.zeros((8, 8, 3), dtype=np.uint8)
    for _ in range(n):
        last = brain.step(blank, blank, hex_L=hex_L, hex_R=hex_R)
    ro = last.readout
    return {
        "dnp03_L": float(ro.dnp03_L),
        "dnp03_R": float(ro.dnp03_R),
        "dna02_L": float(ro.dna02_L),
        "dna02_R": float(ro.dna02_R),
        "hse_L": float(ro.hse_L),
        "hse_R": float(ro.hse_R),
        "dnp01_L": float(ro.dnp01_L),
        "dnp01_R": float(ro.dnp01_R),
        "lc4_L": float(ro.lc4_L),
        "lc4_R": float(ro.lc4_R),
        "lplc2_L": float(ro.lplc2_L),
        "lplc2_R": float(ro.lplc2_R),
    }


def test_side_walls_brighten_ipsilateral_eye():
    """Canted 120° eyes: the matching wall must raise that eye's hex mean."""
    pytest.importorskip("mujoco")
    from threedfly.brain.eyes import stereo_to_hex
    from threedfly.sim import FlyEnvironment
    import importlib.util
    from pathlib import Path

    _acc = Path(__file__).resolve().parents[1] / "scripts" / "closed_loop_acceptance.py"
    _spec = importlib.util.spec_from_file_location("cla_eye_lat", _acc)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)

    env = FlyEnvironment(render_mode="rgb_array", camera_width=80, camera_height=60)
    contrast = {}
    for side in ("L", "R"):
        pose = _mod.lateral_loom_pose(side)
        env.reset(seed=0)
        env.set_pose(pose["pose"], yaw=pose["yaw"])
        hL, hR = stereo_to_hex(*env.get_eye_images())
        contrast[side] = (float(np.std(hL)), float(np.std(hR)))
    env.close()
    # Textured wall is on the ipsilateral eye → higher hex contrast, not mean.
    assert contrast["L"][0] > contrast["L"][1] + 0.02, f"left wall should texture left eye: {contrast}"
    assert contrast["R"][1] > contrast["R"][0] + 0.02, f"right wall should texture right eye: {contrast}"


@needs_stack
def test_hex_swap_mirrors_dn_outputs():
    """Swapping the two hex frames must swap per-side DNp03 / DNa02 / HSE."""
    params = FlightParams(prewarm_steps=16, flyvis_stride=1, flyvis_torch_threads=2)
    brain = ClosedLoopBrain.build(params=params, quick=True)
    gray = np.full(N_HEX, 0.5, dtype=np.float32)
    movie = loom(0.4, x0=-25.0)
    looming = movie[-1]
    blank = np.zeros((8, 8, 3), dtype=np.uint8)
    brain.reset()
    for _ in range(12):
        brain.step(blank, blank, hex_L=gray, hex_R=gray)
    a = _mean_dns(brain, looming, gray, n=10)
    brain.reset()
    for _ in range(12):
        brain.step(blank, blank, hex_L=gray, hex_R=gray)
    b = _mean_dns(brain, gray, looming, n=10)

    # Mirror: the eye that got the loom should drive the matching DN side more.
    d03_a = a["dnp03_L"] - a["dnp03_R"]
    d03_b = b["dnp03_L"] - b["dnp03_R"]
    assert d03_a * d03_b < 0, f"DNp03 L−R did not flip: A={a} B={b}"
    d02_a = a["dna02_L"] - a["dna02_R"]
    d02_b = b["dna02_L"] - b["dna02_R"]
    assert d02_a * d02_b < 0, f"DNa02 L−R did not flip: A={a} B={b}"
    # Left-eye loom should prefer DNp03_L (ipsi) and LC4_L.
    assert a["dnp03_L"] > a["dnp03_R"], f"left-eye loom should raise DNp03_L: {a}"
    assert b["dnp03_R"] > b["dnp03_L"], f"right-eye loom should raise DNp03_R: {b}"
    assert a["lc4_L"] > a["lc4_R"], f"left-eye loom should raise LC4_L: {a}"
    assert b["lc4_R"] > b["lc4_L"], f"right-eye loom should raise LC4_R: {b}"


@needs_stack
@pytest.mark.skipif(
    not core_stack_available(),
    reason="MuJoCo mirror-scene needs the closed-loop stack",
)
def test_lateral_approaches_turn_away_from_threat():
    """DNp01 escapes on true L/R side-loom must change heading away from the wall."""
    pytest.importorskip("mujoco")
    from threedfly.control.closed_loop_metrics import direction_correct_frac
    from threedfly.control.dn_controller import DNFlightController
    from threedfly.control.exploration import heading_from_quaternion
    from threedfly.sim import FlyEnvironment
    import importlib.util
    from pathlib import Path

    _acc = Path(__file__).resolve().parents[1] / "scripts" / "closed_loop_acceptance.py"
    _spec = importlib.util.spec_from_file_location("cla_laterality", _acc)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    params = FlightParams(flyvis_stride=3, flyvis_torch_threads=2)
    brain = ClosedLoopBrain.build(params=params, quick=True)
    env = FlyEnvironment(render_mode="rgb_array", camera_width=80, camera_height=60)
    ctl = DNFlightController(params)
    lateral_loom_object = _mod.lateral_loom_object

    def _approach(side: str):
        spec = lateral_loom_object(side)
        env.reset(seed=0)
        env.set_pose(spec["pose"], yaw=spec["yaw"], vel=spec["vel"])
        env.hide_loom_object()
        brain.reset()
        ctl.reset()
        ctl.bind_hover(env.hover_action)
        left, right = env.get_eye_images()
        warmed = brain.prewarm(left, right, env.get_fly_angular_velocity(), n=params.prewarm_steps)
        if warmed is not None:
            ctl.capture_baseline(warmed.readout)
        env.set_loom_object(spec["loom_start"])
        heads = []
        loom_pos = np.asarray(spec["loom_start"], dtype=float)
        loom_v = np.asarray(spec["loom_vel"], dtype=float)
        brain.params.flyvis_stride = 1
        for _ in range(160):
            loom_pos = loom_pos + loom_v * float(env.control_dt)
            if loom_pos[0] < -0.4:
                env.hide_loom_object()
            else:
                env.set_loom_object(loom_pos)
            res = brain.step(left, right, env.get_fly_angular_velocity())
            act = ctl.compute(res.readout)
            left, right, info = env.step(act, wing_strokes=ctl.last_wing_strokes)
            heads.append(heading_from_quaternion(info["orientation"]))
            if env.check_collision():
                env.recover_from_collision()
        brain.params.flyvis_stride = int(params.flyvis_stride)
        return heads, list(ctl.escape_events)

    for side in ("L", "R"):
        heads, evs = _approach(side)
        assert evs, f"{side} approach produced no DNp01 escape"
        frac = direction_correct_frac(heads, evs, side)
        assert frac >= 0.8, f"{side} direction_correct={frac} events={[(e.step, e.yaw) for e in evs]}"
    env.close()
