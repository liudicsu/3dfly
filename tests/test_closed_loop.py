"""ClosedLoopBrain persistence and handoff with synthetic / fake eyes."""

from __future__ import annotations

import numpy as np
import pytest

from threedfly.brain.closed_loop import (
    ClosedLoopBrain,
    DualFlyvisEyes,
    FakeDualEyes,
    core_stack_available,
)
from threedfly.brain.constants import connectome_available, flyvis_available
from threedfly.brain.handoff import HandoffPlan
from threedfly.brain.rate_core import RateCore
from threedfly.brain.readouts import build_readout_index
from tests.brain_fakes import tiny_signed_connectome


def _tiny_brain(fn=None) -> ClosedLoopBrain:
    conn = tiny_signed_connectome()
    core = RateCore.from_connectome(conn, g_inh=1.0, g_gap=0.0, dng02_bias=0.0, rho=None)
    index = build_readout_index(conn.typ, conn.side, conn.body_ids)
    # Clamp T4a L (index 0) from a single-column fake T4a voltage.
    plan_L = HandoffPlan(
        idx=np.array([0], dtype=np.int64),
        types=np.array(["T4a"], dtype=object),
        cols=np.array([0], dtype=np.int64),
        v_gray=np.array([0.0], dtype=np.float32),
        p99=np.array([1.0], dtype=np.float32),
        rmax=40.0,
    )
    plan_R = HandoffPlan(
        idx=np.zeros(0, dtype=np.int64),
        types=np.zeros(0, dtype=object),
        cols=np.zeros(0, dtype=np.int64),
        v_gray=np.zeros(0, dtype=np.float32),
        p99=np.zeros(0, dtype=np.float32),
    )

    def _default(hL, hR):
        return {"T4a": np.array([0.8], dtype=np.float32)}, {}

    eyes = FakeDualEyes(fn or _default)
    from threedfly.control.flight_params import FlightParams

    return ClosedLoopBrain(
        core=core,
        index=index,
        plans={"L": plan_L, "R": plan_R},
        eyes=eyes,
        halt_idx=np.array([], dtype=np.int64),
        params=FlightParams(flyvis_stride=1),
    )


def test_closed_loop_persists_and_clamps_handoff():
    brain = _tiny_brain()
    blank = np.zeros((16, 16, 3), dtype=np.uint8)
    s1 = brain.step(blank, blank)
    assert brain.n_steps == 1
    assert s1.rates[0] == pytest.approx(32.0, abs=0.5)  # 40 * 0.8
    r_after_first = s1.rates.copy()
    s2 = brain.step(blank, blank)
    assert brain.n_steps == 2
    # State persisted: second step starts from previous rates, clamp still applied.
    assert s2.rates[0] == pytest.approx(32.0, abs=0.5)
    assert brain.eyes.n_steps == 2
    assert r_after_first.shape == s2.rates.shape


def test_closed_loop_reset_clears_core_not_called_per_step():
    brain = _tiny_brain()
    blank = np.zeros((8, 8, 3), dtype=np.uint8)
    brain.step(blank, blank)
    brain.reset()
    assert brain.n_steps == 0
    assert float(brain.core.r.max()) == pytest.approx(0.0)
    assert brain.eyes.n_steps == 0


def test_mirrored_hex_swap_mirrors_fake_eye_outputs():
    """Eye-swap of a laterality cue must swap the returned type traces."""

    def _fn(hL, hR):
        return (
            {"cue": np.array([float(np.mean(hL))], dtype=np.float32)},
            {"cue": np.array([float(np.mean(hR))], dtype=np.float32)},
        )

    eyes = FakeDualEyes(_fn)
    leftish = np.ones(8, dtype=np.float32)
    rightish = np.full(8, 0.1, dtype=np.float32)
    vL, vR = eyes.step_pair(leftish, rightish)
    vL2, vR2 = eyes.step_pair(rightish, leftish)
    assert vL["cue"][0] > vR["cue"][0]
    assert vL2["cue"][0] < vR2["cue"][0]
    assert vL["cue"][0] == pytest.approx(vR2["cue"][0])


def test_dual_eyes_share_one_network_with_independent_state():
    class _Eye:
        def __init__(self):
            self._state = {"n": 0}
            self.resets = 0

        def reset(self):
            self.resets += 1
            self._state = {"n": 0}

        def step(self, hex_frame):
            self._state = {"n": int(self._state["n"]) + 1 + int(np.mean(hex_frame))}
            return {"T4a": np.array([float(self._state["n"])], dtype=np.float32)}

    eye = _Eye()
    pair = DualFlyvisEyes.share(eye)
    pair.reset()
    hL = np.zeros(4, dtype=np.float32)
    hR = np.ones(4, dtype=np.float32)
    vL1, vR1 = pair.step_pair(hL, hR)
    vL2, vR2 = pair.step_pair(hL, hR)
    assert vL1["T4a"][0] != vR1["T4a"][0]
    assert vL2["T4a"][0] > vL1["T4a"][0]
    assert vR2["T4a"][0] > vR1["T4a"][0]


def test_closed_loop_flyvis_stride_holds_voltages():
    brain = _tiny_brain()
    brain.params.flyvis_stride = 2
    blank = np.zeros((8, 8, 3), dtype=np.uint8)
    brain.step(blank, blank)
    brain.step(blank, blank)
    brain.step(blank, blank)
    assert brain.eyes.n_steps == 2  # steps 0 and 2; step 1 held


def test_closed_loop_build_defaults_torch_threads_optional():
    import inspect

    sig = inspect.signature(ClosedLoopBrain.build)
    assert sig.parameters["torch_threads"].default is None


def test_core_stack_available_is_false_without_data_or_flyvis():
    from threedfly.brain.core_bundle import core_bundle_available

    flag = core_stack_available()
    assert flag is (
        flyvis_available() and (connectome_available() or core_bundle_available())
    )


@pytest.mark.skipif(
    not connectome_available() or not flyvis_available(),
    reason="real closed-loop step needs connectome + flyvis",
)
def test_real_closed_loop_one_step():
    brain = ClosedLoopBrain.build(quick=True)
    blank = np.zeros((32, 32, 3), dtype=np.uint8)
    s = brain.step(blank, blank)
    assert s.rates.size == brain.n
    assert s.readout.dng02_amp >= 0.0
    assert s.ms["total"] > 0.0
