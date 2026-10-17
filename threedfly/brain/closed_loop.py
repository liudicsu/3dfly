"""Persistent closed-loop brain: MuJoCo eyes → flyvis → handoff → RateCore.

State (flyvis network + RateCore rates) is kept across steps. Call
``reset()`` only when the simulation itself is reset.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from threedfly.brain.connectome_matrix import build_signed_matrix
from threedfly.brain.constants import (
    DNG02_BIAS_DEFAULT,
    G_GAP_DEFAULT,
    G_INH_DEFAULT,
    HALTERE_RATE,
    RHO_FULL_SIGNED_REFERENCE,
    cache_dir,
    configure_threads,
    connectome_available,
    flyvis_available,
)
from threedfly.brain.core_subgraph import select_core_tplus, slice_connectome
from threedfly.brain.eyes import stereo_to_hex
from threedfly.brain.handoff import (
    FlyvisCalibration,
    FlyvisEye,
    HandoffPlan,
    apply_handoff_plan,
    build_handoff_map,
    compile_handoff_plan,
    haltere_indices,
)
from threedfly.brain.rate_core import RateCore
from threedfly.brain.readouts import DNReadout, ReadoutIndex, build_readout_index, read_dns
from threedfly.control.flight_params import FlightParams

logger = logging.getLogger(__name__)


def core_stack_available(data_dir=None) -> bool:
    return bool(connectome_available(data_dir) and flyvis_available())


class DualFlyvisEyes:
    """Two persistent flyvis eyes (independent state; do not reset per sim step).

    Prefer ``DualFlyvisEyes.share(eye)``: one pretrained network, two states.
    Loading two ``FlyvisEye`` copies doubles init cost and cache pressure on
    a 2-core box.
    """

    def __init__(self, eye_L, eye_R=None):
        self.eye_L = eye_L
        self.eye_R = eye_R if eye_R is not None else eye_L
        self._shared = eye_R is None or eye_R is eye_L
        self._state_L = None
        self._state_R = None

    @classmethod
    def share(cls, eye) -> "DualFlyvisEyes":
        return cls(eye, None)

    def reset(self) -> None:
        self.eye_L.reset()
        if self._shared:
            self._state_L = self.eye_L._state
            self.eye_L.reset()
            self._state_R = self.eye_L._state
        else:
            self.eye_R.reset()
            self._state_L = None
            self._state_R = None

    def step_pair(
        self, hex_L: np.ndarray, hex_R: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        if not self._shared:
            return self.eye_L.step(hex_L), self.eye_R.step(hex_R)
        eye = self.eye_L
        eye._state = self._state_L
        vL = eye.step(hex_L)
        self._state_L = eye._state
        eye._state = self._state_R
        vR = eye.step(hex_R)
        self._state_R = eye._state
        return vL, vR


class FakeDualEyes:
    """Deterministic stand-in for tests (no flyvis)."""

    def __init__(
        self,
        fn: Optional[Callable[[np.ndarray, np.ndarray], Tuple[dict, dict]]] = None,
    ):
        self.fn = fn
        self.n_steps = 0

    def reset(self) -> None:
        self.n_steps = 0

    def step_pair(self, hex_L, hex_R):
        self.n_steps += 1
        if self.fn is not None:
            return self.fn(hex_L, hex_R)
        empty = {}
        return empty, empty


@dataclass
class BrainStep:
    readout: DNReadout
    rates: np.ndarray
    hex_L: np.ndarray
    hex_R: np.ndarray
    ms: Dict[str, float] = field(default_factory=dict)

    def population(self) -> Dict[str, float]:
        r = self.rates
        ro = self.readout
        return {
            "mean_rate": float(r.mean()) if r.size else 0.0,
            "max_rate": float(r.max()) if r.size else 0.0,
            "active_fraction": float((r > 0.01).mean()) if r.size else 0.0,
            "dnp01": float(max(ro.dnp01_L, ro.dnp01_R)),
            "dnp03_L": float(ro.dnp03_L),
            "dnp03_R": float(ro.dnp03_R),
        }


class ClosedLoopBrain:
    """Streaming flyvis + coreTplus RateCore. Do not reset between sim steps."""

    def __init__(
        self,
        core: RateCore,
        index: ReadoutIndex,
        plans: Dict[str, HandoffPlan],
        eyes,
        halt_idx: Optional[np.ndarray] = None,
        params: Optional[FlightParams] = None,
    ):
        self.core = core
        self.index = index
        self.plans = plans
        self.eyes = eyes
        self.halt_idx = np.asarray(halt_idx if halt_idx is not None else [], dtype=np.int64)
        self.params = params or FlightParams()
        self.last_ms: Dict[str, float] = {}
        self.n_steps = 0
        n_l = int(self.plans.get("L", _empty_plan()).idx.size)
        n_r = int(self.plans.get("R", _empty_plan()).idx.size)
        n_h = int(self.halt_idx.size)
        self._n_clamp = n_l + n_r + n_h
        self._split = (n_l, n_l + n_r)
        self._clamp_idx = np.empty(self._n_clamp, dtype=np.int64)
        self._clamp_val = np.empty(self._n_clamp, dtype=np.float32)
        if n_l:
            self._clamp_idx[:n_l] = self.plans["L"].idx
        if n_r:
            self._clamp_idx[n_l : n_l + n_r] = self.plans["R"].idx
        if n_h:
            self._clamp_idx[n_l + n_r :] = self.halt_idx

    @property
    def n(self) -> int:
        return int(self.core.n)

    def reset(self) -> None:
        self.core.reset()
        if hasattr(self.eyes, "reset"):
            self.eyes.reset()
        self.n_steps = 0

    def get_population_activity(self) -> Dict[str, float]:
        r = self.core.r
        return {
            "mean_rate": float(r.mean()),
            "max_rate": float(r.max()),
            "active_fraction": float((r > 0.01).mean()),
        }

    def _haltere_rates(self, angular_velocity: Optional[np.ndarray]) -> np.ndarray:
        n = int(self.halt_idx.size)
        if n == 0:
            return np.zeros(0, dtype=np.float32)
        amp = float(HALTERE_RATE)
        if angular_velocity is not None:
            w = np.atleast_1d(np.asarray(angular_velocity, dtype=float)).ravel()
            wz = float(w[2]) if w.size >= 3 else float(w[-1])
            amp = HALTERE_RATE * (0.5 + 0.5 * np.tanh(abs(wz) / max(self.params.haltere_omega_hz, 1e-6)))
        return np.full(n, amp, dtype=np.float32)

    def step(
        self,
        left_img: np.ndarray,
        right_img: np.ndarray,
        angular_velocity: Optional[np.ndarray] = None,
        hex_L: Optional[np.ndarray] = None,
        hex_R: Optional[np.ndarray] = None,
    ) -> BrainStep:
        t0 = time.perf_counter()
        if hex_L is None or hex_R is None:
            hex_L, hex_R = stereo_to_hex(left_img, right_img)
        t1 = time.perf_counter()
        vL, vR = self.eyes.step_pair(hex_L, hex_R)
        t2 = time.perf_counter()

        _idxL, rL = apply_handoff_plan(self.plans.get("L", _empty_plan()), vL)
        _idxR, rR = apply_handoff_plan(self.plans.get("R", _empty_plan()), vR)
        a, b = self._split
        if rL.size:
            self._clamp_val[:a] = rL
        if rR.size:
            self._clamp_val[a:b] = rR
        if self.halt_idx.size:
            self._clamp_val[b:] = self._haltere_rates(angular_velocity)

        if self._n_clamp:
            self.core.step(clamp_idx=self._clamp_idx, clamp_val=self._clamp_val)
        else:
            self.core.step()
        t3 = time.perf_counter()

        ro = read_dns(self.core.r, self.index)
        ms = {
            "sample": 1000.0 * (t1 - t0),
            "flyvis": 1000.0 * (t2 - t1),
            "brain": 1000.0 * (t3 - t2),
            "total": 1000.0 * (t3 - t0),
        }
        self.last_ms = ms
        self.n_steps += 1
        return BrainStep(readout=ro, rates=self.core.r.copy(), hex_L=hex_L, hex_R=hex_R, ms=ms)

    @classmethod
    def build(
        cls,
        data_dir=None,
        params: Optional[FlightParams] = None,
        quick: bool = False,
        eyes=None,
        calib: Optional[FlyvisCalibration] = None,
        torch_threads: int = 1,
    ) -> "ClosedLoopBrain":
        """Load connectome + flyvis. Raises if data or flyvis is missing."""
        if not connectome_available(data_dir):
            raise FileNotFoundError("data/full_connectome.npz not found")
        configure_threads(scipy_threads=1, torch_threads=max(1, int(torch_threads)))
        params = params or FlightParams()
        print("Building signed connectome + coreTplus…", flush=True)
        conn = build_signed_matrix(data_dir, cache=True)
        core_sel = select_core_tplus(conn)
        sub = slice_connectome(conn, core_sel.keep)
        remap = -np.ones(conn.n, dtype=np.int64)
        remap[core_sel.keep] = np.arange(len(core_sel.keep))
        index = build_readout_index(sub.typ, sub.side, sub.body_ids)
        core = RateCore.from_connectome(
            sub,
            g_inh=G_INH_DEFAULT,
            g_gap=G_GAP_DEFAULT,
            dng02_bias=DNG02_BIAS_DEFAULT,
            rho=RHO_FULL_SIGNED_REFERENCE,
        )
        print(f"  core n={core.n} scale={core.scale:.6g}", flush=True)
        hmap = build_handoff_map(conn)
        if eyes is None:
            if not flyvis_available():
                raise ImportError("flyvis is required for ClosedLoopBrain.build")
            print("Init flyvis (one net, two eye states)…", flush=True)
            eye = FlyvisEye(torch_threads=max(1, int(torch_threads)))
            eyes = DualFlyvisEyes.share(eye)
            cache = cache_dir(data_dir) / "flyvis_cal.npz"
            if calib is None and cache.is_file():
                calib = FlyvisCalibration.load(cache)
                print(f"  loaded calibration {cache}", flush=True)
            if calib is None:
                calib = eye.calibrate()
                cache.parent.mkdir(parents=True, exist_ok=True)
                # Persist even on --quick so the next short run does not recalibrate.
                calib.save(cache)
                print(f"  wrote calibration {cache}", flush=True)
            eyes.reset()
        if calib is None:
            raise ValueError("FlyvisCalibration is required")
        plans = {
            "L": compile_handoff_plan(hmap, calib, "L", remap),
            "R": compile_handoff_plan(hmap, calib, "R", remap),
        }
        halt_full = haltere_indices(conn, remap=None)
        halt = haltere_indices(conn, remap)
        print(
            f"  handoff L={plans['L'].idx.size} R={plans['R'].idx.size} "
            f"halt={halt.size} (full={halt_full.size}"
            f"{'' if halt.size or halt_full.size == 0 else '; not in coreTplus, physics damping fallback'})",
            flush=True,
        )
        return cls(
            core=core,
            index=index,
            plans=plans,
            eyes=eyes,
            halt_idx=halt,
            params=params,
        )


def _empty_plan() -> HandoffPlan:
    return HandoffPlan(
        idx=np.zeros(0, dtype=np.int64),
        types=np.zeros(0, dtype=object),
        cols=np.zeros(0, dtype=np.int64),
        v_gray=np.zeros(0, dtype=np.float32),
        p99=np.zeros(0, dtype=np.float32),
        type_ranges={},
    )
