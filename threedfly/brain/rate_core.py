"""Spectrally scaled firing-rate core (research_r3/r3sim.py S2 + r4 g_inh)."""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np
import scipy.sparse as sp

from threedfly.brain.connectome_matrix import ConnectomeData, spectral_radius
from threedfly.brain.constants import (
    DNG02_BIAS_DEFAULT,
    DT,
    G_GAP_DEFAULT,
    G_INH_DEFAULT,
    G_INH_TARGETS,
    RATE_CAP,
    RHO_FULL_SIGNED_REFERENCE,
    SHUFFLE_SEED,
    TARGET_RHO,
    TAU,
    configure_threads,
)

logger = logging.getLogger(__name__)


def degree_preserving_shuffle(W: sp.spmatrix, seed: int = SHUFFLE_SEED) -> sp.csr_matrix:
    """Permute postsynaptic indices; keeps each synapse's pre and weight (r3sim.py)."""
    C = W.tocoo()
    rng = np.random.default_rng(seed)
    shuffled = sp.csr_matrix(
        (C.data, (rng.permutation(C.row), C.col)),
        shape=W.shape,
    )
    shuffled.sum_duplicates()
    return shuffled


def apply_inhibitory_gain(
    W: sp.spmatrix,
    types: np.ndarray,
    g_inh: float,
    targets: Sequence[str] = G_INH_TARGETS,
) -> sp.csr_matrix:
    """Multiply negative inputs onto LPLC1/2/4 and LC4 rows by ``g_inh``."""
    W = W.tocsr()
    if g_inh == 1.0 or W.nnz == 0:
        return W
    loom = np.isin(np.asarray(types, dtype=object), list(targets))
    C = W.tocoo()
    data = C.data.copy()
    mask = loom[C.row] & (data < 0)
    data[mask] *= float(g_inh)
    return sp.csr_matrix((data, (C.row, C.col)), shape=W.shape, dtype=np.float32)


def gf_gap_matrix(
    types: np.ndarray,
    sides: np.ndarray,
    g_gap: float,
) -> Optional[sp.csr_matrix]:
    """Rectifying rate-domain GF coupling: DNp01 → ipsi TTMn and PSI."""
    if g_gap == 0:
        return None
    n = len(types)
    rows, cols, vals = [], [], []
    for side in ("L", "R"):
        pre = np.flatnonzero((types == "DNp01") & (sides == side))
        for tgt in ("TTMn", "PSI"):
            post = np.flatnonzero((types == tgt) & (sides == side))
            for i in post:
                for j in pre:
                    rows.append(int(i))
                    cols.append(int(j))
                    vals.append(float(g_gap))
    if not rows:
        return None
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float32)


def scale_to_rho(W: sp.spmatrix, target_rho: float = TARGET_RHO) -> tuple[sp.csr_matrix, float, float]:
    """S2: W_m = W * (target_rho / ρ(W_signed)). Returns (Wm, rho, scale)."""
    rho = spectral_radius(W)
    if rho <= 0:
        scale = 0.0
        Wm = W.tocsr().astype(np.float32)
    else:
        scale = float(target_rho / rho)
        Wm = (W * scale).astype(np.float32).tocsr()
    return Wm, float(rho), scale


class RateCore:
    """Leaky rate network: r ← a r + (1−a) clip(W_m r + bias + G r, 0, cap).

    S2 scaling matches research_r3/r3sim.py: ``W_m = W * (target_rho / ρ)``.
    Default ``rho`` is the full signed MaleCNS radius
    (``RHO_FULL_SIGNED_REFERENCE``); r3 uses that same denominator on
    subgraphs. Pass ``rho=None`` to recompute ρ on the matrix being simulated
    (``--rho-mode subgraph``). On coreTplus that is ~3× stronger recurrence
    than research and saturates HS at the rate cap.
    """

    def __init__(
        self,
        W: sp.spmatrix,
        types,
        sides,
        body_ids: Optional[np.ndarray] = None,
        g_inh: float = G_INH_DEFAULT,
        g_gap: float = G_GAP_DEFAULT,
        dng02_bias: float = 0.0,
        shuffle: bool = False,
        target_rho: float = TARGET_RHO,
        tau: float = TAU,
        dt: float = DT,
        cap: float = RATE_CAP,
        scipy_threads: int = 1,
        rho: Optional[float] = RHO_FULL_SIGNED_REFERENCE,
    ):
        configure_threads(scipy_threads=scipy_threads)
        self.typ = np.asarray(types, dtype=object)
        self.side = np.asarray(sides, dtype=object)
        self.body_ids = None if body_ids is None else np.asarray(body_ids)
        self.n = int(W.shape[0])
        self.g_inh = float(g_inh)
        self.g_gap = float(g_gap)
        self.cap = float(cap)
        self.dt = float(dt)
        self.tau = float(tau)
        self.a = float(np.exp(-dt / tau))
        self.shuffle = bool(shuffle)
        self.target_rho = float(target_rho)

        W = W.tocsr().astype(np.float32)
        if shuffle:
            W = degree_preserving_shuffle(W)
        self.W_unscaled = W
        if rho is None:
            self.W_scaled, self.rho_unscaled, self.scale = scale_to_rho(W, target_rho)
        else:
            self.rho_unscaled = float(rho)
            self.scale = float(target_rho / rho) if rho else 0.0
            self.W_scaled = (W * self.scale).astype(np.float32).tocsr()
        self.Wm = apply_inhibitory_gain(self.W_scaled, self.typ, self.g_inh)
        self.G = gf_gap_matrix(self.typ, self.side, self.g_gap)

        self.bias = np.zeros(self.n, dtype=np.float32)
        if dng02_bias:
            self.bias[self.select(r"^DNg02")] += float(dng02_bias)

        self.r = np.zeros(self.n, dtype=np.float32)
        self._clamp_idx: Optional[np.ndarray] = None
        self._clamp_val: Optional[np.ndarray] = None

    @classmethod
    def from_connectome(
        cls,
        connectome: ConnectomeData,
        keep: Optional[np.ndarray] = None,
        **kwargs,
    ) -> "RateCore":
        if keep is None:
            keep = np.arange(connectome.n)
        keep = np.asarray(keep, dtype=int)
        return cls(
            W=connectome.W[keep][:, keep],
            types=connectome.typ[keep],
            sides=connectome.side[keep],
            body_ids=connectome.body_ids[keep],
            **kwargs,
        )

    def select(self, pat: str, side: Optional[str] = None) -> np.ndarray:
        import re

        m = np.array([bool(re.match(pat, str(t))) for t in self.typ])
        if side is not None:
            m &= self.side == side
        return np.flatnonzero(m)

    def set_clamps(self, idx, val) -> None:
        """Persistent clamps applied at the end of every ``step``."""
        self._clamp_idx = np.asarray(idx, dtype=int)
        self._clamp_val = np.asarray(val, dtype=np.float32)

    def clear_clamps(self) -> None:
        self._clamp_idx = None
        self._clamp_val = None

    def reset(self) -> None:
        self.r.fill(0.0)

    def step(
        self,
        r: Optional[np.ndarray] = None,
        clamp_idx=None,
        clamp_val=None,
    ) -> np.ndarray:
        if r is None:
            r = self.r
        x = self.Wm @ r + self.bias
        if self.G is not None:
            x = x + self.G @ r
        r = r * self.a + (1.0 - self.a) * np.clip(x, 0.0, self.cap)
        if clamp_idx is not None:
            r[np.asarray(clamp_idx, dtype=int)] = np.asarray(clamp_val, dtype=np.float32)
        if self._clamp_idx is not None:
            r[self._clamp_idx] = self._clamp_val
        self.r = np.asarray(r, dtype=np.float32)
        return self.r

    def n_at_ceiling(self, eps: float = 1.0) -> int:
        return int((self.r >= self.cap - eps).sum())
