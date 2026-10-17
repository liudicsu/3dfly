"""RateCore: spectral scaling, g_inh rows, clamps, bounded dynamics."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from threedfly.brain.connectome_matrix import spectral_radius
from threedfly.brain.constants import RHO_FULL_SIGNED_REFERENCE, TARGET_RHO
from threedfly.brain.rate_core import RateCore, apply_inhibitory_gain, degree_preserving_shuffle
from tests.brain_fakes import random_rate_graph, tiny_signed_connectome


def test_default_scale_is_full_reference():
    W = sp.csr_matrix(np.array([[0.0, 2.0], [2.0, 0.0]], dtype=np.float32))
    types = np.array(["A", "B"], dtype=object)
    sides = np.array(["L", "R"], dtype=object)
    core = RateCore(W, types, sides, g_inh=1.0, g_gap=0.0, dng02_bias=0.0)
    assert core.scale == pytest.approx(TARGET_RHO / RHO_FULL_SIGNED_REFERENCE)
    assert core.rho_unscaled == pytest.approx(RHO_FULL_SIGNED_REFERENCE)
    # r3 applies 0.9/4085.83 even on a slice; effective subgraph rho stays well below 1.
    assert spectral_radius(core.W_scaled) < 1.0


def test_subgraph_mode_scales_local_rho_to_target():
    W = sp.csr_matrix(np.array([[0.0, 2.0], [2.0, 0.0]], dtype=np.float32))
    types = np.array(["A", "B"], dtype=object)
    sides = np.array(["L", "R"], dtype=object)
    core = RateCore(W, types, sides, g_inh=1.0, g_gap=0.0, dng02_bias=0.0, rho=None)
    rho = spectral_radius(core.W_scaled)
    assert rho == pytest.approx(TARGET_RHO, abs=0.01)
    assert core.rho_unscaled == pytest.approx(2.0, abs=0.01)


def test_g_inh_only_scales_negative_inputs_on_loom_detectors():
    conn = tiny_signed_connectome()
    plain = RateCore.from_connectome(conn, g_inh=1.0, g_gap=0.0)
    gained = RateCore.from_connectome(conn, g_inh=8.0, g_gap=0.0)
    lplc = int(np.flatnonzero((conn.typ == "LPLC2") & (conn.side == "L"))[0])
    hse = int(np.flatnonzero((conn.typ == "HSE") & (conn.side == "L"))[0])
    lpi = int(np.flatnonzero(conn.typ == "LPi43")[0])
    t4 = int(np.flatnonzero((conn.typ == "T4a") & (conn.side == "L"))[0])

    assert gained.Wm[lplc, lpi] == pytest.approx(8.0 * plain.Wm[lplc, lpi])
    assert gained.Wm[lplc, t4] == pytest.approx(plain.Wm[lplc, t4])
    assert gained.Wm[hse, lpi] == pytest.approx(plain.Wm[hse, lpi])
    assert gained.Wm[hse, lpi] < 0
    # LC4 / LPLC1 / LPLC4 rows are the other intended targets
    for tgt in ("LC4", "LPLC1", "LPLC4"):
        row = int(np.flatnonzero(conn.typ == tgt)[0])
        if plain.Wm[row, lpi] < 0:
            assert gained.Wm[row, lpi] == pytest.approx(8.0 * plain.Wm[row, lpi])


def test_apply_inhibitory_gain_helper_masks_rows():
    types = np.array(["LPLC2", "HSE", "LPi43"], dtype=object)
    W = sp.csr_matrix(np.array([[0, 0, -2.0], [0, 0, -3.0], [0, 0, 0]], dtype=np.float32))
    out = apply_inhibitory_gain(W, types, 8.0)
    assert out[0, 2] == pytest.approx(-16.0)
    assert out[1, 2] == pytest.approx(-3.0)


def test_gf_gap_couples_ipsi_dnp01_to_ttmn_and_psi():
    conn = tiny_signed_connectome()
    core = RateCore.from_connectome(conn, g_inh=1.0, g_gap=0.5)
    assert core.G is not None
    dnp = int(np.flatnonzero((conn.typ == "DNp01") & (conn.side == "L"))[0])
    ttmn = int(np.flatnonzero((conn.typ == "TTMn") & (conn.side == "L"))[0])
    psi = int(np.flatnonzero((conn.typ == "PSI") & (conn.side == "L"))[0])
    dnp_r = int(np.flatnonzero((conn.typ == "DNp01") & (conn.side == "R"))[0])
    assert core.G[ttmn, dnp] == pytest.approx(0.5)
    assert core.G[psi, dnp] == pytest.approx(0.5)
    assert core.G[ttmn, dnp_r] == 0.0


def test_dng02_bias_is_tonic_on_prefix():
    conn = tiny_signed_connectome()
    core = RateCore.from_connectome(conn, g_inh=1.0, g_gap=0.0, dng02_bias=5.0)
    idx = core.select(r"^DNg02")
    assert idx.size >= 2
    assert np.all(core.bias[idx] == 5.0)
    others = np.ones(core.n, dtype=bool)
    others[idx] = False
    assert np.all(core.bias[others] == 0.0)


def test_clamping_api_and_step():
    conn = tiny_signed_connectome()
    core = RateCore.from_connectome(conn, g_inh=1.0, g_gap=0.0)
    core.set_clamps([0], [12.5])
    r = core.step()
    assert r[0] == pytest.approx(12.5)
    r2 = core.step(clamp_idx=[1], clamp_val=[3.0])
    assert r2[1] == pytest.approx(3.0)
    assert r2[0] == pytest.approx(12.5)


def test_ratecore_bounded_under_random_input():
    conn = random_rate_graph()
    core = RateCore.from_connectome(conn, g_inh=8.0, g_gap=0.5, dng02_bias=5.0)
    rng = np.random.default_rng(0)
    for _ in range(80):
        idx = rng.choice(core.n, size=8, replace=False)
        val = rng.uniform(0, 40, size=8).astype(np.float32)
        r = core.step(clamp_idx=idx, clamp_val=val)
        assert np.isfinite(r).all()
        assert r.max() <= core.cap + 1e-5
        assert r.min() >= 0.0


def test_synthetic_dynamics_drive_downstream():
    # 2-cycle so rho > 0; clamp neuron 0 and watch neuron 1 rise.
    W = sp.csr_matrix(np.array([[0.0, 0.2], [1.5, 0.0]], dtype=np.float32))
    core = RateCore(
        W,
        types=["T4a", "LPLC2"],
        sides=["L", "L"],
        g_inh=1.0,
        g_gap=0.0,
        target_rho=0.9,
        rho=None,
    )
    for _ in range(40):
        r = core.step(clamp_idx=[0], clamp_val=[10.0])
    assert r[0] == pytest.approx(10.0)
    assert r[1] > 0.05
    assert r[1] <= core.cap


def test_shuffle_preserves_nnz_and_changes_structure():
    conn = tiny_signed_connectome()
    W = conn.W
    sh = degree_preserving_shuffle(W, seed=0)
    assert sh.nnz == W.nnz or sh.nnz <= W.nnz  # sum_duplicates may merge
    assert not np.allclose(sh.toarray(), W.toarray())
    core = RateCore.from_connectome(conn, g_inh=8.0, g_gap=0.0, shuffle=True)
    assert core.shuffle is True
    assert core.scale == pytest.approx(TARGET_RHO / RHO_FULL_SIGNED_REFERENCE)
