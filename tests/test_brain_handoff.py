"""Handoff map, voltage→rate, flyvis skip."""

from __future__ import annotations

import numpy as np
import pytest

from threedfly.brain.constants import N_HEX, RMAX, connectome_available, flyvis_available
from threedfly.brain.connectome_matrix import build_signed_matrix
from threedfly.brain.handoff import (
    FlyvisCalibration,
    FlyvisEye,
    apply_handoff_plan,
    build_handoff_map,
    compile_handoff_plan,
    handoff_rates_for_frame,
    impute_hex,
    voltages_to_rates,
)
from threedfly.brain.stimuli import get_hex_coords, n_hex
from tests.brain_fakes import tiny_signed_connectome


def test_hex_lattice_has_721_cells():
    u, v = get_hex_coords(15)
    assert len(u) == N_HEX == 721
    assert n_hex(15) == 721


def test_voltages_to_rates_matches_spec_formula():
    V = np.array([0.5, 1.5, 2.5, -1.0], dtype=np.float32)
    V_gray = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    p99 = 2.0
    rates = voltages_to_rates(V, V_gray, p99, rmax=RMAX)
    # relu(V-0.5)/2 clipped: 0, 0.5, 1.0, 0  →  0, 75, 150, 0
    np.testing.assert_allclose(rates, [0.0, 75.0, 150.0, 0.0], atol=1e-3)


def test_impute_hex_from_neighbours():
    hex_uv = np.array([[1.0, 2.0], [np.nan, np.nan], [np.nan, np.nan]], dtype=float)
    has = np.array([True, False, False])
    target = np.array([True, True, True])
    # post 1 receives from pre 0; post 2 isolated
    import scipy.sparse as sp

    W = sp.csr_matrix((np.array([4.0]), (np.array([1]), np.array([0]))), shape=(3, 3))
    cur, known = impute_hex(hex_uv, has, target, W, n_iter=4)
    assert known[1]
    np.testing.assert_allclose(cur[1], [1.0, 2.0])
    assert not known[2]


def test_handoff_map_covers_both_eyes_and_tmy9_split():
    conn = tiny_signed_connectome()
    hmap = build_handoff_map(conn)
    assert "L" in hmap.sides_present()
    assert "R" in hmap.sides_present()
    assert hmap.counts["T4a|L"] >= 1
    assert hmap.counts["T4a|R"] >= 1
    # TmY9b starts without hex but is connected to TmY9a
    assert hmap.counts.get("TmY9|L", 0) >= 1
    assert hmap.n_mapped >= 3


def test_compile_handoff_plan_matches_per_frame():
    conn = tiny_signed_connectome()
    hmap = build_handoff_map(conn)
    uv = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    calib = FlyvisCalibration(
        V_gray={"T4a": np.array([0.5, 0.5, 0.5], dtype=np.float32)},
        p99={"T4a": 1.0},
        uv={"T4a": uv},
        rmax=150.0,
    )
    voltages = {"T4a": np.array([0.5, 1.5, 2.5], dtype=np.float32)}
    idx_a, rate_a = handoff_rates_for_frame(voltages, calib, hmap, "L")
    plan = compile_handoff_plan(hmap, calib, "L")
    idx_b, rate_b = apply_handoff_plan(plan, voltages)
    np.testing.assert_array_equal(idx_a, idx_b)
    np.testing.assert_allclose(rate_a, rate_b, atol=1e-5)


def test_flyvis_eye_skips_cleanly_when_missing():
    if flyvis_available():
        pytest.skip("flyvis present — construction downloads a checkpoint; run on the verification machine")
    with pytest.raises(ImportError, match="flyvis"):
        FlyvisEye()


@pytest.mark.skipif(not connectome_available(), reason="data/full_connectome.npz not present")
def test_real_handoff_map_coverage():
    conn = build_signed_matrix(cache=True)
    hmap = build_handoff_map(conn)
    assert hmap.n_mapped >= 25_000
    assert hmap.sides_present() == {"L", "R"}
