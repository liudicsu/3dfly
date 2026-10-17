"""Signed connectome matrix: signs, shape, cache, skip if real data missing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from threedfly.brain.connectome_matrix import (
    build_signed_matrix,
    connectome_from_counts,
    signed_post_pre,
    spectral_radius,
)
from threedfly.brain.constants import RHO_FULL_SIGNED_REFERENCE, connectome_available, nt_sign
from tests.brain_fakes import tiny_signed_connectome


def test_nt_sign_gaba_glutamate_histamine_negative():
    assert nt_sign("gaba") == -1.0
    assert nt_sign("GABA") == -1.0
    assert nt_sign("glutamate") == -1.0
    assert nt_sign("histamine") == -1.0
    assert nt_sign("acetylcholine") == 1.0
    assert nt_sign("unknown") == 1.0


def test_signed_matrix_is_post_by_pre_and_flips_inhibitory_rows_of_A():
    # A is pre × post: neuron 1 (GABA) → neuron 2
    A = sp.csr_matrix(np.array([[0, 0, 0], [0, 0, 5.0], [0, 0, 0]], dtype=np.float32))
    sign = np.array([1.0, -1.0, 1.0], dtype=np.float32)
    W = signed_post_pre(A, sign)
    assert W.shape == (3, 3)
    # W[post, pre] = sign[pre] * A[pre, post]
    assert W[2, 1] == pytest.approx(-5.0)
    assert W[1, 2] == pytest.approx(0.0)


def test_tiny_connectome_signs_and_shape():
    c = tiny_signed_connectome()
    assert c.W.shape[0] == c.W.shape[1] == c.n
    assert c.A_counts.shape == c.W.shape
    assert c.W.format == "csr"
    lpi = int(np.flatnonzero(c.typ == "LPi43")[0])
    lplc = int(np.flatnonzero((c.typ == "LPLC2") & (c.side == "L"))[0])
    # inhibitory pre → LPLC2 is negative in post×pre
    assert c.W[lplc, lpi] < 0
    t4 = int(np.flatnonzero((c.typ == "T4a") & (c.side == "L"))[0])
    assert c.W[lplc, t4] > 0
    assert c.rho_full_reference == RHO_FULL_SIGNED_REFERENCE


def test_build_signed_matrix_from_synthetic_files(tmp_path: Path):
    n = 4
    body_ids = np.array([10, 11, 12, 13], dtype=np.int64)
    A = sp.csr_matrix(
        ([3.0, 2.0], ([0, 1], [1, 2])),
        shape=(n, n),
    )
    raw = tmp_path / "connectome-raw"
    raw.mkdir()
    np.savez(
        tmp_path / "full_connectome.npz",
        body_ids=body_ids,
        adj_data=A.data,
        adj_indices=A.indices,
        adj_indptr=A.indptr,
        adj_shape=A.shape,
    )
    pd.DataFrame(
        {
            "body": body_ids,
            "consensus_nt": ["acetylcholine", "GABA", None, "glutamate"],
            "predicted_nt": ["acetylcholine", "GABA", "histamine", "glutamate"],
        }
    ).to_feather(raw / "body-neurotransmitters-male-cns-v1.0.feather")
    pd.DataFrame(
        {
            "bodyId": body_ids,
            "type": ["T4a", "LPi43", "LPLC2", "DNp01"],
            "somaSide": ["L", "L", "?", "R"],
            "rootSide": ["L", "L", "L", "R"],
            "subclass": ["", "", "", ""],
            "assignedOlHex1": [1.0, np.nan, 2.0, np.nan],
            "assignedOlHex2": [0.0, np.nan, 3.0, np.nan],
        }
    ).to_feather(raw / "body-annotations-male-cns-v1.0-minconf-0.5.feather")

    data = build_signed_matrix(tmp_path, cache=True)
    assert data.n == 4
    assert data.side[2] == "L"  # somaSide=? → rootSide
    assert data.sign[1] == -1.0  # GABA
    assert data.sign[2] == -1.0  # consensus missing → predicted histamine
    assert data.W[2, 1] < 0  # post LPLC2, pre LPi
    # cache round-trip
    data2 = build_signed_matrix(tmp_path, cache=True, rebuild=False)
    assert data2.n == 4
    assert np.allclose(data2.W.data, data.W.data)


def test_spectral_radius_known_matrix():
    W = sp.csr_matrix(np.array([[0.0, 2.0], [2.0, 0.0]]))
    assert spectral_radius(W) == pytest.approx(2.0, abs=1e-6)


@pytest.mark.skipif(not connectome_available(), reason="data/full_connectome.npz not present")
def test_real_connectome_shape_when_available():
    data = build_signed_matrix(cache=True)
    assert data.n > 10_000
    assert data.W.shape == (data.n, data.n)
    assert data.W.nnz > 0
    assert set(np.unique(data.sign)).issubset({-1.0, 1.0})
