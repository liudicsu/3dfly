"""coreTplus type-graph selection and packaged body-ID list."""

from __future__ import annotations

import numpy as np
import pytest

from threedfly.brain.constants import (
    CORE_TPLUS_N_TARGET,
    CORE_TPLUS_N_TYPES,
    INH_SOURCE_TYPES,
    connectome_available,
)
from threedfly.brain.connectome_matrix import build_signed_matrix
from threedfly.brain.core_subgraph import load_core_tplus_body_ids, select_core_tplus
from tests.brain_fakes import tiny_signed_connectome


def test_core_tplus_keeps_path_forced_and_inh_drops_unrelated():
    conn = tiny_signed_connectome()
    sel = select_core_tplus(conn, method="recipe", thr=300, k_handoff_dn=2, k_dn_mn=3)
    kept = set(conn.typ[sel.keep])
    assert "T4a" in kept
    assert "LPLC2" in kept
    assert "DNp01" in kept
    assert "TTMn" in kept
    assert "PSI" in kept
    assert "Li39" in kept  # inh source (the “plus”)
    assert "LPi43" in kept  # forced
    assert "HSE" in kept
    assert "DNg02_a" in kept
    assert "DNp03" in kept
    assert "Foo" not in kept
    assert "Bar" not in kept
    assert "coreTplus" in sel.recipe
    assert "inh_sources" in sel.recipe


def test_dnp03_body_ids_are_included():
    conn = tiny_signed_connectome()
    sel = select_core_tplus(conn, method="recipe")
    kept_ids = set(conn.body_ids[sel.keep].tolist())
    assert 10752 in kept_ids
    assert 10989 in kept_ids


def test_inh_source_types_are_listed():
    assert "Li39" in INH_SOURCE_TYPES
    assert "TmY5a" in INH_SOURCE_TYPES
    assert "Am1" in INH_SOURCE_TYPES
    assert "LPi43" in INH_SOURCE_TYPES
    assert "LPi34" in INH_SOURCE_TYPES
    assert "LPi2d" in INH_SOURCE_TYPES


def test_core_bundle_lpi_t4t5_lplc2_signed():
    """LPi in coreTplus: glutamate → −1 (GluCl); T4/T5 → LPi → LPLC2 present."""
    from threedfly.brain.constants import INHIBITORY_NT
    from threedfly.brain.core_bundle import core_bundle_available, core_bundle_path, load_core_bundle

    if not core_bundle_available():
        pytest.skip("data/core_bundle.npz not present")
    conn, _, _, _ = load_core_bundle(core_bundle_path())
    typ = np.asarray(conn.typ, dtype=object)
    nt = np.asarray(conn.nt, dtype=object)
    sign = np.asarray(conn.sign, dtype=float)
    W = conn.W.tocsr()
    lpi = np.array([t.startswith("LPi") for t in typ.astype(str)])
    assert int(np.count_nonzero(lpi)) > 100
    for name in ("LPi43", "LPi34", "LPi2c", "LPi2d"):
        assert name in set(typ.tolist()), name
    glu = lpi & np.array([str(n).lower() == "glutamate" for n in nt])
    assert int(np.count_nonzero(glu)) > 50
    assert float(np.max(np.abs(sign[glu] + 1.0))) < 1e-6
    t4t5 = np.array([str(t).startswith(("T4", "T5")) for t in typ])
    lplc = typ == "LPLC2"
    fwd = W[np.flatnonzero(lpi)][:, np.flatnonzero(t4t5)]
    back = W[np.flatnonzero(lplc)][:, np.flatnonzero(lpi)]
    assert fwd.nnz > 1000 and float(fwd.data.sum()) > 0.0
    assert back.nnz > 100 and float(back.data.sum()) < 0.0


def test_packaged_core_tplus_bodyids():
    ids = load_core_tplus_body_ids()
    assert ids.shape == (CORE_TPLUS_N_TARGET,)
    assert len(np.unique(ids)) == CORE_TPLUS_N_TARGET


def test_body_ids_method_raises_when_nothing_matches():
    conn = tiny_signed_connectome()
    with pytest.raises(ValueError, match="method='recipe'"):
        select_core_tplus(conn, method="body_ids", body_ids=np.array([999999999], dtype=np.int64))


@pytest.mark.skipif(not connectome_available(), reason="data/full_connectome.npz not present")
def test_real_core_tplus_exact_membership():
    conn = build_signed_matrix(cache=True)
    ref = load_core_tplus_body_ids()
    sel = select_core_tplus(conn)
    assert sel.n_neurons == CORE_TPLUS_N_TARGET
    assert sel.n_types == CORE_TPLUS_N_TYPES
    assert set(conn.body_ids[sel.keep].tolist()) == set(ref.tolist())
    kept = set(conn.typ[sel.keep])
    for t in ("DNp01", "DNp03", "DNa02", "DNp15", "TTMn", "PSI", "Tm9"):
        assert t in kept
    assert any(str(t).startswith("DNg02") for t in kept)
