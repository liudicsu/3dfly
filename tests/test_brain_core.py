"""coreTplus type-graph selection."""

from __future__ import annotations

import numpy as np
import pytest

from threedfly.brain.constants import (
    CORE_TPLUS_N_TARGET,
    CORE_TPLUS_N_TOL,
    INH_SOURCE_TYPES,
    connectome_available,
)
from threedfly.brain.connectome_matrix import build_signed_matrix
from threedfly.brain.core_subgraph import select_core_tplus
from tests.brain_fakes import tiny_signed_connectome


def test_core_tplus_keeps_path_forced_and_inh_drops_unrelated():
    conn = tiny_signed_connectome()
    sel = select_core_tplus(conn, thr=300, k_handoff_dn=2, k_dn_mn=3)
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
    sel = select_core_tplus(conn)
    kept_ids = set(conn.body_ids[sel.keep].tolist())
    assert 10752 in kept_ids
    assert 10989 in kept_ids


def test_inh_source_types_are_listed():
    assert "Li39" in INH_SOURCE_TYPES
    assert "TmY5a" in INH_SOURCE_TYPES
    assert "Am1" in INH_SOURCE_TYPES


@pytest.mark.skipif(not connectome_available(), reason="data/full_connectome.npz not present")
def test_real_core_tplus_count_within_5_percent():
    conn = build_signed_matrix(cache=True)
    sel = select_core_tplus(conn)
    assert abs(sel.n_neurons - CORE_TPLUS_N_TARGET) <= CORE_TPLUS_N_TOL * CORE_TPLUS_N_TARGET
    kept = set(conn.typ[sel.keep])
    for t in ("DNp01", "DNp03", "DNa02", "DNp15", "TTMn", "PSI"):
        assert t in kept
    assert any(str(t).startswith("DNg02") for t in kept)
