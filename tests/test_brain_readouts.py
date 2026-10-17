"""DN readout groups. DNg02 is amplitude only (no L−R yaw)."""

from __future__ import annotations

import numpy as np
import pytest

from threedfly.brain.constants import connectome_available
from threedfly.brain.connectome_matrix import build_signed_matrix
from threedfly.brain.core_subgraph import select_core_tplus, slice_connectome
from threedfly.brain.readouts import DNReadout, build_readout_index, read_dns
from tests.brain_fakes import tiny_signed_connectome


def test_dnp03_one_per_side_on_synthetic():
    conn = tiny_signed_connectome()
    idx = build_readout_index(conn.typ, conn.side, conn.body_ids)
    assert idx.size("DNp03|L") == 1
    assert idx.size("DNp03|R") == 1
    assert conn.body_ids[idx["DNp03|L"][0]] == 10752
    assert conn.body_ids[idx["DNp03|R"][0]] == 10989


def test_dng02_prefix_and_amplitude_is_mean_not_difference():
    conn = tiny_signed_connectome()
    idx = build_readout_index(conn.typ, conn.side, conn.body_ids)
    assert idx.size("DNg02|L") == 2  # a, b
    assert idx.size("DNg02|R") == 1
    rates = np.zeros(conn.n, dtype=np.float32)
    rates[idx["DNg02|L"]] = 10.0
    rates[idx["DNg02|R"]] = 4.0
    ro = read_dns(rates, idx)
    assert ro.dng02_L == pytest.approx(10.0)
    assert ro.dng02_R == pytest.approx(4.0)
    assert ro.dng02_amp == pytest.approx(7.0)
    assert not hasattr(ro, "dng02_yaw")
    assert "dng02_yaw" not in ro.as_dict()


def test_readout_dataclass_fields():
    ro = DNReadout(dnp01_L=1.0, dnp01_R=0.1)
    assert ro.dnp01_L == 1.0
    d = ro.as_dict()
    assert "dnp01_L" in d and "dng02_amp" in d


@pytest.mark.skipif(not connectome_available(), reason="data/full_connectome.npz not present")
def test_real_dn_group_sizes():
    conn = build_signed_matrix(cache=True)
    sel = select_core_tplus(conn)
    sub = slice_connectome(conn, sel.keep)
    idx = build_readout_index(sub.typ, sub.side, sub.body_ids)
    n_dng = idx.size("DNg02|L") + idx.size("DNg02|R")
    assert n_dng == 29
    assert idx.size("DNp03|L") == 1
    assert idx.size("DNp03|R") == 1
