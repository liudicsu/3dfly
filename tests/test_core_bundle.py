"""coreTplus bundle save/load (no full connectome required)."""

from __future__ import annotations

import numpy as np

from threedfly.brain.core_bundle import (
    load_core_bundle,
    remap_handoff_map,
    save_core_bundle,
)
from threedfly.brain.core_subgraph import slice_connectome
from threedfly.brain.handoff import build_handoff_map, compile_handoff_plan, apply_handoff_plan
from threedfly.brain.handoff import FlyvisCalibration
from tests.brain_fakes import tiny_signed_connectome


def test_bundle_roundtrip_tiny_connectome(tmp_path):
    conn = tiny_signed_connectome()
    keep = np.arange(conn.n, dtype=np.int64)
    keep = keep[conn.typ != "Foo"]
    remap = -np.ones(conn.n, dtype=np.int64)
    remap[keep] = np.arange(len(keep))
    sub = slice_connectome(conn, keep)
    hmap = remap_handoff_map(build_handoff_map(conn), remap)
    halt = remap[np.flatnonzero(np.array(["halt" in str(t).lower() for t in conn.typ]))]
    halt = halt[halt >= 0]
    path = tmp_path / "core_bundle.npz"
    save_core_bundle(path, sub, hmap, halt, extra_meta={"recipe": "tiny"})
    assert path.stat().st_size < 50 * 1024 * 1024
    loaded, h2, halt2, meta = load_core_bundle(path)
    assert loaded.n == sub.n
    assert loaded.W.nnz == sub.W.nnz
    assert set(h2.index) == set(hmap.index)
    assert halt2.size == halt.size
    assert meta.get("recipe") == "tiny"
    uv = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    calib = FlyvisCalibration(
        V_gray={"T4a": np.array([0.5, 0.5, 0.5], dtype=np.float32)},
        p99={"T4a": 1.0},
        uv={"T4a": uv},
        rmax=40.0,
    )
    plan = compile_handoff_plan(h2, calib, "L")
    idx, rates = apply_handoff_plan(plan, {"T4a": np.array([0.5, 1.5, 2.5], dtype=np.float32)})
    assert idx.size == plan.idx.size
    assert rates.size == idx.size
