"""Tests for coreTplus brain-panel static assets (UI PR3)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from threedfly.stream.builder import FrameBuilder
from threedfly.viz.brain_assets import (
    ASSET_NAME,
    META_NAME,
    BrainAsset,
    assert_matches_hello,
    body_ids_sha1,
    build_from_connectome,
    default_asset_dir,
    load_asset,
    save_asset,
)

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "data" / "core_bundle.npz"
ANN = ROOT / "data" / "full_connectome.annotations.pkl"
ASSET = default_asset_dir(ROOT) / ASSET_NAME


def _tiny_asset(tmp_path: Path) -> BrainAsset:
    """Synthetic 8-neuron asset (no annotations needed)."""
    ids = np.arange(10001, 10009, dtype=np.int64)
    types = np.array(["T4a", "T4a", "T5a", "LPLC2", "DNp01", "DNg02_a", "HSE", "LC4"], dtype=object)
    sides = np.array(list("LRLRLRLR"), dtype=object)
    pos = np.random.default_rng(0).normal(size=(8, 3)).astype(np.float32) * 1e5 + 3e5
    lo = pos.min(0).astype(np.float32)
    hi = pos.max(0).astype(np.float32)
    span = np.maximum(hi - lo, 1.0)
    u16 = np.clip(np.rint((pos - lo) / span * 65535), 0, 65535).astype(np.uint16)
    type_names, type_codes = np.unique(types, return_inverse=True)
    sc_names = np.array(["visual"], dtype=str)
    asset = BrainAsset(
        n_neurons=8,
        body_ids_sha1=body_ids_sha1(ids),
        positions_nm=pos,
        positions_u16=u16,
        has_soma=np.ones(8, dtype=bool),
        type_names=type_names.astype(str),
        type_codes=type_codes.astype(np.uint16),
        sides=np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.uint8),
        superclass_names=sc_names,
        superclass_codes=np.zeros(8, dtype=np.uint8),
        lo_nm=lo,
        hi_nm=hi,
        key_idx={
            "T4a": np.array([0, 1], dtype=np.int32),
            "DNp01": np.array([4], dtype=np.int32),
            "DNg02*": np.array([5], dtype=np.int32),
        },
    )
    save_asset(asset, tmp_path)
    return asset


def test_body_ids_sha1_matches_frame_builder():
    ids = np.array([10001, 10006, 10010], dtype=np.int64)
    types = np.array(["T4a", "T5a", "LPLC2"], dtype=object)
    fb = FrameBuilder(types=types, body_ids=ids, include_eye_layout=False)
    assert fb.meta()["body_ids_sha1"] == body_ids_sha1(ids)


def test_save_load_roundtrip(tmp_path):
    asset = _tiny_asset(tmp_path)
    loaded = load_asset(tmp_path / ASSET_NAME)
    assert loaded.n_neurons == 8
    assert loaded.body_ids_sha1 == asset.body_ids_sha1
    assert loaded.positions_u16.shape == (8, 3)
    assert loaded.type_codes.dtype == np.uint16
    assert "DNp01" in loaded.key_idx
    meta = json.loads((tmp_path / META_NAME).read_text())
    assert meta["body_ids_sha1"] == asset.body_ids_sha1
    assert meta["n_neurons"] == 8
    # dequantisation stays inside bbox
    decoded = loaded.decode_positions_nm()
    assert decoded.shape == (8, 3)
    assert np.all(decoded >= loaded.lo_nm - 1.0)
    assert np.all(decoded <= loaded.hi_nm + 1.0)


def test_assert_matches_hello(tmp_path):
    asset = _tiny_asset(tmp_path)
    ids = np.arange(10001, 10009, dtype=np.int64)
    types = np.array(["T4a", "T4a", "T5a", "LPLC2", "DNp01", "DNg02_a", "HSE", "LC4"], dtype=object)
    fb = FrameBuilder(types=types, body_ids=ids, include_eye_layout=False)
    assert_matches_hello(asset, fb.meta())
    bad = dict(fb.meta())
    bad["body_ids_sha1"] = "0" * 40
    with pytest.raises(AssertionError, match="body_ids_sha1"):
        assert_matches_hello(asset, bad)


@pytest.mark.skipif(not CORE.is_file() or not ANN.is_file(), reason="core bundle + annotations required")
def test_build_from_real_connectome_hash_matches():
    b = np.load(CORE, allow_pickle=True)
    asset = build_from_connectome(b["body_ids"], b["typ"], b["side"], annotations_path=ANN)
    fb = FrameBuilder(types=b["typ"], sides=b["side"], body_ids=b["body_ids"], include_eye_layout=False)
    assert_matches_hello(asset, fb.meta())
    assert asset.n_neurons == 45434
    assert asset.has_soma.sum() > 40000
    assert "DNp01" in asset.key_idx
    assert "DNg02*" in asset.key_idx
    # positions quantised tightly
    assert asset.positions_u16.dtype == np.uint16
    assert asset.lo_nm.shape == (3,)


@pytest.mark.skipif(not ASSET.is_file(), reason="committed asset not present yet")
def test_committed_asset_matches_core_bundle():
    """The tracked assets/brain file must match the core bundle on this box."""
    if not CORE.is_file():
        pytest.skip("core bundle absent")
    asset = load_asset(ASSET)
    b = np.load(CORE, allow_pickle=True)
    fb = FrameBuilder(types=b["typ"], sides=b["side"], body_ids=b["body_ids"], include_eye_layout=False)
    assert_matches_hello(asset, fb.meta())
    assert asset.n_neurons == int(len(b["body_ids"]))
    # size sanity: compressed npz should stay well under 2 MiB
    assert ASSET.stat().st_size < 2 * 1024 * 1024
