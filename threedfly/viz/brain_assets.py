"""CoreTplus brain-panel static assets (positions, cell types, region shells).

The binary asset is produced by ``scripts/export_brain_assets.py`` and must
carry the same ``body_ids_sha1`` as the protocol-v1 HELLO message
(``FrameBuilder.meta()`` / ``docs/stream_protocol.md``), so the front end can
reject a mismatched connectome order.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ASSET_VERSION = 1
ASSET_NAME = "coreTplus_v1.npz"
META_NAME = "meta.json"

# MaleCNS somaLocation is in 8 nm voxels → multiply by 8 for nm.
VOXEL_NM = 8.0

KEY_TYPES = (
    "DNp01",
    "DNp03",
    "DNa02",
    "DNp15",
    "LPLC2",
    "LC4",
    "T4a",
    "T4b",
    "T4c",
    "T4d",
    "T5a",
    "T5b",
    "T5c",
    "T5d",
    "HSE",
    "TTMn",
    "PSI",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_asset_dir(root: Optional[Path] = None) -> Path:
    return (root or repo_root()) / "assets" / "brain"


def body_ids_sha1(body_ids: Sequence[int]) -> str:
    """SHA-1 of little-endian int64 body ids — matches FrameBuilder.meta()."""
    arr = np.asarray(body_ids, dtype="<i8")
    return hashlib.sha1(arr.tobytes()).hexdigest()


def _as_xyz(val) -> Optional[np.ndarray]:
    if val is None:
        return None
    try:
        if isinstance(val, float) and np.isnan(val):
            return None
    except (TypeError, ValueError):
        pass
    try:
        p = np.asarray(val, dtype=np.float64).ravel()
    except (TypeError, ValueError):
        return None
    if p.shape != (3,) or not np.isfinite(p).all():
        return None
    return p.astype(np.float64)


@dataclass
class BrainAsset:
    """In-memory coreTplus brain-panel asset (RateCore neuron order)."""

    n_neurons: int
    body_ids_sha1: str
    positions_nm: np.ndarray  # (N, 3) float32; NaN where unknown before fill
    positions_u16: np.ndarray  # (N, 3) uint16 quantised into [lo_nm, hi_nm]
    has_soma: np.ndarray  # (N,) bool
    type_names: np.ndarray  # (n_types,) str
    type_codes: np.ndarray  # (N,) uint16
    sides: np.ndarray  # (N,) uint8: 0=L, 1=R, 2=other/unknown
    superclass_names: np.ndarray
    superclass_codes: np.ndarray  # (N,) uint8
    lo_nm: np.ndarray  # (3,) float32
    hi_nm: np.ndarray  # (3,) float32
    key_idx: Dict[str, np.ndarray]
    version: int = ASSET_VERSION

    def decode_positions_nm(self) -> np.ndarray:
        """Dequantize uint16 positions back to nm (float32)."""
        span = (self.hi_nm - self.lo_nm).astype(np.float32)
        span = np.where(span <= 0, 1.0, span)
        out = self.lo_nm.astype(np.float32) + (
            self.positions_u16.astype(np.float32) / 65535.0
        ) * span
        return out

    def meta_dict(self) -> dict:
        return {
            "version": int(self.version),
            "n_neurons": int(self.n_neurons),
            "neuron_order": "coreTplus RateCore index (ClosedLoopBrain.core.r)",
            "body_ids_sha1": self.body_ids_sha1,
            "n_with_soma": int(self.has_soma.sum()),
            "lo_nm": self.lo_nm.astype(float).tolist(),
            "hi_nm": self.hi_nm.astype(float).tolist(),
            "type_names": [str(t) for t in self.type_names.tolist()],
            "superclass_names": [str(s) for s in self.superclass_names.tolist()],
            "side_codes": {"0": "L", "1": "R", "2": "?"},
            "key_idx": {k: v.astype(int).tolist() for k, v in self.key_idx.items()},
            "key_counts": {k: int(v.size) for k, v in self.key_idx.items()},
            "positions": {
                "dtype": "uint16",
                "scheme": "linear",
                "axis_order": "xyz",
                "unit": "nm",
                "note": "decode: lo + u16/65535 * (hi-lo); MaleCNS somaLocation * 8 nm",
            },
            "voxel_nm": VOXEL_NM,
        }


def build_from_connectome(
    body_ids: Sequence[int],
    types: Sequence[str],
    sides: Optional[Sequence[str]] = None,
    annotations_path: Optional[Path] = None,
    connectome_npz: Optional[Path] = None,
) -> BrainAsset:
    """Build a BrainAsset for ``body_ids`` (coreTplus order) from MaleCNS annotations."""
    import pandas as pd

    root = repo_root()
    ann_path = Path(annotations_path) if annotations_path else root / "data" / "full_connectome.annotations.pkl"
    if not ann_path.is_file():
        # Fall back to feather next to connectome if present
        feather = root / "data" / "connectome-raw"
        raise FileNotFoundError(
            f"Need MaleCNS body annotations at {ann_path} to export brain assets"
        )

    ann = pd.read_pickle(ann_path)
    ids = np.asarray(body_ids, dtype=np.int64)
    typ = np.asarray([str(t) for t in types], dtype=object)
    if len(ids) != len(typ):
        raise ValueError(f"body_ids ({len(ids)}) and types ({len(typ)}) length mismatch")
    if sides is None:
        side_arr = np.full(len(ids), "?", dtype=object)
    else:
        side_arr = np.asarray([str(s) for s in sides], dtype=object)

    amap = dict(zip(ann.bodyId.astype(np.int64).tolist(), range(len(ann))))
    soma_col = ann.somaLocation.values
    sc_col = ann.superclass.fillna("unknown").astype(str).values

    positions = np.full((len(ids), 3), np.nan, dtype=np.float64)
    has_soma = np.zeros(len(ids), dtype=bool)
    superclass = np.empty(len(ids), dtype=object)
    for i, bid in enumerate(ids):
        j = amap.get(int(bid))
        if j is None:
            superclass[i] = "unknown"
            continue
        superclass[i] = sc_col[j]
        xyz = _as_xyz(soma_col[j])
        if xyz is not None:
            positions[i] = xyz * VOXEL_NM
            has_soma[i] = True

    # Fill missing somata with per-type mean (else global mean of known somata).
    known = positions[has_soma]
    if known.size == 0:
        raise RuntimeError("no soma positions found in annotations for coreTplus neurons")
    global_mean = known.mean(axis=0)
    type_means: Dict[str, np.ndarray] = {}
    for t in np.unique(typ):
        mask = (typ == t) & has_soma
        if mask.any():
            type_means[str(t)] = positions[mask].mean(axis=0)
    for i in range(len(ids)):
        if has_soma[i]:
            continue
        positions[i] = type_means.get(str(typ[i]), global_mean)

    lo = positions.min(axis=0).astype(np.float32)
    hi = positions.max(axis=0).astype(np.float32)
    span = hi - lo
    span = np.where(span <= 0, 1.0, span).astype(np.float32)
    pos_u16 = np.clip(
        np.rint((positions.astype(np.float32) - lo) / span * 65535.0), 0, 65535
    ).astype(np.uint16)

    # Categorical type / superclass codes
    type_names, type_codes = np.unique(typ, return_inverse=True)
    type_codes = type_codes.astype(np.uint16)
    sc_names, sc_codes = np.unique(superclass.astype(str), return_inverse=True)
    sc_codes = sc_codes.astype(np.uint8)

    side_codes = np.full(len(ids), 2, dtype=np.uint8)
    side_codes[side_arr == "L"] = 0
    side_codes[side_arr == "R"] = 1

    key_idx: Dict[str, np.ndarray] = {}
    for t in KEY_TYPES:
        idx = np.flatnonzero(typ == t).astype(np.int32)
        if idx.size:
            key_idx[t] = idx
    dng = np.flatnonzero([str(t).startswith("DNg02") for t in typ]).astype(np.int32)
    if dng.size:
        key_idx["DNg02*"] = dng
    sapp = np.flatnonzero([str(t).startswith("SApp") for t in typ]).astype(np.int32)
    if sapp.size:
        key_idx["SApp"] = sapp

    return BrainAsset(
        n_neurons=len(ids),
        body_ids_sha1=body_ids_sha1(ids),
        positions_nm=positions.astype(np.float32),
        positions_u16=pos_u16,
        has_soma=has_soma,
        type_names=type_names.astype(str),
        type_codes=type_codes,
        sides=side_codes,
        superclass_names=sc_names.astype(str),
        superclass_codes=sc_codes,
        lo_nm=lo,
        hi_nm=hi,
        key_idx=key_idx,
    )


def save_asset(asset: BrainAsset, out_dir: Path, shells: Optional[Dict[str, Path]] = None) -> Path:
    """Write ``coreTplus_v1.npz`` + ``meta.json`` (+ optional shell GLBs) under out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / ASSET_NAME
    payload = {
        "version": np.array([asset.version], dtype=np.int32),
        "body_ids_sha1": np.array(asset.body_ids_sha1),
        "positions_u16": asset.positions_u16,
        "has_soma": asset.has_soma.astype(np.uint8),
        "type_codes": asset.type_codes,
        "type_names": asset.type_names.astype("U64"),
        "sides": asset.sides,
        "superclass_codes": asset.superclass_codes,
        "superclass_names": asset.superclass_names.astype("U32"),
        "lo_nm": asset.lo_nm,
        "hi_nm": asset.hi_nm,
    }
    # key_* arrays: store as key_idx|<name>
    for name, idx in asset.key_idx.items():
        payload[f"key_idx|{name}"] = idx.astype(np.int32)
    np.savez_compressed(npz_path, **payload)
    meta = asset.meta_dict()
    if shells:
        meta["shells"] = {}
        for name, src in shells.items():
            src = Path(src)
            dest = out_dir / src.name
            if src.resolve() != dest.resolve():
                dest.write_bytes(src.read_bytes())
            meta["shells"][name] = dest.name
    # Flat bins for the browser (avoids in-browser npz inflate).
    (out_dir / "positions_u16.bin").write_bytes(np.ascontiguousarray(asset.positions_u16).tobytes())
    (out_dir / "type_codes.bin").write_bytes(np.ascontiguousarray(asset.type_codes, dtype="<u2").tobytes())
    (out_dir / "has_soma.bin").write_bytes(asset.has_soma.astype(np.uint8).tobytes())
    (out_dir / "sides.bin").write_bytes(np.ascontiguousarray(asset.sides).tobytes())
    meta["bins"] = {
        "positions_u16": "positions_u16.bin",
        "type_codes": "type_codes.bin",
        "has_soma": "has_soma.bin",
        "sides": "sides.bin",
    }
    (out_dir / META_NAME).write_text(json.dumps(meta, indent=2) + "\n")
    return npz_path


def load_asset(path: Optional[Path] = None) -> BrainAsset:
    """Load asset from ``assets/brain/coreTplus_v1.npz`` (or an explicit path)."""
    if path is None:
        path = default_asset_dir() / ASSET_NAME
    path = Path(path)
    data = np.load(path, allow_pickle=False)
    version = int(data["version"][0]) if "version" in data.files else 1
    sha = str(data["body_ids_sha1"].item() if data["body_ids_sha1"].shape == () else data["body_ids_sha1"][()])
    key_idx = {}
    for k in data.files:
        if k.startswith("key_idx|"):
            key_idx[k.split("|", 1)[1]] = np.asarray(data[k], dtype=np.int32)
    return BrainAsset(
        n_neurons=int(data["positions_u16"].shape[0]),
        body_ids_sha1=sha,
        positions_nm=np.zeros((data["positions_u16"].shape[0], 3), dtype=np.float32),
        positions_u16=np.asarray(data["positions_u16"], dtype=np.uint16),
        has_soma=np.asarray(data["has_soma"], dtype=np.uint8).astype(bool),
        type_names=np.asarray(data["type_names"]).astype(str),
        type_codes=np.asarray(data["type_codes"], dtype=np.uint16),
        sides=np.asarray(data["sides"], dtype=np.uint8),
        superclass_names=np.asarray(data["superclass_names"]).astype(str),
        superclass_codes=np.asarray(data["superclass_codes"], dtype=np.uint8),
        lo_nm=np.asarray(data["lo_nm"], dtype=np.float32),
        hi_nm=np.asarray(data["hi_nm"], dtype=np.float32),
        key_idx=key_idx,
        version=version,
    )


def assert_matches_hello(asset: BrainAsset, hello_meta: dict) -> None:
    """Raise AssertionError if asset hash / n_neurons disagree with HELLO meta."""
    if int(hello_meta.get("n_neurons", -1)) != asset.n_neurons:
        raise AssertionError(
            f"n_neurons mismatch: asset={asset.n_neurons} hello={hello_meta.get('n_neurons')}"
        )
    h = hello_meta.get("body_ids_sha1")
    if h is None:
        raise AssertionError("HELLO meta missing body_ids_sha1")
    if h != asset.body_ids_sha1:
        raise AssertionError(f"body_ids_sha1 mismatch: asset={asset.body_ids_sha1} hello={h}")


def maybe_decimate_shells(
    cache_dir: Path,
    out_dir: Path,
    target_cb: int = 12000,
    target_ol: int = 6000,
) -> Dict[str, Path]:
    """Decimate MaleCNS major-shell ngmeshes into GLBs under out_dir (optional).

    Requires ``trimesh`` and ``fast_simplification``. Returns mapping name→path.
    Skips silently if cache files are missing.
    """
    import struct

    cache_dir = Path(cache_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = {"CB": "shell_CB.glb", "OL(L)": "shell_OL_L.glb", "OL(R)": "shell_OL_R.glb"}
    # Prefer already-decimated research prototypes
    proto = repo_root() / "output" / "research_ui" / "prototype" / "assets"
    out: Dict[str, Path] = {}
    for name, fname in names.items():
        proto_f = proto / fname
        if proto_f.is_file():
            dest = out_dir / fname
            dest.write_bytes(proto_f.read_bytes())
            out[name] = dest
            continue
        src = cache_dir / f"{name}.ngmesh"
        if not src.is_file():
            continue
        try:
            import trimesh
            import fast_simplification
        except ImportError:
            break
        raw = src.read_bytes()
        nv = struct.unpack("<I", raw[:4])[0]
        v = np.frombuffer(raw, "<f4", nv * 3, 4).reshape(-1, 3)
        f = np.frombuffer(raw, "<u4", offset=4 + nv * 12).reshape(-1, 3)
        mm = trimesh.Trimesh(v, f.astype(np.int64), process=True)
        mm.merge_vertices()
        tgt = target_cb if name == "CB" else target_ol
        v2, f2 = fast_simplification.simplify(
            mm.vertices.astype(np.float32), mm.faces.astype(np.int32), target_count=tgt, agg=8
        )
        m = trimesh.Trimesh(v2, f2)
        dest = out_dir / fname
        m.export(dest)
        out[name] = dest
    return out
