"""Compact coreTplus bundle so closed-loop can run without the full connectome.

On a machine with ``data/full_connectome.npz``, run
``python scripts/export_core_bundle.py``. The result (``data/core_bundle.npz``,
target < 50 MB) holds the signed subgraph, annotations, remapped handoff map,
DN readout groups, and halt indices. ``ClosedLoopBrain.build`` loads it when
the full connectome is absent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.sparse as sp

from threedfly.brain.connectome_matrix import ConnectomeData
from threedfly.brain.constants import (
    RHO_FULL_SIGNED_REFERENCE,
    cache_dir,
    connectome_available,
    default_data_dir,
)
from threedfly.brain.handoff import HandoffMap

BUNDLE_NAME = "core_bundle.npz"
BUNDLE_VERSION = 1


def core_bundle_path(data_dir=None) -> Path:
    return default_data_dir(data_dir) / BUNDLE_NAME


def core_bundle_available(data_dir=None) -> bool:
    return core_bundle_path(data_dir).is_file()


def save_core_bundle(
    path,
    connectome: ConnectomeData,
    hmap: HandoffMap,
    halt_idx: np.ndarray,
    extra_meta: Optional[dict] = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    W = connectome.W.tocsr()
    payload = {
        "W_data": np.asarray(W.data, dtype=np.float32),
        "W_indices": np.asarray(W.indices, dtype=np.int32),
        "W_indptr": np.asarray(W.indptr, dtype=np.int32),
        "W_shape": np.asarray(W.shape, dtype=np.int64),
        "body_ids": np.asarray(connectome.body_ids),
        "typ": np.asarray(connectome.typ).astype(str),
        "side": np.asarray(connectome.side).astype(str),
        "sign": np.asarray(connectome.sign, dtype=np.float32),
        "nt": np.asarray(connectome.nt).astype(str),
        "subclass": np.asarray(connectome.subclass).astype(str),
        "hex": np.asarray(connectome.hex, dtype=np.float32),
        "halt_idx": np.asarray(halt_idx, dtype=np.int64),
        "rho_full_reference": np.array([float(connectome.rho_full_reference)], dtype=np.float64),
        "version": np.array([BUNDLE_VERSION], dtype=np.int32),
    }
    A = connectome.A_counts.tocsr()
    payload.update(
        {
            "A_data": np.asarray(A.data, dtype=np.float32),
            "A_indices": np.asarray(A.indices, dtype=np.int32),
            "A_indptr": np.asarray(A.indptr, dtype=np.int32),
            "A_shape": np.asarray(A.shape, dtype=np.int64),
        }
    )
    keys = sorted(k for k in hmap.index if "|" in k)
    payload["hmap_keys"] = np.asarray(keys, dtype=object)
    for key in keys:
        payload[f"hmap_idx|{key}"] = np.asarray(hmap.index[key], dtype=np.int64)
        payload[f"hmap_uv|{key}"] = np.asarray(hmap.uv[key], dtype=np.float32)
        payload[f"hmap_count|{key}"] = np.array([int(hmap.counts.get(key, 0))], dtype=np.int32)
    for s, c in hmap.centre.items():
        payload[f"hmap_centre|{s}"] = np.asarray(c, dtype=np.float32)
        payload[f"hmap_scale|{s}"] = np.array([float(hmap.scale.get(s, 1.0))], dtype=np.float32)
    extras = connectome.extras or {}
    extra_names = []
    for name, arr in extras.items():
        extra_names.append(name)
        payload[f"extra|{name}"] = np.asarray(arr).astype(str)
    meta = {
        "n": int(connectome.n),
        "nnz": int(W.nnz),
        "hmap_keys": keys,
        "extra_names": extra_names,
        "rho_full_reference": float(connectome.rho_full_reference),
        "version": BUNDLE_VERSION,
    }
    if extra_meta:
        meta.update(extra_meta)
    payload["meta_json"] = np.array([json.dumps(meta)])
    np.savez_compressed(path, **payload)
    return path


def load_core_bundle(path) -> tuple[ConnectomeData, HandoffMap, np.ndarray, dict]:
    path = Path(path)
    z = np.load(path, allow_pickle=True)
    W = sp.csr_matrix(
        (z["W_data"], z["W_indices"], z["W_indptr"]),
        shape=tuple(int(x) for x in z["W_shape"]),
    )
    if "A_data" in z.files:
        A = sp.csr_matrix(
            (z["A_data"], z["A_indices"], z["A_indptr"]),
            shape=tuple(int(x) for x in z["A_shape"]),
        )
    else:
        A = abs(W.T).tocsr()
    extras = {}
    for name in z.files:
        if name.startswith("extra|"):
            extras[name.split("|", 1)[1]] = np.asarray(z[name], dtype=object)
    conn = ConnectomeData(
        body_ids=np.asarray(z["body_ids"]),
        W=W.astype(np.float32),
        A_counts=A,
        sign=np.asarray(z["sign"], dtype=np.float32),
        typ=np.asarray(z["typ"], dtype=object),
        side=np.asarray(z["side"], dtype=object),
        subclass=np.asarray(z["subclass"], dtype=object),
        hex=np.asarray(z["hex"], dtype=np.float32),
        nt=np.asarray(z["nt"], dtype=object),
        rho_full_reference=float(z["rho_full_reference"][0])
        if "rho_full_reference" in z.files
        else RHO_FULL_SIGNED_REFERENCE,
        extras=extras,
    )
    hmap = HandoffMap()
    keys = [str(k) for k in z["hmap_keys"]] if "hmap_keys" in z.files else []
    for key in keys:
        hmap.index[key] = np.asarray(z[f"hmap_idx|{key}"], dtype=np.int64)
        hmap.uv[key] = np.asarray(z[f"hmap_uv|{key}"], dtype=np.float32)
        hmap.counts[key] = int(z[f"hmap_count|{key}"][0]) if f"hmap_count|{key}" in z.files else int(
            hmap.index[key].size
        )
    for s in "LR":
        if f"hmap_centre|{s}" in z.files:
            hmap.centre[s] = np.asarray(z[f"hmap_centre|{s}"], dtype=np.float32)
            hmap.scale[s] = float(z[f"hmap_scale|{s}"][0])
    hmap.counts["total"] = hmap.n_mapped
    halt = np.asarray(z["halt_idx"], dtype=np.int64) if "halt_idx" in z.files else np.zeros(0, dtype=np.int64)
    meta = json.loads(str(z["meta_json"][0])) if "meta_json" in z.files else {}
    return conn, hmap, halt, meta


def remap_handoff_map(hmap: HandoffMap, remap: np.ndarray) -> HandoffMap:
    """Rewrite full-connectome indices to core indices (drop unmapped)."""
    out = HandoffMap()
    out.centre = dict(hmap.centre)
    out.scale = dict(hmap.scale)
    for key, idx in hmap.index.items():
        mapped = remap[np.asarray(idx, dtype=np.int64)]
        ok = mapped >= 0
        out.index[key] = mapped[ok].astype(np.int64)
        out.uv[key] = np.asarray(hmap.uv[key], dtype=np.float32)[ok]
        out.counts[key] = int(ok.sum())
    out.counts["total"] = out.n_mapped
    return out


def export_closed_loop_bundle(data_dir=None, dest=None) -> Path:
    """Build the bundle from the full connectome (run on the verification box)."""
    if not connectome_available(data_dir):
        raise FileNotFoundError("data/full_connectome.npz is required to export the bundle")
    from threedfly.brain.connectome_matrix import build_signed_matrix
    from threedfly.brain.core_subgraph import select_core_tplus, slice_connectome
    from threedfly.brain.handoff import build_handoff_map, haltere_indices

    dest = Path(dest) if dest is not None else core_bundle_path(data_dir)
    conn = build_signed_matrix(data_dir, cache=True)
    core_sel = select_core_tplus(conn)
    halt_full = haltere_indices(conn, remap=None)
    keep = np.unique(np.concatenate([core_sel.keep, halt_full])).astype(np.int64)
    remap = -np.ones(conn.n, dtype=np.int64)
    remap[keep] = np.arange(len(keep))
    sub = slice_connectome(conn, keep)
    hmap = remap_handoff_map(build_handoff_map(conn), remap)
    halt = remap[halt_full]
    halt = halt[halt >= 0]
    meta = {
        "core_n": int(core_sel.n_neurons),
        "keep_n": int(len(keep)),
        "halt_full": int(halt_full.size),
        "halt_core": int(halt.size),
        "recipe": core_sel.recipe,
        "cache": str(cache_dir(data_dir)),
    }
    save_core_bundle(dest, sub, hmap, halt, extra_meta=meta)
    return dest
