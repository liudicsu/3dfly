"""Signed post×pre MaleCNS matrix (research_r2/prep.py)."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import scipy.sparse as sp

from threedfly.brain.constants import (
    RHO_FULL_SIGNED_REFERENCE,
    ann_feather_path,
    as_str_array,
    cache_dir,
    connectome_npz_path,
    nt_feather_path,
    nt_sign,
)

logger = logging.getLogger(__name__)


@dataclass
class ConnectomeData:
    """Signed connectome plus per-neuron annotations.

    ``W`` is post × pre (input = W @ r). ``A_counts`` is the original
    unsigned pre × post synapse-count CSR from ``full_connectome.npz``.
    """

    body_ids: np.ndarray
    W: sp.csr_matrix
    A_counts: sp.csr_matrix
    sign: np.ndarray
    typ: np.ndarray
    side: np.ndarray
    subclass: np.ndarray
    hex: np.ndarray
    nt: np.ndarray
    rho_full_reference: float = RHO_FULL_SIGNED_REFERENCE
    extras: Dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.W.shape[0])

    def index_of(self, body_id: int) -> Optional[int]:
        hits = np.flatnonzero(self.body_ids == int(body_id))
        return int(hits[0]) if hits.size else None


def _predicted_nt_column(nt: "object") -> Optional[str]:
    for col in ("predicted_nt", "celltype_predicted_nt"):
        if col in nt.columns:
            return col
    return None


def load_counts_npz(path: Path) -> tuple[np.ndarray, sp.csr_matrix]:
    """Load ``full_connectome.npz`` (pre × post counts)."""
    z = np.load(path)
    body_ids = np.asarray(z["body_ids"])
    A = sp.csr_matrix(
        (z["adj_data"], z["adj_indices"], z["adj_indptr"]),
        shape=tuple(z["adj_shape"]),
    )
    return body_ids, A


def signs_from_neurotransmitters(body_ids: np.ndarray, nt_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """consensus_nt, fallback predicted_nt; gaba/glutamate/histamine → −1."""
    import pandas as pd

    nt = pd.read_feather(nt_path)
    if "body" not in nt.columns:
        raise ValueError(f"{nt_path} has no 'body' column")
    pred_col = _predicted_nt_column(nt)
    indexed = nt.set_index("body")
    consensus = indexed["consensus_nt"].astype(object)
    blank = {"", "nan", "none", "unclear", "<na>"}
    consensus = consensus.where(
        ~consensus.astype(str).str.lower().str.strip().isin(blank),
        other=np.nan,
    )
    if pred_col is not None:
        ntc = consensus.fillna(indexed[pred_col].astype(object))
    else:
        ntc = consensus
    ntc = ntc.reindex(body_ids).astype(object).fillna("unknown").astype(str).str.lower()
    names = ntc.to_numpy()
    signs = np.array([nt_sign(x) for x in names], dtype=np.float32)
    return signs, names


def annotations_aligned(body_ids: np.ndarray, ann_path: Path) -> dict[str, np.ndarray]:
    """type, side (somaSide else rootSide), assignedOlHex1/2, subclass."""
    import pandas as pd

    ann = pd.read_feather(ann_path).drop_duplicates("bodyId").set_index("bodyId")
    a = ann.reindex(body_ids)
    for col in ("type", "subclass", "superclass", "somaSide", "rootSide"):
        if col not in a.columns:
            a[col] = ""
        a[col] = a[col].astype(object).fillna("").astype(str)
    soma = a["somaSide"].to_numpy()
    root = a["rootSide"].to_numpy()
    side = np.where(np.isin(soma, ["L", "R"]), soma, root)
    hex1 = (
        a["assignedOlHex1"].to_numpy(dtype=float)
        if "assignedOlHex1" in a.columns
        else np.full(len(body_ids), np.nan)
    )
    hex2 = (
        a["assignedOlHex2"].to_numpy(dtype=float)
        if "assignedOlHex2" in a.columns
        else np.full(len(body_ids), np.nan)
    )
    extras: Dict[str, np.ndarray] = {}
    for extra_col in ("superclass", "celltype", "instance", "class"):
        if extra_col in a.columns:
            extras[extra_col] = as_str_array(a[extra_col].astype(object).fillna("").to_numpy())
    return {
        "typ": as_str_array(a["type"].to_numpy()),
        "side": as_str_array(side),
        "subclass": as_str_array(a["subclass"].to_numpy()),
        "hex": np.stack([hex1, hex2], axis=1).astype(np.float32),
        "extras": extras,
    }


def signed_post_pre(A_counts: sp.csr_matrix, sign: np.ndarray) -> sp.csr_matrix:
    """``W = (diag(sign) @ A).T`` — post × pre, input = W @ r."""
    Ws = sp.diags(np.asarray(sign, dtype=np.float32)) @ A_counts.tocsr()
    return Ws.T.tocsr().astype(np.float32)


def spectral_radius(W: sp.spmatrix, maxiter: int = 4000) -> float:
    """Largest-magnitude eigenvalue of a (possibly signed) sparse matrix."""
    W = W.tocsr().astype(np.float64)
    n = W.shape[0]
    if n == 0 or W.nnz == 0:
        return 0.0
    if n <= 256:
        vals = np.linalg.eigvals(W.toarray())
        return float(np.max(np.abs(vals)))
    try:
        import scipy.sparse.linalg as sla

        vals = sla.eigs(
            W,
            k=1,
            which="LM",
            return_eigenvectors=False,
            maxiter=max(maxiter, n),
        )
        return float(np.abs(vals[0]))
    except Exception as exc:  # pragma: no cover - eigs can fail on awkward spectra
        logger.warning("eigs failed (%s); falling back to power iteration", exc)
        return _power_iteration(W)


def _power_iteration(W: sp.spmatrix, niter: int = 200) -> float:
    rng = np.random.default_rng(0)
    v = rng.standard_normal(W.shape[0])
    v /= np.linalg.norm(v) + 1e-12
    rho = 0.0
    for _ in range(niter):
        w = W @ v
        nrm = float(np.linalg.norm(w))
        if nrm == 0.0:
            return 0.0
        v = w / nrm
        rho = nrm
    return float(rho)


def connectome_from_counts(
    body_ids: np.ndarray,
    A_counts: sp.spmatrix,
    types,
    sides,
    transmitters,
    hexes: Optional[np.ndarray] = None,
    subclass=None,
    rho_full_reference: float = RHO_FULL_SIGNED_REFERENCE,
) -> ConnectomeData:
    """Build a ``ConnectomeData`` from in-memory arrays (tests / synthetic)."""
    body_ids = np.asarray(body_ids)
    A_counts = sp.csr_matrix(A_counts)
    n = A_counts.shape[0]
    typ = as_str_array(types)
    side = as_str_array(sides)
    nt = as_str_array(transmitters)
    sign = np.array([nt_sign(x) for x in nt], dtype=np.float32)
    if hexes is None:
        hexes = np.full((n, 2), np.nan, dtype=np.float32)
    else:
        hexes = np.asarray(hexes, dtype=np.float32)
    if subclass is None:
        subclass = np.array([""] * n, dtype=object)
    else:
        subclass = as_str_array(subclass)
    W = signed_post_pre(A_counts, sign)
    return ConnectomeData(
        body_ids=body_ids,
        W=W,
        A_counts=A_counts.tocsr(),
        sign=sign,
        typ=typ,
        side=side,
        subclass=subclass,
        hex=hexes,
        nt=nt,
        rho_full_reference=rho_full_reference,
    )


def _cache_paths(cdir: Path) -> dict[str, Path]:
    return {
        "W": cdir / "W_signed_postpre.npz",
        "A": cdir / "A_counts_prepost.npz",
        "meta": cdir / "neuron_meta.npz",
        "info": cdir / "connectome_info.json",
    }


def _save_cache(data: ConnectomeData, cdir: Path) -> None:
    cdir.mkdir(parents=True, exist_ok=True)
    paths = _cache_paths(cdir)
    sp.save_npz(paths["W"], data.W)
    sp.save_npz(paths["A"], data.A_counts)
    np.savez_compressed(
        paths["meta"],
        body_ids=data.body_ids,
        sign=data.sign,
        typ=data.typ.astype(str),
        side=data.side.astype(str),
        subclass=data.subclass.astype(str),
        hex=data.hex,
        nt=data.nt.astype(str),
        **{f"extra_{k}": np.asarray(v).astype(str) for k, v in (data.extras or {}).items()},
    )
    paths["info"].write_text(
        json.dumps(
            {
                "n": data.n,
                "nnz": int(data.W.nnz),
                "rho_full_reference": data.rho_full_reference,
            },
            indent=2,
        )
    )


def _load_cache(cdir: Path) -> Optional[ConnectomeData]:
    paths = _cache_paths(cdir)
    if not all(p.is_file() for p in paths.values()):
        return None
    W = sp.load_npz(paths["W"]).tocsr()
    A = sp.load_npz(paths["A"]).tocsr()
    meta = np.load(paths["meta"], allow_pickle=True)
    info = json.loads(paths["info"].read_text())
    return ConnectomeData(
        body_ids=meta["body_ids"],
        W=W.astype(np.float32),
        A_counts=A,
        sign=meta["sign"].astype(np.float32),
        typ=as_str_array(meta["typ"]),
        side=as_str_array(meta["side"]),
        subclass=as_str_array(meta["subclass"]),
        hex=meta["hex"].astype(np.float32),
        nt=as_str_array(meta["nt"]),
        rho_full_reference=float(info.get("rho_full_reference", RHO_FULL_SIGNED_REFERENCE)),
        extras={
            k[6:]: as_str_array(meta[k])
            for k in meta.files
            if k.startswith("extra_")
        },
    )


def build_signed_matrix(
    data_dir=None,
    cache: bool = True,
    rebuild: bool = False,
) -> ConnectomeData:
    """Load counts + NT + annotations and return a signed post×pre CSR.

    Expects ``data/full_connectome.npz`` and the two MaleCNS feathers.
    Caches to ``data/cache/`` (gitignored).
    """
    cdir = cache_dir(data_dir)
    if cache and not rebuild:
        cached = _load_cache(cdir)
        if cached is not None:
            if not cached.extras:
                ann_path = ann_feather_path(data_dir)
                if ann_path.is_file():
                    try:
                        ann = annotations_aligned(cached.body_ids, ann_path)
                        cached.extras = ann.get("extras") or {}
                    except Exception:
                        logger.debug("Could not attach annotation extras to cache", exc_info=True)
            logger.info("Loaded signed connectome cache (%d neurons)", cached.n)
            return cached

    npz = connectome_npz_path(data_dir)
    nt_path = nt_feather_path(data_dir)
    ann_path = ann_feather_path(data_dir)
    for path in (npz, nt_path, ann_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"Connectome file not found: {path}. "
                "Place data/full_connectome.npz and data/connectome-raw/*.feather on the machine."
            )

    body_ids, A = load_counts_npz(npz)
    sign, nt_names = signs_from_neurotransmitters(body_ids, nt_path)
    ann = annotations_aligned(body_ids, ann_path)
    W = signed_post_pre(A, sign)
    data = ConnectomeData(
        body_ids=body_ids,
        W=W,
        A_counts=A.tocsr(),
        sign=sign,
        typ=ann["typ"],
        side=ann["side"],
        subclass=ann["subclass"],
        hex=ann["hex"],
        nt=nt_names,
        extras=ann.get("extras") or {},
    )
    if cache:
        _save_cache(data, cdir)
        logger.info("Cached signed connectome to %s", cdir)
    return data
