"""coreTplus cell-type subgraph (research body-ID list + type-graph fallback)."""

from __future__ import annotations

import gzip
import logging
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import numpy as np
import scipy.sparse as sp

from threedfly.brain.connectome_matrix import ConnectomeData
from threedfly.brain.constants import (
    CORE_K_DN_MN,
    CORE_K_HANDOFF_DN,
    CORE_THR,
    CORE_TPLUS_N_TARGET,
    CORE_TPLUS_N_TOL,
    CORE_TPLUS_N_TYPES,
    DNP03_BODY,
    FORCED_TYPES,
    HANDOFF_CONNECTOME_TYPES,
    INH_SOURCE_TYPES,
    REQUIRED_EXACT_TYPES,
    REQUIRED_PREFIXES,
    TARGET_DN_TYPES,
    WING_MN_EXACT,
)

logger = logging.getLogger(__name__)


@dataclass
class CoreSelection:
    keep: np.ndarray
    keep_types: tuple
    n_neurons: int
    n_types: int
    nnz: int
    recipe: str
    extras: dict = field(default_factory=dict)

    def within_target(self, target: int = CORE_TPLUS_N_TARGET, tol: float = CORE_TPLUS_N_TOL) -> bool:
        return abs(self.n_neurons - target) <= tol * target


def _unique_types(typ: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cleaned = np.array(["__untyped" if (t is None or str(t) == "") else str(t) for t in typ], dtype=object)
    ut, inv = np.unique(cleaned, return_inverse=True)
    return ut, inv


def type_count_graph(W_abs_pre_post: sp.spmatrix, inv: np.ndarray, n_types: int) -> sp.csr_matrix:
    """Type × type synapse-count graph (sides merged)."""
    n = W_abs_pre_post.shape[0]
    M = sp.csr_matrix((np.ones(n, dtype=np.float64), (np.arange(n), inv)), shape=(n, n_types))
    return (M.T @ W_abs_pre_post @ M).tocsr()


def _bfs(A: sp.spmatrix, sources: Sequence[int], K: int) -> np.ndarray:
    """Hop distances from ``sources`` following reversed edges of A (downstream of src).

    ``A`` is type×type with A[i, j] = i (pre) → j (post). ``A.T @ frontier``
    marks posts of currently marked pres, matching research_r4/core_type.py.
    """
    nt = A.shape[0]
    d = np.full(nt, 99, dtype=np.int32)
    if not sources:
        return d
    src = np.asarray(list(sources), dtype=int)
    src = src[(src >= 0) & (src < nt)]
    d[src] = 0
    fr = np.zeros(nt, dtype=bool)
    fr[src] = True
    AT = A.T.tocsr()
    for k in range(1, K + 1):
        nxt = np.asarray(AT @ fr.astype(np.float64)).ravel() > 0
        new = nxt & (d == 99)
        d[new] = k
        fr = new
    return d


def _ids_for(names: Iterable[str], tid: dict[str, int]) -> list[int]:
    return [tid[t] for t in names if t in tid]


def is_wing_mn(name: str) -> bool:
    return name in WING_MN_EXACT or str(name).endswith(" MN")


def is_required_type(name: str) -> bool:
    if name in REQUIRED_EXACT_TYPES or name in INH_SOURCE_TYPES:
        return True
    return any(str(name).startswith(p) for p in REQUIRED_PREFIXES + ("DNg02",))


def core_tplus_bodyids_resource():
    return files("threedfly.brain.data").joinpath("coreTplus_bodyids.txt.gz")


def load_core_tplus_body_ids(
    path: Optional[Union[str, Path]] = None,
) -> np.ndarray:
    """Research coreTplus MaleCNS body IDs (45,230 unique)."""
    if path is None:
        with core_tplus_bodyids_resource().open("rb") as fh:
            with gzip.open(fh, "rt") as gz:
                return np.fromiter((int(line) for line in gz if line.strip()), dtype=np.int64)
    p = str(path)
    if p.endswith(".gz"):
        with gzip.open(p, "rt") as gz:
            return np.fromiter((int(line) for line in gz if line.strip()), dtype=np.int64)
    return np.loadtxt(p, dtype=np.int64)


def _selection_from_keep(connectome: ConnectomeData, keep: np.ndarray, recipe: str, extras: dict) -> CoreSelection:
    keep = np.unique(np.asarray(keep, dtype=np.int64))
    W_sub = connectome.W[keep][:, keep]
    keep_types = tuple(sorted({str(t) for t in connectome.typ[keep] if str(t)}))
    return CoreSelection(
        keep=keep,
        keep_types=keep_types,
        n_neurons=int(len(keep)),
        n_types=len(keep_types),
        nnz=int(W_sub.nnz),
        recipe=recipe,
        extras=extras,
    )


def select_core_tplus_body_ids(
    connectome: ConnectomeData,
    body_ids: Optional[np.ndarray] = None,
) -> CoreSelection:
    """Map the packaged research body-ID list onto ``connectome.body_ids``."""
    ref = np.asarray(load_core_tplus_body_ids() if body_ids is None else body_ids, dtype=np.int64)
    order = np.argsort(connectome.body_ids, kind="mergesort")
    sorted_ids = connectome.body_ids[order]
    pos = np.searchsorted(sorted_ids, ref)
    in_range = pos < len(sorted_ids)
    pos_c = np.clip(pos, 0, max(len(sorted_ids) - 1, 0))
    matched = in_range & (sorted_ids[pos_c] == ref)
    keep = order[pos_c[matched]]
    keep.sort()
    n_miss = int((~matched).sum())
    if keep.size == 0:
        raise ValueError(
            "No coreTplus body IDs matched this connectome. "
            "For synthetic graphs pass method='recipe'."
        )
    recipe = (
        f"coreTplus body-ID list ({len(ref)} packaged IDs) mapped onto connectome "
        f"body_ids; matched={len(keep)} missing={n_miss} "
        f"(research {CORE_TPLUS_N_TARGET} neurons / {CORE_TPLUS_N_TYPES} types)."
    )
    logger.info(recipe)
    if n_miss:
        logger.warning("coreTplus body-ID map missed %d of %d IDs", n_miss, len(ref))
    return _selection_from_keep(
        connectome,
        keep,
        recipe,
        extras={"method": "body_ids", "n_ref": int(len(ref)), "n_missing": n_miss},
    )


def select_core_tplus(
    connectome: ConnectomeData,
    method: str = "body_ids",
    thr: int = CORE_THR,
    k_handoff_dn: int = CORE_K_HANDOFF_DN,
    k_dn_mn: int = CORE_K_DN_MN,
    extra_types: Sequence[str] = (),
    body_ids: Optional[np.ndarray] = None,
) -> CoreSelection:
    """Select the coreTplus subgraph.

    Default ``method='body_ids'`` loads the research list
    ``threedfly/brain/data/coreTplus_bodyids.txt.gz`` (45,230 MaleCNS IDs)
    and maps them through ``connectome.body_ids``.

    ``method='recipe'`` is the type-graph reconstruction (thr/K + forced +
    inh_sources). Use it for synthetic graphs that do not carry MaleCNS IDs.
    """
    if method == "body_ids":
        return select_core_tplus_body_ids(connectome, body_ids=body_ids)
    if method != "recipe":
        raise ValueError(f"unknown coreTplus method {method!r}; use 'body_ids' or 'recipe'")
    return select_core_tplus_recipe(
        connectome,
        thr=thr,
        k_handoff_dn=k_handoff_dn,
        k_dn_mn=k_dn_mn,
        extra_types=extra_types,
    )


def select_core_tplus_recipe(
    connectome: ConnectomeData,
    thr: int = CORE_THR,
    k_handoff_dn: int = CORE_K_HANDOFF_DN,
    k_dn_mn: int = CORE_K_DN_MN,
    extra_types: Sequence[str] = (),
) -> CoreSelection:
    """Keep all neurons of types on handoff→DN and DN→wing-MN paths, plus forced / inhibitors.

    Recipe (logged):
      1. Type graph = sum of |synapses| between types (pre×post, sides merged).
      2. Keep types on a handoff→DN path whose hop-sum ≤ ``k_handoff_dn``
         using edges with ≥ ``thr`` synapses (default thr=300, K=2).
      3. Keep types on a DN→wing-MN path with hop-sum ≤ ``k_dn_mn`` (default 3).
      4. Force every handoff, target-DN, and wing-MN type (research ``hs+dn+mn``).
      5. Force HS / LPLC / LC4 / LPi43 / LPi34.
      6. Force inhibitory sources onto LPLC2/LC4 from research_r4/inh_sources.json
         — this is the “plus” in coreTplus; coreT_thr300_K2 alone fails T3.
      7. Force required DNs / DNg02* / TTMn / PSI / DLMn* and spec body IDs.
    """
    typ = connectome.typ
    ut, inv = _unique_types(typ)
    tid = {str(t): i for i, t in enumerate(ut)}
    W_abs_pre_post = abs(connectome.W).T.tocsr()
    T = type_count_graph(W_abs_pre_post, inv, len(ut))
    A = (T >= thr).astype(np.float64).tocsr()

    handoff = _ids_for(HANDOFF_CONNECTOME_TYPES, tid)
    dns = _ids_for(TARGET_DN_TYPES, tid)
    dns += [i for i, t in enumerate(ut) if str(t).startswith("DNg02")]
    mns = [i for i, t in enumerate(ut) if is_wing_mn(str(t))]

    df = _bfs(A, handoff, k_handoff_dn)
    db = _bfs(A.T.tocsr(), dns, k_handoff_dn)
    keep_t = set(np.flatnonzero((df + db) <= k_handoff_dn).tolist())

    df2 = _bfs(A, dns, k_dn_mn)
    db2 = _bfs(A.T.tocsr(), mns, k_dn_mn)
    keep_t |= set(np.flatnonzero((df2 + db2) <= k_dn_mn).tolist())

    forced = set(FORCED_TYPES) | set(INH_SOURCE_TYPES) | set(REQUIRED_EXACT_TYPES) | set(extra_types)
    for t in list(ut):
        if is_required_type(str(t)):
            forced.add(str(t))
    keep_t |= set(handoff + dns + mns)
    keep_t |= set(_ids_for(forced, tid))
    keep_t.discard(tid.get("__untyped", -1))

    type_mask = np.isin(inv, list(keep_t))
    keep = np.flatnonzero(type_mask)

    # Force-include the two identified DNp03 body IDs if present.
    extra_bodies = []
    for bid in DNP03_BODY.values():
        idx = connectome.index_of(int(bid))
        if idx is not None and idx not in keep:
            extra_bodies.append(idx)
    if extra_bodies:
        keep = np.unique(np.concatenate([keep, np.asarray(extra_bodies, dtype=int)]))

    W_sub = connectome.W[keep][:, keep]
    keep_types = tuple(sorted(str(ut[i]) for i in keep_t if str(ut[i]) != "__untyped"))
    recipe = (
        f"coreTplus = type-graph |syn|≥{thr}; types on handoff→DN paths with "
        f"hop-sum≤{k_handoff_dn}; plus DN→wing-MN hop-sum≤{k_dn_mn}; "
        f"plus forced {list(FORCED_TYPES)}; plus inh_sources onto LPLC2/LC4 "
        f"{list(INH_SOURCE_TYPES)}; plus required DNs/DNg02*/TTMn/PSI/DLMn* "
        f"and DNp03 bodies {DNP03_BODY}. "
        f"n_types={len(keep_types)} n_neurons={len(keep)} nnz={int(W_sub.nnz)} "
        f"(target {CORE_TPLUS_N_TARGET} ±{CORE_TPLUS_N_TOL:.0%})."
    )
    logger.info(recipe)
    if abs(len(keep) - CORE_TPLUS_N_TARGET) > CORE_TPLUS_N_TOL * CORE_TPLUS_N_TARGET:
        logger.warning(
            "coreTplus neuron count %d is outside ±5%% of %d — recipe logged above",
            len(keep),
            CORE_TPLUS_N_TARGET,
        )
    return CoreSelection(
        keep=keep.astype(np.int64),
        keep_types=keep_types,
        n_neurons=int(len(keep)),
        n_types=len(keep_types),
        nnz=int(W_sub.nnz),
        recipe=recipe,
        extras={"method": "recipe", "thr": thr, "k_handoff_dn": k_handoff_dn, "k_dn_mn": k_dn_mn},
    )


def slice_connectome(connectome: ConnectomeData, keep: np.ndarray) -> ConnectomeData:
    """Restrict a connectome to ``keep`` indices, preserving metadata."""
    keep = np.asarray(keep, dtype=int)
    return ConnectomeData(
        body_ids=connectome.body_ids[keep],
        W=connectome.W[keep][:, keep].tocsr(),
        A_counts=connectome.A_counts[keep][:, keep].tocsr(),
        sign=connectome.sign[keep],
        typ=connectome.typ[keep],
        side=connectome.side[keep],
        subclass=connectome.subclass[keep],
        hex=connectome.hex[keep],
        nt=connectome.nt[keep],
        rho_full_reference=connectome.rho_full_reference,
    )
