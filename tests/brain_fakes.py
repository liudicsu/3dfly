"""Synthetic MaleCNS-like graphs for brain unit tests (no big data files)."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from threedfly.brain.connectome_matrix import connectome_from_counts


def tiny_signed_connectome():
    """Handful of named cells covering signs, g_inh targets, DNs, handoff hex."""
    # index: 0 T4a L ACh, 1 LPLC2 L ACh, 2 LPi43 L Glu, 3 HSE L ACh,
    #        4 DNp01 L ACh, 5 TTMn L ACh, 6 PSI L ACh, 7 DNp03 L ACh (body 10752),
    #        8 DNa02 L ACh, 9 DNp15 L ACh, 10 DNg02_a L ACh, 11 DNg02_b L ACh,
    #        12 T4a R, 13 LPLC2 R, 14 DNp01 R, 15 DNp03 R (body 10989),
    #        16 DNg02_a R, 17 HSE R, 18 DNa02 R, 19 halt L, 20 LC4 L, 21 LPLC1 L,
    #        22 LPLC4 L, 23 Foo L (should drop from core), 24 Bar L,
    #        25 Li39 L Glu, 26 TmY9a L, 27 TmY9b L
    names = [
        "T4a",
        "LPLC2",
        "LPi43",
        "HSE",
        "DNp01",
        "TTMn",
        "PSI",
        "DNp03",
        "DNa02",
        "DNp15",
        "DNg02_a",
        "DNg02_b",
        "T4a",
        "LPLC2",
        "DNp01",
        "DNp03",
        "DNg02_a",
        "HSE",
        "DNa02",
        "haltere_aff",
        "LC4",
        "LPLC1",
        "LPLC4",
        "Foo",
        "Bar",
        "Li39",
        "TmY9a",
        "TmY9b",
    ]
    sides = (
        ["L"] * 12
        + ["R"] * 7
        + ["L", "L", "L", "L", "L", "L", "L", "L", "L"]
    )
    nts = ["acetylcholine"] * len(names)
    nts[2] = "glutamate"
    nts[25] = "gaba"
    subclass = [""] * len(names)
    subclass[19] = "haltere"
    body_ids = np.arange(1000, 1000 + len(names), dtype=np.int64)
    body_ids[7] = 10752
    body_ids[15] = 10989

    hexes = np.full((len(names), 2), np.nan, dtype=np.float32)
    hexes[0] = (10.0, 4.0)  # T4a L
    hexes[12] = (8.0, 3.0)  # T4a R
    hexes[26] = (11.0, 5.0)  # TmY9a L
    # TmY9b L has no hex — imputation from TmY9a if connected

    n = len(names)
    A = sp.lil_matrix((n, n), dtype=np.float32)
    # pre → post counts (A is pre × post)
    A[0, 1] = 400  # T4a → LPLC2
    A[1, 4] = 400  # LPLC2 → DNp01
    A[4, 5] = 50  # DNp01 → TTMn
    A[4, 6] = 50  # DNp01 → PSI
    A[2, 1] = 80  # LPi43 → LPLC2 (inhibitory)
    A[25, 1] = 60  # Li39 → LPLC2
    A[2, 3] = 40  # LPi43 → HSE (inhibitory, must NOT get g_inh)
    A[0, 3] = 30  # T4a → HSE
    A[12, 13] = 400
    A[13, 14] = 400
    A[23, 24] = 400  # Foo → Bar (unrelated)
    A[26, 27] = 10  # TmY9a → TmY9b for hex imputation
    A[0, 20] = 20  # T4a → LC4
    A[2, 20] = 15  # LPi → LC4
    return connectome_from_counts(
        body_ids,
        A.tocsr(),
        names,
        sides,
        nts,
        hexes=hexes,
        subclass=subclass,
    )


def random_rate_graph(n: int = 48, seed: int = 1):
    rng = np.random.default_rng(seed)
    A = sp.random(n, n, density=0.08, format="csr", dtype=np.float32, random_state=seed)
    A.setdiag(0)
    types = np.array([f"T{i%5}" for i in range(n)], dtype=object)
    types[0] = "LPLC2"
    types[1] = "LC4"
    types[2] = "DNp01"
    sides = np.array(["L" if i % 2 == 0 else "R" for i in range(n)], dtype=object)
    nts = np.array(["acetylcholine"] * n, dtype=object)
    nts[3::7] = "gaba"
    return connectome_from_counts(np.arange(n), A, types, sides, nts)
