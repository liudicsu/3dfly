"""Shared constants for the M1 flyvis-handoff + coreTplus rate core."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional

# Full signed MaleCNS matrix spectral radius from research_r3/r3sim.py.
# Default S2 scale is TARGET_RHO / this value even on subgraphs (r3sim does the
# same). Pass rho=None / --rho-mode subgraph to recompute ρ on the slice.
RHO_FULL_SIGNED_REFERENCE = 4085.83
TARGET_RHO = 0.9

INHIBITORY_NT = frozenset({"gaba", "glutamate", "histamine"})

DT = 0.01  # s
TAU = 0.05  # s
RATE_CAP = 50.0  # Hz
RMAX = 150.0  # Hz, flyvis voltage → rate
HALTERE_RATE = 40.0  # Hz
G_INH_DEFAULT = 8.0
G_GAP_DEFAULT = 0.5
DNG02_BIAS_DEFAULT = 5.0
SHUFFLE_SEED = 0

# flyvis lattice
HEX_EXTENT = 15
N_HEX = 3 * HEX_EXTENT * (HEX_EXTENT + 1) + 1  # 721
HEX_DEG = 5.8 / (3**0.5)  # research_r4/fv4_run.py

# coreTplus type-graph recipe (research_r4/core_type.py + inh_sources.json)
CORE_THR = 300
CORE_K_HANDOFF_DN = 2
CORE_K_DN_MN = 3
CORE_TPLUS_N_TARGET = 45230
CORE_TPLUS_N_TYPES = 292
CORE_TPLUS_N_TOL = 0.05
CORE_TPLUS_NNZ_TARGET = 1_280_000

# DNp03 identities called out in the M1 spec.
DNP03_BODY = {"L": 10752, "R": 10989}

# flyvis (TuragaLab) pretrained checkpoint used in r3/r4.
FLYVIS_CHECKPOINT = "flow/0000/000"

# 32 flyvis cell types handed off in research_r4/fv4_run.py.
HANDOFF_FLYVIS_TYPES = (
    "T2",
    "T2a",
    "T3",
    "T4a",
    "T4b",
    "T4c",
    "T4d",
    "T5a",
    "T5b",
    "T5c",
    "T5d",
    "Tm1",
    "Tm2",
    "Tm3",
    "Tm4",
    "Tm5a",
    "Tm5b",
    "Tm5c",
    "Tm9",
    "Tm16",
    "Tm20",
    "TmY3",
    "TmY4",
    "TmY5a",
    "TmY10",
    "TmY13",
    "TmY14",
    "TmY15",
    "TmY18",
    "Tm5Y",
    "Tm30",
    "TmY9",
)

# TmY9 is split into a/b in MaleCNS.
FLYVIS_TO_CONNECTOME = {t: (t,) for t in HANDOFF_FLYVIS_TYPES}
FLYVIS_TO_CONNECTOME["TmY9"] = ("TmY9a", "TmY9b")

HANDOFF_CONNECTOME_TYPES = tuple(
    sorted({ct for types in FLYVIS_TO_CONNECTOME.values() for ct in types})
)

# Target DNs for the type-graph walk (research_r4/core_type.py).
TARGET_DN_TYPES = (
    "DNp01",
    "DNp03",
    "DNa02",
    "DNp15",
    "DNHS1",
    "DNp20",
    "DNp22",
    "DNg41",
    "DNa04",
    "DNa15",
    "DNb01",
    "DNa03",
    "DNp06",
    "DNp04",
    "DNp02",
    "DNp11",
)

WING_MN_EXACT = (
    "TTMn",
    "PSI",
    "DLMn a, b",
    "DLMn c-f",
    "DVMn 1a-c",
    "MNwm35",
    "MNwm36",
)

FORCED_TYPES = (
    "HSE",
    "HSN",
    "HSS",
    "LPLC2",
    "LC4",
    "LPLC1",
    "LPi43",
    "LPi34",
)

# Inhibitory sources onto LPLC2 / LC4 during rotation (research_r4/inh_sources.json).
# This is what makes the subgraph "coreTplus" rather than coreT_thr300_K2.
INH_SOURCES_LPLC2 = (
    "TmY5a",
    "PVLP011",
    "Tm5c",
    "LPi21",
    "Li17",
    "Li39",
    "LPi2b",
    "LPi43",
    "LPi12",
    "LPi2c",
    "LPi4b",
    "LPi34",
    "MeVC23",
    "Tlp13",
    "LPT50",
)
INH_SOURCES_LC4 = (
    "Li39",
    "Li28",
    "Am1",
    "MeLo8",
    "Li29",
    "Li17",
    "TmY15",
    "TmY19a",
    "Li25",
    "TmY19b",
    "Li16",
    "Li30",
    "Pm7_Li28",
    "Tm5c",
    "Li26",
)
INH_SOURCE_TYPES = tuple(sorted(set(INH_SOURCES_LPLC2) | set(INH_SOURCES_LC4)))

G_INH_TARGETS = ("LPLC1", "LPLC2", "LPLC4", "LC4")

REQUIRED_EXACT_TYPES = (
    "DNp01",
    "DNp03",
    "DNa02",
    "DNp15",
    "DNHS1",
    "TTMn",
    "PSI",
    *FORCED_TYPES,
)

REQUIRED_PREFIXES = ("DNg02", "DLMn")

# Acceptance thresholds (M1 spec).
T1_DNP01_MIN = 5.0
T1_DNP03_MIN = 3.0
T2_END_FRAC_MAX = 0.1
T3_ROT_FRAC_MAX = 0.25
T3_EXP_FRAC_MAX = 0.25
FRONTAL_FRAC_MIN = 0.3
SHUFFLE_T3_FAIL_MIN = 1.0
BRAIN_MS_MAX = 3.0
FLYVIS_MS_MAX = 8.0


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_data_dir(data_dir: Optional[os.PathLike] = None) -> Path:
    if data_dir is not None:
        return Path(data_dir)
    env = os.environ.get("THREEFLY_DATA")
    if env:
        return Path(env)
    return repo_root() / "data"


def connectome_npz_path(data_dir: Optional[os.PathLike] = None) -> Path:
    return default_data_dir(data_dir) / "full_connectome.npz"


def nt_feather_path(data_dir: Optional[os.PathLike] = None) -> Path:
    return (
        default_data_dir(data_dir)
        / "connectome-raw"
        / "body-neurotransmitters-male-cns-v1.0.feather"
    )


def ann_feather_path(data_dir: Optional[os.PathLike] = None) -> Path:
    return (
        default_data_dir(data_dir)
        / "connectome-raw"
        / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
    )


def cache_dir(data_dir: Optional[os.PathLike] = None) -> Path:
    return default_data_dir(data_dir) / "cache"


def connectome_available(data_dir: Optional[os.PathLike] = None) -> bool:
    return connectome_npz_path(data_dir).is_file()


def flyvis_available() -> bool:
    try:
        import flyvis  # noqa: F401

        return True
    except ImportError:
        return False


def is_inhibitory_nt(name: str) -> bool:
    return str(name).lower().strip() in INHIBITORY_NT


def nt_sign(name: str) -> float:
    return -1.0 if is_inhibitory_nt(name) else 1.0


def configure_threads(scipy_threads: int = 1, torch_threads: int = 1) -> None:
    """Pin BLAS / torch thread counts (CPU-only M1 runtime)."""
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[key] = str(scipy_threads)
    try:
        import torch

        torch.set_num_threads(int(torch_threads))
        if hasattr(torch, "set_num_interop_threads"):
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass
    except ImportError:
        pass


def as_str_array(values: Iterable) -> "object":
    import numpy as np

    return np.asarray(["" if v is None else str(v) for v in values], dtype=object)
