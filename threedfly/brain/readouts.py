"""Descending-neuron readouts. No hand-coded escape/avoidance rules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from threedfly.brain.constants import DNP03_BODY


@dataclass
class DNReadout:
    """Per-side DN means. DNg02 is a bilateral amplitude (mean L+R), not L−R yaw."""

    dnp01_L: float = 0.0
    dnp01_R: float = 0.0
    dnp03_L: float = 0.0
    dnp03_R: float = 0.0
    dna02_L: float = 0.0
    dna02_R: float = 0.0
    dnp15_L: float = 0.0
    dnp15_R: float = 0.0
    hse_L: float = 0.0
    hse_R: float = 0.0
    dng02_L: float = 0.0
    dng02_R: float = 0.0
    dng02_amp: float = 0.0  # mean(L, R) wing-amplitude proxy; NO L−R yaw term
    lplc2_L: float = 0.0
    lplc2_R: float = 0.0
    # Per-eye hex mean. Room walls are bright filled fields; DNp03 tracks
    # compact dark looms (hex-swap) but not these, so laterality uses eyes.
    eye_L: float = 0.0
    eye_R: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return self.__dict__.copy()


@dataclass
class ReadoutIndex:
    groups: Dict[str, np.ndarray] = field(default_factory=dict)

    def __contains__(self, key: str) -> bool:
        return key in self.groups

    def __getitem__(self, key: str) -> np.ndarray:
        return self.groups[key]

    def size(self, key: str) -> int:
        return int(len(self.groups.get(key, ())))


def _mean(rates: np.ndarray, idx: np.ndarray) -> float:
    if idx is None or len(idx) == 0:
        return 0.0
    return float(np.mean(rates[idx]))


def _select(typ: np.ndarray, side: np.ndarray, name: str, s: str, prefix: bool = False) -> np.ndarray:
    if prefix:
        m = np.array([str(t).startswith(name) for t in typ])
    else:
        m = typ == name
    m &= side == s
    return np.flatnonzero(m)


def build_readout_index(
    types,
    sides,
    body_ids: Optional[np.ndarray] = None,
) -> ReadoutIndex:
    """Index DNp01, DNp03, DNa02, DNp15/DNHS1, HSE, DNg02 prefix, per side."""
    typ = np.asarray(types, dtype=object)
    side = np.asarray(sides, dtype=object)
    g: Dict[str, np.ndarray] = {}
    for s in "LR":
        g[f"DNp01|{s}"] = _select(typ, side, "DNp01", s)
        dnp03 = _select(typ, side, "DNp03", s)
        if dnp03.size == 0 and body_ids is not None:
            bid = DNP03_BODY.get(s)
            if bid is not None:
                hit = np.flatnonzero(np.asarray(body_ids) == int(bid))
                if hit.size:
                    dnp03 = hit
        g[f"DNp03|{s}"] = dnp03
        g[f"DNa02|{s}"] = _select(typ, side, "DNa02", s)
        dnp15 = _select(typ, side, "DNp15", s)
        if dnp15.size == 0:
            dnp15 = _select(typ, side, "DNHS1", s)
        g[f"DNp15|{s}"] = dnp15
        g[f"HSE|{s}"] = _select(typ, side, "HSE", s)
        g[f"DNg02|{s}"] = _select(typ, side, "DNg02", s, prefix=True)
        g[f"LPLC2|{s}"] = _select(typ, side, "LPLC2", s)
        g[f"LC4|{s}"] = _select(typ, side, "LC4", s)
        g[f"LPi43|{s}"] = _select(typ, side, "LPi43", s)
    return ReadoutIndex(groups=g)


def read_dns(rates: np.ndarray, index: ReadoutIndex) -> DNReadout:
    """Population means. DNg02 amplitude is mean(L, R); there is no DNg02 L−R yaw."""
    L = _mean(rates, index.groups.get("DNg02|L", np.array([], dtype=int)))
    R = _mean(rates, index.groups.get("DNg02|R", np.array([], dtype=int)))
    return DNReadout(
        dnp01_L=_mean(rates, index.groups.get("DNp01|L", np.zeros(0, dtype=int))),
        dnp01_R=_mean(rates, index.groups.get("DNp01|R", np.zeros(0, dtype=int))),
        dnp03_L=_mean(rates, index.groups.get("DNp03|L", np.zeros(0, dtype=int))),
        dnp03_R=_mean(rates, index.groups.get("DNp03|R", np.zeros(0, dtype=int))),
        dna02_L=_mean(rates, index.groups.get("DNa02|L", np.zeros(0, dtype=int))),
        dna02_R=_mean(rates, index.groups.get("DNa02|R", np.zeros(0, dtype=int))),
        dnp15_L=_mean(rates, index.groups.get("DNp15|L", np.zeros(0, dtype=int))),
        dnp15_R=_mean(rates, index.groups.get("DNp15|R", np.zeros(0, dtype=int))),
        hse_L=_mean(rates, index.groups.get("HSE|L", np.zeros(0, dtype=int))),
        hse_R=_mean(rates, index.groups.get("HSE|R", np.zeros(0, dtype=int))),
        dng02_L=L,
        dng02_R=R,
        dng02_amp=0.5 * (L + R),
        lplc2_L=_mean(rates, index.groups.get("LPLC2|L", np.zeros(0, dtype=int))),
        lplc2_R=_mean(rates, index.groups.get("LPLC2|R", np.zeros(0, dtype=int))),
    )


def group_sizes(index: ReadoutIndex) -> Dict[str, int]:
    return {k: int(len(v)) for k, v in index.groups.items()}
