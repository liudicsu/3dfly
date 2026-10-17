"""Offline T1–T7 metrics matching research_r4/e2e4.py + M1 spec."""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional

import numpy as np

from threedfly.brain.constants import (
    FRONTAL_FRAC_MIN,
    SHUFFLE_T3_FAIL_MIN,
    T1_DNP01_MIN,
    T1_DNP03_MIN,
    T2_END_FRAC_MAX,
    T3_EXP_FRAC_MAX,
    T3_ROT_FRAC_MAX,
)

TRACE_KEYS = (
    "DNp01|L",
    "DNp01|R",
    "DNp03|L",
    "DNp03|R",
    "HSE|L",
    "HSE|R",
    "DNa02|L",
    "DNa02|R",
    "DNp15|L",
    "DNp15|R",
    "LPLC2|L",
    "LPLC2|R",
    "LC4|L",
    "LC4|R",
)


def peak_delta(trace: Iterable[float], base_n: int = 30, stim_n: int = 100) -> float:
    v = np.asarray(list(trace), dtype=float)
    return float(v[base_n : base_n + stim_n].max() - v[:base_n].mean())


def end_delta(trace: Iterable[float], base_n: int = 30, end_n: int = 20) -> float:
    v = np.asarray(list(trace), dtype=float)
    return float(v[-end_n:].mean() - v[:base_n].mean())


def yaw_lr(traces: Mapping[str, Iterable[float]], key: str, base_n: int = 30, stim_n: int = 100) -> float:
    """Stimulus-window Δ(L) − Δ(R). CW should be < 0 for HSE and DNa02."""
    L = np.asarray(list(traces[f"{key}|L"]), dtype=float)
    R = np.asarray(list(traces[f"{key}|R"]), dtype=float)
    dL = L[base_n : base_n + stim_n].mean() - L[:base_n].mean()
    dR = R[base_n : base_n + stim_n].mean() - R[:base_n].mean()
    return float(dL - dR)


def summarize_traces(
    scenarios: Mapping[str, Mapping[str, object]],
    nceil_rest: int = 0,
) -> Dict[str, object]:
    """``scenarios[name]['trace'][key]`` is a list starting 30 steps before stimulus."""

    def tr(sc: str, k: str) -> List[float]:
        return list(scenarios[sc]["trace"][k])

    def pk(sc: str, k: str) -> float:
        return peak_delta(tr(sc, k))

    ipsi_p01 = (pk("loomL", "DNp01|L") + pk("loomR", "DNp01|R")) / 2.0
    contra_p01 = (pk("loomL", "DNp01|R") + pk("loomR", "DNp01|L")) / 2.0
    ipsi_p03 = (pk("loomL", "DNp03|L") + pk("loomR", "DNp03|R")) / 2.0
    contra_p03 = (pk("loomL", "DNp03|R") + pk("loomR", "DNp03|L")) / 2.0
    rot = max(pk(c, k) for c in ("CW", "CCW") for k in ("DNp01|L", "DNp01|R"))
    exp = max(pk("expand", k) for k in ("DNp01|L", "DNp01|R"))
    loomF = max(pk("loomF", "DNp01|L"), pk("loomF", "DNp01|R"))
    t2 = end_delta(tr("loomL", "DNp01|L")) / max(pk("loomL", "DNp01|L"), 1e-3)

    S: Dict[str, object] = {
        "loom_ipsi_DNp01": round(ipsi_p01, 3),
        "loom_contra_DNp01": round(contra_p01, 3),
        "T1_ratio_DNp01": round(ipsi_p01 / max(contra_p01, 1e-3), 1),
        "T1_ratio_DNp03": round(ipsi_p03 / max(contra_p03, 1e-3), 1),
        "rot_DNp01_max": round(rot, 3),
        "exp_DNp01_max": round(exp, 3),
        "T3_rot_frac": round(rot / max(ipsi_p01, 1e-3), 3),
        "T3_exp_frac": round(exp / max(ipsi_p01, 1e-3), 3),
        "T2_DNp01_end_frac": round(t2, 3),
        "nceil_rest": int(nceil_rest),
        "loomF_DNp01": (round(pk("loomF", "DNp01|L"), 3), round(pk("loomF", "DNp01|R"), 3)),
        "loomF_frac_of_lateral": round(loomF / max(ipsi_p01, 1e-3), 3),
    }
    if "HSE|L" in scenarios.get("CW", {}).get("trace", {}):
        for name in ("HSE", "DNa02", "DNp15"):
            if f"{name}|L" in scenarios["CW"]["trace"]:
                S[f"yaw_{name}_CW_CCW"] = (
                    round(yaw_lr(scenarios["CW"]["trace"], name), 2),
                    round(yaw_lr(scenarios["CCW"]["trace"], name), 2),
                )
    return S


def verdict(summary: Mapping[str, object], shuffle_summary: Optional[Mapping[str, object]] = None) -> Dict[str, bool]:
    yaw = summary.get("yaw_HSE_CW_CCW")
    yaw_dna = summary.get("yaw_DNa02_CW_CCW")
    yaw_ok = True
    if yaw:
        yaw_ok = yaw_ok and yaw[0] < 0 and yaw[1] > 0
    if yaw_dna:
        yaw_ok = yaw_ok and yaw_dna[0] < 0 and yaw_dna[1] > 0
    out = {
        "T1_DNp01": float(summary["T1_ratio_DNp01"]) >= T1_DNP01_MIN,
        "T1_DNp03": float(summary["T1_ratio_DNp03"]) >= T1_DNP03_MIN,
        "T2_no_latch": float(summary["T2_DNp01_end_frac"]) <= T2_END_FRAC_MAX
        and int(summary.get("nceil_rest", 0)) == 0,
        "T3_rotation": float(summary["T3_rot_frac"]) <= T3_ROT_FRAC_MAX,
        "T3b_expand": float(summary["T3_exp_frac"]) <= T3_EXP_FRAC_MAX,
        "yaw_sign": yaw_ok,
        "frontal_loom": float(summary["loomF_frac_of_lateral"]) >= FRONTAL_FRAC_MIN,
    }
    if shuffle_summary is not None:
        out["shuffle_fails_T3"] = float(shuffle_summary["T3_rot_frac"]) > SHUFFLE_T3_FAIL_MIN
    out["all"] = all(out.values())
    return out
