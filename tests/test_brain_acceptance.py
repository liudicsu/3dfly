"""T1–T7 metric helpers (synthetic traces; no connectome / flyvis)."""

from __future__ import annotations

import numpy as np
import pytest

from threedfly.brain.acceptance import peak_delta, summarize_traces, verdict, yaw_lr


def _trace(base, peak, end=None, n=160):
    end = peak if end is None else end
    v = np.full(n, float(base))
    v[30:80] = float(peak)
    v[-20:] = float(end)
    return v.tolist()


def test_peak_and_yaw_helpers():
    v = _trace(1.0, 6.0)
    assert peak_delta(v) == pytest.approx(5.0)
    traces = {"HSE|L": _trace(0, -2), "HSE|R": _trace(0, 4)}
    # ΔL=-2, ΔR=4 → L−R = -6 (CW sign)
    assert yaw_lr(traces, "HSE") < 0


def test_summarize_and_verdict_pass_pattern():
    def sc(p01L, p01R, p03L=2.0, p03R=0.1, hse=None):
        tr = {
            "DNp01|L": _trace(0, p01L, end=0.0),
            "DNp01|R": _trace(0, p01R, end=0.0),
            "DNp03|L": _trace(0, p03L),
            "DNp03|R": _trace(0, p03R),
        }
        if hse is not None:
            tr["HSE|L"] = _trace(0, hse[0])
            tr["HSE|R"] = _trace(0, hse[1])
            tr["DNa02|L"] = _trace(0, hse[0] / 10)
            tr["DNa02|R"] = _trace(0, hse[1] / 10)
        return {"trace": tr}

    scenarios = {
        "loomL": sc(10, 0.01, p03L=2.0, p03R=0.1),
        "loomR": sc(0.01, 10, p03L=0.1, p03R=2.0),
        "CW": sc(1.0, 1.0, hse=(-5, 5)),
        "CCW": sc(1.0, 1.0, hse=(5, -5)),
        "expand": sc(1.2, 1.2),
        "loomF": sc(5.0, 5.0),
        "gray": sc(0, 0),
    }
    S = summarize_traces(scenarios, nceil_rest=0)
    assert S["T1_ratio_DNp01"] >= 5
    assert S["T3_rot_frac"] <= 0.25
    V = verdict(S, {"T3_rot_frac": 2.2})
    assert V["T1_DNp01"]
    assert V["T3_rotation"]
    assert V["shuffle_fails_T3"]
    assert V["yaw_sign"]
    assert V["all"]


def test_verdict_fails_when_rotation_triggers_escape():
    def sc(p):
        return {
            "trace": {
                "DNp01|L": _trace(0, p),
                "DNp01|R": _trace(0, p),
                "DNp03|L": _trace(0, p),
                "DNp03|R": _trace(0, p),
            }
        }

    scenarios = {
        "loomL": sc(2),
        "loomR": sc(2),
        "CW": sc(3),
        "CCW": sc(3),
        "expand": sc(3),
        "loomF": sc(0.1),
        "gray": sc(0),
    }
    S = summarize_traces(scenarios, nceil_rest=0)
    V = verdict(S)
    assert V["T3_rotation"] is False
    assert V["all"] is False
