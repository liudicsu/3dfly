"""GF readout: score math, fitter, and recompute of the frozen calibration."""

from __future__ import annotations

import numpy as np
import pytest

from threedfly.control.flight_params import FlightParams
from threedfly.control.gf_readout import (
    CANONICAL,
    GF_CITATION,
    GF_TARGET_DEG,
    GF_WINDOW_HI_DEG,
    GF_WINDOW_LO_DEG,
    TRACES_PATH,
    first_fire_angle,
    fit_gf_readout,
    load_canonical_traces,
    pd_score,
    rise_velocity,
)


def test_rise_velocity_uses_physical_window_not_one_frame():
    dt = 0.01
    hist = [0.0, 0.10]  # 10 Hz/s if one-frame
    # 40 ms window needs 4 steps; with only 1 interval it falls back.
    assert abs(rise_velocity(hist, dt, 0.04) - 10.0) < 1e-9
    hist = [0.0, 0.01, 0.02, 0.03, 0.04]  # 1 Hz/s over 40 ms
    assert abs(rise_velocity(hist, dt, 0.04) - 1.0) < 1e-9
    assert rise_velocity([0.2], dt, 0.04) == 0.0


def test_pd_score_is_rise_plus_k_times_vel():
    assert pd_score(0.20, 1.5, 0.04) == pytest.approx(0.26)


def _synthetic_canonical(dt=0.01):
    """Three identical accelerating rises that cross 0.20 at 50° when k=0."""
    theta = np.linspace(10.0, 90.0, 81)
    # rise grows so rise(50°)=0.20, rise(40°)=0.14, rise(60°)=0.26
    rise = 0.20 * (theta / 50.0)
    traces = {}
    for spec in CANONICAL:
        traces[spec["name"]] = {
            "theta_deg": [float(x) for x in theta],
            "rise": [float(x) for x in rise],
            "dt": dt,
        }
    return traces


def test_fit_picks_pd_near_target_on_synthetic_canonical():
    traces = _synthetic_canonical()
    fit = fit_gf_readout(traces, dt=0.01)
    assert fit.n_in_window == 3
    assert fit.form in ("pd", "integrate")
    for name, ang in fit.fire_angles_deg.items():
        assert GF_WINDOW_LO_DEG <= ang <= GF_WINDOW_HI_DEG, (name, ang)
    rms = fit.rms_to_target
    assert rms <= 8.0
    # k=0, threshold≈0.20 is a perfect PD solution on this set.
    if fit.form == "pd":
        assert abs(fit.k) <= 0.08
        assert 0.12 <= fit.threshold <= 0.28


def test_first_fire_angle_none_when_score_never_crosses():
    theta = [10.0, 30.0, 50.0, 70.0]
    rise = [0.0, 0.02, 0.03, 0.04]
    assert first_fire_angle(theta, rise, 0.01, "pd", 0.0, 0.20) is None


def test_frozen_params_match_refit_on_canonical_traces():
    """Recompute the official fit from the committed traces; must match freeze."""
    if not TRACES_PATH.is_file():
        pytest.skip("gf_readout_traces.json not written yet")
    traces = load_canonical_traces()
    dt = float(next(iter(traces.values())).get("dt", 0.01))
    fit = fit_gf_readout(traces, dt=dt)
    p = FlightParams()
    assert fit.form == p.gf_form
    assert fit.k == pytest.approx(p.gf_k, abs=1e-9)
    assert fit.threshold == pytest.approx(p.gf_threshold, abs=1e-9)
    assert fit.vel_window_s == pytest.approx(p.gf_vel_window_s, abs=1e-9)
    # Binocular hex loom is stronger; the frozen fit keeps L/R in 40–60°
    # and both-eyes just under 40°. No (k, threshold) puts all three in
    # the window — that is the calibration result, not a retune.
    assert fit.n_in_window == 2
    assert GF_WINDOW_LO_DEG <= fit.fire_angles_deg["L_az30"] <= GF_WINDOW_HI_DEG
    assert GF_WINDOW_LO_DEG <= fit.fire_angles_deg["R_az-30"] <= GF_WINDOW_HI_DEG
    assert 35.0 <= fit.fire_angles_deg["both_x0_0"] < GF_WINDOW_LO_DEG
    assert abs(np.mean(list(fit.fire_angles_deg.values())) - GF_TARGET_DEG) <= 12.0
    assert "von Reyn" in GF_CITATION
    assert "calibrate_gf_readout" in p.notes["escape"]
