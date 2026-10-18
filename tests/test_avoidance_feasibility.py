"""Kinematic feasibility model and response-window behavioural score."""

from __future__ import annotations

import numpy as np

from threedfly.control.closed_loop_metrics import (
    LOOM_RADIUS_M,
    MISS_SURFACE_M,
    avoidance_feasibility,
    behavioural_response,
    max_lateral_accel_mps2,
    max_lateral_displacement_m,
    net_heading_delta,
    wander_yaw_threshold,
)

A_LAT = max_lateral_accel_mps2(0.05, 0.05, 0.001696)
W_MAX = np.radians(2011.0)


def _feas(lv, lead_steps, az=45.0, v0=0.9):
    return avoidance_feasibility(
        decision_step=100 - lead_steps,
        ca_step=100,
        dt=0.01,
        approach_speed_mps=LOOM_RADIUS_M / lv,
        azimuth_decision_deg=az,
        v0_mps=v0,
        a_lat_mps2=A_LAT,
        yaw_rate_max_rad_s=W_MAX,
    )


def test_lateral_accel_upper_bound_includes_spare_lift():
    a = max_lateral_accel_mps2(0.05, 0.05, 0.001696)
    assert 29.0 < 0.05 / 0.001696 < a < 2.0 * 0.05 / 0.001696


def test_displacement_monotone_and_turn_delays_accel():
    s0 = max_lateral_displacement_m(0.1, v0_mps=0.0, a_lat_mps2=50.0, turn_rad=0.0, yaw_rate_max_rad_s=W_MAX)
    s1 = max_lateral_displacement_m(0.1, v0_mps=0.0, a_lat_mps2=50.0, turn_rad=np.pi / 2, yaw_rate_max_rad_s=W_MAX)
    assert s0 == np.float64(0.5 * 50.0 * 0.01) or abs(s0 - 0.25) < 1e-9
    assert s1 < s0
    assert max_lateral_displacement_m(0.2, v0_mps=0.0, a_lat_mps2=50.0, turn_rad=0.0, yaw_rate_max_rad_s=W_MAX) > s0


def test_fast_loom_at_gf_window_is_infeasible_slow_is_feasible():
    need = LOOM_RADIUS_M + MISS_SURFACE_M
    fast = _feas(0.04, 8)  # 80 ms before closest approach
    assert fast["feasible"] is False and fast["label"] == "infeasible"
    assert fast["max_lateral_m"] < need
    assert fast["feasible_at_gf50"] is False  # 14.5 m/s: 50° is ~86 ms out
    slow = _feas(0.08, 30)  # 300 ms lead, 7 m/s
    assert slow["feasible"] is True
    assert slow["feasible_at_gf50"] is True
    # Faster looms need the decision when the disc is smaller.
    assert _feas(0.02, 5)["latest_decision_angle_deg"] < _feas(0.08, 5)["latest_decision_angle_deg"]


def test_no_decision_is_labelled():
    f = avoidance_feasibility(
        decision_step=None, ca_step=50, dt=0.01, approach_speed_mps=14.5,
        azimuth_decision_deg=None, v0_mps=0.9, a_lat_mps2=A_LAT, yaw_rate_max_rad_s=W_MAX,
    )
    assert f["label"] == "no_decision" and f["feasible"] is False


def test_net_heading_is_unwrapped():
    # 300° right turn from a left loom: away, not a 60° left turn.
    h = np.radians(np.linspace(0.0, -300.0, 31))
    h = np.arctan2(np.sin(h), np.cos(h))
    assert net_heading_delta(h) < -np.radians(250.0)


def test_behavioural_window_success_and_late_fail():
    dt = 0.01
    h = np.zeros(60)
    # Saccade away (right) from a left threat starting at step 20.
    h[21:] = -np.radians(np.minimum(np.arange(1, 40) * 20.0, 120.0))
    ok = behavioural_response(headings=h, dt=dt, ca_step=40, gf_step=None, threat_side="L", wander_rad_s=5.0)
    assert ok["ok"] and ok["decision_source"] == "saccade" and ok["away"]
    late = behavioural_response(headings=h, dt=dt, ca_step=40, gf_step=45, threat_side="L", wander_rad_s=100.0)
    assert late["decision_step"] == 45 and not late["on_time"] and not late["ok"]
    toward = behavioural_response(headings=-h, dt=dt, ca_step=40, gf_step=None, threat_side="L", wander_rad_s=5.0)
    assert not toward["ok"] and toward["toward"]


def test_gf_fire_counts_as_decision_but_needs_turn_above_wander():
    h = np.zeros(60)
    b = behavioural_response(headings=h, dt=0.01, ca_step=40, gf_step=30, threat_side="R", wander_rad_s=1.0)
    assert b["decision_source"] == "gf" and b["on_time"] and not b["above_wander"] and not b["ok"]


def test_wander_threshold_pooled_and_peak():
    rng = np.random.default_rng(0)
    runs = [np.cumsum(rng.normal(0, 0.01, 200)) for _ in range(6)]
    w = wander_yaw_threshold(runs, [0.02] * 6)
    assert w["n_runs"] == 6
    assert w["threshold_rad_s"] > w["pooled_mean"]
    assert w["peak_threshold_rad_s"] >= w["peak_mean"]
