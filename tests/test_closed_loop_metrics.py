"""Closed-loop acceptance metrics: NumPy 2 ptp, per-step walls, heading direction."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from threedfly.control.closed_loop_metrics import (
    PR16_WALLS_PER_100,
    body_azimuth_deg,
    classify_loom,
    coverage,
    direction_correct_frac,
    escape_turned_away,
    geometry_aware_event_ok,
    geometry_aware_frac,
    walls_per_100,
    wrap_heading_delta,
    xy_extent,
)
from threedfly.control.dn_controller import EscapeEvent


ROOT = Path(__file__).resolve().parents[1]


def test_xy_extent_uses_np_ptp_not_ndarray_method():
    traj = np.array([[0.0, -1.0, 1.5], [4.9, 1.5, 1.5], [2.0, 0.2, 1.5]])
    dx, dy = xy_extent(traj)
    assert dx == pytest.approx(4.9)
    assert dy == pytest.approx(2.5)
    assert xy_extent(np.zeros((0, 3))) == (0.0, 0.0)


def test_walls_per_100_normalises_against_pr16_2000():
    assert walls_per_100(81, 2000) == pytest.approx(PR16_WALLS_PER_100)
    assert PR16_WALLS_PER_100 == pytest.approx(4.05)
    # 21 contacts in 400 steps is *worse* than 81/2000, not a win.
    assert walls_per_100(21, 400) > PR16_WALLS_PER_100
    assert walls_per_100(191, 2000) > PR16_WALLS_PER_100


def test_coverage_counts_occupied_bins():
    traj = np.array([[0.0, 0.0, 1.5], [0.1, 0.1, 1.5], [4.9, 0.0, 1.5]])
    cov = coverage(traj)
    assert 0.0 < cov < 0.1


def test_escape_turned_away_uses_heading_not_label():
    # +yaw / increasing heading = left turn. Left threat → heading decrease.
    left_away = [0.0, -0.05, -0.12, -0.20]
    right_away = [0.0, 0.05, 0.12, 0.20]
    assert escape_turned_away(left_away, 0, "L")
    assert not escape_turned_away(left_away, 0, "R")
    assert escape_turned_away(right_away, 0, "R")
    assert not escape_turned_away(right_away, 0, "L")
    # A "L" label with no heading change is not correct (old metric counted ties as L).
    assert not escape_turned_away([0.2, 0.2, 0.2], 0, "L")


def test_direction_correct_frac_uses_ground_truth_threat_not_label_or_yaw():
    # Heading decreases = right turn = away from a LEFT threat.
    headings = [0.10, 0.08, 0.02, -0.05]
    # Event labeled L with yaw=0 — old metric counted the label / yaw fallback.
    events = [
        EscapeEvent(step=0, dnp01=2.2, threat_side="L", yaw=0.0, engineered=False),
        {"step": 0, "dnp01": 2.0, "threat_side": "L", "yaw": -0.8, "engineered": False},
    ]
    assert direction_correct_frac(headings, events, threat_side="L") == pytest.approx(1.0)
    # Same events, but the scenario threat is RIGHT: turning right is INTO the wall.
    assert direction_correct_frac(headings, events, threat_side="R") == pytest.approx(0.0)
    # Flat heading + commanded yaw must not count (no yaw fallback).
    flat = [0.2, 0.2, 0.2]
    yaw_only = [EscapeEvent(step=0, dnp01=2.2, threat_side="L", yaw=-0.8, engineered=False)]
    assert direction_correct_frac(flat, yaw_only, threat_side="L") == pytest.approx(0.0)


def test_wrap_heading_delta_across_pi():
    assert wrap_heading_delta(-3.0, 3.0) == pytest.approx(-6.0 + 2 * np.pi, abs=1e-6)


def test_body_azimuth_left_is_positive():
    assert body_azimuth_deg((0.0, 1.0), (0.0, 0.0), 0.0) == pytest.approx(90.0)
    assert body_azimuth_deg((1.0, 0.0), (0.0, 0.0), 0.0) == pytest.approx(0.0)
    assert body_azimuth_deg((0.0, -1.0), (0.0, 0.0), 0.0) == pytest.approx(-90.0)


def test_classify_loom_uses_30deg_lateral():
    assert classify_loom(29.9) == "frontal"
    assert classify_loom(30.0) == "lateral"
    assert classify_loom(-45.0) == "lateral"
    assert classify_loom(None) == "unknown"


def test_geometry_aware_frontal_must_not_turn_into():
    straight = [0.0, 0.0, 0.0, 0.0]
    into_left = [0.0, 0.10, 0.20, 0.30]
    away_left = [0.0, -0.10, -0.20, -0.30]
    assert geometry_aware_event_ok(straight, 0, "L", azimuth_onset_deg=10.0)
    assert not geometry_aware_event_ok(into_left, 0, "L", azimuth_onset_deg=10.0)
    assert geometry_aware_event_ok(away_left, 0, "L", azimuth_onset_deg=45.0)
    assert not geometry_aware_event_ok(straight, 0, "L", azimuth_onset_deg=45.0)


def test_geometry_aware_frac_frontal_brake_without_escape():
    assert geometry_aware_frac([], [], "L", 15.0, escaped=False, braked=True) == pytest.approx(1.0)
    assert geometry_aware_frac([], [], "L", 45.0, escaped=False, braked=True) == pytest.approx(0.0)
    assert geometry_aware_frac([], [], "L", 45.0, escaped=False, braked=False) == pytest.approx(0.0)


def test_acceptance_script_metric_code_runs_on_numpy2():
    """Execute the acceptance script's metric helpers (ndarray.ptp is gone on NumPy 2)."""
    path = ROOT / "scripts" / "closed_loop_acceptance.py"
    spec = importlib.util.spec_from_file_location("closed_loop_acceptance_metrics", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    traj = np.array([[0.0, 0.0, 1.5], [4.9, -1.2, 1.5]], dtype=float)
    dx, dy = mod.xy_extent(traj)
    assert dx == pytest.approx(4.9)
    assert dy == pytest.approx(1.2)
    assert mod.walls_per_100(81, 2000) == pytest.approx(4.05)
    assert mod.beats_pr16_walls(21, 2000)
    assert not mod.beats_pr16_walls(21, 400)
    assert not mod.beats_pr16_walls(191, 2000)
    pose_l = mod.lateral_loom_pose("L")
    pose_r = mod.lateral_loom_pose("R")
    # Mostly +X, with a component toward the side wall (distance decreases).
    assert 0.2 < pose_l["yaw"] < 0.9
    assert -0.9 < pose_r["yaw"] < -0.2
    assert pose_l["pose"][1] > 2.5
    assert pose_r["pose"][1] < -2.5
    assert pose_l["vel"][1] > 0.0
    assert pose_r["vel"][1] < 0.0
    # Old scenario faced +Y (π/2) and drove into the frontal wall.
    assert abs(pose_l["yaw"] - np.pi / 2) > 0.7
    yt = mod._yaw_term_summary(
        [
            {"dna02": 0.2, "dnp15": 0.0, "hse": -0.1, "dnp03": -0.4, "vpn_loom": -0.2, "eye": 0.0, "escape_kick": 0.0, "near_wall": 1.0},
            {"dna02": 0.0, "dnp15": 0.0, "hse": 0.0, "dnp03": 0.0, "vpn_loom": 0.0, "eye": 0.8, "escape_kick": 0.0, "near_wall": 0.0},
        ]
    )
    assert yt["near_wall_steps"] == 1
    assert yt["eye_share_near_wall"] == pytest.approx(0.0)
    assert yt["brain_share_near_wall"] > 0.9
    open_p = mod.open_space_pose()
    assert open_p["yaw"] == pytest.approx(0.0)
    assert open_p["max_steps"] >= 300
    assert open_p["pose"][0] < -2.5 and abs(open_p["pose"][1]) < 0.1
    assert open_p["vel"][0] >= 0.35
    head = mod.headon_wall_pose()
    assert head["pose"][0] > 1.5 and abs(head["pose"][1]) < 0.1
    assert head["vel"][0] > 0.3
    loom_l = mod.lateral_loom_object("L")
    loom_r = mod.lateral_loom_object("R")
    assert loom_l["loom_start"][1] > 1.5
    assert loom_r["loom_start"][1] < -1.5
    assert loom_l["loom_vel"][1] < -0.5
    assert loom_r["loom_vel"][1] > 0.5
    vars_l = {v["tag"]: v for v in mod.loom_variants("L")}
    vars_r = {v["tag"]: v for v in mod.loom_variants("R")}
    assert set(vars_l) == {"far1.4x", "slow0.7x", "offset_x+0.8", "stride_hold"}
    assert vars_l["far1.4x"]["loom_start"][1] > loom_l["loom_start"][1]
    assert vars_r["offset_x+0.8"]["loom_start"][0] > loom_r["loom_start"][0]
    assert vars_l["stride_hold"]["stride"] == 2
    vd = mod._variant_direction(
        {
            "side_loom_left": {"escape_direction_correct": 1.0},
            "loomR_far1.4x": {"escape_direction_correct": 1.0},
            "loomR_slow0.7x": {"escape_direction_correct": 0.0},
            "room": {"escape_direction_correct": None},
        }
    )
    assert vd["n"] == 3
    assert vd["mean"] == pytest.approx(2.0 / 3.0)
    assert vd["pass"] is False
    vg = mod._variant_geometry_aware(
        {
            "side_loom_left": {
                "escape_direction_correct": 1.0,
                "geometry_aware_correct": 1.0,
                "loom_class": "lateral",
                "azimuth_onset_deg": 45.0,
                "brain_escapes": 1,
            },
            "loomR_far1.4x": {
                "escape_direction_correct": 0.0,
                "geometry_aware_correct": 1.0,
                "loom_class": "frontal",
                "azimuth_onset_deg": 12.0,
                "brain_escapes": 1,
            },
            "room": {"geometry_aware_correct": None},
        }
    )
    assert vg["n"] == 2
    assert vg["mean"] == pytest.approx(1.0)
    assert vg["pass"] is True
