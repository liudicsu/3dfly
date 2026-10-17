"""Closed-loop acceptance metrics: NumPy 2 ptp, per-step walls, heading direction."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from threedfly.control.closed_loop_metrics import (
    PR16_WALLS_PER_100,
    coverage,
    direction_correct_frac,
    escape_turned_away,
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


def test_direction_correct_frac_heading_and_dicts():
    headings = [0.10, 0.08, 0.02, -0.05]
    events = [
        EscapeEvent(step=0, dnp01=2.2, threat_side="L", yaw=0.0, engineered=False),
        {"step": 0, "dnp01": 2.0, "threat_side": "L", "yaw": 0.0, "engineered": False},
    ]
    assert direction_correct_frac(headings, events[:1]) == pytest.approx(1.0)
    assert direction_correct_frac(headings, events[1:]) == pytest.approx(1.0)
    wrong = [EscapeEvent(step=0, dnp01=2.2, threat_side="L", yaw=0.4, engineered=False)]
    # Positive yaw is a left turn — toward a left threat, not away.
    # Headings still decrease, so heading wins; use a heading that increases.
    assert direction_correct_frac([0.0, 0.2, 0.35], wrong) == pytest.approx(0.0)


def test_wrap_heading_delta_across_pi():
    assert wrap_heading_delta(-3.0, 3.0) == pytest.approx(-6.0 + 2 * np.pi, abs=1e-6)


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
    assert 0.2 < pose_l["yaw"] < 0.8
    assert -0.8 < pose_r["yaw"] < -0.2
    assert pose_l["pose"][1] > 2.5
    assert pose_r["pose"][1] < -2.5
    assert pose_l["vel"][1] > 0.0
    assert pose_r["vel"][1] < 0.0
    # Old scenario faced +Y (π/2) and drove into the frontal wall.
    assert abs(pose_l["yaw"] - np.pi / 2) > 0.7
