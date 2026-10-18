"""Closed-loop acceptance metrics: NumPy 2 ptp, per-step walls, heading direction."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from threedfly.control.closed_loop_metrics import (
    PR16_WALLS_PER_100,
    MISS_SURFACE_M,
    body_azimuth_deg,
    peak_yaw_rate_rad_s,
    classify_loom,
    closest_approach_step,
    collision_course_object,
    coverage,
    direction_correct_frac,
    dnp01_baseline_shows_loom,
    flight_avoidance_ok,
    heading_away_from_side,
    min_center_distance_m,
    net_heading_delta,
    passed_far_side,
    yaw_term_away_integrals,
    escape_turned_away,
    fire_minus_closest,
    geometry_aware_event_ok,
    geometry_aware_frac,
    object_behind_fly,
    onset_azimuth_band,
    visible_approaching_ok,
    wall_ttc_s,
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


def test_onset_azimuth_bands_and_collision_course():
    assert onset_azimuth_band(12.0) == "frontal"
    assert onset_azimuth_band(20.0) == "frontolat"
    assert onset_azimuth_band(45.0) == "frontolat"
    assert onset_azimuth_band(70.0) == "frontolat"
    assert onset_azimuth_band(90.0) == "eccentric"
    spec = collision_course_object(35.0, 2.4, 0.04)
    assert spec["threat_side"] == "L"
    assert spec["loom_start"][1] > 0
    # Aimed at the origin: velocity is anti-parallel to the start vector.
    start = np.array(spec["loom_start"][:2])
    vel = np.array(spec["loom_vel"][:2])
    assert float(np.dot(start, vel)) < 0
    spec_r = collision_course_object(-55.0, 2.4, 0.08)
    assert spec_r["threat_side"] == "R"
    assert spec_r["loom_start"][1] < 0


def test_wall_ttc_yaw220_from_west_is_minus_x_wall():
    rec = wall_ttc_s((-2.0, 0.0, 1.5), 2.20, 0.40)
    assert rec["wall"] == "-X"
    assert rec["distance_m"] > 4.5
    assert rec["ttc_s"] > 8.0
    assert rec["expansion_rate_1_per_s"] == pytest.approx(0.40 / rec["distance_m"])


def test_escape_turned_into_is_opposite_of_away():
    into_left = [0.0, 0.08, 0.16, 0.24]
    away_left = [0.0, -0.08, -0.16, -0.24]
    from threedfly.control.closed_loop_metrics import escape_turned_into

    assert escape_turned_into(into_left, 0, "L")
    assert not escape_turned_into(away_left, 0, "L")
    assert not escape_turned_into(into_left, 0, "R")


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


def test_closest_approach_and_visible_approaching_gate():
    dists = [3.0, 2.0, 1.2, 0.8, 0.9, 1.5]
    ca = closest_approach_step(dists)
    assert ca == 3
    assert fire_minus_closest(3, ca) == 0
    assert fire_minus_closest(5, ca) == 2
    assert visible_approaching_ok(2, ca, visible=True)
    assert visible_approaching_ok(3, ca, visible=True)
    assert not visible_approaching_ok(4, ca, visible=True)
    assert not visible_approaching_ok(2, ca, visible=False)
    assert not visible_approaching_ok(None, ca, visible=True)
    assert closest_approach_step([]) is None
    assert closest_approach_step([np.nan, np.nan]) is None


def test_object_behind_fly_uses_body_x_not_world_x():
    # Object at world x=-1, fly at origin heading +X → behind.
    assert object_behind_fly((-1.0, 0.0, 1.5), (0.0, 0.0, 1.5), 0.0)
    # Same world x, fly facing −X → object is in front.
    assert not object_behind_fly((-1.0, 0.0, 1.5), (0.0, 0.0, 1.5), np.pi)
    # Off-origin start: world x=+3 is behind a fly at x=+4 heading +X.
    assert object_behind_fly((3.0, 1.0, 1.5), (4.0, 1.0, 1.5), 0.0)


def test_geometry_aware_frac_frontal_brake_without_escape():
    assert geometry_aware_frac([], [], "L", 15.0, escaped=False, braked=True) == pytest.approx(1.0)
    assert geometry_aware_frac([], [], "L", 45.0, escaped=False, braked=True) == pytest.approx(0.0)
    assert geometry_aware_frac([], [], "L", 45.0, escaped=False, braked=False) == pytest.approx(0.0)


def test_acceptance_does_not_teleport_or_reuse_last_visible_geo():
    src = (ROOT / "scripts" / "closed_loop_acceptance.py").read_text()
    assert "last_visible_geo" not in src
    assert "Do not teleport/hide" in src
    assert "fire_minus_closest" in src
    assert "visible_approaching_ok" in src


def test_noobj_heading_control_requires_zero_fires():
    path = ROOT / "scripts" / "closed_loop_acceptance.py"
    spec = importlib.util.spec_from_file_location("cla_noobj", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    summary = mod._noobj_heading_summary(
        {
            "ctrl_noobj_yaw0": {"brain_escapes": 0, "would_fire": 0, "dnp01_rise_max": 0.1, "steps": 220},
            "ctrl_noobj_yaw1": {"brain_escapes": 0, "would_fire": 0, "dnp01_rise_max": 0.1, "steps": 220},
            "ctrl_noobj_yaw2": {"brain_escapes": 0, "would_fire": 0, "dnp01_rise_max": 0.1, "steps": 220},
            "ctrl_noobj_yaw3": {"brain_escapes": 0, "would_fire": 0, "dnp01_rise_max": 0.1, "steps": 220},
            "room": {"brain_escapes": 4},
        }
    )
    assert summary["n"] == 4
    assert summary["fires"] == 0
    assert summary["pass"] is True
    open6 = {f"ctrl_noobj_h{i}": {"brain_escapes": 0, "would_fire": 0, "dnp01_rise_max": 0.1, "steps": 220} for i in range(6)}
    open6["ctrl_wallloom_yaw220"] = {"brain_escapes": 1, "would_fire": 1, "dnp01_rise_max": 0.4, "steps": 220}
    s6 = mod._noobj_heading_summary(open6)
    assert s6["n"] == 6
    assert s6["fires"] == 0
    assert s6["pass"] is True
    fail = mod._noobj_heading_summary(
        {"ctrl_noobj_yaw0": {"brain_escapes": 1, "would_fire": 1, "dnp01_rise_max": 0.3, "steps": 220}}
    )
    assert fail["pass"] is False
    room = {"escape_events": [{"threat_side": "L"}, {"threat_side": "L"}, {"threat_side": "L"}]}
    assert mod._escape_one_sided(room) is True
    mixed = {"escape_events": [{"threat_side": "L"}, {"threat_side": "R"}, {"threat_side": "L"}]}
    assert mod._escape_one_sided(mixed) is False


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
    fl = {c["tag"]: c for c in mod.frontolateral_gate_cases()}
    assert any(c["az_design_deg"] == 35.0 for c in fl.values())
    assert any(c["az_design_deg"] == -55.0 for c in fl.values())
    assert {c["lv_s"] for c in fl.values()} >= {0.02, 0.04, 0.08}
    from threedfly.control.closed_loop_metrics import (
        LOOM_RADIUS_M,
        angular_size_deg,
        start_distance_for_angular_size,
        visible_loom_placement,
    )

    d5 = start_distance_for_angular_size(5.0)
    assert angular_size_deg(d5, LOOM_RADIUS_M) == pytest.approx(5.0, abs=0.05)
    for c in fl.values():
        assert angular_size_deg(c["distance_m"], LOOM_RADIUS_M) <= 5.05
        assert c.get("present_at_prewarm") is True
    vis, rad = visible_loom_placement((13.3, 0.0, 1.5), (0.0, 0.0, 1.5))
    assert vis[0] <= 4.61
    assert rad < 0.58
    statics = {c["tag"]: c for c in mod.static_onset_gate_cases()}
    assert statics
    assert any(c.get("appear_after_prewarm") for c in statics.values())
    recedes = mod.receding_gate_cases()
    assert len(recedes) == 2
    assert recedes[0]["loom_vel"][0] > 0.0  # moving away from origin along +X/+Y
    assert all(c.get("present_at_prewarm") is True for c in recedes)
    assert all(c.get("appear_after_prewarm") is False for c in recedes)
    unseen = mod.unseen_frontolat_cases()
    assert len(unseen) == 16
    assert {abs(c["az_design_deg"]) for c in unseen} >= {25.0, 40.0, 50.0, 65.0}
    rnd = mod.random_frontolateral_cases()
    assert [c["seed"] for c in rnd] == list(mod.FRONTOLAT_RANDOM_SEEDS)
    rnd2 = mod.random_frontolateral_cases()
    assert rnd[0]["loom_start"] == rnd2[0]["loom_start"]
    for c in rnd:
        assert 20.0 <= abs(c["az_design_deg"]) <= 70.0
    assert {c["lv_s"] for c in rnd} >= {0.02, 0.04, 0.08}
    assert {c["threat_side"] for c in rnd} >= {"L", "R"}
    fronts = mod.frontal_gate_cases()
    assert all(abs(c["az_design_deg"]) < 20.0 for c in fronts)
    assert len(mod.OPEN_NOOBJ_YAWS) == 6
    gate = mod._frontolat_gate(
        {
            "fl_L_az35_lv0.04_d2.4": {
                "flight_avoidance_ok": True,
                "avoidance_mode": "pre_gf_saccade",
                "yaw_driver": "dnp03",
                "brain_escapes": 0,
            },
            "fl_R_az35_lv0.04_d2.4": {
                "flight_avoidance_ok": False,
                "avoidance_mode": "none",
                "brain_escapes": 1,
                "approaching_at_fire": False,
            },
            "side_loom_left": {
                "flight_avoidance_ok": True,
                "brain_escapes": 1,
            },
        }
    )
    assert gate["n"] == 2
    assert gate["mean"] == pytest.approx(0.5)
    assert gate["pass"] is False
    assert gate["n_pre_gf_saccade"] == 1
    assert gate["yaw_drivers"]["dnp03"] == 1
    official = mod._official_pass(
        {
            "frontolat_gate": {"pass": True},
            "frontal_gate": {"pass": True},
            "ctrl_noobj_headings": {"pass": True},
            "static_onset_gate": {"pass": False},
            "recede_gate": {"pass": True},
        },
        type("P", (), {"use_engineered_eye_laterality": False, "use_engineered_looming_residual": False})(),
    )
    assert official["pass"] is True
    assert "static_onset" not in official["gates"]
    official_fail = mod._official_pass(
        {
            "frontolat_gate": {"pass": False},
            "frontal_gate": {"pass": True},
            "ctrl_noobj_headings": {"pass": True},
            "static_onset_gate": {"pass": True},
            "recede_gate": {"pass": True},
        },
        type("P", (), {"use_engineered_eye_laterality": False, "use_engineered_looming_residual": False})(),
    )
    assert official_fail["pass"] is False
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
    vd_miss = mod._variant_direction(
        {
            "side_loom_left": {"escape_direction_correct": None},
            "loomR_far1.4x": {"escape_direction_correct": 1.0},
        }
    )
    assert vd_miss["n"] == 2
    assert vd_miss["mean"] == pytest.approx(0.5)
    assert vd_miss["per_scenario"]["side_loom_left"] == 0.0
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


def test_flight_avoidance_miss_and_far_side():
    """Surface clearance > 0.1 m + heading-away or far-side; GF is not required."""
    assert min_center_distance_m([2.0, 0.4, 1.1]) == pytest.approx(0.4)
    assert MISS_SURFACE_M == pytest.approx(0.1)
    assert peak_yaw_rate_rad_s([0.0, 0.1, 0.2], 0.01) == pytest.approx(10.0)
    rec = flight_avoidance_ok(
        min_center_m=0.80,
        headings=[0.0, -0.05, -0.15],
        threat_side="L",
        azimuths=[35.0, 50.0, 80.0],
        ca_step=2,
    )
    assert rec["ok"] is True
    assert rec["missed"] is True
    assert rec["min_surface_m"] == pytest.approx(0.22)
    assert rec["heading_away"] is True
    assert rec["passed_far_side"] is True
    # Centre 0.4 m clips the 0.58 m disc — that is a collision.
    clip = flight_avoidance_ok(
        min_center_m=0.40,
        headings=[0.0, -0.2],
        threat_side="L",
        azimuths=[80.0, 90.0],
        ca_step=1,
    )
    assert clip["ok"] is False
    assert clip["missed"] is False
    hit = flight_avoidance_ok(
        min_center_m=0.05,
        headings=[0.0, -0.2],
        threat_side="L",
        azimuths=[80.0, 90.0],
        ca_step=1,
    )
    assert hit["ok"] is False
    assert hit["missed"] is False
    # Right threat, heading increase is away; object stays negative az.
    right = flight_avoidance_ok(
        min_center_m=0.80,
        headings=[0.0, 0.04, 0.12],
        threat_side="R",
        azimuths=[-35.0, -55.0, -90.0],
        ca_step=2,
    )
    assert right["ok"] is True
    assert heading_away_from_side(net_heading_delta([0.0, 0.12]), "R")
    assert passed_far_side([-35.0, -90.0], "R", ca_step=1)
    integ = yaw_term_away_integrals(
        [
            {"dnp03": -0.2, "dna02": 0.01, "dnp15": 0.0, "hse": 0.0, "vpn_loom": -0.05},
            {"dnp03": -0.3, "dna02": 0.01, "dnp15": 0.0, "hse": 0.0, "vpn_loom": -0.04},
        ],
        "L",
        end_step=1,
    )
    assert integ["driver"] == "dnp03"
    assert integ["integrals"]["dnp03"] > integ["integrals"]["vpn_loom"]


def test_far_disc_hidden_for_dnp01_baseline_large_recede_stays():
    """≤5° approach discs do not sit in the DNp01 rest snapshot; recede does."""
    assert dnp01_baseline_shows_loom(
        present_at_prewarm=True, appear_after_prewarm=False, start_ang_deg=5.0
    ) is False
    assert dnp01_baseline_shows_loom(
        present_at_prewarm=True, appear_after_prewarm=False, start_ang_deg=4.9
    ) is False
    assert dnp01_baseline_shows_loom(
        present_at_prewarm=True, appear_after_prewarm=False, start_ang_deg=42.0
    ) is True
    assert dnp01_baseline_shows_loom(
        present_at_prewarm=False, appear_after_prewarm=True, start_ang_deg=34.0
    ) is False
