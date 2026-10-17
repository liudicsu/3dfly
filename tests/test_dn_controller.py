"""DNFlightController: DNp01 escape, DNp03 contralateral turn, no DNg02 yaw."""

from __future__ import annotations

import numpy as np
import pytest

from threedfly.brain.readouts import DNReadout
from threedfly.control.dn_controller import DNFlightController
from threedfly.control.flight_params import FlightParams


def _ctl(**kwargs) -> DNFlightController:
    kwargs.setdefault("action_alpha", 1.0)
    kwargs.setdefault("dnp01_warmup_steps", 1)
    p = FlightParams(**kwargs)
    c = DNFlightController(p)
    c.bind_hover(0.32)
    return c


def test_escape_only_from_dnp01():
    c = _ctl()
    rest = DNReadout(dng02_amp=5.0, dng02_L=5.0, dng02_R=5.0)
    c.compute(rest)
    assert c.brain_escape_count == 0
    burst = DNReadout(dnp01_L=20.0, dnp01_R=1.0, dng02_amp=5.0, dng02_L=5.0, dng02_R=5.0)
    act = c.compute(burst)
    assert c.brain_escape_count == 1
    assert c.engineered_escape_count == 0
    assert act[3] > 0.2  # pitch-up climb


def test_engineered_residual_off_by_default():
    c = _ctl()
    rest = DNReadout(dng02_amp=5.0)
    c.compute(rest, looming_left=5.0, looming_right=5.0)
    c.compute(rest, looming_left=5.0, looming_right=5.0)
    assert c.brain_escape_count == 0
    assert c.engineered_escape_count == 0
    assert c.escape_timer == 0


def test_engineered_residual_opt_in():
    c = _ctl(use_engineered_looming_residual=True)
    rest = DNReadout(dng02_amp=5.0)
    c.compute(rest, looming_left=0.0, looming_right=0.0)
    c.compute(rest, looming_left=4.0, looming_right=0.0)
    assert c.engineered_escape_count >= 1
    assert c.brain_escape_count == 0


def test_dnp03_left_yaws_right():
    """Left DNp03 (ipsi loom) → contralateral / right turn (negative yaw)."""
    c = _ctl()
    c.compute(DNReadout(dng02_amp=5.0))
    left = DNReadout(
        dnp01_L=20.0,
        dnp03_L=15.0,
        dnp03_R=1.0,
        dng02_amp=5.0,
    )
    yaw_l = float(c.compute(left)[4])
    c.reset()
    c.compute(DNReadout(dng02_amp=5.0))
    right = DNReadout(
        dnp01_R=20.0,
        dnp03_L=1.0,
        dnp03_R=15.0,
        dng02_amp=5.0,
    )
    yaw_r = float(c.compute(right)[4])
    assert yaw_l < -0.05, f"DNp03_L should yaw right (neg), got {yaw_l:.3f}"
    assert yaw_r > 0.05, f"DNp03_R should yaw left (pos), got {yaw_r:.3f}"
    assert yaw_l * yaw_r < 0
    assert c.escape_events[-1].threat_side == "R"


def test_dng02_does_not_contribute_yaw():
    c = _ctl()
    # Strong L/R DNg02 imbalance, no steering DNs, no escape.
    ro = DNReadout(dng02_L=20.0, dng02_R=0.0, dng02_amp=10.0)
    yaw = float(c.compute(ro)[4])
    assert abs(yaw) < 1e-6
    assert c.last_wing_strokes[0] == pytest.approx(c.last_wing_strokes[1])


def test_dn_yaw_drives_differential_strokes_not_dng02():
    """+yaw (left turn) → more left stroke. DNg02 only sets the mean."""
    c = _ctl()
    c.compute(DNReadout(dng02_amp=5.0))
    left_turn = DNReadout(dnp03_L=0.10, dnp03_R=1.30, dng02_amp=5.0)
    c.compute(left_turn)
    assert c.last_wing_strokes[0] > c.last_wing_strokes[1] + 0.02
    c.reset()
    c.compute(DNReadout(dng02_amp=5.0))
    right_turn = DNReadout(dnp03_L=1.30, dnp03_R=0.10, dng02_amp=5.0)
    c.compute(right_turn)
    assert c.last_wing_strokes[1] > c.last_wing_strokes[0] + 0.02


def test_dna02_lr_steers_without_escape():
    c = _ctl()
    left = DNReadout(dna02_L=8.0, dna02_R=0.0, dng02_amp=5.0)
    yaw_l = float(c.compute(left)[4])
    c.reset()
    right = DNReadout(dna02_L=0.0, dna02_R=8.0, dng02_amp=5.0)
    yaw_r = float(c.compute(right)[4])
    assert yaw_l > 0.05
    assert yaw_r < -0.05
    assert c.brain_escape_count == 0


def test_no_false_escape_at_rest():
    c = _ctl()
    rest = DNReadout(dng02_amp=5.0, dng02_L=5.0, dng02_R=5.0, dna02_L=0.2, dna02_R=0.2)
    for _ in range(40):
        c.compute(rest)
    assert c.brain_escape_count == 0
    assert c.dnp01_latch_frac == 0.0


def test_latch_fraction_counts_cap():
    c = _ctl()
    c.compute(DNReadout(dnp01_L=50.0, dng02_amp=5.0))
    assert c.dnp01_latch_frac == pytest.approx(1.0)


def test_escape_from_m1_scale_rise_not_absolute_12hz():
    """Closed-loop DNp01 sits at ~2 Hz; 12 Hz is unreachable. Fire on EMA rise."""
    c = _ctl()
    cruise = DNReadout(dnp01_L=2.0, dnp01_R=1.8, dng02_amp=5.0)
    for _ in range(30):
        c.compute(cruise)
    assert c.brain_escape_count == 0
    loom = DNReadout(dnp01_L=3.1, dnp01_R=2.0, dnp03_L=1.3, dnp03_R=0.2, dng02_amp=5.0)
    act = c.compute(loom)
    assert c.brain_escape_count == 1
    assert c.escape_events[-1].threat_side == "L"
    assert act[4] < -0.05  # contralateral / right turn
    assert act[3] > 0.2


def test_warmup_blocks_takeoff_false_escape():
    c = _ctl(dnp01_warmup_steps=20)
    takeoff = DNReadout(dnp01_L=2.2, dng02_amp=5.0)
    for _ in range(15):
        c.compute(takeoff)
    assert c.brain_escape_count == 0


def test_dnp03_continuous_steer_without_escape_or_2hz_gate():
    """In-loop DNp03 ≤ 1.34 Hz; a 2 Hz gate would zero the dodge."""
    c = _ctl()
    left = DNReadout(dnp03_L=1.30, dnp03_R=0.10, dng02_amp=5.0)
    yaw_l = float(c.compute(left)[4])
    c.reset()
    right = DNReadout(dnp03_L=0.10, dnp03_R=1.30, dng02_amp=5.0)
    yaw_r = float(c.compute(right)[4])
    assert c.brain_escape_count == 0
    assert yaw_l < -0.05
    assert yaw_r > 0.05


def test_stereo_looming_does_not_overwrite_dnp03_threat():
    c = _ctl()
    rest = DNReadout(dng02_amp=5.0)
    c.compute(rest)
    burst = DNReadout(
        dnp01_L=3.2,
        dnp03_L=0.2,
        dnp03_R=1.3,
        dng02_amp=5.0,
    )
    c.compute(burst, looming_left=5.0, looming_right=0.0)
    assert c.brain_escape_count == 1
    assert c.escape_events[-1].threat_side == "R"


def test_dna02_normalised_to_m1_lr_scale():
    c = _ctl()
    # M1 DNa02 L−R peak delta is 0.54 Hz — must not require 8 Hz to steer.
    ro = DNReadout(dna02_L=0.54, dna02_R=0.0, dng02_amp=5.0)
    yaw = float(c.compute(ro)[4])
    assert yaw > 0.20


def test_no_escapes_in_first_100_steps_of_open_flight():
    """Startup artifact: rest EMA at 0 vs flyvis settling 1.7 Hz must not fire."""
    c = _ctl(dnp01_warmup_steps=40)
    for i in range(100):
        hz = 0.0 if i < 8 else 1.75
        c.compute(
            DNReadout(
                dnp01_L=hz,
                dnp01_R=hz * 0.95,
                dnp03_L=0.75,
                dnp03_R=0.35,
                dng02_amp=5.0,
            )
        )
    assert c.brain_escape_count == 0


def test_frozen_calibrated_rest_catches_slow_loom():
    c = _ctl(dnp01_warmup_steps=10)
    cruise = DNReadout(dnp01_L=1.80, dnp01_R=1.70, dng02_amp=5.0)
    for _ in range(20):
        c.compute(cruise)
    assert c.brain_escape_count == 0
    # 1.8 → 2.45 is the r2 side_loom_right rise the moving EMA ate.
    loom = DNReadout(dnp01_L=2.45, dnp01_R=1.90, dnp03_R=1.2, dnp03_L=0.3, dng02_amp=5.0)
    c.compute(loom)
    assert c.brain_escape_count == 1
    assert c.escape_events[-1].threat_side == "R"


def test_per_side_baseline_removes_dc_bias():
    c = _ctl(dnp01_warmup_steps=0)
    biased = DNReadout(dnp03_L=0.75, dnp03_R=0.35, dng02_amp=5.0, dnp01_L=1.8, dnp01_R=1.7)
    c.capture_baseline(biased)
    yaw = float(c.compute(biased)[4])
    assert abs(yaw) < 0.05
    left_loom = DNReadout(dnp03_L=1.30, dnp03_R=0.35, dng02_amp=5.0, dnp01_L=1.8, dnp01_R=1.7)
    assert float(c.compute(left_loom)[4]) < -0.05


def test_last8_rest_allows_loom_above_prewarm_peak():
    """Last prewarm frame can be the DC peak; rest is the last-8 mean."""
    c = _ctl(dnp01_warmup_steps=0, dnp01_post_baseline_steps=2, dnp01_escape_hold_steps=2)
    peak = DNReadout(dnp01_L=2.34, dnp01_R=2.10, dng02_amp=5.0)
    c.capture_baseline(peak, dnp01_rest=2.04)
    for _ in range(6):
        c.compute(DNReadout(dnp01_L=2.40, dnp01_R=2.10, dng02_amp=5.0))
    assert c.brain_escape_count == 0
    loom = DNReadout(dnp01_L=2.70, dnp01_R=2.20, dnp03_L=1.2, dnp03_R=0.4, dng02_amp=5.0)
    for _ in range(4):
        c.compute(loom)
    assert c.brain_escape_count == 1


def test_post_baseline_delay_skips_takeoff_transient():
    c = _ctl(dnp01_warmup_steps=0, dnp01_post_baseline_steps=8, dnp01_escape_hold_steps=2)  # short delay for the unit test
    rest = DNReadout(dnp01_L=1.80, dnp01_R=1.70, dng02_amp=5.0)
    c.capture_baseline(rest)
    transient = DNReadout(dnp01_L=2.50, dnp01_R=1.90, dnp03_L=1.2, dnp03_R=0.3, dng02_amp=5.0)
    for _ in range(7):
        c.compute(transient)
    assert c.brain_escape_count == 0
    c.compute(transient)
    c.compute(transient)
    assert c.brain_escape_count == 1


def test_baseline_eye_laterality_locks_approach_threat():
    c = _ctl(dnp01_warmup_steps=0, dnp01_post_baseline_steps=0, dnp01_escape_hold_steps=1)
    rest = DNReadout(eye_L=0.22, eye_R=0.10, dnp03_L=0.2, dnp03_R=0.8, dng02_amp=5.0)
    c.capture_baseline(rest)
    assert c._approach_threat == "L"
    flipped = DNReadout(
        dnp01_L=3.2,
        eye_L=0.22,
        eye_R=0.10,
        dnp03_L=0.2,
        dnp03_R=0.8,
        dng02_amp=5.0,
    )
    act = c.compute(flipped)
    assert c.escape_events[-1].threat_side == "L"
    assert act[4] < -0.05


def test_brighter_ipsilateral_eye_overrides_dnp03_for_threat():
    """Room walls are bright fields; DNp03 may stay L-biased, the eye mean must not."""
    c = _ctl(dnp01_warmup_steps=0, dnp01_post_baseline_steps=0, dnp01_escape_hold_steps=1)
    rest = DNReadout(dnp03_L=0.50, dnp03_R=0.30, eye_L=0.15, eye_R=0.15, dng02_amp=5.0)
    c.capture_baseline(rest)
    c.compute(rest)
    right_wall = DNReadout(
        dnp01_L=3.2,
        dnp03_L=0.70,
        dnp03_R=0.40,
        eye_L=0.10,
        eye_R=0.22,
        dng02_amp=5.0,
    )
    act = c.compute(right_wall)
    assert c.brain_escape_count == 1
    assert c.escape_events[-1].threat_side == "R"
    assert act[4] > 0.05


def test_lplc2_can_override_dnp03_dc_for_threat_side():
    """Right-wall loom: DNp03 may stay L-biased; LPLC2_R must still label R."""
    c = _ctl(dnp01_warmup_steps=0, dnp01_post_baseline_steps=0, dnp01_escape_hold_steps=1)
    rest = DNReadout(dnp03_L=0.40, dnp03_R=0.30, lplc2_L=1.0, lplc2_R=1.0, dng02_amp=5.0)
    c.capture_baseline(rest)
    c.compute(rest)
    right_wall = DNReadout(
        dnp01_L=3.2,
        dnp03_L=0.55,
        dnp03_R=0.40,
        lplc2_L=1.1,
        lplc2_R=2.4,
        dng02_amp=5.0,
    )
    act = c.compute(right_wall)
    assert c.brain_escape_count == 1
    assert c.escape_events[-1].threat_side == "R"
    assert act[4] > 0.05


def test_rising_edge_does_not_re_fire_while_dnp01_stays_high():
    """r3: frozen rest + pinned wall kept rise ≥ 0.55 and fired every cooldown."""
    c = _ctl(dnp01_warmup_steps=4)
    cruise = DNReadout(dnp01_L=1.80, dnp01_R=1.70, dng02_amp=5.0)
    for _ in range(10):
        c.compute(cruise)
    assert c.brain_escape_count == 0
    loom = DNReadout(dnp01_L=2.50, dnp01_R=1.90, dnp03_L=1.2, dnp03_R=0.3, dng02_amp=5.0)
    for _ in range(40):
        c.compute(loom)
    assert c.brain_escape_count == 1
    for _ in range(8):
        c.compute(cruise)
    assert c.brain_escape_count == 1
    c.compute(loom)
    assert c.brain_escape_count == 2
