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
