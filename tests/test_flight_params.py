"""Engineered flight parameters live in one module, calibrated from M1 stats."""

from threedfly.control.flight_params import (
    M1_DNA02_LR_HZ,
    M1_DNP01_LOOM_IPSI_HZ,
    M1_DNP01_ROT_MAX_HZ,
    M1_HSE_LR_HZ,
    FlightParams,
)


def test_residual_off_and_no_dng02_yaw_gain():
    p = FlightParams()
    assert p.use_engineered_looming_residual is False
    assert p.use_engineered_eye_laterality is False
    d = p.as_dict()
    assert "dng02_yaw_gain" not in d
    assert "dnp03_active_hz" not in d
    assert d["dnp03_yaw_gain"] > 0
    assert "dng02" in p.notes["dng02"].lower() or "NO L" in p.notes["dng02"]


def test_escape_threshold_sits_between_m1_rotation_and_loom():
    p = FlightParams()
    assert 0.15 < p.dnp01_escape_delta_hz < 0.45
    assert p.dnp01_loom_hz == M1_DNP01_LOOM_IPSI_HZ
    assert p.dnp01_rot_hz == M1_DNP01_ROT_MAX_HZ
    assert p.dna02_scale_hz == M1_DNA02_LR_HZ
    assert p.hse_scale_hz == M1_HSE_LR_HZ
    assert p.dnp01_freeze_rest is True
    assert p.prewarm_steps >= 80
    assert p.side_loom_prewarm_steps >= p.prewarm_steps
    assert p.flyvis_stride >= 2
    assert p.dnp01_rest_tau_steps >= 80.0
    assert p.dnp01_loom_lat_min > 0.0
    assert p.dnp01_slope_min > 0.0
    assert p.threat_integrate_steps >= 8
    assert p.threat_lock_consistent >= 3
    assert p.threat_lock_rise_hz >= 0.10
    assert p.dnp01_vpn_delta_hz < p.dnp01_escape_delta_hz
    assert p.vpn_amp_min > 0.0
    assert p.lat_rest_tau_steps >= 10.0
    assert "0.20" in p.notes["escape"] or "laterality" in p.notes["escape"]
    assert "20" in p.notes["geometry"] and "70" in p.notes["geometry"]
