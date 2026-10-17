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
    d = p.as_dict()
    assert "dng02_yaw_gain" not in d
    assert "dnp03_active_hz" not in d
    assert d["dnp03_yaw_gain"] > 0
    assert "dng02" in p.notes["dng02"].lower() or "NO L" in p.notes["dng02"]


def test_escape_threshold_sits_between_m1_rotation_and_loom():
    p = FlightParams()
    assert M1_DNP01_ROT_MAX_HZ < p.dnp01_escape_delta_hz < M1_DNP01_LOOM_IPSI_HZ
    assert p.dnp01_loom_hz == M1_DNP01_LOOM_IPSI_HZ
    assert p.dnp01_rot_hz == M1_DNP01_ROT_MAX_HZ
    assert p.dna02_scale_hz == M1_DNA02_LR_HZ
    assert p.hse_scale_hz == M1_HSE_LR_HZ
    assert p.dnp01_freeze_rest is True
    assert "0.55" in p.notes["escape"] or "frozen" in p.notes["escape"]
