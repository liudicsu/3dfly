"""Engineered flight parameters live in one module."""

from threedfly.control.flight_params import FlightParams


def test_residual_off_and_no_dng02_yaw_gain():
    p = FlightParams()
    assert p.use_engineered_looming_residual is False
    d = p.as_dict()
    assert "dng02_yaw_gain" not in d
    assert d["dnp01_escape_hz"] == 12.0
    assert d["dnp03_yaw_gain"] > 0
    assert "dng02" in p.notes["dng02"].lower() or "NO L" in p.notes["dng02"]
