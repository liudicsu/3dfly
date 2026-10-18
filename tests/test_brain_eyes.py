"""BoxEye-equivalent hex sampling (no flyvis / connectome)."""

from __future__ import annotations

import numpy as np
import pytest

from threedfly.brain.constants import N_HEX
from threedfly.brain.eyes import box_eye_uv, image_to_hex, stereo_to_hex


def test_box_eye_has_721_cells():
    uv = box_eye_uv()
    assert uv.shape == (N_HEX, 2)
    assert uv.min() >= 0.0 and uv.max() <= 1.0


def test_image_to_hex_samples_blob_laterality():
    h = w = 64
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # Bright patch on the left half of the image (small u).
    img[:, : w // 4] = 255
    hex_frame = image_to_hex(img)
    assert hex_frame.shape == (N_HEX,)
    assert hex_frame.max() > 0.5
    uv = box_eye_uv()
    left = hex_frame[uv[:, 0] < 0.35].mean()
    right = hex_frame[uv[:, 0] > 0.65].mean()
    assert left > right + 0.2


def test_right_eye_looks_forward_not_backward():
    """Root cause of the r2 DNp03_L bias: right camera Y was not mirrored."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "threedfly" / "sim" / "mjcf.py"
    spec = importlib.util.spec_from_file_location("threedfly_mjcf_camera", path)
    mjcf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mjcf)
    look_L = mjcf.camera_look_direction(mjcf.parse_eye_xyaxes(mjcf.FLY_MJCF, "left_eye"))
    look_R = mjcf.camera_look_direction(mjcf.parse_eye_xyaxes(mjcf.FLY_MJCF, "right_eye"))
    assert look_L[0] > 0.7, f"left eye should look +X, got {look_L}"
    assert look_R[0] > 0.7, f"right eye should look +X, got {look_R}"
    assert look_L[0] * look_R[0] > 0
    # Outward cant: left toward +Y, right toward −Y, sagittal mirrors.
    assert look_L[1] > 0.3, f"left eye should cant toward +Y, got {look_L}"
    assert look_R[1] < -0.3, f"right eye should cant toward -Y, got {look_R}"
    assert abs(look_L[1] + look_R[1]) < 0.05
    assert "fovy=\"120\"" in mjcf.FLY_MJCF
    xml_L = mjcf.parse_eye_xyaxes(mjcf.FLY_MJCF, "left_eye")
    xml_R = mjcf.parse_eye_xyaxes(mjcf.FLY_MJCF, "right_eye")
    assert xml_L == pytest.approx(mjcf.eye_xyaxes("L"), abs=0.002)
    assert xml_R == pytest.approx(mjcf.eye_xyaxes("R"), abs=0.002)


def test_walls_use_high_contrast_checker():
    from threedfly.sim import mjcf as mjcf_mod

    xml = mjcf_mod.FLY_MJCF
    assert "wall_check" in xml
    assert 'builtin="checker"' in xml
    assert 'texrepeat="48 20"' in xml
    # Flat beige rgba must not wash out the texture on the room walls.
    assert "rgba=\"0.8 0.8 0.9 1\"" not in xml
    assert 'name="loom_target"' in xml
    assert 'mocap="true"' in xml
    assert 'size="0.58"' in xml
    # Solid dark disc (Klapoetke / flyvis loom). Fine checkers average
    # to gray at hex resolution (~5.8°) until ~80° of disk.
    assert "loom_mat" in xml
    assert 'rgba="0.02 0.02 0.03 1"' in xml
    assert "loom_check" not in xml


def test_mirror_stereo_images_helper_removed():
    """Swap+fliplr is not a rendered mirror (DNp01 err 0.33 vs 0.03)."""
    from threedfly.brain import eyes as eyes_mod

    assert not hasattr(eyes_mod, "mirror_stereo_images")


def test_hex_fovy_matches_camera():
    from threedfly.brain.eyes import EYE_FOVY_DEG
    from threedfly.sim import mjcf as mjcf_mod

    assert EYE_FOVY_DEG == pytest.approx(mjcf_mod.EYE_FOVY_DEG)
    assert EYE_FOVY_DEG == pytest.approx(120.0)


def test_stereo_to_hex_pair():
    left = np.full((32, 32, 3), 128, dtype=np.uint8)
    right = np.zeros((32, 32, 3), dtype=np.uint8)
    hL, hR = stereo_to_hex(left, right)
    assert hL.shape == hR.shape == (N_HEX,)
    assert hL.mean() > 0.4
    assert hR.mean() == pytest.approx(0.0)


def test_right_eye_hex_mirrors_u_for_mirrored_camera():
    """Right camera X is sagittally mirrored: a left-of-image blob maps to large u."""
    h = w = 64
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, : w // 4] = 255  # left columns
    left = np.zeros_like(img)
    hL, hR = stereo_to_hex(left, img)
    uv = box_eye_uv()
    lo = hR[uv[:, 0] < 0.35].mean()
    hi = hR[uv[:, 0] > 0.65].mean()
    assert hi > lo + 0.2
    assert float(hL.max()) == pytest.approx(0.0)
