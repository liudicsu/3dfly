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


def test_stereo_to_hex_pair():
    left = np.full((32, 32, 3), 128, dtype=np.uint8)
    right = np.zeros((32, 32, 3), dtype=np.uint8)
    hL, hR = stereo_to_hex(left, right)
    assert hL.shape == hR.shape == (N_HEX,)
    assert hL.mean() > 0.4
    assert hR.mean() == pytest.approx(0.0)
