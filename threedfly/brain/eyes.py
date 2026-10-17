"""Sample MuJoCo eye images onto the flyvis 721-hex lattice (BoxEye equivalent).

flyvis ``BoxEye`` maps the cube-range hex lattice (extent=15 → 721 cells)
into a square image. We do the same from rendered left/right cameras:
hex angular coordinates → pixel UV using the eye ``fovy`` (120°).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional, Tuple

import numpy as np

from threedfly.brain.constants import HEX_EXTENT, N_HEX
from threedfly.brain.stimuli import hex_degrees


def _sample_image_hex(image: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Bilinear-sample gray/first-channel image at UV in [0, 1]."""
    if image.ndim == 3:
        gray = image.mean(axis=2)
    else:
        gray = image
    h, w = gray.shape[:2]
    xf = np.clip(np.asarray(uv[:, 0], dtype=np.float32) * (w - 1), 0.0, float(w - 1))
    yf = np.clip(np.asarray(uv[:, 1], dtype=np.float32) * (h - 1), 0.0, float(h - 1))
    x0 = np.floor(xf).astype(np.int32)
    y0 = np.floor(yf).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    wx = xf - x0.astype(np.float32)
    wy = yf - y0.astype(np.float32)
    g = np.asarray(gray, dtype=np.float32)
    samples = (
        g[y0, x0] * (1.0 - wx) * (1.0 - wy)
        + g[y0, x1] * wx * (1.0 - wy)
        + g[y1, x0] * (1.0 - wx) * wy
        + g[y1, x1] * wx * wy
    )
    peak = 255.0 if float(samples.max() if samples.size else 0.0) > 1.5 else 1.0
    return np.clip(samples / peak, 0.0, 1.0)

# MuJoCo left_eye / right_eye cameras (threedfly/sim/mjcf.py). Must match fovy.
EYE_FOVY_DEG = 120.0


@lru_cache(maxsize=4)
def box_eye_uv(extent: int = HEX_EXTENT, fovy_deg: float = EYE_FOVY_DEG) -> np.ndarray:
    """(721, 2) UV in [0, 1] for the flyvis hex lattice packed into a square FOV."""
    _, _, xd, yd = hex_degrees(extent)
    # hex_degrees returns degrees from the lattice centre (flyvis hex_to_pixel × HEX_DEG).
    half = float(fovy_deg) * 0.5
    u = 0.5 + xd / (2.0 * half)
    v = 0.5 - yd / (2.0 * half)
    return np.clip(np.stack([u, v], axis=1), 0.0, 1.0).astype(np.float32)


def image_to_hex(
    image: np.ndarray,
    uv: Optional[np.ndarray] = None,
    fovy_deg: float = EYE_FOVY_DEG,
) -> np.ndarray:
    """Grayscale-sample one RGB/gray image onto 721 hex cells in [0, 1]."""
    if uv is None:
        uv = box_eye_uv(fovy_deg=fovy_deg)
    samples = _sample_image_hex(np.asarray(image), uv)
    if samples.size != N_HEX:
        raise ValueError(f"hex sample size {samples.size} != {N_HEX}")
    return samples.astype(np.float32)


def stereo_to_hex(
    left_img: np.ndarray,
    right_img: np.ndarray,
    uv: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Left and right MuJoCo eye images → (721,), (721,) hex frames.

    Right camera: X is sagittally mirrored and Y is flipped so look stays
    +X. That is a horizontal remap in the image, not a vertical one.
    Open-loop L vs mirrored-R objects: a **U-flip** on the right hex
    (no V-flip) plus bilinear sampling keeps LC4/DNp01 L↔R within ~20%
    on the official 45° approach. V-flip alone drives the wrong side.
    """
    if uv is None:
        uv = box_eye_uv()
    uv_r = np.asarray(uv, dtype=np.float32).copy()
    uv_r[:, 0] = 1.0 - uv_r[:, 0]
    return image_to_hex(left_img, uv=uv), image_to_hex(right_img, uv=uv_r)


def mirror_stereo_images(left_img: np.ndarray, right_img: np.ndarray):
    """Mirror the scene: swap eyes and flip each image left–right."""
    return np.fliplr(np.asarray(right_img)), np.fliplr(np.asarray(left_img))
