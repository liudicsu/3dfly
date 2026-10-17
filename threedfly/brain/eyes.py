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
    """Sample gray/first-channel image at UV in [0, 1]. Local copy (no cv2)."""
    if image.ndim == 3:
        gray = image.mean(axis=2)
    else:
        gray = image
    h, w = gray.shape[:2]
    xs = np.clip((uv[:, 0] * (w - 1)).astype(np.int32), 0, w - 1)
    ys = np.clip((uv[:, 1] * (h - 1)).astype(np.int32), 0, h - 1)
    samples = gray[ys, xs].astype(np.float32)
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

    Horizontal: do **not** flip U. After the r3 right-camera X flip both
    eyes have image-right = nasal and look +X.

    Vertical: the same r3 Y flip that restored look=+X inverts the right
    image (camera Y ≈ −Z). flyvis loom/T4/T5 expect upright flow, so V is
    mirrored for the right eye only.
    """
    if uv is None:
        uv = box_eye_uv()
    uv_r = np.asarray(uv, dtype=np.float32).copy()
    uv_r[:, 1] = 1.0 - uv_r[:, 1]
    return image_to_hex(left_img, uv=uv), image_to_hex(right_img, uv=uv_r)
