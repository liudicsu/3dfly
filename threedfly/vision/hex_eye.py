"""Hexagonal ommatidial lattice approximating Drosophila compound eyes.

MaleCNS annotation tables shipped with 3dfly do not include optic-lobe hex
column coordinates (no hexX/hexY/column/ommatidium fields). Sampling therefore
uses a synthetic hex lattice: ~750 ommatidia per eye with ~5° biological
spacing packed into each 90° eye camera image.

This replaces the old 10×10 rectangular grid that fed R1–R6.
"""

from typing import Optional

import numpy as np

# Drosophila: ~750 ommatidia/eye, ~4.5–5.5° interommatidial angle.
N_OMMATIDIA_PER_EYE = 750
HEX_SPACING_DEG = 5.0


def _hex_radius_for_count(n: int) -> int:
    """Smallest hex radius R with 1 + 3 R (R+1) >= n."""
    r = 0
    while 1 + 3 * r * (r + 1) < n:
        r += 1
    return r


def hex_lattice_uv(
    n_ommatidia: int = N_OMMATIDIA_PER_EYE,
    spacing_deg: float = HEX_SPACING_DEG,
) -> np.ndarray:
    """
    Hex lattice of `n_ommatidia` points as UV coordinates in [0, 1]².

    Axial hex coordinates (q, r) are converted to Cartesian, the closest
    `n_ommatidia` sites to the origin are kept, then scaled into the image.
    `spacing_deg` is the biological interommatidial angle used to space the
    lattice before normalisation (the 90° cameras pack ~750 samples into the
    image, denser than a 5° lattice over 90° would be).
    """
    n_ommatidia = int(n_ommatidia)
    radius = _hex_radius_for_count(n_ommatidia)
    pts = []
    for q in range(-radius, radius + 1):
        r_min = max(-radius, -q - radius)
        r_max = min(radius, -q + radius)
        for r in range(r_min, r_max + 1):
            x = spacing_deg * (3.0 / 2.0 * q)
            y = spacing_deg * (np.sqrt(3.0) / 2.0 * q + np.sqrt(3.0) * r)
            pts.append((x, y))
    coords = np.asarray(pts, dtype=np.float64)
    dist = np.linalg.norm(coords, axis=1)
    order = np.argsort(dist)[:n_ommatidia]
    coords = coords[order]
    extent = float(np.max(np.abs(coords))) + 1e-12
    uv = 0.5 + 0.48 * coords / extent
    return np.clip(uv, 0.0, 1.0).astype(np.float32)


def sample_image_hex(
    image: np.ndarray,
    uv: np.ndarray,
) -> np.ndarray:
    """
    Sample a grayscale (or first-channel) image at hex UV coordinates.

    Args:
        image: (H, W) or (H, W, C) array
        uv: (N, 2) coordinates in [0, 1]

    Returns:
        (N,) samples in [0, 1]
    """
    if image.ndim == 3:
        gray = image.mean(axis=2)
    else:
        gray = image
    h, w = gray.shape[:2]
    xs = np.clip((uv[:, 0] * (w - 1)).astype(np.int32), 0, w - 1)
    ys = np.clip((uv[:, 1] * (h - 1)).astype(np.int32), 0, h - 1)
    samples = gray[ys, xs].astype(np.float32)
    peak = 255.0 if samples.max() > 1.5 else 1.0
    return np.clip(samples / peak, 0.0, 1.0)


def hex_assignment_from_annotations(
    uv: np.ndarray,
    hex_xy: np.ndarray,
) -> Optional[np.ndarray]:
    """Map lattice indices onto annotated hex column (x, y) if provided."""
    if hex_xy is None or len(hex_xy) == 0:
        return None
    # Nearest lattice site for each annotated column (not used unless columns exist).
    cols = np.asarray(hex_xy, dtype=np.float64)
    cols = cols - cols.min(axis=0)
    span = np.maximum(cols.max(axis=0), 1e-9)
    cols_uv = cols / span
    d = ((uv[None, :, :] - cols_uv[:, None, :]) ** 2).sum(axis=-1)
    return d.argmin(axis=1).astype(np.int32)
