"""Sample MuJoCo eye images onto the flyvis 721-hex lattice.

Eye input is rendered pixels (or an explicit hex movie). No sim object
pose, radius, or mocap state is read here.

Each hex ommatidium has a body-frame direction taken from the flyvis
lattice (``hex_degrees``). That lattice is the frontal ~150° of each eye;
its origin is the binocular field, not the camera optical axis.

Drosophila: each eye covers roughly −15°…+170° azimuth ipsilateral with
~10–20° frontal overlap (Buchner / Heisenberg). flyvis only has the
frontal 150° patch, so hex (xd, yd) maps to body (az=±xd, el=yd). Hexes
outside that eye's biological envelope stay mid-gray: a +35° object is
left-eye only (right eye's contra limit is +15°). Cameras look along +X
with a wide square FOV so the remaining rays hit the image.

``box_eye_uv`` (legacy) packed the lattice into a square pinhole. Combined
with a 40° outward / 20° down cant that put the frontal 0–70° sector on
the nasal/upper periphery — LPLC2/LC4's least sensitive hexes.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional, Tuple

import numpy as np

from threedfly.brain.constants import HEX_EXTENT, N_HEX
from threedfly.brain.stimuli import hex_degrees

# Must match threedfly/sim/mjcf.py (forward wide-FOV, no cant).
EYE_FOVY_DEG = 150.0
EYE_CANT_DEG = 0.0
EYE_DOWN_DEG = 0.0
# Biological envelope (for coverage plots / tests; sampling is unclamped).
BIO_AZ_CONTRA_DEG = 15.0
BIO_AZ_IPSI_DEG = 170.0
UNSEEN_GRAY = 0.5
# Drosophila acceptance angle Δρ ≈ 5° FWHM (Stavenga / Land). σ = FWHM / 2.355.
ACCEPT_FWHM_DEG = 5.0
ACCEPT_SIGMA_DEG = ACCEPT_FWHM_DEG / 2.35482004503
ACCEPT_N = 5


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


@lru_cache(maxsize=4)
def box_eye_uv(extent: int = HEX_EXTENT, fovy_deg: float = EYE_FOVY_DEG) -> np.ndarray:
    """Legacy (721, 2) UV: pack the hex lattice into a square FOV.

    Kept for tests and the coverage diagnosis of the old front end.
    ``stereo_to_hex`` no longer uses this.
    """
    _, _, xd, yd = hex_degrees(extent)
    half = float(fovy_deg) * 0.5
    u = 0.5 + xd / (2.0 * half)
    v = 0.5 - yd / (2.0 * half)
    return np.clip(np.stack([u, v], axis=1), 0.0, 1.0).astype(np.float32)


def hex_body_azel_deg(side: str, extent: int = HEX_EXTENT) -> tuple[np.ndarray, np.ndarray]:
    """Body-frame azimuth / elevation (deg) of each flyvis hex.

    +azimuth = left (+Y). Left-eye +xd looks left; right-eye +xd looks right
    so a mirrored object falls on the same lattice site of the other eye.
    Lattice centre (0, 0) is body-frontal — the LPLC2/LC4-sensitive site.
    """
    if side not in ("L", "R"):
        raise ValueError(f"side must be L or R, got {side!r}")
    _, _, xd, yd = hex_degrees(extent)
    az = xd if side == "L" else -xd
    return np.asarray(az, dtype=np.float32), np.asarray(yd, dtype=np.float32)


def azel_to_body_dir(az_deg, el_deg) -> np.ndarray:
    """Unit vectors, body frame (+X forward, +Y left, +Z up). (N, 3)."""
    az = np.deg2rad(np.asarray(az_deg, dtype=float))
    el = np.deg2rad(np.asarray(el_deg, dtype=float))
    c = np.cos(el)
    return np.stack([c * np.cos(az), c * np.sin(az), np.sin(el)], axis=-1)


def eye_camera_axes(
    side: str = "L",
    cant_deg: float = EYE_CANT_DEG,
    down_deg: float = EYE_DOWN_DEG,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(look, x_right, y_up) in the body frame. MuJoCo camera looks along −Z."""
    a = np.deg2rad(float(cant_deg))
    d = np.deg2rad(float(down_deg))
    # Outward cant: left +Y, right −Y. down_deg tilts look below the horizon.
    sgn = 1.0 if side == "L" else -1.0
    look = np.array(
        [np.cos(d) * np.cos(a), sgn * np.cos(d) * np.sin(a), -np.sin(d)],
        dtype=float,
    )
    look = look / float(np.linalg.norm(look))
    z = -look
    x = np.cross(np.array([0.0, 0.0, 1.0], dtype=float), z)
    n = float(np.linalg.norm(x))
    if n < 1e-8:
        x = np.array([0.0, -1.0, 0.0], dtype=float)
    else:
        x = x / n
    y = np.cross(z, x)
    y = y / float(np.linalg.norm(y))
    return look, x, y


def project_dirs_to_uv(
    dirs: np.ndarray,
    look: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    fovy_deg: float,
    aspect: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Pinhole project body-frame directions. Returns (uv (N,2), valid (N,))."""
    d = np.asarray(dirs, dtype=float).reshape(-1, 3)
    xc = d @ np.asarray(x_axis, dtype=float)
    yc = d @ np.asarray(y_axis, dtype=float)
    zc = d @ (-np.asarray(look, dtype=float))  # camera +Z is backward
    in_front = zc < -1e-6
    tan_half = float(np.tan(np.deg2rad(float(fovy_deg) * 0.5)))
    asp = max(float(aspect), 1e-6)
    x_ndc = np.divide(xc, -zc, out=np.zeros_like(xc), where=in_front) / (asp * tan_half)
    y_ndc = np.divide(yc, -zc, out=np.zeros_like(yc), where=in_front) / tan_half
    u = 0.5 + 0.5 * x_ndc
    v = 0.5 - 0.5 * y_ndc
    on_sensor = (u >= 0.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0)
    valid = in_front & on_sensor
    uv = np.stack([np.clip(u, 0.0, 1.0), np.clip(v, 0.0, 1.0)], axis=1).astype(np.float32)
    return uv, valid.astype(bool)


def hex_in_bio_envelope(side: str, az_deg) -> np.ndarray:
    """True where *az_deg* lies on that eye's Drosophila azimuth range.

    Left: −15° (contra) … +170° (ipsi). Right: mirrored. flyvis only
    spans ~±75°, so the unused rear ~95° stays empty; the contra cut is
    what makes a frontolat loom monocular.
    """
    if side not in ("L", "R"):
        raise ValueError(f"side must be L or R, got {side!r}")
    az = np.asarray(az_deg, dtype=float)
    if side == "L":
        return (az >= -BIO_AZ_CONTRA_DEG) & (az <= BIO_AZ_IPSI_DEG)
    return (az <= BIO_AZ_CONTRA_DEG) & (az >= -BIO_AZ_IPSI_DEG)


def hex_camera_uv(
    side: str,
    height: int,
    width: int,
    fovy_deg: float = EYE_FOVY_DEG,
    cant_deg: float = EYE_CANT_DEG,
    down_deg: float = EYE_DOWN_DEG,
) -> tuple[np.ndarray, np.ndarray]:
    """UV + validity for each hex through the named eye camera."""
    az, el = hex_body_azel_deg(side)
    look, x_ax, y_ax = eye_camera_axes(side, cant_deg=cant_deg, down_deg=down_deg)
    aspect = float(width) / max(float(height), 1.0)
    uv, valid = project_dirs_to_uv(azel_to_body_dir(az, el), look, x_ax, y_ax, fovy_deg, aspect)
    return uv, valid & hex_in_bio_envelope(side, az)


@lru_cache(maxsize=8)
def hex_accept_uv(
    side: str,
    height: int,
    width: int,
    fovy_deg: float = EYE_FOVY_DEG,
    sigma_deg: float = ACCEPT_SIGMA_DEG,
    n_off: int = ACCEPT_N,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-hex Gaussian acceptance samples.

    Returns ``(uv (N,K,2), sample_valid (N,K), weights (K,), hex_valid (N,))``.
    Offsets are in body az/el degrees; no object geometry is used.
    """
    az, el = hex_body_azel_deg(side)
    look, x_ax, y_ax = eye_camera_axes(side)
    aspect = float(width) / max(float(height), 1.0)
    sig = max(float(sigma_deg), 1e-3)
    offs = np.linspace(-2.0 * sig, 2.0 * sig, int(n_off))
    daz, del_ = np.meshgrid(offs, offs, indexing="xy")
    daz = daz.ravel().astype(np.float32)
    del_ = del_.ravel().astype(np.float32)
    w = np.exp(-0.5 * (daz * daz + del_ * del_) / (sig * sig)).astype(np.float32)
    az_s = az[:, None] + daz[None, :]
    el_s = el[:, None] + del_[None, :]
    uv, ok = project_dirs_to_uv(
        azel_to_body_dir(az_s.ravel(), el_s.ravel()), look, x_ax, y_ax, fovy_deg, aspect
    )
    n_hex = int(az.size)
    n_k = int(daz.size)
    uv = uv.reshape(n_hex, n_k, 2)
    ok = ok.reshape(n_hex, n_k)
    hex_ok = hex_in_bio_envelope(side, az) & ok.any(axis=1)
    return uv, ok, w, hex_ok


def _sample_image_hex_weighted(
    image: np.ndarray,
    uv: np.ndarray,
    sample_valid: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Gaussian-weighted bilinear samples. ``uv`` is (N, K, 2)."""
    n, k, _ = uv.shape
    samples = _sample_image_hex(image, uv.reshape(n * k, 2)).reshape(n, k)
    w = np.asarray(weights, dtype=np.float32)[None, :] * np.asarray(sample_valid, dtype=np.float32)
    den = w.sum(axis=1)
    num = (samples * w).sum(axis=1)
    out = np.full(n, UNSEEN_GRAY, dtype=np.float32)
    good = den > 1e-8
    out[good] = (num[good] / den[good]).astype(np.float32)
    return out


def _luminance_to_gray(hex_frame: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Map the rendered background to flyvis gray=0.5 (dark stays dark).

    Two-point histogram stretch: p80 → 0.5, p5 stays below it. A growing
    dark object must not become the background (median would). Floor the
    span so a flat field is not amplified. Generic — no object geometry.
    """
    out = np.asarray(hex_frame, dtype=np.float32).copy()
    m = np.asarray(valid, dtype=bool)
    if np.count_nonzero(m) < 8:
        return out
    lo = float(np.percentile(out[m], 5.0))
    bg = float(np.percentile(out[m], 80.0))
    span = max(bg - lo, 0.12)
    out[m] = np.clip((out[m] - bg) / span * UNSEEN_GRAY + UNSEEN_GRAY, 0.0, 1.0)
    return out


def hex_coverage(side: str) -> dict:
    """Body-frame az/el actually sampled by each hex (for plots / tests)."""
    az, el = hex_body_azel_deg(side)
    look, x_ax, y_ax = eye_camera_axes(side)
    uv, valid = hex_camera_uv(side, 160, 160)
    bio = hex_in_bio_envelope(side, az)
    return {
        "side": side,
        "az_deg": az,
        "el_deg": el,
        "uv": uv,
        "valid": valid,
        "bio": bio,
        "look": look,
        "x_axis": x_ax,
        "y_axis": y_ax,
        "n_valid": int(np.count_nonzero(valid)),
        "n": int(az.size),
        "n_bio": int(np.count_nonzero(bio)),
    }


def legacy_packed_azel_deg(
    side: str,
    fovy_deg: float = 120.0,
    cant_deg: float = 40.0,
    down_deg: float = 20.0,
    aspect: float = 160.0 / 120.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Body az/el implied by the *old* pack-lattice-into-image front end.

    Image centre → hex (0, 0) → camera optical axis (40° out, 20° down).
    Used only to diagnose why 3D looms missed LPLC2.
    """
    uv = box_eye_uv(fovy_deg=fovy_deg).copy()
    if side == "R":
        uv[:, 0] = 1.0 - uv[:, 0]
    look, x_ax, y_ax = eye_camera_axes(side, cant_deg=cant_deg, down_deg=down_deg)
    tan_half = float(np.tan(np.deg2rad(float(fovy_deg) * 0.5)))
    x_ndc = (2.0 * uv[:, 0] - 1.0) * aspect * tan_half
    y_ndc = (1.0 - 2.0 * uv[:, 1]) * tan_half
    dirs = (
        x_ndc[:, None] * x_ax[None, :]
        + y_ndc[:, None] * y_ax[None, :]
        + look[None, :]
    )
    dirs = dirs / np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-9)
    az = np.degrees(np.arctan2(dirs[:, 1], dirs[:, 0]))
    el = np.degrees(np.arcsin(np.clip(dirs[:, 2], -1.0, 1.0)))
    return az.astype(np.float32), el.astype(np.float32), uv


def image_to_hex(
    image: np.ndarray,
    uv: Optional[np.ndarray] = None,
    fovy_deg: float = EYE_FOVY_DEG,
    valid: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Grayscale-sample one RGB/gray image onto 721 hex cells in [0, 1]."""
    if uv is None:
        uv = box_eye_uv(fovy_deg=fovy_deg)
    samples = _sample_image_hex(np.asarray(image), uv)
    if valid is not None:
        samples = np.where(np.asarray(valid, dtype=bool), samples, UNSEEN_GRAY)
    if samples.size != N_HEX:
        raise ValueError(f"hex sample size {samples.size} != {N_HEX}")
    return samples.astype(np.float32)


def _image_to_hex_accept(image: np.ndarray, side: str) -> np.ndarray:
    """Sample one rendered eye through the ~5° Gaussian acceptance lobe."""
    image = np.asarray(image)
    h, w = int(image.shape[0]), int(image.shape[1])
    uv, sample_ok, weights, hex_ok = hex_accept_uv(side, h, w)
    hex_frame = _sample_image_hex_weighted(image, uv, sample_ok, weights)
    hex_frame = np.where(hex_ok, hex_frame, UNSEEN_GRAY).astype(np.float32)
    return _luminance_to_gray(hex_frame, hex_ok)


def stereo_to_hex(
    left_img: np.ndarray,
    right_img: np.ndarray,
    uv: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Left and right rendered eye images → (721,), (721,) hex frames.

    Official path: each hex averages a 5° FWHM Gaussian acceptance lobe
    over the rendered pixels, then valid hexes are shifted so the
    background (80th percentile) sits at flyvis gray=0.5. Generic for
    every geom (walls, object, sky). No sim object pose/radius is read
    here — the camera image is the only scene input.
    """
    left_img = np.asarray(left_img)
    right_img = np.asarray(right_img)
    if uv is not None:
        # Explicit UV (tests): left as-is, right U-flipped only when the
        # caller passed the packed lattice, not a direction map.
        uv = np.asarray(uv, dtype=np.float32)
        uv_r = uv.copy()
        uv_r[:, 0] = 1.0 - uv_r[:, 0]
        return image_to_hex(left_img, uv=uv), image_to_hex(right_img, uv=uv_r)
    return _image_to_hex_accept(left_img, "L"), _image_to_hex_accept(right_img, "R")
