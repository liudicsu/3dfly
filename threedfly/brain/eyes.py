"""Sample MuJoCo eye images onto the flyvis 721-hex lattice.

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
# Must match threedfly/sim/mjcf.py loom_geom radius.
LOOM_RADIUS_M = 0.58
LOOM_HEX = 0.0


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


def rays_hit_sphere(dirs: np.ndarray, center, radius: float) -> np.ndarray:
    """True if a ray from the origin along *dirs* hits the sphere.

    Origin inside the sphere counts as a hit on every ray (full-field dark).
    """
    d = np.asarray(dirs, dtype=float).reshape(-1, 3)
    n = np.linalg.norm(d, axis=1, keepdims=True)
    d = d / np.maximum(n, 1e-12)
    c = np.asarray(center, dtype=float).reshape(3)
    r = float(radius)
    c2 = float(np.dot(c, c))
    if c2 <= r * r:
        return np.ones(d.shape[0], dtype=bool)
    dc = d @ c
    disc = dc * dc - (c2 - r * r)
    return (disc >= 0.0) & (dc > 0.0)


def angular_loom_mask(
    side: str,
    center,
    radius: float = LOOM_RADIUS_M,
    thmax_deg: float = 90.0,
) -> np.ndarray:
    """Hex-disc mask: ``hypot(az − az0, el − el0) ≤ θ/2``, θ = 2 atan(r/d) ≤ 90°.

    Same geometry as ``stimuli.loom()``. Origin inside the sphere uses θ=90°.
    """
    az, el = hex_body_azel_deg(side)
    c = np.asarray(center, dtype=float).reshape(3)
    dist = float(np.linalg.norm(c))
    if dist < 1e-6:
        az0, el0, theta = 0.0, 0.0, float(thmax_deg)
    else:
        az0 = float(np.degrees(np.arctan2(c[1], c[0])))
        el0 = float(np.degrees(np.arcsin(np.clip(c[2] / dist, -1.0, 1.0))))
        theta = float(2.0 * np.degrees(np.arctan(float(radius) / dist)))
        if dist <= float(radius):
            theta = float(thmax_deg)
        theta = min(theta, float(thmax_deg))
    return (np.hypot(az - az0, el - el0) <= 0.5 * theta) & hex_in_bio_envelope(side, az)


def paint_loom_rays(
    hex_frame: np.ndarray,
    side: str,
    center,
    radius: float = LOOM_RADIUS_M,
    valid: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Paint a hard dark disc on hexes that look at the loom (flyvis loom())."""
    hit = angular_loom_mask(side, center, radius)
    if valid is not None:
        hit = hit & np.asarray(valid, dtype=bool)
    out = np.asarray(hex_frame, dtype=np.float32).copy()
    out[hit] = LOOM_HEX
    return out


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


def stereo_to_hex(
    left_img: np.ndarray,
    right_img: np.ndarray,
    uv: Optional[np.ndarray] = None,
    loom_body=None,
    loom_radius: float = LOOM_RADIUS_M,
) -> Tuple[np.ndarray, np.ndarray]:
    """Left and right MuJoCo eye images → (721,), (721,) hex frames.

    Each hex samples its own body-frame ray through the matching camera
    (inverse pinhole). Hexes outside the camera frustum stay mid-gray.

    ``loom_body`` is the sphere centre in the body frame (or a pair of
    per-eye centres). Hexes whose ray hits the sphere are painted black —
    hard ommatidial edges, and full-field dark once the fly is inside.
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
    hL, wL = int(left_img.shape[0]), int(left_img.shape[1])
    hR, wR = int(right_img.shape[0]), int(right_img.shape[1])
    uv_L, ok_L = hex_camera_uv("L", hL, wL)
    uv_R, ok_R = hex_camera_uv("R", hR, wR)
    hex_L = image_to_hex(left_img, uv=uv_L, valid=ok_L)
    hex_R = image_to_hex(right_img, uv=uv_R, valid=ok_R)
    if loom_body is None:
        return hex_L, hex_R
    body = np.asarray(loom_body, dtype=float)
    if body.ndim == 1:
        cL = cR = body
    else:
        cL, cR = body[0], body[1]
    r = float(loom_radius)
    return (
        paint_loom_rays(hex_L, "L", cL, r),
        paint_loom_rays(hex_R, "R", cR, r),
    )
