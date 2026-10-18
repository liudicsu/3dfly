"""Sample MuJoCo eye images onto the flyvis 721-hex lattice.

Eye input is rendered pixels (or an explicit hex movie). No sim object
pose, radius, or mocap state is read here.

Each hex ommatidium has a body-frame direction taken from the flyvis
lattice (``hex_degrees``). The 721-hex lattice is a ±75° projection of
one compound eye: lattice (0, 0) stays body-frontal (LPLC2/LC4), 1:1
degrees so a 40° disc still covers several hexes. One 150° +X pinhole
drops the ipsilateral rim, so each eye has a forward camera and an
ipsilateral side camera (cant 90°). The rear 75–165° of the biological
eye has no flyvis cells.

    Drosophila: each eye covers roughly −15°…+165° azimuth ipsilateral with
~10–20° frontal overlap (Buchner / Heisenberg). Hexes outside that
envelope stay mid-gray: a +35° object is left-eye only (right eye's
contra limit is +15°).

``box_eye_uv`` (legacy) packed the lattice into a square pinhole. Combined
with a 40° outward / 20° down cant that put the frontal 0–70° sector on
the nasal/upper periphery — LPLC2/LC4's least sensitive hexes.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional, Tuple

import numpy as np

from threedfly.brain.constants import DT, HEX_DEG, HEX_EXTENT, N_HEX
from threedfly.brain.stimuli import hex_degrees

# Must match threedfly/sim/mjcf.py (forward + ipsilateral side cameras).
EYE_FOVY_DEG = 150.0
EYE_CANT_DEG = 0.0
EYE_DOWN_DEG = 0.0
EYE_SIDE_CANT_DEG = 90.0
# Biological envelope (Buchner / Heisenberg). Lattice (0,0) = frontal;
# xd/yd are scaled onto this extent so rear hexes are real directions.
BIO_AZ_CONTRA_DEG = 15.0
BIO_AZ_IPSI_DEG = 165.0
BIO_EL_MAX_DEG = 70.0
UNSEEN_GRAY = 0.5
# Scene-independent photoreceptor rest (flyvis gray / calib mean).
LUM_REST_INIT = 0.5
# flyvis neighbor spacing (hex_to_pixel × HEX_DEG). BoxEye averages a
# box of this size, then samples the hex centre — not a 5° Gaussian on
# top of an already 5.8° lattice (that delayed DNp01 61° → 147°).
HEX_SPACING_DEG = float(HEX_DEG) * (3.0 ** 0.5)  # 5.8°
# Legacy Gaussian kept for coverage / diagnosis only.
ACCEPT_FWHM_DEG = 5.0
ACCEPT_SIGMA_DEG = ACCEPT_FWHM_DEG / 2.35482004503
ACCEPT_N = 5
# flyvis ContrastBrightness around a p80 background. 1.10 is in the
# training log-normal (c≈1, std 0.2). Disc is matte black.
FLYVIS_CONTRAST_GAIN = 1.10
# Photoreceptor-like luminance rest. Rest follows input with
# exp(−dt/τ), τ = 5 s. Initialised to the calibration mean, never to
# frame 1 (that mapped a static scene to exactly 0.5 on frame 2).
LUM_TAU_S = 5.0
LUM_DT_S = 0.02  # default visual / control tick; override per step.


def _sample_image_hex(image: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Bilinear-sample gray/first-channel image at UV in [0, 1].

    Official cameras are 320 px / 150° (0.47°/px). Bilinear then adds
    ≪1° — the 5° FWHM Gaussian is the acceptance, not a second blur
    on a 1° raster. Nearest-neighbor at 160 px broke L/R hex symmetry.
    """
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


@lru_cache(maxsize=4)
def _lattice_xy_deg(extent: int = HEX_EXTENT) -> tuple[np.ndarray, np.ndarray]:
    _, _, xd, yd = hex_degrees(extent)
    return np.asarray(xd, dtype=np.float32), np.asarray(yd, dtype=np.float32)


def lattice_span_deg(extent: int = HEX_EXTENT) -> tuple[float, float]:
    """Raw flyvis (xd, yd) half-spans in degrees (≈75°)."""
    xd, yd = _lattice_xy_deg(extent)
    return float(np.max(np.abs(xd))), float(np.max(np.abs(yd)))


def hex_world_spacing_deg(extent: int = HEX_EXTENT) -> float:
    """Neighbor spacing in body degrees. flyvis is a 5.8° lattice."""
    return float(HEX_SPACING_DEG)


def hex_body_azel_deg(side: str, extent: int = HEX_EXTENT) -> tuple[np.ndarray, np.ndarray]:
    """Body-frame azimuth / elevation (deg) of each flyvis hex.

    +azimuth = left (+Y). Lattice centre (0, 0) stays body-frontal
    (LPLC2/LC4). flyvis is a ±75° projection of the eye — 1:1 degrees
    so a 40° disc still covers several hexes. The rear 75–165° of the
    biological eye has no flyvis cells. Right-eye +xd looks right.
    A second (90° cant) camera covers hexes a single 150° +X pinhole
    drops at the rim.
    """
    if side not in ("L", "R"):
        raise ValueError(f"side must be L or R, got {side!r}")
    xd, yd = _lattice_xy_deg(extent)
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

    Left: −15° (contra) … +165° (ipsi). Right: mirrored. The lattice is
    scaled onto that extent; the contra cut makes a frontolat loom
    monocular.
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


class _BgRest:
    """Per-hex luminance rest. Starts at the calibration mean, not frame 1."""

    def __init__(self) -> None:
        self.hex_L: Optional[np.ndarray] = None
        self.hex_R: Optional[np.ndarray] = None

    def reset(self) -> None:
        self.hex_L = None
        self.hex_R = None

    def update_hex(
        self, side: str, hex_frame: np.ndarray, dt: Optional[float] = None
    ) -> np.ndarray:
        """Advance rest toward ``hex_frame`` with exp(−dt/τ). Return rest."""
        x = np.asarray(hex_frame, dtype=np.float32)
        step = float(LUM_DT_S if dt is None else dt)
        alpha = float(1.0 - np.exp(-step / float(LUM_TAU_S)))
        prev = self.hex_L if side == "L" else self.hex_R
        if prev is None or prev.shape != x.shape:
            rest = np.full(x.shape, float(LUM_REST_INIT), dtype=np.float32)
        else:
            rest = prev
        rest = (rest + alpha * (x - rest)).astype(np.float32)
        if side == "L":
            self.hex_L = rest
        else:
            self.hex_R = rest
        return rest


BG_REST = _BgRest()


def reset_luminance_rest() -> None:
    """Call at brain reset / prewarm so each scenario starts clean."""
    BG_REST.reset()


def raw_stereo_hex(
    left_img: np.ndarray,
    right_img: np.ndarray,
    left_side_img: Optional[np.ndarray] = None,
    right_side_img: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """BoxEye hex luminance. Does not touch photoreceptor rest."""
    hex_L, ok_L = _sample_hex_box(left_img, "L")
    hex_R, ok_R = _sample_hex_box(right_img, "R")
    if left_side_img is not None:
        s, ok_s = _sample_hex_box(left_side_img, "L", cant_deg=EYE_SIDE_CANT_DEG)
        fill = (~ok_L) & ok_s
        hex_L = np.where(fill, s, hex_L).astype(np.float32)
        ok_L = ok_L | ok_s
    if right_side_img is not None:
        s, ok_s = _sample_hex_box(right_side_img, "R", cant_deg=EYE_SIDE_CANT_DEG)
        fill = (~ok_R) & ok_s
        hex_R = np.where(fill, s, hex_R).astype(np.float32)
        ok_R = ok_R | ok_s
    return hex_L, hex_R


def init_luminance_rest(hex_L: np.ndarray, hex_R: np.ndarray) -> None:
    """τ→∞ equivalent: set per-hex rest to this frame. Next luminance ≈ 0.5.

    A several-τ prewarm at τ=5 s would reach the same DC-free field.
    This is that initialisation, not a sim-object paint. Call only on a
    no-object frame so a present disc is not grayed out.
    """
    BG_REST.hex_L = np.asarray(hex_L, dtype=np.float32).copy()
    BG_REST.hex_R = np.asarray(hex_R, dtype=np.float32).copy()


def init_luminance_rest_from_images(
    left_img: np.ndarray,
    right_img: np.ndarray,
    left_side_img: Optional[np.ndarray] = None,
    right_side_img: Optional[np.ndarray] = None,
) -> None:
    """Initialise photoreceptor rest from rendered pixels only (no object state)."""
    init_luminance_rest(
        *raw_stereo_hex(left_img, right_img, left_side_img, right_side_img)
    )


# Back-compat aliases.
snap_luminance_rest = init_luminance_rest
snap_luminance_rest_from_images = init_luminance_rest_from_images


def _luminance_to_gray(
    hex_frame: np.ndarray,
    valid: np.ndarray,
    side: str = "L",
    dt: Optional[float] = None,
) -> np.ndarray:
    """Match flyvis input stats: rest → 0.5, fixed contrast gain.

    Rest starts at the calibration mean (scene-independent) and follows
    the hex luminance with exp(−dt/τ), τ = 5 s. A static scene keeps
    ~37% of its initial contrast at t = τ. p99 handoff stays on the
    fixed calib set. Generic; no object geometry.
    """
    out = np.asarray(hex_frame, dtype=np.float32).copy()
    m = np.asarray(valid, dtype=bool)
    if np.count_nonzero(m) < 8:
        return out
    rest = BG_REST.update_hex(side, out, dt=dt)
    out[m] = np.clip(
        UNSEEN_GRAY + FLYVIS_CONTRAST_GAIN * (out[m] - rest[m]), 0.0, 1.0
    )
    return out


def hex_coverage(side: str) -> dict:
    """Body-frame az/el actually sampled by each hex (for plots / tests)."""
    az, el = hex_body_azel_deg(side)
    look, x_ax, y_ax = eye_camera_axes(side)
    uv, valid_f = hex_camera_uv(side, 160, 160)
    _, valid_s = hex_camera_uv(side, 160, 160, cant_deg=EYE_SIDE_CANT_DEG)
    valid = valid_f | valid_s
    bio = hex_in_bio_envelope(side, az)
    return {
        "side": side,
        "az_deg": az,
        "el_deg": el,
        "uv": uv,
        "valid": valid,
        "valid_front": valid_f,
        "valid_side": valid_s,
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


def hex_box_kernel_px(height: int, fovy_deg: float = EYE_FOVY_DEG) -> int:
    """flyvis BoxEye kernel: one (scaled) hex-spacing box in camera pixels."""
    deg_per_px = float(fovy_deg) / max(float(height), 1.0)
    return max(1, int(round(hex_world_spacing_deg() / max(deg_per_px, 1e-6))))


def _box_filter_gray(image: np.ndarray, k: int) -> np.ndarray:
    """Mean box filter (flyvis BoxEye conv / kernel_size²)."""
    if image.ndim == 3:
        gray = image.mean(axis=2)
    else:
        gray = image
    g = np.asarray(gray, dtype=np.float32)
    peak = 255.0 if float(g.max() if g.size else 0.0) > 1.5 else 1.0
    g = np.clip(g / peak, 0.0, 1.0)
    if int(k) <= 1:
        return g
    from scipy.ndimage import uniform_filter

    return uniform_filter(g, size=int(k), mode="nearest").astype(np.float32)


def _sample_hex_box(
    image: np.ndarray, side: str, cant_deg: float = EYE_CANT_DEG
) -> tuple[np.ndarray, np.ndarray]:
    """Box-average + hex-centre sample. Returns (hex, valid)."""
    image = np.asarray(image)
    h, w = int(image.shape[0]), int(image.shape[1])
    k = hex_box_kernel_px(h)
    filt = _box_filter_gray(image, k)
    uv, hex_ok = hex_camera_uv(side, h, w, cant_deg=cant_deg)
    hex_frame = _sample_image_hex(filt, uv)
    hex_frame = np.where(hex_ok, hex_frame, UNSEEN_GRAY).astype(np.float32)
    return hex_frame, hex_ok


def _image_to_hex_box(
    image: np.ndarray,
    side: str,
    side_image: Optional[np.ndarray] = None,
    dt: Optional[float] = None,
) -> np.ndarray:
    """flyvis BoxEye on the front camera, fill rear hexes from the side view."""
    hex_frame, hex_ok = _sample_hex_box(image, side, cant_deg=EYE_CANT_DEG)
    if side_image is not None:
        hex_s, ok_s = _sample_hex_box(side_image, side, cant_deg=EYE_SIDE_CANT_DEG)
        fill = (~hex_ok) & ok_s
        hex_frame = np.where(fill, hex_s, hex_frame).astype(np.float32)
        hex_ok = hex_ok | ok_s
    return _luminance_to_gray(hex_frame, hex_ok, side=side, dt=dt)


def stereo_to_hex(
    left_img: np.ndarray,
    right_img: np.ndarray,
    uv: Optional[np.ndarray] = None,
    left_side_img: Optional[np.ndarray] = None,
    right_side_img: Optional[np.ndarray] = None,
    dt: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Left and right rendered eye images → (721,), (721,) hex frames.

    Official path: flyvis BoxEye (scaled hex-spacing box average, then
    hex centre) on the forward camera, with rear hexes filled from the
    ipsilateral side camera. Then photoreceptor rest (τ=5 s, init at
    the calibration mean). Generic for every geom. No sim object
    pose/radius.
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
    return (
        _image_to_hex_box(left_img, "L", side_image=left_side_img, dt=dt),
        _image_to_hex_box(right_img, "R", side_image=right_side_img, dt=dt),
    )
