"""Synthetic hex-lattice movies matching research_r4/fv4_run.py."""

from __future__ import annotations

from typing import Dict

import numpy as np

from threedfly.brain.constants import DT, HEX_DEG, HEX_EXTENT, N_HEX


def get_hex_coords(extent: int = HEX_EXTENT) -> tuple[np.ndarray, np.ndarray]:
    """flyvis ``get_hex_coords``: cube-range hex lattice."""
    u, v = [], []
    for q in range(-extent, extent + 1):
        for r in range(max(-extent, -extent - q), min(extent, extent - q) + 1):
            u.append(q)
            v.append(r)
    return np.asarray(u, dtype=np.int32), np.asarray(v, dtype=np.int32)


def hex_to_pixel(u, v) -> tuple[np.ndarray, np.ndarray]:
    """flyvis default ``hex_to_pixel`` (used when flyvis is not installed)."""
    try:
        from flyvis.utils.hex_utils import hex_to_pixel as _h

        return _h(u, v)
    except ImportError:
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        return 1.5 * v, -np.sqrt(3.0) * (u + v / 2.0)


def hex_degrees(extent: int = HEX_EXTENT) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    u, v = get_hex_coords(extent)
    x, y = hex_to_pixel(u, v)
    return u, v, x * HEX_DEG, y * HEX_DEG


def n_hex(extent: int = HEX_EXTENT) -> int:
    u, _ = get_hex_coords(extent)
    assert len(u) == (3 * extent * (extent + 1) + 1) or extent == HEX_EXTENT
    return int(len(u))


def gray(T: float, value: float = 0.5, dt: float = DT, n: int = N_HEX) -> np.ndarray:
    return np.full((int(round(T / dt)), n), value, dtype=np.float32)


def grating(
    T: float,
    direction: float,
    period: float = 30.0,
    speed: float = 90.0,
    dt: float = DT,
) -> np.ndarray:
    _, _, xd, yd = hex_degrees()
    t = np.arange(int(round(T / dt))) * dt
    return (
        0.5 + 0.25 * np.sign(np.sin(2 * np.pi * (xd[None, :] - direction * speed * t[:, None]) / period))
    ).astype(np.float32)


def loom(
    T: float,
    lv: float = 0.04,
    th0: float = 10.0,
    thmax: float = 90.0,
    x0: float = 0.0,
    dt: float = DT,
) -> np.ndarray:
    _, _, xd, yd = hex_degrees()
    t = np.arange(int(round(T / dt))) * dt
    t0 = lv / np.tan(np.deg2rad(th0) / 2.0)
    tt = np.minimum(t, t0 - lv / np.tan(np.deg2rad(thmax) / 2.0))
    th = 2.0 * np.rad2deg(np.arctan(lv / (t0 - tt)))
    r = np.hypot(xd - x0, yd)
    return np.where(r[None, :] <= th[:, None] / 2.0, 0.0, 0.5).astype(np.float32)


def expansion(
    T: float,
    x0: float = 75.0,
    period: float = 30.0,
    speed: float = 90.0,
    dt: float = DT,
) -> np.ndarray:
    """Concentric rings expanding from a frontal FOE (fv4_run.py)."""
    _, _, xd, yd = hex_degrees()
    t = np.arange(int(round(T / dt))) * dt
    rho = np.hypot(xd - x0, yd)
    return (
        0.5 + 0.25 * np.sign(np.sin(2 * np.pi * (rho[None, :] - speed * t[:, None]) / period))
    ).astype(np.float32)


def calibration_conditions(pre: float = 1.0, stim: float = 1.0, dt: float = DT) -> Dict[str, np.ndarray]:
    """Gray + loom + gratings + expansion used to fit V_gray / p99.9."""
    return {
        "gray": gray(pre + stim + 1.0, dt=dt),
        "loom": np.concatenate([gray(pre, dt=dt), loom(0.6, dt=dt), gray(stim + 1.0 - 0.6, dt=dt)]),
        "grat_px": np.concatenate([gray(pre, dt=dt), grating(stim, +1, dt=dt), gray(1.0, dt=dt)]),
        "grat_nx": np.concatenate([gray(pre, dt=dt), grating(stim, -1, dt=dt), gray(1.0, dt=dt)]),
        "loom_front": np.concatenate(
            [gray(pre, dt=dt), loom(0.6, x0=60.0, dt=dt), gray(stim + 1.0 - 0.6, dt=dt)]
        ),
        "expand": np.concatenate([gray(pre, dt=dt), expansion(stim, dt=dt), gray(1.0, dt=dt)]),
    }


def acceptance_eye_movies(pre: float = 1.0, stim: float = 1.0, dt: float = DT) -> Dict[str, np.ndarray]:
    """Per-eye hex movies. Scenarios pair these as in e2e4.py."""
    return calibration_conditions(pre=pre, stim=stim, dt=dt)


# (left_cond, right_cond) — research_r4/e2e4.py
SCENARIOS = {
    "loomL": ("loom", "gray"),
    "loomR": ("gray", "loom"),
    "CW": ("grat_px", "grat_nx"),
    "CCW": ("grat_nx", "grat_px"),
    "expand": ("expand", "expand"),
    "loomF": ("loom_front", "loom_front"),
    "gray": ("gray", "gray"),
}
