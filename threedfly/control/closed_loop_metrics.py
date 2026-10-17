"""Closed-loop acceptance metrics (no flyvis / MuJoCo import).

Wall-contact comparisons are always per 100 steps so a 400-step run is not
scored against PR #16's 81/2000. Escape direction is the signed heading
change after the event, not the threat-side label.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import numpy as np

ROOM_XY = 5.0
BIN_M = 0.5
# PR #16 v5: 81 contacts in 2000 steps.
PR16_WALLS = 81
PR16_STEPS = 2000
PR16_WALLS_PER_100 = 100.0 * PR16_WALLS / PR16_STEPS  # 4.05


def xy_extent(traj: np.ndarray) -> tuple[float, float]:
    """NumPy 2-safe range of XY (``ndarray.ptp`` was removed)."""
    if traj is None or getattr(traj, "size", 0) == 0:
        return 0.0, 0.0
    t = np.asarray(traj, dtype=float)
    if t.ndim != 2 or t.shape[1] < 2:
        return 0.0, 0.0
    return float(np.ptp(t[:, 0])), float(np.ptp(t[:, 1]))


def coverage(traj: np.ndarray, room_xy: float = ROOM_XY, bin_m: float = BIN_M) -> float:
    if traj is None or getattr(traj, "size", 0) == 0:
        return 0.0
    t = np.asarray(traj, dtype=float)
    nbin = int(2 * room_xy / bin_m)
    gx = np.clip(((t[:, 0] + room_xy) / bin_m).astype(int), 0, nbin - 1)
    gy = np.clip(((t[:, 1] + room_xy) / bin_m).astype(int), 0, nbin - 1)
    visited = len(set(zip(gx.tolist(), gy.tolist())))
    return float(visited) / float(nbin * nbin)


def walls_per_100(n_contacts: int, n_steps: int) -> float:
    if n_steps <= 0:
        return 0.0
    return 100.0 * float(n_contacts) / float(n_steps)


def wrap_heading_delta(h1: float, h0: float) -> float:
    return float(np.arctan2(np.sin(h1 - h0), np.cos(h1 - h0)))


def escape_turned_away(
    headings: Sequence[float],
    event_step: int,
    threat_side: str,
    window: int = 8,
    min_abs: float = 0.02,
) -> bool:
    """True if heading after the burst turns away from the looming side.

    Body frame: +yaw / increasing heading = left turn. Left threat → right
    turn → heading decrease.
    """
    h = np.asarray(list(headings), dtype=float)
    if h.size == 0:
        return False
    i0 = int(np.clip(event_step, 0, h.size - 1))
    i1 = int(np.clip(event_step + window, 0, h.size - 1))
    d = wrap_heading_delta(float(h[i1]), float(h[i0]))
    if threat_side == "L":
        return d < -min_abs
    if threat_side == "R":
        return d > min_abs
    return False


def _as_event(e):
    if isinstance(e, dict):
        class _E:
            pass

        o = _E()
        o.step = e["step"]
        o.threat_side = e.get("threat_side", "")
        o.engineered = bool(e.get("engineered", False))
        o.yaw = float(e.get("yaw", 0.0))
        return o
    return e


def direction_correct_frac(
    headings: Sequence[float],
    events: Iterable,
    window: int = 8,
) -> float:
    evs = []
    for raw in events:
        e = _as_event(raw)
        if getattr(e, "engineered", False):
            continue
        evs.append(e)
    if not evs:
        return float("nan")
    ok = 0
    for e in evs:
        side = getattr(e, "threat_side", "")
        step = int(getattr(e, "step", 0))
        yaw = float(getattr(e, "yaw", 0.0))
        by_heading = escape_turned_away(headings, step, side, window=window)
        # Command yaw is a fallback when the heading buffer is too short.
        by_yaw = (side == "L" and yaw < -0.02) or (side == "R" and yaw > 0.02)
        if by_heading or by_yaw:
            ok += 1
    return float(ok) / float(len(evs))
