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
# Must match mjcf loom_geom sphere radius.
LOOM_RADIUS_M = 0.58
# |body azimuth| at rise onset. ≥ this is a side loom; below is frontal GF.
LATERAL_AZ_DEG = 30.0


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


def body_azimuth_deg(obj_xy, fly_xy, heading_rad: float) -> float:
    """Object bearing in the body frame (deg). +azimuth = left (+Y)."""
    rel = np.asarray(obj_xy, dtype=float)[:2] - np.asarray(fly_xy, dtype=float)[:2]
    world = float(np.arctan2(rel[1], rel[0]))
    return float(np.degrees(wrap_heading_delta(world, float(heading_rad))))


def object_behind_fly(obj_xyz, fly_xyz, heading_rad: float, behind_m: float = -0.4) -> bool:
    """True when the object is behind the fly (body +X), not a world-x cut."""
    rel = np.asarray(obj_xyz, dtype=float)[:2] - np.asarray(fly_xyz, dtype=float)[:2]
    fwd = np.array([np.cos(float(heading_rad)), np.sin(float(heading_rad))], dtype=float)
    return float(np.dot(rel, fwd)) < float(behind_m)


def angular_size_deg(distance: float, radius: float = LOOM_RADIUS_M) -> float:
    d = max(float(distance), 1e-6)
    return float(2.0 * np.degrees(np.arctan(float(radius) / d)))


def loom_geometry(obj_xyz, fly_xyz, heading_rad: float, radius: float = LOOM_RADIUS_M) -> dict:
    obj = np.asarray(obj_xyz, dtype=float)
    fly = np.asarray(fly_xyz, dtype=float)
    dist = float(np.linalg.norm(obj[:2] - fly[:2]))
    az = body_azimuth_deg(obj[:2], fly[:2], heading_rad)
    return {
        "azimuth_deg": az,
        "distance_m": dist,
        "angular_size_deg": angular_size_deg(dist, radius),
        "class": "lateral" if abs(az) >= LATERAL_AZ_DEG else "frontal",
    }


def classify_loom(azimuth_deg: Optional[float], lateral_deg: float = LATERAL_AZ_DEG) -> str:
    if azimuth_deg is None or not np.isfinite(float(azimuth_deg)):
        return "unknown"
    return "lateral" if abs(float(azimuth_deg)) >= float(lateral_deg) else "frontal"


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
    threat_side: str,
    window: int = 8,
) -> float:
    """Fraction of events that turned away from the *scenario* threat side.

    Does not use the event's DNp03 label or commanded yaw (those are
    self-consistent even when the fly turns into the wall).
    """
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
        step = int(getattr(e, "step", 0))
        if escape_turned_away(headings, step, threat_side, window=window):
            ok += 1
    return float(ok) / float(len(evs))


def escape_turned_into(
    headings: Sequence[float],
    event_step: int,
    threat_side: str,
    window: int = 8,
    min_abs: float = 0.02,
) -> bool:
    """True if heading after the burst turns toward the looming side."""
    h = np.asarray(list(headings), dtype=float)
    if h.size == 0:
        return False
    i0 = int(np.clip(event_step, 0, h.size - 1))
    i1 = int(np.clip(event_step + window, 0, h.size - 1))
    d = wrap_heading_delta(float(h[i1]), float(h[i0]))
    if threat_side == "L":
        return d > min_abs
    if threat_side == "R":
        return d < -min_abs
    return False


def geometry_aware_event_ok(
    headings: Sequence[float],
    event_step: int,
    threat_side: str,
    azimuth_onset_deg: Optional[float],
    window: int = 8,
) -> bool:
    """Lateral: must turn away. Frontal: escape without turning into the object."""
    cls = classify_loom(azimuth_onset_deg)
    into = escape_turned_into(headings, event_step, threat_side, window=window)
    if cls == "frontal":
        return not into
    return escape_turned_away(headings, event_step, threat_side, window=window)


def geometry_aware_frac(
    headings: Sequence[float],
    events: Iterable,
    threat_side: str,
    azimuth_onset_deg: Optional[float],
    escaped: bool,
    braked: bool = False,
    window: int = 8,
) -> float:
    """Scenario score. Lateral no-escape is 0; frontal no-escape passes if braked."""
    cls = classify_loom(azimuth_onset_deg)
    evs = []
    for raw in events:
        e = _as_event(raw)
        if getattr(e, "engineered", False):
            continue
        evs.append(e)
    if not evs:
        if cls == "frontal" and braked:
            return 1.0
        return 0.0 if threat_side else float("nan")
    ok = 0
    for e in evs:
        step = int(getattr(e, "step", 0))
        if geometry_aware_event_ok(headings, step, threat_side, azimuth_onset_deg, window=window):
            ok += 1
    return float(ok) / float(len(evs))
