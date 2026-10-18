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
# Official looms start no larger than this (grow from a point / start far).
LOOM_START_ANG_MAX_DEG = 5.0
# Place a matching-angle sphere inside the room when the virtual object is outside.
ROOM_LOOM_CLEARANCE_M = 4.6
# |body azimuth| at rise onset. ≥ this is a side loom; below is frontal GF.
LATERAL_AZ_DEG = 30.0
# Acceptance bands (von Reyn / Klapoetke): frontolateral collision-course
# looms gate the official escape score. Eccentric |az|≳70° is report-only
# because LPLC2/GF are frontally biased.
FRONTAL_AZ_MAX_DEG = 20.0
FRONTOLAT_AZ_MIN_DEG = 20.0
FRONTOLAT_AZ_MAX_DEG = 70.0


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


def start_distance_for_angular_size(
    ang_deg: float, radius: float = LOOM_RADIUS_M
) -> float:
    """Distance (m) at which a sphere of ``radius`` subtends ``ang_deg``."""
    return float(radius) / max(float(np.tan(np.radians(0.5 * float(ang_deg)))), 1e-9)


def visible_loom_placement(
    virtual_xyz,
    fly_xyz,
    radius: float = LOOM_RADIUS_M,
    max_range: float = ROOM_LOOM_CLEARANCE_M,
) -> tuple[tuple[float, float, float], float]:
    """In-room sphere on the same bearing whose visual angle matches ``virtual_xyz``.

    The room is ±5 m, so a 5° start of a 0.58 m disc sits outside the walls.
    Eyes see this placed sphere; metrics keep the virtual collision-course.
    """
    virt = np.asarray(virtual_xyz, dtype=float).reshape(3)
    fly = np.asarray(fly_xyz, dtype=float).reshape(3)
    rel = virt - fly
    d_virt = float(np.hypot(rel[0], rel[1]))
    az = float(np.arctan2(rel[1], rel[0]))
    theta = 2.0 * np.arctan(float(radius) / max(d_virt, 1e-6))
    d_place = min(d_virt, float(max_range))
    r_place = float(d_place) * float(np.tan(0.5 * theta))
    pos = fly.copy()
    pos[0] += d_place * float(np.cos(az))
    pos[1] += d_place * float(np.sin(az))
    pos[2] = virt[2]
    return (float(pos[0]), float(pos[1]), float(pos[2])), float(max(r_place, 1e-4))


def angular_size_deg(distance: float, radius: float = LOOM_RADIUS_M) -> float:
    d = max(float(distance), 1e-6)
    return float(2.0 * np.degrees(np.arctan(float(radius) / d)))


def dnp01_baseline_shows_loom(
    *,
    present_at_prewarm: bool,
    appear_after_prewarm: bool,
    start_ang_deg: Optional[float],
    start_ang_max_deg: float = LOOM_START_ANG_MAX_DEG,
) -> bool:
    """Whether the object is visible while DNp01 rest is captured.

    Biological GF has no loom-present subtraction. Prewarm snaps
    photoreceptor rest to this frame (DC → mid-gray) and freezes DNp01
    there. A far (≤5°) disc in that snapshot is grayed out and the
    inbound climb is absorbed (camera rest 0.50–0.64 vs uniform-field
    0.008 Hz). Hide those discs, snap/settle, then show them. An
    already-large receding disc stays visible so we do not inject a
    40° onset.
    """
    if appear_after_prewarm or not present_at_prewarm:
        return False
    if start_ang_deg is not None and float(start_ang_deg) <= float(start_ang_max_deg) + 0.51:
        return False
    return True


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


def closest_approach_step(distances) -> Optional[int]:
    """Step of minimum object–fly distance. None if there is no finite sample."""
    d = np.asarray(list(distances), dtype=float)
    if d.size == 0 or not np.any(np.isfinite(d)):
        return None
    return int(np.nanargmin(d))


def fire_minus_closest(fire_step: Optional[int], closest_step: Optional[int]) -> Optional[int]:
    """``fire_step - closest_approach_step``. Negative / zero = on the way in."""
    if fire_step is None or closest_step is None:
        return None
    return int(fire_step) - int(closest_step)


def visible_approaching_ok(
    fire_step: Optional[int],
    closest_step: Optional[int],
    *,
    visible: bool,
) -> bool:
    """True only for a fire while the object is still visible and closing.

    An escape after closest approach is a fail: disappearance / recession
    is not looming (Klapoetke / von Reyn GF window is on the approach).
    """
    if not visible or fire_step is None or closest_step is None:
        return False
    return int(fire_step) <= int(closest_step)


def onset_azimuth_band(azimuth_deg: Optional[float]) -> str:
    """frontal / frontolat / eccentric / unknown. Gate uses frontolat."""
    if azimuth_deg is None or not np.isfinite(float(azimuth_deg)):
        return "unknown"
    a = abs(float(azimuth_deg))
    if a < FRONTAL_AZ_MAX_DEG:
        return "frontal"
    if a <= FRONTOLAT_AZ_MAX_DEG:
        return "frontolat"
    return "eccentric"


def nearest_wall_ray(xyz, heading_rad: float, room_xy: float = ROOM_XY) -> dict:
    """First room wall along the heading. Distance (m) and time-of-flight at 1 m/s."""
    p = np.asarray(xyz, dtype=float)[:2]
    u = np.array([np.cos(float(heading_rad)), np.sin(float(heading_rad))], dtype=float)
    best = None
    faces = (
        ("+X", np.array([1.0, 0.0]), float(room_xy)),
        ("-X", np.array([-1.0, 0.0]), float(room_xy)),
        ("+Y", np.array([0.0, 1.0]), float(room_xy)),
        ("-Y", np.array([0.0, -1.0]), float(room_xy)),
    )
    for name, nrm, lim in faces:
        denom = float(np.dot(u, nrm))
        if denom <= 1e-9:
            continue
        dist = (lim - float(np.dot(p, nrm))) / denom
        if dist <= 0:
            continue
        hit = p + dist * u
        if max(abs(hit[0]), abs(hit[1])) > float(room_xy) + 1e-6:
            continue
        rec = {"wall": name, "distance_m": float(dist), "hit_xy": (float(hit[0]), float(hit[1]))}
        if best is None or rec["distance_m"] < best["distance_m"]:
            best = rec
    if best is None:
        return {"wall": None, "distance_m": None, "ttc_s": None, "hit_xy": None}
    return best


def wall_ttc_s(xyz, heading_rad: float, speed_mps: float, room_xy: float = ROOM_XY) -> dict:
    rec = nearest_wall_ray(xyz, heading_rad, room_xy=room_xy)
    dist = rec.get("distance_m")
    rec["speed_mps"] = float(speed_mps)
    rec["ttc_s"] = None if dist is None else float(dist) / max(float(speed_mps), 1e-6)
    rec["expansion_rate_1_per_s"] = (
        None if dist is None else float(speed_mps) / max(float(dist), 1e-6)
    )
    return rec


def collision_course_object(
    az_deg: float,
    distance_m: float,
    lv_s: float,
    radius: float = LOOM_RADIUS_M,
    fly_xyz=(0.0, 0.0, 1.5),
    fly_yaw: float = 0.0,
) -> dict:
    """Dark disc aimed at the fly (impact parameter 0). ``lv_s`` is l/v in seconds."""
    az = np.radians(float(az_deg)) + float(fly_yaw)
    fly = np.asarray(fly_xyz, dtype=float)
    start = fly.copy()
    start[0] += float(distance_m) * float(np.cos(az))
    start[1] += float(distance_m) * float(np.sin(az))
    speed = float(radius) / max(float(lv_s), 1e-6)
    vel = (-speed * float(np.cos(az)), -speed * float(np.sin(az)), 0.0)
    if abs(float(az_deg)) < 1e-6:
        side = "F"
    else:
        side = "L" if float(az_deg) > 0 else "R"
    return {
        "pose": (float(fly[0]), float(fly[1]), float(fly[2])),
        "yaw": float(fly_yaw),
        "vel": (0.02, 0.0, 0.0),
        "loom_start": (float(start[0]), float(start[1]), float(start[2])),
        "loom_vel": vel,
        "az_design_deg": float(az_deg),
        "lv_s": float(lv_s),
        "distance_m": float(distance_m),
        "closing_speed_mps": float(speed),
        "threat_side": side,
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
