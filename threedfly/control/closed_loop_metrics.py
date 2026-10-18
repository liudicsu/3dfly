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
# Closed-loop flight miss: surface clearance of the 0.58 m visual disc.
# d_centre − 0.58 > 0.1 m. A 0.1 m centre miss is still a collision.
MISS_SURFACE_M = 0.1
MISS_CENTER_M = LOOM_RADIUS_M + MISS_SURFACE_M  # 0.68 m; kept as the equivalent centre
# Net heading away from the true side. ~4.6°; no-object wander is the control.
NET_HEADING_AWAY_RAD = 0.08
# Object stayed ipsilateral at closest approach (did not cross in front).
FAR_SIDE_AZ_DEG = 40.0
YAW_TERM_KEYS = ("dna02", "dnp15", "hse", "dnp03", "vpn_loom")


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

    Far (≤5°) approach discs are hidden so photoreceptor rest can be
    initialised to the no-object scene (τ→∞ equivalent of the 5 s LMC
    high-pass) and DNp01 rest is the uniform-field rate. An already-large
    receding disc stays visible and that initialisation is *skipped*, so
    the standing disc is not grayed out and then re-appears as an onset.
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


def min_center_distance_m(distances) -> Optional[float]:
    d = np.asarray(list(distances), dtype=float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return None
    return float(np.min(d))


def min_surface_clearance_m(distances, radius: float = LOOM_RADIUS_M) -> Optional[float]:
    d = min_center_distance_m(distances)
    if d is None:
        return None
    return float(d) - float(radius)


def net_heading_delta(headings) -> float:
    """Cumulative (unwrapped) heading change, first → last sample.

    Summing wrapped per-step deltas keeps a >180° away turn signed as
    away. The old end-minus-start wrap scored a 300° right turn from a
    left loom as a 60° *left* turn (box r11: "L az30 turned toward").
    """
    h = np.asarray(list(headings), dtype=float)
    if h.size < 2:
        return 0.0
    d = np.diff(h)
    return float(np.sum(np.arctan2(np.sin(d), np.cos(d))))


def heading_away_from_side(
    delta_rad: float,
    threat_side: str,
    min_abs: float = NET_HEADING_AWAY_RAD,
) -> bool:
    """+yaw = left. Left threat → heading decrease is away."""
    if threat_side == "L":
        return float(delta_rad) < -float(min_abs)
    if threat_side == "R":
        return float(delta_rad) > float(min_abs)
    return False


def passed_far_side(
    azimuths,
    threat_side: str,
    ca_step: Optional[int] = None,
    min_az_deg: float = FAR_SIDE_AZ_DEG,
) -> bool:
    """True when at closest approach the object is still on the threat side."""
    az = np.asarray(list(azimuths), dtype=float)
    if az.size == 0:
        return False
    i = int(az.size - 1) if ca_step is None else int(ca_step)
    i = int(np.clip(i, 0, az.size - 1))
    a = float(az[i])
    if not np.isfinite(a):
        return False
    if threat_side == "L":
        return a >= float(min_az_deg)
    if threat_side == "R":
        return a <= -float(min_az_deg)
    return False


def yaw_rates(headings, dt: float) -> np.ndarray:
    """Wrapped heading derivative (rad/s)."""
    h = np.asarray(list(headings), dtype=float)
    if h.size < 2:
        return np.zeros(0, dtype=float)
    dh = np.arctan2(np.sin(np.diff(h)), np.cos(np.diff(h)))
    return dh / max(float(dt), 1e-9)


def peak_yaw_rate_rad_s(headings, dt: float) -> float:
    w = yaw_rates(headings, dt)
    return 0.0 if w.size == 0 else float(np.max(np.abs(w)))


def mean_abs_yaw_rate_rad_s(headings, dt: float) -> float:
    w = yaw_rates(headings, dt)
    return 0.0 if w.size == 0 else float(np.mean(np.abs(w)))


def flight_avoidance_ok(
    *,
    min_center_m: Optional[float],
    headings,
    threat_side: str,
    azimuths=None,
    ca_step: Optional[int] = None,
    miss_m: float = MISS_SURFACE_M,
    radius: float = LOOM_RADIUS_M,
) -> dict:
    """Flight success: surface miss and turn the object to the far side.

    GF fire is not required (Tammero & Dickinson 2002; Muijres et al. 2014:
    flying Drosophila mostly bank away; GF takeoff is primarily from
    standing). ``missed`` is ``d_centre − 0.58 > 0.1 m``. A 0.1 m centre
    miss is still a collision with the visual disc.
    """
    surface = None if min_center_m is None else float(min_center_m) - float(radius)
    missed = surface is not None and float(surface) > float(miss_m)
    dhead = net_heading_delta(headings)
    away = heading_away_from_side(dhead, threat_side)
    far = passed_far_side(azimuths or [], threat_side, ca_step=ca_step)
    return {
        "ok": bool(missed and (away or far)),
        "missed": bool(missed),
        "min_center_m": None if min_center_m is None else float(min_center_m),
        "min_surface_m": None if surface is None else float(surface),
        "net_heading_rad": float(dhead),
        "heading_away": bool(away),
        "passed_far_side": bool(far),
    }


def yaw_term_away_integrals(rows, threat_side: str, end_step: Optional[int] = None) -> dict:
    """Signed integral of yaw terms in the away direction. Driver = max."""
    sign = -1.0 if threat_side == "L" else (1.0 if threat_side == "R" else 0.0)
    subset = list(rows)
    if end_step is not None:
        subset = subset[: int(end_step) + 1]
    integrals = {
        k: float(sum(sign * float(r.get(k, 0.0) or 0.0) for r in subset))
        for k in YAW_TERM_KEYS
    }
    driver = max(integrals, key=integrals.get) if integrals else None
    if driver is not None and integrals[driver] <= 0.0:
        driver = None
    return {"integrals": integrals, "driver": driver}


def avoidance_mode(*, avoidance_ok: bool, gf_fired_approaching: bool) -> str:
    if gf_fired_approaching:
        return "gf_escape"
    if avoidance_ok:
        return "pre_gf_saccade"
    return "none"


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


# --- Avoidance feasibility (frontolat flight gate) ---------------------------
# Kinematic *upper bound* on how far the fly can move perpendicular to a
# collision-course object's approach line between its decision (first GF
# fire or saccade onset) and closest approach. Generous on purpose so an
# "infeasible" label is robust:
#   * the fly's whole current speed counts as already lateral (v0·T),
#   * forward thrust *plus* all spare banked lift accelerate it laterally,
#   * the only delay is rotating the thrust axis to perpendicular at the
#     calibrated peak saccade yaw rate (no smoothing / sensorimotor lag),
#   * drag is ignored.
# A centre miss of r + 0.1 m is needed for 0.1 m surface clearance because
# the object is aimed at the fly (impact parameter 0) and it is ~10–40×
# faster than the fly, so only fly motion perpendicular to the object's
# velocity changes the miss distance.
# Calibrated plant: scripts/calibrate_saccade_plant.py (unit DNa02 L−R,
# 90° in ~80 ms, peak ~2011 °/s at ×12). Used only if the run does not
# re-measure it.
PLANT_PEAK_YAW_DEG_S = 2011.0
GF_WINDOW_DEG = 50.0  # von Reyn 2014: GF spike ~40–60° subtense
RESPONSE_WINDOW_S = 0.10
MIN_AWAY_RAD = 0.02


def max_lateral_accel_mps2(
    thrust_max_n: float, lift_max_n: float, mass_kg: float, g: float = 9.81
) -> float:
    """Upper bound: full forward thrust + all lift not needed for weight."""
    w = float(mass_kg) * float(g)
    spare = float(np.sqrt(max(float(lift_max_n) ** 2 - w * w, 0.0)))
    return (float(thrust_max_n) + spare) / max(float(mass_kg), 1e-12)


def max_lateral_displacement_m(
    t_avail_s: float,
    *,
    v0_mps: float,
    a_lat_mps2: float,
    turn_rad: float,
    yaw_rate_max_rad_s: float,
) -> float:
    t = max(0.0, float(t_avail_s))
    t_turn = max(0.0, float(turn_rad)) / max(float(yaw_rate_max_rad_s), 1e-9)
    t_acc = max(0.0, t - t_turn)
    return float(v0_mps) * t + 0.5 * float(a_lat_mps2) * t_acc * t_acc


def required_decision_lead_s(
    *,
    need_m: float,
    v0_mps: float,
    a_lat_mps2: float,
    turn_rad: float,
    yaw_rate_max_rad_s: float,
) -> float:
    """Smallest decision→closest-approach time that reaches ``need_m``."""
    lo, hi = 0.0, 10.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        s = max_lateral_displacement_m(
            mid, v0_mps=v0_mps, a_lat_mps2=a_lat_mps2, turn_rad=turn_rad,
            yaw_rate_max_rad_s=yaw_rate_max_rad_s,
        )
        if s >= need_m:
            hi = mid
        else:
            lo = mid
    return float(hi)


def avoidance_feasibility(
    *,
    decision_step: Optional[int],
    ca_step: Optional[int],
    dt: float,
    approach_speed_mps: float,
    azimuth_decision_deg: Optional[float],
    v0_mps: float,
    a_lat_mps2: float,
    yaw_rate_max_rad_s: float,
    radius: float = LOOM_RADIUS_M,
    miss_surface_m: float = MISS_SURFACE_M,
) -> dict:
    """Can 0.1 m surface clearance be reached at all after this decision?

    Also reports the latest decision (as object angular size) that would
    have allowed it, and feasibility had the decision come at the GF
    window (50°).
    """
    need = float(radius) + float(miss_surface_m)
    az = 45.0 if azimuth_decision_deg is None or not np.isfinite(float(azimuth_decision_deg)) else float(azimuth_decision_deg)
    turn = np.radians(max(0.0, 90.0 - abs(az)))
    kw = dict(v0_mps=v0_mps, a_lat_mps2=a_lat_mps2, turn_rad=turn, yaw_rate_max_rad_s=yaw_rate_max_rad_s)
    t_req = required_decision_lead_s(need_m=need, **kw)
    u = max(float(approach_speed_mps), 1e-6)
    # Object centre distance at the latest workable decision.
    d_req = u * t_req + 0.0
    ang_req = angular_size_deg(max(d_req, float(radius) * 1.0001), radius)
    d50 = start_distance_for_angular_size(GF_WINDOW_DEG, radius)
    t50 = d50 / u
    s50 = max_lateral_displacement_m(t50, **kw)
    out = {
        "need_center_m": need,
        "approach_speed_mps": float(u),
        "v0_mps": float(v0_mps),
        "a_lat_max_mps2": float(a_lat_mps2),
        "yaw_rate_max_deg_s": float(np.degrees(yaw_rate_max_rad_s)),
        "turn_needed_deg": float(np.degrees(turn)),
        "required_lead_s": float(t_req),
        "latest_decision_angle_deg": float(ang_req),
        "lead_at_gf50_s": float(t50),
        "max_lateral_at_gf50_m": float(s50),
        "feasible_at_gf50": bool(s50 >= need),
        "decision_step": None if decision_step is None else int(decision_step),
        "ca_step": None if ca_step is None else int(ca_step),
    }
    if decision_step is None or ca_step is None:
        out.update({"t_avail_s": None, "max_lateral_m": None, "feasible": False, "label": "no_decision"})
        return out
    t_av = max(0.0, (int(ca_step) - int(decision_step)) * float(dt))
    s = max_lateral_displacement_m(t_av, **kw)
    feas = bool(s >= need)
    out.update({
        "t_avail_s": float(t_av),
        "max_lateral_m": float(s),
        "feasible": feas,
        "label": "feasible" if feas else "infeasible",
    })
    return out


def wander_yaw_threshold(headings_runs, dts) -> Optional[dict]:
    """No-object wander: pooled per-step |yaw rate| mean + 2 sd (rad/s).

    Also mean + 2 sd of per-run *peak* |yaw rate| (stricter, reported).
    """
    pooled, peaks = [], []
    for h, dt in zip(headings_runs, dts):
        w = np.abs(yaw_rates(h, dt))
        if w.size:
            pooled.append(w)
            peaks.append(float(np.max(w)))
    if not pooled:
        return None
    allw = np.concatenate(pooled)
    return {
        "pooled_mean": float(np.mean(allw)),
        "pooled_std": float(np.std(allw)),
        "threshold_rad_s": float(np.mean(allw) + 2.0 * np.std(allw)),
        "peak_mean": float(np.mean(peaks)),
        "peak_std": float(np.std(peaks)),
        "peak_threshold_rad_s": float(np.mean(peaks) + 2.0 * np.std(peaks)),
        "n_runs": int(len(pooled)),
    }


def behavioural_response(
    *,
    headings,
    dt: float,
    ca_step: Optional[int],
    gf_step: Optional[int],
    threat_side: str,
    wander_rad_s: float,
    window_s: float = RESPONSE_WINDOW_S,
    min_away_rad: float = MIN_AWAY_RAD,
    first_step: int = 0,
) -> dict:
    """Response-window score (not the wrapped end-of-trial heading).

    decision = first GF fire or first |yaw rate| > no-object wander
    (whichever is earlier, searched up to closest approach).
    ok = decision before closest approach AND peak |yaw rate| in
    [decision, decision + window] > wander AND heading change over that
    window is away from the TRUE side.
    """
    h = np.asarray(list(headings), dtype=float)
    w = yaw_rates(h, dt)
    sacc = None
    if ca_step is not None and w.size:
        end = min(int(ca_step), int(w.size) - 1)
        for i in range(max(0, int(first_step)), end + 1):
            if abs(float(w[i])) > float(wander_rad_s):
                sacc = int(i)
                break
    cands = [s for s in (gf_step, sacc) if s is not None]
    decision = int(min(cands)) if cands else None
    source = None
    if decision is not None:
        source = "gf" if (gf_step is not None and int(gf_step) == decision) else "saccade"
    on_time = decision is not None and ca_step is not None and int(decision) < int(ca_step)
    peak = 0.0
    dh = 0.0
    if decision is not None and h.size >= 2:
        n = max(1, int(round(float(window_s) / max(float(dt), 1e-9))))
        i0 = int(np.clip(decision, 0, h.size - 1))
        i1 = int(np.clip(decision + n, 0, h.size - 1))
        seg = w[i0: max(i0 + 1, i1)] if w.size else np.zeros(0)
        peak = float(np.max(np.abs(seg))) if seg.size else 0.0
        dh = net_heading_delta(h[i0: i1 + 1])
    away = heading_away_from_side(dh, threat_side, min_abs=min_away_rad)
    above = peak > float(wander_rad_s)
    return {
        "ok": bool(on_time and above and away),
        "decision_step": decision,
        "decision_source": source,
        "saccade_onset_step": sacc,
        "gf_step": None if gf_step is None else int(gf_step),
        "on_time": bool(on_time),
        "peak_yaw_rate_rad_s": float(peak),
        "above_wander": bool(above),
        "window_heading_rad": float(dh),
        "away": bool(away),
        "toward": bool(heading_away_from_side(-dh, threat_side, min_abs=min_away_rad)),
    }
