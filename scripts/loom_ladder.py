#!/usr/bin/env python3
"""Diagnose why camera frontolat fires after CA while hex ±30° fires at 49–55°.

Same stimulus on every rung: az +35°, l/v = 40 ms, start ≤5°.
GF readout stays frozen. No gate retune.

    L1  hex canonical (flyvis loom() at x0=+35, th0=5°)
    L2  camera, fly held still, uniform gray room
    L3  camera, fly still, 1/f walls
    L4  camera, fly translating at cruise, open-loop (no steering), 1/f walls
    L5  full closed loop (official frontolat path)

Each step logs vs angular size: hex disc value, photoreceptor rest,
LC4/LPLC2 L/R, DNp01 raw, rest_L/R, rise, gf score, plus fire-gate flags.

    python3 scripts/loom_ladder.py --output output/m2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threedfly.brain.constants import DT, N_HEX
from threedfly.brain.eyes import BG_REST, _image_to_hex_box, _sample_hex_box, UNSEEN_GRAY
from threedfly.brain.stimuli import gray, hex_degrees, loom, loom_theta_deg
from threedfly.control.closed_loop_metrics import (
    LOOM_RADIUS_M,
    LOOM_START_ANG_MAX_DEG,
    closest_approach_step,
    collision_course_object,
    fire_minus_closest,
    loom_geometry,
    start_distance_for_angular_size,
    visible_loom_placement,
)
from threedfly.control.dn_controller import DNFlightController
from threedfly.control.flight_params import FlightParams
from threedfly.control.gf_readout import first_fire_angle

AZ = 35.0
LV = 0.04
TH0 = float(LOOM_START_ANG_MAX_DEG)
CRUISE = 0.30
DT_LOOM = 0.01
N_STEPS = 120
PRE_HEX = 0.40
STIM_HEX = 0.90


def _hex_xy() -> tuple[np.ndarray, np.ndarray]:
    _, _, xd, yd = hex_degrees()
    return xd, yd


def _disc_mask(theta_deg: float, x0: float = AZ) -> np.ndarray:
    xd, yd = _hex_xy()
    r = np.hypot(xd - float(x0), yd)
    return r <= (0.5 * max(float(theta_deg), 1.0))


def _nearest_hex(x0: float = AZ) -> int:
    xd, yd = _hex_xy()
    return int(np.argmin(np.hypot(xd - float(x0), yd)))


def sample_hex_raw(left_img, right_img, left_side, right_side):
    """BoxEye hex only. Does not touch photoreceptor rest (brain.step owns τ)."""
    hex_L, ok_L = _sample_hex_box(left_img, "L")
    hex_R, ok_R = _sample_hex_box(right_img, "R")
    if left_side is not None:
        s, ok_s = _sample_hex_box(left_side, "L", cant_deg=90.0)
        fill = (~ok_L) & ok_s
        hex_L = np.where(fill, s, hex_L).astype(np.float32)
        ok_L = ok_L | ok_s
    if right_side is not None:
        s, ok_s = _sample_hex_box(right_side, "R", cant_deg=90.0)
        fill = (~ok_R) & ok_s
        hex_R = np.where(fill, s, hex_R).astype(np.float32)
        ok_R = ok_R | ok_s
    return hex_L, hex_R, ok_L, ok_R


def _disc_stats(layers, theta_deg: float, side: str = "L") -> dict:
    mask = _disc_mask(theta_deg)
    key = "L" if side == "L" else "R"
    raw = layers[f"raw_{key}"]
    rest = layers[f"rest_{key}"]
    pr = layers[f"pr_{key}"]
    if rest is None:
        rest = np.full(raw.shape, 0.5, dtype=np.float32)
    n = int(np.count_nonzero(mask))
    if n < 1:
        i = _nearest_hex()
        mask = np.zeros(raw.shape, dtype=bool)
        mask[i] = True
        n = 1
    return {
        "hex_disc": float(np.mean(raw[mask])),
        "hex_disc_min": float(np.min(raw[mask])),
        "pr_rest_disc": float(np.mean(rest[mask])),
        "pr_hex_disc": float(np.mean(pr[mask])),
        "n_disc_hex": n,
    }


def _row(step, theta, ctl, ro, disc, extra=None) -> dict:
    rec = {
        "step": int(step),
        "angular_size_deg": float(theta),
        "hex_disc": disc.get("hex_disc"),
        "hex_disc_min": disc.get("hex_disc_min"),
        "pr_rest_disc": disc.get("pr_rest_disc"),
        "pr_hex_disc": disc.get("pr_hex_disc"),
        "n_disc_hex": disc.get("n_disc_hex"),
        "lc4_L": float(ro.lc4_L),
        "lc4_R": float(ro.lc4_R),
        "lplc2_L": float(ro.lplc2_L),
        "lplc2_R": float(ro.lplc2_R),
        "dnp01_L": float(ro.dnp01_L),
        "dnp01_R": float(ro.dnp01_R),
        "dnp01_raw": float(max(ro.dnp01_L, ro.dnp01_R)),
        "rest_L": float(ctl.dnp01_rest_L or 0.0),
        "rest_R": float(ctl.dnp01_rest_R or 0.0),
        "rise": float(ctl.last_rise),
        "gf_score": float(getattr(ctl, "last_gf_score", 0.0)),
        "gf_threshold": float(ctl.params.gf_threshold),
        "lat": float(ctl.last_lat_integrated),
        "vpn_amp": float(ctl.last_vpn_amp),
        "vpn_slope": float(ctl.last_vpn_slope),
        "hold": int(ctl._rise_hold),
        "steep": bool(ctl._steep_climb),
        "armed": bool(ctl._escape_armed),
        "escapes": int(ctl.brain_escape_count),
        "would_fire": int(ctl.would_fire_count),
    }
    if extra:
        rec.update(extra)
    return rec


def _summarise(name: str, rows: list, distances=None) -> dict:
    p = FlightParams()
    theta = [r["angular_size_deg"] for r in rows]
    rise = [r["rise"] for r in rows]
    gf = [r["gf_score"] for r in rows]
    fire = next((r for r in rows if int(r["escapes"]) > 0), None)
    score_cross = next(
        (r for r in rows if float(r["gf_score"]) >= float(p.gf_threshold) and float(r["angular_size_deg"]) >= 12.0),
        None,
    )
    at50 = min(rows, key=lambda r: abs(float(r["angular_size_deg"]) - 50.0)) if rows else None
    ca = closest_approach_step(distances) if distances is not None else None
    fire_step = None if fire is None else int(fire["step"])
    return {
        "name": name,
        "n": len(rows),
        "fire_step": fire_step,
        "fire_ang_deg": None if fire is None else float(fire["angular_size_deg"]),
        "fire_rise": None if fire is None else float(fire["rise"]),
        "fire_gf": None if fire is None else float(fire["gf_score"]),
        "score_cross_step": None if score_cross is None else int(score_cross["step"]),
        "score_cross_ang_deg": None if score_cross is None else float(score_cross["angular_size_deg"]),
        "dCA": fire_minus_closest(fire_step, ca),
        "ca_step": ca,
        "lag_score_to_fire": (
            None
            if fire_step is None or score_cross is None
            else int(fire_step) - int(score_cross["step"])
        ),
        "at50": None
        if at50 is None
        else {
            k: at50[k]
            for k in (
                "angular_size_deg",
                "hex_disc",
                "pr_rest_disc",
                "pr_hex_disc",
                "dnp01_raw",
                "rest_L",
                "rest_R",
                "rise",
                "gf_score",
                "lc4_L",
                "lplc2_L",
            )
        },
        "peak_rise": max((float(r["rise"]) for r in rows), default=0.0),
        "peak_gf": max((float(r["gf_score"]) for r in rows), default=0.0),
        "gf_k": p.gf_k,
        "gf_threshold": p.gf_threshold,
        "gf_form": p.gf_form,
    }


def run_l1(brain, ctl) -> tuple[list, dict]:
    """Hex-canonical loom at az +35°, start 5°, fly still, gray."""
    from threedfly.brain.eyes import reset_luminance_rest

    movie = np.concatenate(
        [gray(PRE_HEX, dt=DT_LOOM), loom(STIM_HEX, lv=LV, th0=TH0, x0=AZ, dt=DT_LOOM)]
    )
    theta = np.concatenate(
        [
            np.full(int(round(PRE_HEX / DT_LOOM)), 0.0, dtype=np.float32),
            loom_theta_deg(STIM_HEX, lv=LV, th0=TH0, dt=DT_LOOM),
        ]
    )
    blank = gray(PRE_HEX + STIM_HEX, dt=DT_LOOM)
    brain.reset()
    ctl.reset()
    reset_luminance_rest()
    g = gray(0.2, dt=DT_LOOM)[0]
    last = None
    for _ in range(20):
        last = brain.step(g, g, None, hex_L=g, hex_R=g)
    if last is not None:
        ctl.capture_baseline(last.readout)
    dummy = np.zeros((8, 8), dtype=np.float32)
    rows = []
    for i in range(movie.shape[0]):
        res = brain.step(dummy, dummy, None, hex_L=movie[i], hex_R=blank[i])
        ctl.compute(res.readout)
        mask = _disc_mask(float(theta[i]))
        n = int(np.count_nonzero(mask))
        disc = {
            "hex_disc": float(np.mean(movie[i][mask])) if n else float(movie[i].min()),
            "hex_disc_min": float(np.min(movie[i][mask])) if n else float(movie[i].min()),
            "pr_rest_disc": 0.5,
            "pr_hex_disc": float(np.mean(movie[i][mask])) if n else float(movie[i].min()),
            "n_disc_hex": n,
        }
        rows.append(_row(i, float(theta[i]), ctl, res.readout, disc))
    return rows, _summarise("L1_hex", rows)


def _set_room_gray(env, gray: float = 0.5):
    loom_id = int(getattr(env, "_loom_geom_id", -1))
    fly_ids = set(int(g) for g in getattr(env, "_fly_geom_ids", ()))
    saved = {
        "rgba": env.model.geom_rgba.copy(),
        "matid": env.model.geom_matid.copy(),
    }
    for gid in range(env.model.ngeom):
        if gid == loom_id or gid in fly_ids:
            continue
        env.model.geom_matid[gid] = -1
        env.model.geom_rgba[gid] = np.array([gray, gray, gray, 1.0], dtype=np.float32)
    return saved


def _restore_room(env, saved) -> None:
    env.model.geom_rgba[:] = saved["rgba"]
    env.model.geom_matid[:] = saved["matid"]


def _show_loom(env, virtual_xyz, fly_xyz):
    vis, rad = visible_loom_placement(virtual_xyz, fly_xyz, radius=LOOM_RADIUS_M)
    env.set_loom_object(vis, radius=rad)


def run_camera_rung(
    name: str,
    env,
    brain,
    ctl,
    *,
    gray_room: bool,
    hold_still: bool,
    open_loop_cruise: bool,
    closed_loop: bool,
    no_object_baseline: bool = True,
) -> tuple[list, dict]:
    from threedfly.brain.eyes import reset_luminance_rest

    spec = collision_course_object(AZ, start_distance_for_angular_size(TH0), LV)
    fly0 = np.array(spec["pose"], dtype=float)
    loom0 = np.array(spec["loom_start"], dtype=float)
    loom_v = np.array(spec["loom_vel"], dtype=float)
    yaw0 = float(spec["yaw"])
    cruise = np.array([CRUISE, 0.0, 0.0], dtype=float)

    saved = _set_room_gray(env) if gray_room else None
    try:
        env.reset(seed=0)
        env.set_pose(fly0, yaw=yaw0, vel=(0.0, 0.0, 0.0))
        env.hide_loom_object()
        brain.reset()
        ctl.reset()
        ctl.bind_hover(env.hover_action)
        reset_luminance_rest()
        prev_stride = int(brain.params.flyvis_stride)
        prev_vdt = float(getattr(brain, "visual_dt", 0.02))
        prev_dt, prev_sub = env.control_dt, env.n_substeps
        brain.params.flyvis_stride = 1
        brain.visual_dt = DT_LOOM
        env.control_dt = DT_LOOM
        env.n_substeps = max(1, int(round(DT_LOOM / float(env.model.opt.timestep))))

        fly = fly0.copy()
        loom = loom0.copy()
        if no_object_baseline:
            env.hide_loom_object()
        else:
            _show_loom(env, loom, fly)
        left, right = env.get_eye_images()
        side = env.last_side_images
        n_pre = int(ctl.params.prewarm_steps)
        warmed = brain.prewarm(
            left, right, env.get_fly_angular_velocity(), n=n_pre,
            left_side_img=side[0], right_side_img=side[1],
        )
        if warmed is not None:
            ctl.capture_baseline(warmed.readout)
        if no_object_baseline:
            _show_loom(env, loom, fly)
            left, right = env.get_eye_images()
            side = env.last_side_images

        from threedfly.control.exploration import heading_from_quaternion

        rows = []
        distances = []
        yaw_now = yaw0
        for step in range(N_STEPS):
            loom = loom0 + loom_v * (step * DT_LOOM)
            if hold_still:
                fly = fly0.copy()
                env.set_pose(fly, yaw=yaw0, vel=(0.0, 0.0, 0.0))
            elif open_loop_cruise:
                fly = fly0 + cruise * (step * DT_LOOM)
                loom = loom + (fly - fly0)
                env.set_pose(fly, yaw=yaw0, vel=tuple(cruise))
            else:
                # L5: physics owns the fly. Do not reset pose.
                fly = np.asarray(env.get_fly_position(), dtype=float)
                yaw_now = heading_from_quaternion(env.get_fly_orientation())
            _show_loom(env, loom, fly)
            left, right = env.get_eye_images()
            side = env.last_side_images
            raw_L, raw_R, ok_L, ok_R = sample_hex_raw(left, right, side[0], side[1])
            geo = loom_geometry(loom, fly, yaw_now)
            res = brain.step(
                left, right, env.get_fly_angular_velocity(),
                left_side_img=side[0], right_side_img=side[1],
            )
            layers = {
                "raw_L": raw_L,
                "raw_R": raw_R,
                "pr_L": res.hex_L,
                "pr_R": res.hex_R,
                "rest_L": None if BG_REST.hex_L is None else BG_REST.hex_L.copy(),
                "rest_R": None if BG_REST.hex_R is None else BG_REST.hex_R.copy(),
                "ok_L": ok_L,
                "ok_R": ok_R,
            }
            disc = _disc_stats(layers, float(geo["angular_size_deg"]), "L")
            action = ctl.compute(res.readout)
            if closed_loop:
                _left, _right, info = env.step(action, wing_strokes=ctl.last_wing_strokes)
                fly = np.asarray(info["position"], dtype=float)
                yaw_now = heading_from_quaternion(info["orientation"])
                geo = loom_geometry(loom, fly, yaw_now)
            distances.append(float(geo["distance_m"]))
            extra = {
                "azimuth_deg": float(geo["azimuth_deg"]),
                "distance_m": float(geo["distance_m"]),
                "visible": abs(float(geo["azimuth_deg"])) < 90.0,
            }
            rows.append(_row(step, float(geo["angular_size_deg"]), ctl, res.readout, disc, extra))
        brain.params.flyvis_stride = prev_stride
        brain.visual_dt = prev_vdt
        env.control_dt = prev_dt
        env.n_substeps = prev_sub
    finally:
        if saved is not None:
            _restore_room(env, saved)
    return rows, _summarise(name, rows, distances)


def _print_table(summaries: list) -> None:
    print("\n=== loom ladder  az=+35°  l/v=40 ms  start≤5°  GF frozen ===")
    print(
        f"{'rung':<22} {'fire°':>7} {'dCA':>5} {'xGF°':>7} {'lag':>4} "
        f"{'restL':>6} {'hex50':>6} {'pr50':>6} {'rise50':>7} {'gf50':>6} {'peakR':>6}"
    )
    for s in summaries:
        a50 = s.get("at50") or {}
        print(
            f"{s['name']:<22} "
            f"{'-' if s['fire_ang_deg'] is None else f'{s['fire_ang_deg']:.1f}':>7} "
            f"{'-' if s['dCA'] is None else f'{s['dCA']:+d}':>5} "
            f"{'-' if s['score_cross_ang_deg'] is None else f'{s['score_cross_ang_deg']:.1f}':>7} "
            f"{'-' if s['lag_score_to_fire'] is None else f'{s['lag_score_to_fire']:+d}':>4} "
            f"{'-' if a50.get('rest_L') is None else f'{a50['rest_L']:.3f}':>6} "
            f"{'-' if a50.get('hex_disc') is None else f'{a50['hex_disc']:.3f}':>6} "
            f"{'-' if a50.get('pr_rest_disc') is None else f'{a50['pr_rest_disc']:.3f}':>6} "
            f"{'-' if a50.get('rise') is None else f'{a50['rise']:.3f}':>7} "
            f"{'-' if a50.get('gf_score') is None else f'{a50['gf_score']:.3f}':>6} "
            f"{s['peak_rise']:>6.3f}"
        )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default=str(ROOT / "output" / "m2"))
    p.add_argument("--skip", default="", help="comma rungs to skip, e.g. L5")
    p.add_argument(
        "--fix-baseline",
        action="store_true",
        default=True,
        help="Capture DNp01 rest on a no-object DC-removed prewarm (official)",
    )
    p.add_argument(
        "--loom-at-prewarm",
        action="store_true",
        help="Leave the ≤5° disc up during prewarm (old buggy protocol)",
    )
    args = p.parse_args(argv)
    skip = {s.strip().upper() for s in args.skip.split(",") if s.strip()}

    from threedfly.brain.closed_loop import ClosedLoopBrain, core_stack_available
    from threedfly.brain.constants import flyvis_available
    from threedfly.sim import FlyEnvironment

    if not flyvis_available() or not core_stack_available():
        print("ERROR: ladder needs flyvis + core bundle", file=sys.stderr)
        return 2

    params = FlightParams()
    print(
        f"GF frozen form={params.gf_form} k={params.gf_k} thr={params.gf_threshold} "
        f"(do not retune)",
        flush=True,
    )
    print("Building closed-loop brain…", flush=True)
    brain = ClosedLoopBrain.build(params=params)
    brain.params.flyvis_stride = 1
    brain.visual_dt = DT_LOOM
    ctl = DNFlightController(params)
    env = FlyEnvironment(render_mode="rgb_array")

    all_rows = {}
    summaries = []

    if "L1" not in skip:
        print("L1 hex canonical…", flush=True)
        rows, sm = run_l1(brain, ctl)
        all_rows["L1_hex"] = rows
        summaries.append(sm)
        print(f"  fire={sm['fire_ang_deg']} score_cross={sm['score_cross_ang_deg']} peak_rise={sm['peak_rise']:.3f}", flush=True)

    camera = (
        ("L2", "L2_cam_still_gray", True, True, False, False),
        ("L3", "L3_cam_still_1f", False, True, False, False),
        ("L4", "L4_cam_cruise_ol", False, False, True, False),
        ("L5", "L5_closed_loop", False, False, False, True),
    )
    for tag, name, gray_room, still, cruise, closed in camera:
        if tag in skip:
            continue
        print(f"{tag} {name}…", flush=True)
        rows, sm = run_camera_rung(
            name, env, brain, ctl,
            gray_room=gray_room,
            hold_still=still,
            open_loop_cruise=cruise,
            closed_loop=closed,
            no_object_baseline=bool(args.fix_baseline) and not bool(args.loom_at_prewarm),
        )
        all_rows[name] = rows
        summaries.append(sm)
        print(
            f"  fire={sm['fire_ang_deg']} dCA={sm['dCA']} "
            f"score_cross={sm['score_cross_ang_deg']} lag={sm['lag_score_to_fire']} "
            f"peak_rise={sm['peak_rise']:.3f}",
            flush=True,
        )

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "stimulus": {"az_deg": AZ, "lv_s": LV, "th0_deg": TH0, "dt": DT_LOOM},
        "gf": {
            "form": params.gf_form,
            "k": params.gf_k,
            "threshold": params.gf_threshold,
            "frozen": True,
        },
        "summaries": summaries,
        "rows": all_rows,
        "hypotheses": {
            "a": "DNp01 / laterality rest absorbs the slow 5° growth",
            "b": "photoreceptor τ=5 s adapts the slowly darkening disc",
            "c": "pipeline latency (filters + hold + lock) is a fixed 5–9 steps",
            "d": "self-motion flow from 1/f walls (L4/L5 only)",
        },
    }
    (out / "loom_ladder.json").write_text(json.dumps(payload, indent=2))
    _print_table(summaries)
    print(f"wrote {out / 'loom_ladder.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
