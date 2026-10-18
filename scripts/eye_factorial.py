#!/usr/bin/env python3
"""Factorial open-loop: why camera DNp01 is ~5× weaker than hex_mono.

Fly held still unless C. az=35°, l/v 0.04 and 0.08. Camera path except D.

  A. uniform gray room (no wall texture), bg ≈ hex_mono 0.5
  B. A + checker walls (current MJCF)
  C. B + self-motion (fly translating at cruise)
  D. hex_mono with the same bg luminance/contrast as B's camera frames

    python scripts/eye_factorial.py --output output/m2
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
from threedfly.brain.eyes import hex_body_azel_deg, hex_in_bio_envelope, stereo_to_hex
from threedfly.brain.stimuli import hex_degrees, loom, loom_theta_deg
from threedfly.control.closed_loop_metrics import LOOM_RADIUS_M, angular_size_deg
from threedfly.control.flight_params import FlightParams

T4_KEYS = ("T4a", "T4b", "T4c", "T4d")
T5_KEYS = ("T5a", "T5b", "T5c", "T5d")
AZ_DEG = 35.0
DIST0 = 3.6
CRUISE_MPS = 0.45
PRE_S = 0.30
STIM_S = 0.70


def _mean_types(volts: dict, names) -> float:
    vals = []
    for n in names:
        a = volts.get(n)
        if a is not None and getattr(a, "size", 0):
            vals.append(float(np.mean(a)))
    return float(np.mean(vals)) if vals else 0.0


def flatten_room_gray(env) -> None:
    """Drop wall/floor/obstacle textures to mid-gray. Loom + fly unchanged."""
    import mujoco

    skip = set(env._fly_geom_ids)
    try:
        loom_gid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "loom_geom")
        if loom_gid >= 0:
            skip.add(int(loom_gid))
    except Exception:
        pass
    for gid in range(env.model.ngeom):
        if gid in skip:
            continue
        env.model.geom_matid[gid] = -1
        env.model.geom_rgba[gid] = np.array([0.5, 0.5, 0.5, 1.0], dtype=np.float32)


def hex_stats(hL: np.ndarray, ang_deg: float = 40.0) -> dict:
    _, _, xd, yd = hex_degrees()
    az, el = hex_body_azel_deg("L")
    disc = np.hypot(xd - AZ_DEG, yd) <= 0.5 * float(ang_deg)
    valid = hex_in_bio_envelope("L", az)
    bg = valid & (~disc) & (np.abs(el) < 40.0)
    return {
        "mean": float(np.mean(hL[valid])) if valid.any() else float(np.mean(hL)),
        "std": float(np.std(hL[valid])) if valid.any() else float(np.std(hL)),
        "min": float(np.min(hL[valid])) if valid.any() else float(np.min(hL)),
        "disc_mean": float(np.mean(hL[disc])) if disc.any() else None,
        "bg_mean": float(np.mean(hL[bg])) if bg.any() else None,
        "bg_std": float(np.std(hL[bg])) if bg.any() else None,
    }


def run_trace(brain, hex_L_seq, hex_R_seq, pre=20, pre_hex=None):
    from threedfly.control.dn_controller import DNFlightController

    ctl = DNFlightController(FlightParams())
    brain.reset()
    dummy = np.zeros((8, 8, 3), dtype=np.uint8)
    if pre_hex is None:
        g = np.full(N_HEX, 0.5, dtype=np.float32)
        hL0, hR0 = g, g
    else:
        hL0, hR0 = pre_hex
    last = None
    for _ in range(pre):
        last = brain.step(dummy, dummy, hex_L=hL0, hex_R=hR0)
    if last is not None:
        ctl.capture_baseline(last.readout)
    rows = []
    for hL, hR in zip(hex_L_seq, hex_R_seq):
        res = brain.step(dummy, dummy, hex_L=hL, hex_R=hR)
        ctl.compute(res.readout)
        vL, vR = (None, None)
        if brain._flyvis_hold is not None:
            vL, vR = brain._flyvis_hold
        ro = res.readout
        rows.append(
            {
                "dnp01_L": float(ro.dnp01_L),
                "dnp01_R": float(ro.dnp01_R),
                "dnp01_rise": float(ctl.last_rise),
                "would_fire": int(ctl.would_fire_count),
                "lc4_L": float(ro.lc4_L),
                "lc4_R": float(ro.lc4_R),
                "lplc2_L": float(ro.lplc2_L),
                "lplc2_R": float(ro.lplc2_R),
                "t4_L": _mean_types(vL or {}, T4_KEYS),
                "t5_L": _mean_types(vL or {}, T5_KEYS),
                "hex_L_mean": float(np.mean(hL)),
                "hex_L_std": float(np.std(hL)),
                "hex_L_min": float(np.min(hL)),
            }
        )
    return rows


def onset(rows, key, thresh, theta):
    for r, t in zip(rows, theta):
        if float(t) > 5.0 and float(r[key]) >= thresh:
            return float(t)
    return None


def summarise(rows, theta) -> dict:
    return {
        "peak_dnp01_rise": max(r["dnp01_rise"] for r in rows),
        "peak_lc4_L": max(r["lc4_L"] for r in rows),
        "peak_lplc2_L": max(r["lplc2_L"] for r in rows),
        "peak_t4_L": max(r["t4_L"] for r in rows),
        "peak_t5_L": max(r["t5_L"] for r in rows),
        "dnp01_onset_0.20": onset(rows, "dnp01_rise", 0.20, theta),
        "lplc2_onset_0.10": onset(rows, "lplc2_L", 0.10, theta),
        "lc4_onset_0.10": onset(rows, "lc4_L", 0.10, theta),
        "would_fire": max(r["would_fire"] for r in rows),
        "hex_mean": float(np.mean([r["hex_L_mean"] for r in rows])),
        "hex_std": float(np.mean([r["hex_L_std"] for r in rows])),
    }


def camera_movie(env, lv, dist0, dt, pre, stim, translate=False, cruise=CRUISE_MPS):
    speed = LOOM_RADIUS_M / max(lv, 1e-6)
    az = np.radians(AZ_DEG)
    n_pre = int(round(pre / dt))
    n = int(round(stim / dt))
    hex_L, hex_R, theta = [], [], []
    fly = np.array([0.0, 0.0, 1.5], dtype=float)
    env.set_pose(tuple(fly), yaw=0.0, vel=(cruise, 0.0, 0.0) if translate else (0.0, 0.0, 0.0))
    env.hide_loom_object()
    room = stereo_to_hex(*env.get_eye_images())
    for _ in range(n_pre):
        if translate:
            fly[0] += cruise * dt
            env.set_pose(tuple(fly), yaw=0.0, vel=(cruise, 0.0, 0.0))
            hL0, hR0 = stereo_to_hex(*env.get_eye_images())
        else:
            hL0, hR0 = room
        hex_L.append(hL0)
        hex_R.append(hR0)
        theta.append(0.0)
    pos = np.array([dist0 * np.cos(az), dist0 * np.sin(az), 1.5], dtype=float)
    vel = np.array([-speed * np.cos(az), -speed * np.sin(az), 0.0], dtype=float)
    for _ in range(n):
        if translate:
            fly[0] += cruise * dt
            env.set_pose(tuple(fly), yaw=0.0, vel=(cruise, 0.0, 0.0))
        env.set_loom_object(pos)
        hL, hR = stereo_to_hex(*env.get_eye_images())
        hex_L.append(hL)
        hex_R.append(hR)
        rel = pos[:2] - fly[:2]
        theta.append(angular_size_deg(float(np.linalg.norm(rel))))
        pos = pos + vel * dt
    return (
        np.asarray(hex_L),
        np.asarray(hex_R),
        np.asarray(theta, dtype=np.float32),
        room,
    )


def hex_ref_movie(lv, dt, pre, stim):
    movie = loom(stim, lv=lv, x0=AZ_DEG, dt=dt)
    th = loom_theta_deg(stim, lv=lv, dt=dt)
    gray = np.full((movie.shape[0], N_HEX), 0.5, dtype=np.float32)
    pre_n = int(round(pre / dt))
    pre_L = np.repeat(gray[:1], pre_n, axis=0)
    return (
        np.concatenate([pre_L, movie]),
        np.concatenate([pre_L, gray]),
        np.concatenate([np.zeros(pre_n, dtype=np.float32), th]),
    )


def hex_matched_movie(lv, dt, pre, stim, bg_mean: float, bg_std: float, seed: int = 0):
    """Hard hex disc on a background with the camera frame's mean and std."""
    movie = loom(stim, lv=lv, x0=AZ_DEG, dt=dt)
    th = loom_theta_deg(stim, lv=lv, dt=dt)
    rng = np.random.default_rng(seed)
    noise = rng.normal(bg_mean, max(float(bg_std), 1e-4), size=movie.shape).astype(np.float32)
    noise = np.clip(noise, 0.0, 1.0)
    matched = np.where(movie < 0.25, 0.0, noise).astype(np.float32)
    pre_n = int(round(pre / dt))
    pre_bg = np.clip(
        rng.normal(bg_mean, max(float(bg_std), 1e-4), size=(pre_n, N_HEX)), 0.0, 1.0
    ).astype(np.float32)
    gray_R = np.full((pre_n + movie.shape[0], N_HEX), 0.5, dtype=np.float32)
    return (
        np.concatenate([pre_bg, matched]),
        gray_R,
        np.concatenate([np.zeros(pre_n, dtype=np.float32), th]),
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default=str(ROOT / "output" / "m2"))
    p.add_argument("--lv", type=float, nargs="*", default=(0.04, 0.08))
    args = p.parse_args(argv)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    from threedfly.brain.closed_loop import ClosedLoopBrain, core_stack_available
    from threedfly.brain.constants import flyvis_available
    from threedfly.sim import FlyEnvironment

    if not flyvis_available() or not core_stack_available():
        print("need flyvis + core bundle", file=sys.stderr)
        return 2

    params = FlightParams()
    print("Building brain…", flush=True)
    brain = ClosedLoopBrain.build(params=params)
    brain.params.flyvis_stride = 1

    table = {}
    for lv in args.lv:
        print(f"\n=== l/v={lv} ===", flush=True)
        env = FlyEnvironment(render_mode="rgb_array")
        env.reset(seed=0)
        env.set_pose((0.0, 0.0, 1.5), yaw=0.0, vel=(0.0, 0.0, 0.0))

        # B first so D can match its camera bg (current closed-loop room).
        env_b = FlyEnvironment(render_mode="rgb_array")
        env_b.reset(seed=0)
        env_b.set_pose((0.0, 0.0, 1.5), yaw=0.0, vel=(0.0, 0.0, 0.0))
        hL_b, hR_b, th_b, room_b = camera_movie(env_b, lv, DIST0, DT, PRE_S, STIM_S, translate=False)
        stats_b = hex_stats(hL_b[int(round(PRE_S / DT)) + 10])
        rows_b = run_trace(brain, hL_b, hR_b, pre_hex=room_b)
        table[f"B_checkers_lv{lv:g}"] = {**summarise(rows_b, th_b), "frame": stats_b, "kind": "camera"}
        print(f"  B checkers {table[f'B_checkers_lv{lv:g}']}", flush=True)
        env_b.close()

        flatten_room_gray(env)
        env.set_pose((0.0, 0.0, 1.5), yaw=0.0, vel=(0.0, 0.0, 0.0))
        env.hide_loom_object()
        hL_a, hR_a, th_a, room_a = camera_movie(env, lv, DIST0, DT, PRE_S, STIM_S, translate=False)
        stats_a = hex_stats(hL_a[int(round(PRE_S / DT)) + 10])
        rows_a = run_trace(brain, hL_a, hR_a, pre_hex=room_a)
        table[f"A_gray_lv{lv:g}"] = {**summarise(rows_a, th_a), "frame": stats_a, "kind": "camera"}
        print(f"  A gray     {table[f'A_gray_lv{lv:g}']}", flush=True)
        env.close()

        env_c = FlyEnvironment(render_mode="rgb_array")
        env_c.reset(seed=0)
        hL_c, hR_c, th_c, room_c = camera_movie(env_c, lv, DIST0, DT, PRE_S, STIM_S, translate=True)
        stats_c = hex_stats(hL_c[int(round(PRE_S / DT)) + 10])
        rows_c = run_trace(brain, hL_c, hR_c, pre_hex=room_c)
        table[f"C_motion_lv{lv:g}"] = {**summarise(rows_c, th_c), "frame": stats_c, "kind": "camera"}
        print(f"  C motion   {table[f'C_motion_lv{lv:g}']}", flush=True)
        env_c.close()

        bg_mean = float(stats_b.get("bg_mean") or 0.5)
        bg_std = float(stats_b.get("bg_std") or 0.0)
        hL_d, hR_d, th_d = hex_matched_movie(lv, DT, PRE_S, STIM_S, bg_mean, bg_std)
        rows_d = run_trace(brain, hL_d, hR_d, pre_hex=(hL_d[0], hR_d[0]))
        table[f"D_hex_matched_lv{lv:g}"] = {
            **summarise(rows_d, th_d),
            "frame": hex_stats(hL_d[int(round(PRE_S / DT)) + 10]),
            "matched_bg_mean": bg_mean,
            "matched_bg_std": bg_std,
            "kind": "hex",
        }
        print(f"  D matched  {table[f'D_hex_matched_lv{lv:g}']}", flush=True)

        hL_r, hR_r, th_r = hex_ref_movie(lv, DT, PRE_S, STIM_S)
        rows_r = run_trace(brain, hL_r, hR_r)
        table[f"hex_ref_lv{lv:g}"] = {
            **summarise(rows_r, th_r),
            "frame": hex_stats(hL_r[int(round(PRE_S / DT)) + 10]),
            "kind": "hex",
        }
        print(f"  hex_ref    {table[f'hex_ref_lv{lv:g}']}", flush=True)

    payload = {
        "az_deg": AZ_DEG,
        "dist0": DIST0,
        "dt": DT,
        "painted": False,
        "conditions": table,
    }
    (out / "eye_factorial.json").write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out / 'eye_factorial.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
