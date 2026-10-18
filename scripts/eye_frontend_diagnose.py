#!/usr/bin/env python3
"""Diagnose render→hex vs synthetic hex discs (no brain-weight changes).

1. Coverage map: each hex's body-frame az/el (new rays vs legacy 40°/120° pack).
2. Same frontolat loom (az 35°, l/v 0.04): hex disc vs 3D camera→hex at 20/40/60°.
   Camera path is rendered pixels only (Gaussian accept + luminance). No paint.

    python scripts/eye_frontend_diagnose.py --output output/m2
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
from threedfly.brain.eyes import (
    BIO_AZ_CONTRA_DEG,
    BIO_AZ_IPSI_DEG,
    hex_body_azel_deg,
    hex_coverage,
    hex_in_bio_envelope,
    legacy_packed_azel_deg,
    stereo_to_hex,
)
from threedfly.brain.stimuli import hex_degrees, loom, loom_theta_deg
from threedfly.control.closed_loop_metrics import LOOM_RADIUS_M, angular_size_deg
from threedfly.control.flight_params import FlightParams

T4_KEYS = ("T4a", "T4b", "T4c", "T4d")
T5_KEYS = ("T5a", "T5b", "T5c", "T5d")


def _hex_image(frame: np.ndarray, side: str = "L") -> np.ndarray:
    """Scatter the 721-hex frame into a square image for display."""
    _, _, xd, yd = hex_degrees()
    az, el = hex_body_azel_deg(side)
    xs = xd if side == "L" else -xd
    n = 96
    lim = 90.0
    img = np.full((n, n), 0.5, dtype=np.float32)
    w = (lim + xs) / (2.0 * lim) * (n - 1)
    h = (lim - el) / (2.0 * lim) * (n - 1)
    xi = np.clip(np.rint(w).astype(int), 0, n - 1)
    yi = np.clip(np.rint(h).astype(int), 0, n - 1)
    img[yi, xi] = np.clip(np.asarray(frame, dtype=np.float32), 0.0, 1.0)
    return img


def plot_coverage(path: Path) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 9.0), sharex=False, sharey=False)
    summary = {}
    for col, side in enumerate(("L", "R")):
        cov = hex_coverage(side)
        az_old, el_old, _ = legacy_packed_azel_deg(side)
        i0 = int(np.argmin(cov["az_deg"] ** 2 + cov["el_deg"] ** 2))
        az0, el0 = hex_body_azel_deg(side)
        i_old = int(np.argmin(az0 ** 2 + el0 ** 2))
        summary[side] = {
            "n_valid": cov["n_valid"],
            "n_bio": cov["n_bio"],
            "az_min": float(np.min(cov["az_deg"])),
            "az_max": float(np.max(cov["az_deg"])),
            "el_min": float(np.min(cov["el_deg"])),
            "el_max": float(np.max(cov["el_deg"])),
            "legacy_az_at_centre": float(az_old[i_old]),
            "legacy_el_at_centre": float(el_old[i_old]),
            "new_az_at_centre": float(cov["az_deg"][i0]),
            "valid_at_az35": bool(
                hex_in_bio_envelope(side, np.array([35.0 if side == "L" else -35.0]))[0]
            ),
        }
        ax = axes[0, col]
        colors = np.where(cov["valid"], 1.0, np.where(cov["bio"], 0.45, 0.0))
        ax.scatter(cov["az_deg"], cov["el_deg"], c=colors, s=8, cmap="coolwarm", vmin=0, vmax=1)
        lo = -BIO_AZ_CONTRA_DEG if side == "L" else -BIO_AZ_IPSI_DEG
        hi = BIO_AZ_IPSI_DEG if side == "L" else BIO_AZ_CONTRA_DEG
        ax.axvspan(lo, hi, color="tab:green", alpha=0.08, label="bio envelope")
        if side == "L":
            ax.axvspan(20, 70, color="tab:orange", alpha=0.12)
        else:
            ax.axvspan(-70, -20, color="tab:orange", alpha=0.12)
        ax.axvline(0, color="k", lw=0.6)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(f"{side} hex rays (new): centre az={summary[side]['new_az_at_centre']:.1f}°")
        ax.set_xlabel("body azimuth (deg, +left)")
        ax.set_ylabel("elevation (deg)")
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_aspect("equal")
        ax = axes[1, col]
        ax.scatter(az_old, el_old, c="tab:red", s=8, alpha=0.7)
        ax.axvline(0, color="k", lw=0.6)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(
            f"{side} legacy pack 40°/20°/120°: centre az={summary[side]['legacy_az_at_centre']:.1f}° "
            f"el={summary[side]['legacy_el_at_centre']:.1f}°"
        )
        ax.set_xlabel("body azimuth (deg)")
        ax.set_ylabel("elevation (deg)")
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_aspect("equal")
    fig.suptitle("Hex coverage: flyvis lattice in the body frame")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return summary


def _dist_for_ang(ang_deg: float, radius: float = LOOM_RADIUS_M) -> float:
    return float(radius) / max(float(np.tan(np.radians(0.5 * ang_deg))), 1e-6)


def camera_hex_at_size(env, az_deg: float, ang_deg: float, fly_xyz=(0.0, 0.0, 1.5)):
    d = _dist_for_ang(ang_deg)
    az = np.radians(az_deg)
    pos = np.array(fly_xyz, dtype=float)
    pos[0] += d * np.cos(az)
    pos[1] += d * np.sin(az)
    env.set_loom_object(pos)
    left, right = env.get_eye_images()
    hL, hR = stereo_to_hex(left, right)
    return hL, hR, left, right, pos, d


def hex_disc_at_size(ang_deg: float, x0: float, binocular: bool):
    th = float(ang_deg)
    _, _, xd, yd = hex_degrees()
    r = np.hypot(xd - x0, yd)
    disc = np.where(r <= th / 2.0, 0.0, 0.5).astype(np.float32)
    gray = np.full(N_HEX, 0.5, dtype=np.float32)
    if binocular:
        rR = np.hypot(xd - (-x0), yd)
        disc_R = np.where(rR <= th / 2.0, 0.0, 0.5).astype(np.float32)
        return disc, disc_R
    return disc, gray


def _mean_types(volts: dict, names) -> float:
    vals = []
    for n in names:
        a = volts.get(n)
        if a is not None and getattr(a, "size", 0):
            vals.append(float(np.mean(a)))
    return float(np.mean(vals)) if vals else 0.0


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
                "t4_R": _mean_types(vR or {}, T4_KEYS),
                "t5_R": _mean_types(vR or {}, T5_KEYS),
                "hex_L_mean": float(np.mean(hL)),
                "hex_L_min": float(np.min(hL)),
                "lat": float(ctl.last_lat_now),
            }
        )
    return rows


def build_hex_movie(lv: float, x0: float, binocular: bool, dt: float, pre: float, stim: float):
    movie = loom(stim, lv=lv, x0=x0, dt=dt)
    theta = loom_theta_deg(stim, lv=lv, dt=dt)
    gray = np.full((movie.shape[0], N_HEX), 0.5, dtype=np.float32)
    if binocular:
        movie_R = loom(stim, lv=lv, x0=-x0, dt=dt)
    else:
        movie_R = gray
    pre_n = int(round(pre / dt))
    pre_L = np.repeat(gray[:1], pre_n, axis=0)
    pre_R = np.repeat(gray[:1], pre_n, axis=0)
    pre_th = np.zeros(pre_n, dtype=np.float32)
    return (
        np.concatenate([pre_L, movie]),
        np.concatenate([pre_R, movie_R]),
        np.concatenate([pre_th, theta]),
    )


def build_camera_movie(env, az_deg: float, lv: float, dist0: float, dt: float, pre: float, stim: float):
    speed = LOOM_RADIUS_M / max(lv, 1e-6)
    az = np.radians(az_deg)
    n_pre = int(round(pre / dt))
    n = int(round(stim / dt))
    hex_L, hex_R, theta = [], [], []
    env.hide_loom_object()
    room = stereo_to_hex(*env.get_eye_images())
    for _ in range(n_pre):
        hex_L.append(room[0])
        hex_R.append(room[1])
        theta.append(0.0)
    pos = np.array([dist0 * np.cos(az), dist0 * np.sin(az), 1.5], dtype=float)
    vel = np.array([-speed * np.cos(az), -speed * np.sin(az), 0.0], dtype=float)
    for i in range(n):
        env.set_loom_object(pos)
        hL, hR = stereo_to_hex(*env.get_eye_images())
        hex_L.append(hL)
        hex_R.append(hR)
        theta.append(angular_size_deg(float(np.linalg.norm(pos[:2]))))
        pos = pos + vel * dt
    return np.asarray(hex_L), np.asarray(hex_R), np.asarray(theta, dtype=np.float32), room


def plot_compare(frames, traces, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    angs = (20, 40, 60)
    names = [n for n in ("hex_mono", "hex_bino", "camera") if n in frames]
    fig, axes = plt.subplots(6, 3, figsize=(11.0, 13.5))
    for j, name in enumerate(names):
        for i, ang in enumerate(angs):
            ax = axes[i, j]
            ax.imshow(frames[name][ang], cmap="gray", vmin=0, vmax=1, origin="upper")
            ax.set_title(f"{name} {ang}°", fontsize=8)
            ax.axis("off")
    keys = (
        ("t4_L", "T4 L"),
        ("t5_L", "T5 L"),
        ("lplc2_L", "LPLC2 L"),
        ("lc4_L", "LC4 L"),
        ("dnp01_rise", "DNp01 rise"),
        ("hex_L_mean", "hex L mean"),
    )
    for k, (key, title) in enumerate(keys):
        ax = axes[3 + (k // 3), k % 3]
        for name, rec in traces.items():
            ax.plot(rec["theta"], [r[key] for r in rec["rows"]], lw=1.2, label=name)
        ax.axvspan(40, 60, color="tab:green", alpha=0.12)
        ax.set_xlim(8, 90)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("angular size (deg)")
        ax.grid(True, alpha=0.3)
        if k == 0:
            ax.legend(fontsize=7)
    fig.suptitle("az=35° l/v=0.04: synthetic hex disc vs 3D camera→hex")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _rel_err(a, b) -> float:
    return abs(float(a) - float(b)) / max(abs(float(a)), abs(float(b)), 1e-6)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default=str(ROOT / "output" / "m2"))
    args = p.parse_args(argv)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    cov = plot_coverage(out / "eye_coverage.png")
    print("coverage", json.dumps(cov, indent=2), flush=True)

    from threedfly.brain.closed_loop import ClosedLoopBrain, core_stack_available
    from threedfly.brain.constants import flyvis_available
    from threedfly.sim import FlyEnvironment

    if not flyvis_available() or not core_stack_available():
        print("skip traces (no flyvis/core)", flush=True)
        return 0

    params = FlightParams()
    print("Building brain…", flush=True)
    brain = ClosedLoopBrain.build(params=params)
    brain.params.flyvis_stride = 1
    from threedfly.brain.constants import N_HEX as _N

    dummy = np.zeros((8, 8, 3), dtype=np.uint8)
    g = np.full(_N, 0.5, dtype=np.float32)
    brain.reset()
    last = None
    for _ in range(20):
        last = brain.step(dummy, dummy, hex_L=g, hex_R=g)
    rest_dc = {
        "gray_hex_dnp01_L": float(last.readout.dnp01_L),
        "gray_hex_dnp01_R": float(last.readout.dnp01_R),
        "gray_hex_dnp01_L_minus_R": float(last.readout.dnp01_L) - float(last.readout.dnp01_R),
        "note": (
            "Identical gray hex on both eyes. Any L−R is a brain DC; "
            "rise uses per-side rest so it cannot hide a contralateral loom."
        ),
    }
    print("  rest DC (gray hex)", rest_dc, flush=True)
    env = FlyEnvironment(render_mode="rgb_array", camera_width=160, camera_height=160)
    env.reset(seed=0)
    env.set_pose((0.0, 0.0, 1.5), yaw=0.0, vel=(0.0, 0.0, 0.0))

    frames = {"hex_mono": {}, "hex_bino": {}, "camera": {}}
    frame_stats = {}
    _, _, xd, yd = hex_degrees()
    for ang in (20, 40, 60):
        hL, hR = hex_disc_at_size(ang, x0=35.0, binocular=False)
        frames["hex_mono"][ang] = _hex_image(hL, "L")
        hLb, hRb = hex_disc_at_size(ang, x0=35.0, binocular=True)
        frames["hex_bino"][ang] = _hex_image(hLb, "L")
        hLc, hRc, left_img, right_img, *_ = camera_hex_at_size(env, 35.0, ang)
        frames["camera"][ang] = _hex_image(hLc, "L")
        disc = np.hypot(xd - 35.0, yd) <= ang / 2.0
        frame_stats[str(ang)] = {
            "hex_mono_disc_mean": float(hL[disc].mean()),
            "camera_disc_mean": float(hLc[disc].mean()),
            "camera_R_disc_mean": float(hRc[disc].mean()),
            "hex_mono_bg_mean": float(hL[~disc].mean()),
            "camera_bg_mean": float(hLc[~disc].mean()),
            "disc_mean_rel_err": _rel_err(hL[disc].mean(), hLc[disc].mean()) if disc.any() else None,
            "n_disc": int(disc.sum()),
            "imgL_mean": float(left_img.mean() / 255.0),
        }
        print(
            f"  frame {ang}° hex_mono disc={frame_stats[str(ang)]['hex_mono_disc_mean']:.3f} "
            f"cam L={frame_stats[str(ang)]['camera_disc_mean']:.3f} "
            f"cam R={frame_stats[str(ang)]['camera_R_disc_mean']:.3f}",
            flush=True,
        )

    pre, stim, lv, x0 = 0.3, 0.7, 0.04, 35.0
    traces = {}
    hL, hR, th = build_hex_movie(lv, x0, False, DT, pre, stim)
    traces["hex_mono_10ms"] = {"theta": th, "rows": run_trace(brain, hL, hR)}
    print("  hex mono done", flush=True)
    hL, hR, th = build_hex_movie(lv, x0, True, DT, pre, stim)
    traces["hex_bino_10ms"] = {"theta": th, "rows": run_trace(brain, hL, hR)}
    print("  hex bino done", flush=True)
    env.reset(seed=0)
    env.set_pose((0.0, 0.0, 1.5), yaw=0.0, vel=(0.0, 0.0, 0.0))
    hL, hR, th, room = build_camera_movie(env, 35.0, lv, 3.6, DT, pre, stim)
    traces["camera_10ms"] = {"theta": th, "rows": run_trace(brain, hL, hR, pre_hex=room)}
    print("  camera 10ms done", flush=True)

    def onset(rows, key, thresh, theta):
        for r, t in zip(rows, theta):
            if float(t) > 5.0 and float(r[key]) >= thresh:
                return float(t)
        return None

    onsets = {}
    for name, rec in traces.items():
        onsets[name] = {
            "dnp01_rise_0.20": onset(rec["rows"], "dnp01_rise", 0.20, rec["theta"]),
            "lc4_L_0.10": onset(rec["rows"], "lc4_L", 0.10, rec["theta"]),
            "lplc2_L_0.10": onset(rec["rows"], "lplc2_L", 0.10, rec["theta"]),
            "peak_dnp01_rise": max(r["dnp01_rise"] for r in rec["rows"]),
            "peak_lc4_L": max(r["lc4_L"] for r in rec["rows"]),
            "peak_lplc2_L": max(r["lplc2_L"] for r in rec["rows"]),
            "peak_t4_L": max(r["t4_L"] for r in rec["rows"]),
            "peak_t5_L": max(r["t5_L"] for r in rec["rows"]),
            "would_fire": max(r["would_fire"] for r in rec["rows"]),
        }
        print(f"  {name} {onsets[name]}", flush=True)

    match = {}
    ref = onsets["hex_mono_10ms"]
    cam = onsets["camera_10ms"]
    for key in ("peak_dnp01_rise", "peak_lc4_L", "peak_lplc2_L", "peak_t4_L", "peak_t5_L"):
        match[key] = _rel_err(ref[key], cam[key])
    match["within_20pct"] = all(match[k] <= 0.20 for k in ("peak_dnp01_rise", "peak_lc4_L", "peak_lplc2_L"))
    print("  match hex_mono vs camera", match, flush=True)

    plot_compare(frames, traces, out / "hex_vs_camera.png")
    env.close()
    payload = {
        "coverage": cov,
        "onsets": onsets,
        "frame_stats": frame_stats,
        "match": match,
        "rest_dc": rest_dc,
        "painted": False,
    }
    (out / "eye_frontend.json").write_text(json.dumps(payload, indent=2))
    print(f"wrote {out / 'eye_coverage.png'} {out / 'hex_vs_camera.png'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
