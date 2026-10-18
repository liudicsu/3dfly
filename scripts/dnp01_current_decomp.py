#!/usr/bin/env python3
"""DNp01 input-current decomposition: hex_ref vs camera at matched angles."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location("eye_factorial", ROOT / "scripts" / "eye_factorial.py")
_fac = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_fac)

from threedfly.control.dn_controller import DNFlightController
from threedfly.control.flight_params import FlightParams

ANGLES = (20.0, 40.0, 60.0, 90.0)

GROUPS = (
    ("LC4", lambda t: t == "LC4"),
    ("LPLC2", lambda t: t == "LPLC2"),
    ("LPLC1", lambda t: t == "LPLC1"),
    ("other_LC_LPLC", lambda t: (t.startswith("LC") or t.startswith("LPLC")) and t not in ("LC4", "LPLC2", "LPLC1")),
    ("LPi", lambda t: t.startswith("LPi")),
    ("Li_inh", lambda t: t.startswith("Li") or t.startswith("Am") or t.startswith("Pm")),
    ("T4T5", lambda t: t.startswith("T4") or t.startswith("T5")),
    ("other_DN", lambda t: t.startswith("DN")),
    ("HS", lambda t: t.startswith("HS")),
    ("other_inh", lambda t: t.startswith(("TmY", "Tm5", "Tlp", "LPT", "MeLo", "MeVC"))),
)


def _group_masks(typ: np.ndarray) -> dict:
    names = np.asarray(typ, dtype=object)
    used = np.zeros(len(names), dtype=bool)
    out = {}
    for label, pred in GROUPS:
        m = np.array([bool(pred(str(t))) for t in names])
        out[label] = m
        used |= m
    out["other"] = ~used
    return out


def _current_into(Wm, r, post_idx, masks) -> dict:
    if post_idx.size == 0:
        return {k: 0.0 for k in list(masks) + ["total", "exc", "inh"]}
    row = np.asarray(Wm[post_idx].dot(r) if False else Wm[post_idx] @ r)
    # Split by pre type: sum_i,j W[i,j]*r[j] for i in post
    Wsub = Wm[post_idx]  # n_post x n
    wr = np.asarray(Wsub.multiply(r[None, :]).sum(axis=0)).ravel() if hasattr(Wsub, "multiply") else None
    if wr is None:
        Wd = Wsub.toarray()
        wr = (Wd * r[None, :]).sum(axis=0)
    else:
        # multiply is elementwise with broadcast? csr.multiply(r) multiplies columns
        Wd = Wsub.tocsr()
        wr = np.asarray(Wd.multiply(r).sum(axis=0)).ravel()
    rec = {}
    for k, m in masks.items():
        rec[k] = float(wr[m].sum())
    rec["total"] = float(wr.sum())
    rec["exc"] = float(wr[wr > 0].sum())
    rec["inh"] = float(wr[wr < 0].sum())
    return rec


def _lplc2_onto_gf(Wm, r, typ, post_idx, hex_xy, loom_az=35.0) -> dict:
    """Which LPLC2 cells drive DNp01, vs which fire, vs disc RF."""
    pre = np.flatnonzero(np.asarray(typ, dtype=object) == "LPLC2")
    if post_idx.size == 0 or pre.size == 0:
        return {}
    Wd = Wm[post_idx][:, pre].toarray()
    contrib = (Wd * r[pre][None, :]).sum(axis=0)
    rates = r[pre]
    xy = hex_xy[pre] if hex_xy is not None else None
    top = np.argsort(-np.abs(contrib))[:8]
    rec = {
        "n": int(pre.size),
        "rate_mean": float(rates.mean()),
        "rate_p90": float(np.percentile(rates, 90)),
        "rate_max": float(rates.max()),
        "I_sum": float(contrib.sum()),
        "I_pos": float(contrib[contrib > 0].sum()),
        "I_neg": float(contrib[contrib < 0].sum()),
        "n_firing": int((rates >= 0.10).sum()),
        "n_driving": int((contrib > 0.01).sum()),
    }
    if xy is not None:
        az = xy[:, 0]
        el = xy[:, 1]
        near = np.hypot(az - loom_az, el) <= 25.0
        rec["rate_near_disc"] = float(rates[near].mean()) if near.any() else 0.0
        rec["I_near_disc"] = float(contrib[near].sum()) if near.any() else 0.0
        rec["n_near"] = int(near.sum())
        rec["top_az"] = [float(az[i]) for i in top]
        rec["top_el"] = [float(el[i]) for i in top]
        rec["top_rate"] = [float(rates[i]) for i in top]
        rec["top_I"] = [float(contrib[i]) for i in top]
    return rec


def _hex_xy(conn, n: int) -> np.ndarray:
    h = np.asarray(conn.hex, dtype=np.float32)
    if h.ndim == 2 and h.shape[1] >= 2:
        return h[:, :2]
    return np.full((n, 2), np.nan, dtype=np.float32)


def nearest_frame(theta, target: float) -> int:
    """Nearest *approaching* frame (theta still growing). Avoids post-collision."""
    th = np.asarray(theta, dtype=float)
    if th.size >= 3:
        rising = np.r_[True, np.diff(th) >= -1e-3]
        cand = np.flatnonzero(rising)
        if cand.size:
            return int(cand[np.argmin(np.abs(th[cand] - float(target)))])
    return int(np.argmin(np.abs(th - float(target))))


def run_path(brain, ctl, hex_L, hex_R, theta, pre_hex, label: str) -> dict:
    from threedfly.brain.eyes import reset_luminance_rest

    reset_luminance_rest()
    brain.reset()
    ctl.reset()
    dummy = np.zeros((8, 8, 3), dtype=np.uint8)
    last = None
    for _ in range(24):
        last = brain.step(dummy, dummy, hex_L=pre_hex[0], hex_R=pre_hex[1])
    if last is not None:
        ctl.capture_baseline(last.readout)
    rest = {
        "dnp01_L": float(last.readout.dnp01_L),
        "dnp01_R": float(last.readout.dnp01_R),
        "lc4_L": float(last.readout.lc4_L),
        "lplc2_L": float(last.readout.lplc2_L),
    }
    core = brain.core
    typ = np.asarray(core.typ, dtype=object)
    masks = _group_masks(typ)
    post_L = core.select(r"^DNp01$", side="L")
    post_R = core.select(r"^DNp01$", side="R")
    ttmn = core.select(r"^TTMn$")
    psi = core.select(r"^PSI$")
    hex_xy = _hex_xy(getattr(brain, "connectome", None) or type("C", (), {"hex": getattr(core, "hex", np.zeros((core.n, 2)))})(), core.n)
    if hasattr(brain, "_conn_hex"):
        hex_xy = brain._conn_hex

    snaps = []
    picked = set()
    for ang in ANGLES:
        i = nearest_frame(theta, ang)
        if i in picked:
            continue
        picked.add(i)
        for j in range(len(snaps), i + 1):
            res = brain.step(dummy, dummy, hex_L=hex_L[j], hex_R=hex_R[j])
            ctl.compute(res.readout)
        r = core.r
        I_L = _current_into(core.Wm, r, post_L, masks)
        I_R = _current_into(core.Wm, r, post_R, masks)
        I_tt = _current_into(core.Wm, r, ttmn, masks)
        I_psi = _current_into(core.Wm, r, psi, masks)
        gap = {}
        if core.G is not None:
            g = np.asarray(core.G[np.concatenate([ttmn, psi])] @ r).ravel() if (ttmn.size + psi.size) else np.zeros(0)
            gap = {"partners_gap_mean": float(g.mean()) if g.size else 0.0}
        snaps.append(
            {
                "want_deg": ang,
                "got_deg": float(theta[i]),
                "step": int(i),
                "dnp01_L": float(res.readout.dnp01_L),
                "dnp01_R": float(res.readout.dnp01_R),
                "rise": float(ctl.last_rise),
                "rest_L": float(ctl.dnp01_rest_L or 0.0),
                "rest_R": float(ctl.dnp01_rest_R or 0.0),
                "lc4_L": float(res.readout.lc4_L),
                "lc4_R": float(res.readout.lc4_R),
                "lplc2_L": float(res.readout.lplc2_L),
                "lplc2_R": float(res.readout.lplc2_R),
                "hex_mean": float(np.mean(hex_L[i])),
                "hex_min": float(np.min(hex_L[i])),
                "I_DNp01_L": I_L,
                "I_DNp01_R": I_R,
                "I_TTMn": I_tt,
                "I_PSI": I_psi,
                "gap": gap,
                "LPLC2_onto_DNp01_L": _lplc2_onto_gf(core.Wm, r, typ, post_L, hex_xy),
            }
        )
    return {"label": label, "rest": rest, "angles": snaps}


def attach_hex(brain) -> None:
    from threedfly.brain.core_bundle import core_bundle_path, load_core_bundle

    conn, _, _, _ = load_core_bundle(core_bundle_path())
    brain._conn_hex = np.asarray(conn.hex, dtype=np.float32)
    if brain._conn_hex.ndim == 1:
        brain._conn_hex = np.full((brain.core.n, 2), np.nan, dtype=np.float32)


def main() -> int:
    from threedfly.brain.closed_loop import ClosedLoopBrain, core_stack_available
    from threedfly.brain.constants import flyvis_available
    from threedfly.sim import FlyEnvironment

    if not flyvis_available() or not core_stack_available():
        return 2
    params = FlightParams()
    params.flyvis_stride = 1
    brain = ClosedLoopBrain.build(params=params)
    attach_hex(brain)
    ctl = DNFlightController(params)
    env = FlyEnvironment(render_mode="rgb_array")
    env.reset(seed=0)
    env.set_pose((0.0, 0.0, 1.5), yaw=0.0, vel=(0.0, 0.0, 0.0))
    hL, hR, th, room = _fac.camera_movie(env, 0.04, _fac.DIST0, _fac.DT, _fac.PRE_S, _fac.STIM_S, translate=False)
    env.close()
    cam = run_path(brain, ctl, hL, hR, th, room, "camera")
    hL_r, hR_r, th_r = _fac.hex_ref_movie(0.04, _fac.DT, _fac.PRE_S, _fac.STIM_S)
    href = run_path(brain, ctl, hL_r, hR_r, th_r, (hL_r[0], hR_r[0]), "hex_ref")
    out = {"camera": cam, "hex_ref": href}
    path = ROOT / "output" / "m2" / "dnp01_current_decomp.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
