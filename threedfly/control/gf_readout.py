"""Giant-fiber readout: one calibration on the canonical flyvis loom.

von Reyn et al. 2014 (Nat Neurosci) and 2017 (Curr Biol) show that GF /
DNp01 encodes angular size + angular velocity, and that the takeoff
decision sits around 40–60° angular size at l/v = 40 ms.

The 0.20 Hz DNp01-rise bar was never fit to that window. This module
replaces it with a readout chosen once on a fixed canonical set:

    flyvis ``loom()``, l/v = 40 ms, fly still, gray background
        * both eyes, centred (x0 = 0)
        * left eye only, x0 = +30°
        * right eye only, x0 = −30°

Two candidate forms (same units so they can be compared):

    pd:         score = rise + k · d(rise)/dt
    integrate:  score = ∫ max(0, rise + k · d(rise)/dt) dt

``d(rise)/dt`` is a ``gf_vel_window_s`` (default 40 ms) finite difference
so a one-frame spike is not treated as a loom. ``k`` is in seconds;
``rise`` is Hz. The form and ``(k, threshold)`` that put all three
canonical first-crossings nearest 50° (and inside 40–60°) are frozen in
``FlightParams``. Re-run ``scripts/calibrate_gf_readout.py`` only if the
brain stack itself changes.

No other closed-loop gain is fit here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from threedfly.brain.constants import DT
from threedfly.brain.stimuli import gray, loom, loom_theta_deg

# Literature target (von Reyn et al. 2014/2017).
GF_TARGET_DEG = 50.0
GF_WINDOW_LO_DEG = 40.0
GF_WINDOW_HI_DEG = 60.0
GF_LV_S = 0.04
GF_VEL_WINDOW_S = 0.04
GF_CITATION = (
    "von Reyn et al. 2014 Nat Neurosci 17:1594; "
    "von Reyn et al. 2017 Curr Biol 27:3489"
)
GF_SCRIPT = "scripts/calibrate_gf_readout.py"

CANONICAL = (
    {"name": "both_x0_0", "x0": 0.0, "eyes": "both"},
    {"name": "L_az30", "x0": 30.0, "eyes": "L"},
    {"name": "R_az-30", "x0": -30.0, "eyes": "R"},
)

TRACES_PATH = Path(__file__).with_name("gf_readout_traces.json")


@dataclass
class GFFit:
    """Frozen readout chosen on the canonical hex-loom set."""

    form: str
    k: float
    threshold: float
    vel_window_s: float = GF_VEL_WINDOW_S
    lv_s: float = GF_LV_S
    target_deg: float = GF_TARGET_DEG
    window_lo_deg: float = GF_WINDOW_LO_DEG
    window_hi_deg: float = GF_WINDOW_HI_DEG
    citation: str = GF_CITATION
    script: str = GF_SCRIPT
    fire_angles_deg: Dict[str, float] = field(default_factory=dict)
    rms_to_target: float = float("nan")
    n_in_window: int = 0
    rise_at_target_hz: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def rise_velocity(
    rise_hist: Sequence[float],
    dt: float,
    window_s: float = GF_VEL_WINDOW_S,
) -> float:
    """Hz/s over a fixed physical window (not a one-frame spike)."""
    if len(rise_hist) < 2 or dt <= 0.0:
        return 0.0
    n = max(1, int(round(float(window_s) / float(dt))))
    n = min(n, len(rise_hist) - 1)
    return (float(rise_hist[-1]) - float(rise_hist[-1 - n])) / (n * float(dt))


def pd_score(rise: float, vel: float, k: float) -> float:
    return float(rise) + float(k) * float(vel)


def first_fire_angle(
    theta_deg: Sequence[float],
    rise: Sequence[float],
    dt: float,
    form: str,
    k: float,
    threshold: float,
    vel_window_s: float = GF_VEL_WINDOW_S,
) -> Optional[float]:
    """Angular size at the first score crossing. None if never."""
    integ = 0.0
    hist: List[float] = []
    for i, (th, r) in enumerate(zip(theta_deg, rise)):
        hist.append(float(r))
        vel = rise_velocity(hist, dt, vel_window_s)
        inst = pd_score(r, vel, k)
        if form == "integrate":
            integ += max(0.0, inst) * float(dt)
            score = integ
        else:
            score = inst
        if score >= float(threshold) and float(th) >= 12.0:
            return float(th)
    return None


def _rms(angles: Iterable[float], target: float) -> float:
    vals = [float(a) for a in angles]
    if not vals:
        return float("inf")
    return float(np.sqrt(np.mean([(a - target) ** 2 for a in vals])))


def _n_in_window(angles: Iterable[float], lo: float, hi: float) -> int:
    return int(sum(1 for a in angles if lo <= float(a) <= hi))


def fit_gf_readout(
    traces: Dict[str, Dict[str, List[float]]],
    dt: float = DT,
    target_deg: float = GF_TARGET_DEG,
    window_lo: float = GF_WINDOW_LO_DEG,
    window_hi: float = GF_WINDOW_HI_DEG,
    vel_window_s: float = GF_VEL_WINDOW_S,
) -> GFFit:
    """Grid-search form / k / threshold on the canonical traces only.

    Preference: all three fires in 40–60°, then RMS to 50°, then the PD
    form (static objects do not accumulate), then smaller |k|.

    On the real canonical traces the binocular hex loom is stronger
    (rise@50° ≈ 0.25 Hz vs ≈ 0.18 Hz at ±30°). No (form, k, threshold)
    puts all three first-crossings in 40–60°; the frozen fit keeps L/R
    in the window and lets the centred loom fire a few degrees early.
    """
    names = [c["name"] for c in CANONICAL]
    missing = [n for n in names if n not in traces]
    if missing:
        raise KeyError(f"canonical traces missing {missing}")

    candidates: List[Tuple[tuple, GFFit]] = []
    grids = (
        (
            "pd",
            np.linspace(0.0, 0.16, 33),
            np.linspace(0.04, 0.48, 45),
        ),
        (
            "integrate",
            np.linspace(0.0, 0.16, 33),
            np.linspace(0.001, 0.090, 45),
        ),
    )
    for form, ks, ths in grids:
        for k in ks:
            for thr in ths:
                angles: Dict[str, float] = {}
                ok = True
                for name in names:
                    ang = first_fire_angle(
                        traces[name]["theta_deg"],
                        traces[name]["rise"],
                        dt,
                        form,
                        float(k),
                        float(thr),
                        vel_window_s,
                    )
                    if ang is None:
                        ok = False
                        break
                    angles[name] = float(ang)
                if not ok:
                    continue
                n_win = _n_in_window(angles.values(), window_lo, window_hi)
                rms = _rms(angles.values(), target_deg)
                fit = GFFit(
                    form=form,
                    k=float(k),
                    threshold=float(thr),
                    vel_window_s=float(vel_window_s),
                    target_deg=float(target_deg),
                    window_lo_deg=float(window_lo),
                    window_hi_deg=float(window_hi),
                    fire_angles_deg=angles,
                    rms_to_target=rms,
                    n_in_window=n_win,
                )
                # Sort key: more in-window, lower RMS, prefer pd, smaller k.
                prefer_pd = 0 if form == "pd" else 1
                candidates.append(((-n_win, rms, prefer_pd, abs(float(k)), float(thr)), fit))

    if not candidates:
        raise RuntimeError(
            "GF readout grid found no (form, k, threshold) that fires "
            "on all three canonical looms"
        )
    candidates.sort(key=lambda item: item[0])
    best = candidates[0][1]
    best.rise_at_target_hz = {
        name: _rise_at_angle(traces[name]["theta_deg"], traces[name]["rise"], target_deg)
        for name in names
    }
    return best


def _rise_at_angle(
    theta_deg: Sequence[float],
    rise: Sequence[float],
    target_deg: float,
) -> float:
    th = np.asarray(theta_deg, dtype=float)
    r = np.asarray(rise, dtype=float)
    if th.size == 0:
        return float("nan")
    i = int(np.argmin(np.abs(th - float(target_deg))))
    return float(r[i])


def score_trace(
    rise: Sequence[float],
    dt: float,
    form: str,
    k: float,
    vel_window_s: float = GF_VEL_WINDOW_S,
) -> np.ndarray:
    hist: List[float] = []
    integ = 0.0
    out = np.zeros(len(rise), dtype=np.float64)
    for i, r in enumerate(rise):
        hist.append(float(r))
        vel = rise_velocity(hist, dt, vel_window_s)
        inst = pd_score(r, vel, k)
        if form == "integrate":
            integ += max(0.0, inst) * float(dt)
            out[i] = integ
        else:
            out[i] = inst
    return out


def collect_canonical_traces(
    brain,
    ctl,
    *,
    lv: float = GF_LV_S,
    pre: float = 0.40,
    stim: float = 0.80,
    gray_prewarm: int = 20,
    dt: float = DT,
) -> Dict[str, Dict[str, List[float]]]:
    """Drive the three canonical hex looms; return rise and θ traces.

    Fly still. Standard gray background. ``ctl.capture_baseline`` after a
    gray hex prewarm so rise is the same quantity the closed-loop
    controller uses. Open-loop hex movies — no cameras, no object pose.
    """
    dummy = np.zeros((8, 8), dtype=np.float32)
    traces: Dict[str, Dict[str, List[float]]] = {}
    for spec in CANONICAL:
        movie = np.concatenate(
            [gray(pre, dt=dt), loom(stim, lv=lv, x0=float(spec["x0"]), dt=dt)]
        )
        theta = np.concatenate(
            [
                np.full(int(round(pre / dt)), 0.0, dtype=np.float32),
                loom_theta_deg(stim, lv=lv, dt=dt),
            ]
        )
        blank = gray(pre + stim, dt=dt)
        eyes = spec["eyes"]
        left = movie if eyes in ("both", "L") else blank
        right = movie if eyes in ("both", "R") else blank
        brain.reset()
        ctl.reset()
        g = gray(0.2, dt=dt)[0]
        last = None
        for _ in range(int(gray_prewarm)):
            last = brain.step(g, g, None, hex_L=g, hex_R=g)
        if last is not None:
            ctl.capture_baseline(last.readout)
        rise: List[float] = []
        dnp01: List[float] = []
        for i in range(left.shape[0]):
            res = brain.step(dummy, dummy, None, hex_L=left[i], hex_R=right[i])
            ctl.compute(res.readout)
            rise.append(float(ctl.last_rise))
            dnp01.append(max(float(res.readout.dnp01_L), float(res.readout.dnp01_R)))
        traces[spec["name"]] = {
            "theta_deg": [float(x) for x in theta],
            "rise": rise,
            "dnp01": dnp01,
            "x0": float(spec["x0"]),
            "eyes": eyes,
            "lv": float(lv),
            "dt": float(dt),
        }
    return traces


def traces_to_jsonable(traces: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "citation": GF_CITATION,
        "script": GF_SCRIPT,
        "canonical": list(CANONICAL),
        "traces": traces,
    }


def load_canonical_traces(path: Optional[Path] = None) -> Dict[str, Dict[str, List[float]]]:
    import json

    p = Path(path) if path is not None else TRACES_PATH
    blob = json.loads(p.read_text())
    return blob["traces"]


def apply_fit_to_params(params, fit: GFFit) -> None:
    """Write the frozen readout onto a ``FlightParams`` instance."""
    params.gf_form = str(fit.form)
    params.gf_k = float(fit.k)
    params.gf_threshold = float(fit.threshold)
    params.gf_vel_window_s = float(fit.vel_window_s)
    params.gf_target_deg = float(fit.target_deg)
    params.dnp01_escape_delta_hz = float(fit.threshold) if fit.form == "pd" else float(params.dnp01_escape_delta_hz)
