#!/usr/bin/env python3
"""Offline M1 acceptance: flyvis hex stimuli → coreTplus rate core → T1–T7.

Single command (verification machine with connectome + flyvis):

    python scripts/brain_acceptance.py
    python scripts/brain_acceptance.py --quick   # short movies, unofficial metrics
    python scripts/brain_acceptance.py --rho-mode subgraph  # local ρ→0.9 (not research)

Writes ``output/m1/acceptance.json`` and a trace plot. Prints T1–T7 plus
ms/step (brain-only and flyvis). Skips nothing here — missing data is an error
so the verification run fails loudly. Unit tests skip when data/flyvis are absent.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threedfly.brain.acceptance import TRACE_KEYS, summarize_traces, verdict
from threedfly.brain.constants import (
    CORE_TPLUS_N_TARGET,
    DNG02_BIAS_DEFAULT,
    DT,
    G_GAP_DEFAULT,
    G_INH_DEFAULT,
    HALTERE_RATE,
    RHO_FULL_SIGNED_REFERENCE,
    connectome_available,
    flyvis_available,
)
from threedfly.brain.connectome_matrix import build_signed_matrix
from threedfly.brain.core_subgraph import select_core_tplus, slice_connectome
from threedfly.brain.handoff import (
    FlyvisCalibration,
    FlyvisEye,
    apply_handoff_plan,
    build_handoff_map,
    compile_handoff_plan,
    haltere_indices,
)
from threedfly.brain.rate_core import RateCore
from threedfly.brain.readouts import build_readout_index
from threedfly.brain.stimuli import SCENARIOS, acceptance_eye_movies


def _remap(n_full: int, keep: np.ndarray) -> np.ndarray:
    remap = -np.ones(n_full, dtype=np.int64)
    remap[keep] = np.arange(len(keep))
    return remap


def _voltages_at(
    cond_acts: dict,
    cond: str,
    frame: int,
) -> dict:
    return {t: arr[frame] for t, arr in cond_acts[cond].items()}


def _run_core(
    core: RateCore,
    index,
    cond_acts,
    plans,
    halt,
    settle: int = 50,
    pre_steps: int = 100,
    flyvis_ms: float = 0.0,
    label: str = "brain",
) -> dict:
    traces = {}
    nceil_rest = 0
    brain_ms = []
    nT = next(iter(next(iter(cond_acts.values())).values())).shape[0]
    n_scen = len(SCENARIOS)
    print(
        f"{label}: {n_scen} scenarios × {settle + nT} steps (settle={settle}, movie={nT})",
        flush=True,
    )
    for si, (name, (cl, cr)) in enumerate(SCENARIOS.items(), 1):
        core.reset()
        tr = {k: [] for k in TRACE_KEYS if k in index}
        scen_ms = []
        t_sc = time.perf_counter()
        for k in range(settle + nT):
            f = max(0, k - settle)
            VL = _voltages_at(cond_acts, cl, f)
            VR = _voltages_at(cond_acts, cr, f)
            IL, RL = apply_handoff_plan(plans["L"], VL)
            IR, RR = apply_handoff_plan(plans["R"], VR)
            ci = np.concatenate([IL, IR, halt]) if halt.size else np.concatenate([IL, IR])
            cv = np.concatenate(
                [RL, RR, np.full(len(halt), HALTERE_RATE, dtype=np.float32)]
            )
            t0 = time.perf_counter()
            r = core.step(clamp_idx=ci, clamp_val=cv)
            dt_ms = 1000.0 * (time.perf_counter() - t0)
            brain_ms.append(dt_ms)
            scen_ms.append(dt_ms)
            rest_k = min(settle + 20, settle + nT - 1)
            if name == "gray" and k == rest_k:
                nceil_rest = int((r >= core.cap - 1).sum())
            for key in tr:
                tr[key].append(float(r[index[key]].mean()) if index.size(key) else 0.0)
        s0 = settle + int(pre_steps)
        start = max(0, s0 - 30)
        traces[name] = {
            "trace": {k: np.round(np.array(v)[start:], 3).tolist() for k, v in tr.items()},
        }
        elapsed = time.perf_counter() - t_sc
        print(
            f"  [{si}/{n_scen}] {name}: {settle + nT} steps, "
            f"{float(np.median(scen_ms)):.2f} ms/step ({elapsed:.1f}s)",
            flush=True,
        )
    summary = summarize_traces(traces, nceil_rest=nceil_rest)
    summary["ms_per_step_brain"] = round(float(np.median(brain_ms)), 2)
    summary["ms_per_frame_flyvis"] = round(float(flyvis_ms), 2)
    summary["ms_per_step_with_flyvis"] = round(
        float(np.median(brain_ms)) + 2.0 * float(flyvis_ms), 2
    )
    return {"scenarios": traces, "summary": summary}


def _plot(path: Path, result: dict) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=False)
    pairs = (("loomL", "loom R vs L DNp01"), ("loomR", "loomR"), ("CW", "CW"), ("expand", "expand"))
    for ax, (name, title) in zip(axes.ravel(), pairs):
        tr = result["scenarios"][name]["trace"]
        ax.plot(tr["DNp01|L"], label="DNp01 L")
        ax.plot(tr["DNp01|R"], label="DNp01 R")
        ax.set_title(title)
        ax.legend(fontsize=7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--output", default=str(ROOT / "output" / "m1" / "acceptance.json"))
    p.add_argument("--g-inh", type=float, default=G_INH_DEFAULT)
    p.add_argument("--g-gap", type=float, default=G_GAP_DEFAULT)
    p.add_argument("--dng02-bias", type=float, default=DNG02_BIAS_DEFAULT)
    p.add_argument("--no-shuffle-control", action="store_true")
    p.add_argument(
        "--quick",
        action="store_true",
        help="Short movies + no shuffle so a 2-core box finishes in a few minutes (metrics unofficial)",
    )
    p.add_argument(
        "--rho-mode",
        choices=("full_reference", "subgraph"),
        default="full_reference",
        help=(
            "S2 scale: full_reference (default, 0.9/4085.83, matches r3 even on "
            "subgraphs) or subgraph (rescale the slice itself to ρ=0.9; stronger "
            "recurrence, saturates HS)"
        ),
    )
    args = p.parse_args(argv)

    if not connectome_available(args.data_dir):
        print(
            "ERROR: data/full_connectome.npz not found. "
            "This script is the real-data acceptance run.",
            file=sys.stderr,
        )
        return 2
    if not flyvis_available():
        print(
            "ERROR: flyvis is not installed. "
            "On Python 3.12: pip install -e '.[brain]'. "
            "On Python 3.13 flyvis is not on PyPI; use a 3.12 venv (see README).",
            file=sys.stderr,
        )
        return 2

    print("Building signed connectome matrix…")
    conn = build_signed_matrix(args.data_dir, cache=True)
    print(f"  n={conn.n} nnz={conn.W.nnz} rho_full_ref={RHO_FULL_SIGNED_REFERENCE}")

    print("Selecting coreTplus…")
    core_sel = select_core_tplus(conn)
    print(" ", core_sel.recipe)
    if not core_sel.within_target():
        print(
            f"  WARNING: n={core_sel.n_neurons} outside ±5% of {CORE_TPLUS_N_TARGET}",
            file=sys.stderr,
        )
    sub = slice_connectome(conn, core_sel.keep)
    remap = _remap(conn.n, core_sel.keep)
    index = build_readout_index(sub.typ, sub.side, sub.body_ids)
    print(
        "  DNg02",
        index.size("DNg02|L") + index.size("DNg02|R"),
        "DNp03 L/R",
        index.size("DNp03|L"),
        index.size("DNp03|R"),
    )

    rho_arg = None if args.rho_mode == "subgraph" else RHO_FULL_SIGNED_REFERENCE
    print("Building RateCore (S2, g_inh=%s, rho_mode=%s)…" % (args.g_inh, args.rho_mode))
    core = RateCore.from_connectome(
        sub,
        g_inh=args.g_inh,
        g_gap=args.g_gap,
        dng02_bias=args.dng02_bias,
        shuffle=False,
        rho=rho_arg,
    )
    print(f"  rho_unscaled={core.rho_unscaled:.3f} scale={core.scale:.6g} n={core.n}")

    print("Handoff map (assignedOlHex + imputation)…")
    hmap = build_handoff_map(conn)
    print(f"  mapped neurons={hmap.n_mapped} sides={sorted(hmap.sides_present())}")

    print("Init flyvis + calibration (batch simulate, as fv4_run.py)…", flush=True)
    eye = FlyvisEye(torch_threads=2)
    cache = Path(args.output).parent / "flyvis_cal.npz"
    if args.quick:
        movies = acceptance_eye_movies(pre=0.3, stim=0.3, loom_T=0.3, tail=0.2)
        settle, pre_steps = 15, 30
        print("  --quick: pre=0.3s stim=0.3s settle=15, shuffle off (metrics unofficial)", flush=True)
    else:
        movies = acceptance_eye_movies()
        settle, pre_steps = 50, int(round(1.0 / DT))
    if cache.is_file():
        calib = FlyvisCalibration.load(cache)
        print(f"  loaded calibration {cache}", flush=True)
    else:
        calib = eye.calibrate(conditions=movies)
        if not args.quick:
            calib.save(cache)
            print(f"  wrote calibration {cache}", flush=True)
        else:
            print("  --quick: calibration not cached", flush=True)
    cond_acts = {}
    fly_ms = []
    for name, movie in movies.items():
        eye.reset()
        acts, ms = eye.simulate_movie(movie)
        cond_acts[name] = acts
        fly_ms.append(ms)
        print(f"  flyvis {name}: {movie.shape[0]} frames, {ms:.2f} ms/frame", flush=True)
    halt = haltere_indices(conn, remap)
    print("Compiling handoff column map (once)…", flush=True)
    t_plan = time.perf_counter()
    plans = {
        "L": compile_handoff_plan(hmap, calib, "L", remap),
        "R": compile_handoff_plan(hmap, calib, "R", remap),
    }
    print(
        f"  L={plans['L'].idx.size} R={plans['R'].idx.size} mapped "
        f"({time.perf_counter() - t_plan:.1f}s)",
        flush=True,
    )

    print("Running brain scenarios…", flush=True)
    result = _run_core(
        core,
        index,
        cond_acts,
        plans,
        halt,
        settle=settle,
        pre_steps=pre_steps,
        flyvis_ms=float(np.median(fly_ms)),
        label="brain",
    )
    shuffle_summary = None
    run_shuffle = not args.no_shuffle_control and not args.quick
    if run_shuffle:
        print("Shuffle control…", flush=True)
        shuf = RateCore.from_connectome(
            sub,
            g_inh=args.g_inh,
            g_gap=args.g_gap,
            dng02_bias=args.dng02_bias,
            shuffle=True,
            rho=rho_arg,
        )
        sh = _run_core(
            shuf,
            index,
            cond_acts,
            plans,
            halt,
            settle=settle,
            pre_steps=pre_steps,
            flyvis_ms=float(np.median(fly_ms)),
            label="shuffle",
        )
        shuffle_summary = sh["summary"]

    V = verdict(result["summary"], shuffle_summary)
    out = {
        "n_neurons": core_sel.n_neurons,
        "n_types": core_sel.n_types,
        "nnz": core_sel.nnz,
        "recipe": core_sel.recipe,
        "rho_sub": core.rho_unscaled,
        "rho_mode": args.rho_mode,
        "g_inh": args.g_inh,
        "g_gap": args.g_gap,
        "dng02_bias": args.dng02_bias,
        "quick": bool(args.quick),
        "handoff_mapped": hmap.n_mapped,
        "summary": result["summary"],
        "shuffle_summary": shuffle_summary,
        "verdict": V,
        "reference_coreTplus_g_inh8": {
            "T1_DNp01": "2127x",
            "T1_DNp03": "8.0x",
            "T3_rot": 0.152,
            "T3_exp": 0.163,
            "loom_DNp01_Hz": 2.13,
            "loomF_frac": 0.50,
            "ms_step": 2.08,
            "T2_end": 0.001,
            "yaw_HSE_CW_CCW": [-19.7, 20.5],
            "yaw_DNa02_CW_CCW": [-0.54, 0.36],
        },
        "scenarios": result["scenarios"],
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    _plot(out_path.with_suffix(".png"), result)

    print("\n=== M1 acceptance (compare to coreTplus + g_inh8) ===")
    for k, v in result["summary"].items():
        print(f"  {k}: {v}")
    if shuffle_summary:
        print("  shuffle T3_rot_frac:", shuffle_summary.get("T3_rot_frac"))
    print("  verdict:", json.dumps(V))
    print(f"  wrote {out_path}")
    return 0 if V.get("all") else 1


if __name__ == "__main__":
    sys.exit(main())
