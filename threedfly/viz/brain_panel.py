"""Brain-activity summaries for the interactive dashboard.

The old panel drew 200 evenly spaced neurons out of ~45k as 1-px bars, so
it looked empty: the coreTplus rates are sparse (≈5 % of cells above
0.01 Hz) and a stride sample mostly lands on silent cells. This module
reduces the full rate vector to things a person can read:

* population stats (mean / max / active fraction / number active),
* mean rate per cell-type family along the visual→motor pathway
  (Tm → T4/T5 → LPi → LPLC2/LC4 → DNp01/DNp03/DNa02 → DNg02 → TTMn/PSI),
* the most active cell types right now,
* rolling DN readout traces (for ``viser`` uPlot charts).

Everything is pure numpy so it is unit-testable without viser.
"""

from __future__ import annotations

import html
import re
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

# (label, regex on cell type). Ordered along the eye → wing signal flow.
PATHWAY_GROUPS: Tuple[Tuple[str, str], ...] = (
    ("Tm/TmY (eye input)", r"^Tm"),
    ("T2/T3", r"^T[23]"),
    ("T4 (ON motion)", r"^T4"),
    ("T5 (OFF motion)", r"^T5"),
    ("LPi", r"^LPi"),
    ("LPLC1", r"^LPLC1$"),
    ("LPLC2", r"^LPLC2$"),
    ("LC4", r"^LC4$"),
    ("HSE", r"^HSE$"),
    ("DNp01 (giant fiber)", r"^DNp01$"),
    ("DNp03", r"^DNp03$"),
    ("DNa02", r"^DNa02$"),
    ("DNp15", r"^(DNp15|DNHS1)$"),
    ("DNg02 (wing amp)", r"^DNg02"),
    ("TTMn/PSI", r"^(TTMn|PSI)$"),
    ("SApp (haltere)", r"^SApp$"),
)

# DN readout traces: (label, DNReadout attribute(s) averaged, colour).
DN_TRACES: Tuple[Tuple[str, Tuple[str, ...], str], ...] = (
    ("DNp01 L", ("dnp01_L",), "#e74c3c"),
    ("DNp01 R", ("dnp01_R",), "#c0392b"),
    ("DNp03 L", ("dnp03_L",), "#3498db"),
    ("DNp03 R", ("dnp03_R",), "#1f618d"),
    ("DNa02 L", ("dna02_L",), "#2ecc71"),
    ("DNa02 R", ("dna02_R",), "#1e8449"),
    ("LPLC2 L", ("lplc2_L",), "#f39c12"),
    ("LPLC2 R", ("lplc2_R",), "#b9770e"),
)
WING_TRACES: Tuple[Tuple[str, Tuple[str, ...], str], ...] = (
    ("DNg02 L", ("dng02_L",), "#8e44ad"),
    ("DNg02 R", ("dng02_R",), "#5b2c6f"),
)

ACTIVE_THRESHOLD = 0.01  # Hz; same threshold as ClosedLoopBrain.get_population_activity
LOG_FLOOR = 1e-3  # Hz; bars use log10(1 + r / LOG_FLOOR)


@dataclass
class PopulationStats:
    n: int
    mean: float
    max: float
    n_active: int
    active_fraction: float
    p99: float


def population_stats(rates: np.ndarray, threshold: float = ACTIVE_THRESHOLD) -> PopulationStats:
    r = np.asarray(rates, dtype=np.float32).ravel()
    if r.size == 0:
        return PopulationStats(0, 0.0, 0.0, 0, 0.0, 0.0)
    act = int(np.count_nonzero(r > threshold))
    return PopulationStats(
        n=int(r.size),
        mean=float(r.mean()),
        max=float(r.max()),
        n_active=act,
        active_fraction=act / float(r.size),
        p99=float(np.percentile(r, 99)),
    )


def top_active_indices(rates: np.ndarray, k: int = 200) -> np.ndarray:
    """Indices of the ``k`` most active neurons, sorted by index (for bar plots)."""
    r = np.asarray(rates).ravel()
    k = int(min(max(k, 0), r.size))
    if k == 0:
        return np.zeros(0, dtype=np.int64)
    idx = np.argpartition(-r, k - 1)[:k]
    return np.sort(idx).astype(np.int64)


def log_bar_fraction(value: float, vmax: float, floor: float = LOG_FLOOR) -> float:
    """0..1 bar length on a log scale so 1e-3 Hz and 20 Hz are both visible."""
    v = max(float(value), 0.0)
    top = np.log10(1.0 + max(float(vmax), floor) / floor)
    return float(np.clip(np.log10(1.0 + v / floor) / top, 0.0, 1.0))


class BrainPanelModel:
    """Turns the per-step rate vector + DN readout into dashboard data."""

    def __init__(
        self,
        types: Optional[Sequence[str]] = None,
        groups: Sequence[Tuple[str, str]] = PATHWAY_GROUPS,
        history: int = 300,
    ):
        self.types = None if types is None else np.asarray(types, dtype=object)
        self.history = int(history)
        self.group_index: Dict[str, np.ndarray] = {}
        self.type_codes: Optional[np.ndarray] = None
        self.type_names: List[str] = []
        self.type_counts: Optional[np.ndarray] = None
        if self.types is not None:
            tstr = np.asarray([str(t) for t in self.types], dtype=object)
            for label, pat in groups:
                rx = re.compile(pat)
                idx = np.flatnonzero([bool(rx.match(t)) for t in tstr]).astype(np.int64)
                if idx.size:
                    self.group_index[label] = idx
            names, codes = np.unique(tstr, return_inverse=True)
            self.type_names = [str(n) if str(n) not in ("", "nan", "None") else "(untyped)" for n in names]
            self.type_codes = codes.astype(np.int64)
            self.type_counts = np.bincount(self.type_codes, minlength=len(names))
        self.t: Deque[float] = deque(maxlen=self.history)
        self.traces: Dict[str, Deque[float]] = {
            lab: deque(maxlen=self.history) for lab, _, _ in DN_TRACES + WING_TRACES
        }
        self.last_rates: Optional[np.ndarray] = None
        self.last_stats: Optional[PopulationStats] = None
        self.n_updates = 0

    # ------------------------------------------------------------------ data
    def update(self, t: float, rates: np.ndarray, readout=None) -> None:
        r = np.asarray(rates, dtype=np.float32).ravel()
        self.last_rates = r
        self.last_stats = population_stats(r)
        self.n_updates += 1
        if readout is not None:
            self.t.append(float(t))
            for lab, attrs, _ in DN_TRACES + WING_TRACES:
                vals = [float(getattr(readout, a, 0.0)) for a in attrs]
                self.traces[lab].append(float(np.mean(vals)))

    def reset_history(self) -> None:
        self.t.clear()
        for d in self.traces.values():
            d.clear()

    def group_means(self, rates: Optional[np.ndarray] = None) -> Dict[str, float]:
        r = self.last_rates if rates is None else np.asarray(rates).ravel()
        if r is None:
            return {}
        return {lab: float(r[idx].mean()) for lab, idx in self.group_index.items()}

    def top_types(self, k: int = 10, min_size: int = 1, rates=None) -> List[Tuple[str, float, int]]:
        """Most active cell types by mean rate: (type, mean Hz, n cells)."""
        r = self.last_rates if rates is None else np.asarray(rates).ravel()
        if r is None or self.type_codes is None:
            return []
        sums = np.bincount(self.type_codes, weights=r.astype(np.float64), minlength=len(self.type_names))
        means = sums / np.maximum(self.type_counts, 1)
        ok = self.type_counts >= int(min_size)
        order = [i for i in np.argsort(-means) if ok[i]][: int(k)]
        return [(self.type_names[i], float(means[i]), int(self.type_counts[i])) for i in order]

    def trace_arrays(self, which: str = "dn") -> Tuple[np.ndarray, ...]:
        spec = DN_TRACES if which == "dn" else WING_TRACES
        t = np.asarray(self.t, dtype=np.float64)
        if t.size == 0:
            t = np.zeros(1)
            return (t,) + tuple(np.zeros(1) for _ in spec)
        return (t,) + tuple(np.asarray(self.traces[lab], dtype=np.float64) for lab, _, _ in spec)

    # ------------------------------------------------------------ rendering
    def stats_markdown(self) -> str:
        s = self.last_stats
        if s is None:
            return "**Brain:** waiting for first step…"
        return (
            f"**{s.n:,} neurons** · active (>{ACTIVE_THRESHOLD:g} Hz): "
            f"**{s.n_active:,}** ({100 * s.active_fraction:.1f}%)  \n"
            f"mean {s.mean:.3g} Hz · p99 {s.p99:.3g} Hz · max {s.max:.3g} Hz · "
            f"updates {self.n_updates}"
        )

    def groups_html(self, vmax: Optional[float] = None) -> str:
        gm = self.group_means()
        if not gm:
            return "<div style='font-size:12px;color:#888'>no cell-type groups</div>"
        top = vmax if vmax is not None else max(max(gm.values()), 1.0)
        rows = []
        for lab, val in gm.items():
            n = int(self.group_index[lab].size)
            frac = log_bar_fraction(val, top)
            rows.append(
                "<div style='display:flex;align-items:center;font:11px monospace;margin:1px 0'>"
                f"<span style='width:132px;white-space:nowrap;overflow:hidden'>{html.escape(lab)}</span>"
                "<span style='flex:1;background:#eee;height:10px;margin:0 4px;position:relative'>"
                f"<span style='position:absolute;left:0;top:0;bottom:0;width:{100 * frac:.1f}%;"
                f"background:{_bar_colour(frac)}'></span></span>"
                f"<span style='width:62px;text-align:right'>{val:.2e}</span>"
                f"<span style='width:62px;text-align:right;color:#888;padding-left:6px'>n={n}</span></div>"
            )
        head = (
            "<div style='font:bold 12px sans-serif;margin:2px 0 4px'>Mean rate by cell-type family "
            "(Hz, log bar)</div>"
        )
        return head + "".join(rows)

    def top_types_html(self, k: int = 10) -> str:
        rows = self.top_types(k)
        if not rows:
            return ""
        body = "".join(
            f"<tr><td>{html.escape(n)}</td><td style='text-align:right'>{m:.3g}</td>"
            f"<td style='text-align:right;color:#888'>{c}</td></tr>"
            for n, m, c in rows
        )
        return (
            "<div style='font:bold 12px sans-serif;margin:6px 0 2px'>Most active cell types</div>"
            "<table style='font:11px monospace;width:100%;border-collapse:collapse'>"
            "<tr style='color:#888'><td>type</td><td style='text-align:right'>mean Hz</td>"
            f"<td style='text-align:right'>n</td></tr>{body}</table>"
        )


def _bar_colour(frac: float) -> str:
    # viridis-ish ramp: dark purple → teal → yellow
    stops = np.array([[68, 1, 84], [33, 145, 140], [253, 231, 37]], dtype=float)
    f = float(np.clip(frac, 0, 1)) * 2
    i = min(int(f), 1)
    c = stops[i] + (stops[i + 1] - stops[i]) * (f - i)
    return "rgb({},{},{})".format(*[int(v) for v in c])
