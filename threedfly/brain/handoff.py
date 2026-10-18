"""flyvis → MaleCNS handoff (research_r4/handoff_map4.py + fv4_run.py)."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import scipy.sparse as sp

from threedfly.brain.connectome_matrix import ConnectomeData
from threedfly.brain.constants import (
    DT,
    FLYVIS_CHECKPOINT,
    FLYVIS_TO_CONNECTOME,
    HALTERE_RATE,
    HANDOFF_FLYVIS_TYPES,
    HEX_EXTENT,
    N_HEX,
    RMAX,
    configure_threads,
    flyvis_available,
)
from threedfly.brain.stimuli import calibration_conditions, get_hex_coords

logger = logging.getLogger(__name__)


def voltages_to_rates(
    V: np.ndarray,
    V_gray: np.ndarray,
    p99: float,
    rmax: float = RMAX,
) -> np.ndarray:
    """Rate = RMAX · clip(relu(V − V_gray) / p99.9, 0, 1)."""
    denom = float(p99) + 1e-6
    return (rmax * np.clip(np.maximum(V - V_gray, 0.0) / denom, 0.0, 1.0)).astype(np.float32)


def impute_hex(
    hex_uv: np.ndarray,
    has: np.ndarray,
    target: np.ndarray,
    W_abs_post_pre: sp.spmatrix,
    n_iter: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Iterative neighbour imputation for neurons lacking assignedOlHex."""
    cur = hex_uv.copy()
    known = has.copy()
    W = abs(W_abs_post_pre).tocsr()
    for _ in range(n_iter):
        need = np.flatnonzero(target & ~known)
        if need.size == 0:
            break
        known_idx = np.flatnonzero(known)
        if known_idx.size == 0:
            break
        Wn = W[need][:, known_idx]
        s = np.asarray(Wn.sum(axis=1)).ravel()
        ok = s > 0
        if not np.any(ok):
            break
        est = (Wn @ np.nan_to_num(cur[known_idx])) / np.maximum(s, 1.0)[:, None]
        cur[need[ok]] = est[ok]
        known[need[ok]] = True
    return cur, known


@dataclass
class HandoffMap:
    """Per-eye maps from flyvis types to connectome neurons + scaled (u, v)."""

    index: Dict[str, np.ndarray] = field(default_factory=dict)  # "T4a|L" → neuron idx
    uv: Dict[str, np.ndarray] = field(default_factory=dict)
    counts: Dict[str, int] = field(default_factory=dict)
    centre: Dict[str, np.ndarray] = field(default_factory=dict)
    scale: Dict[str, float] = field(default_factory=dict)

    @property
    def n_mapped(self) -> int:
        return int(sum(self.counts.get(k, 0) for k in self.counts if "|" in k))

    def sides_present(self) -> set[str]:
        return {k.split("|")[1] for k in self.counts if "|" in k}


def build_handoff_map(connectome: ConnectomeData, n_iter: int = 4) -> HandoffMap:
    """assignedOlHex + neighbour imputation; centre = median, scale = 15 / p99."""
    typ = connectome.typ
    side = connectome.side
    h = np.asarray(connectome.hex, dtype=float)
    has = ~np.isnan(h[:, 0])
    mapped_types = [t for ts in FLYVIS_TO_CONNECTOME.values() for t in ts]
    target = np.isin(typ, mapped_types)
    cur, known = impute_hex(h, has, target, abs(connectome.W), n_iter=n_iter)

    out = HandoffMap()
    for s in "LR":
        m = has & (side == s)
        if not np.any(m):
            continue
        c = np.median(h[m], axis=0)
        p99 = float(np.percentile(np.abs(h[m] - c), 99, axis=0).max())
        sc = 15.0 / (p99 if p99 > 0 else 1.0)
        out.centre[s] = c.astype(np.float32)
        out.scale[s] = float(sc)
        for fvt, mts in FLYVIS_TO_CONNECTOME.items():
            idx = np.flatnonzero(np.isin(typ, list(mts)) & (side == s) & known)
            key = f"{fvt}|{s}"
            out.index[key] = idx.astype(np.int64)
            out.uv[key] = ((cur[idx] - c) * sc).astype(np.float32)
            out.counts[key] = int(len(idx))
    out.counts["total"] = out.n_mapped
    return out


def nearest_flyvis_columns(neuron_uv: np.ndarray, flyvis_uv: np.ndarray) -> np.ndarray:
    """Map each connectome neuron to the nearest flyvis column in (u, v)."""
    if neuron_uv.size == 0:
        return np.zeros(0, dtype=np.int64)
    d = ((neuron_uv[:, None, :] - flyvis_uv[None, :, :].astype(float)) ** 2).sum(-1)
    return d.argmin(axis=1).astype(np.int64)


@dataclass
class FlyvisCalibration:
    V_gray: Dict[str, np.ndarray]
    p99: Dict[str, float]
    uv: Dict[str, np.ndarray]
    rmax: float = RMAX

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {f"V_gray|{t}": v for t, v in self.V_gray.items()}
        payload.update({f"uv|{t}": v for t, v in self.uv.items()})
        payload.update({f"p99|{t}": np.array([p], dtype=np.float32) for t, p in self.p99.items()})
        np.savez_compressed(path, **payload)
        path.with_suffix(".json").write_text(
            json.dumps({"types": list(self.V_gray), "rmax": self.rmax, "p99": self.p99}, indent=2)
        )

    @classmethod
    def load(cls, path: Path) -> "FlyvisCalibration":
        z = np.load(path)
        types = sorted({k.split("|", 1)[1] for k in z.files if k.startswith("V_gray|")})
        return cls(
            V_gray={t: z[f"V_gray|{t}"] for t in types},
            p99={t: float(z[f"p99|{t}"][0]) for t in types},
            uv={t: z[f"uv|{t}"] for t in types},
        )


class FlyvisEye:
    """Streaming wrapper around a pretrained flyvis eye (optional dependency)."""

    def __init__(
        self,
        checkpoint: str = FLYVIS_CHECKPOINT,
        dt: float = DT,
        torch_threads: int = 1,
        network=None,
    ):
        if network is None and not flyvis_available():
            raise ImportError("flyvis is required for FlyvisEye. pip install 'threedfly[brain]'")
        configure_threads(torch_threads=torch_threads)
        self._torch_threads = int(torch_threads)
        self.dt = float(dt)
        self.checkpoint = checkpoint
        self._net = network
        self._state = None
        self.types: Optional[np.ndarray] = None
        self.type_index: Dict[str, np.ndarray] = {}
        self.uv: Dict[str, np.ndarray] = {}
        if network is None:
            self._init_network()
        else:
            self._bind_network(network)

    def _init_network(self) -> None:
        import torch
        from flyvis import NetworkView

        n_th = max(1, int(getattr(self, "_torch_threads", 1) or 1))
        torch.set_num_threads(n_th)
        if hasattr(torch, "set_num_interop_threads"):
            try:
                torch.set_num_interop_threads(min(n_th, 2))
            except RuntimeError:
                pass
        net = NetworkView(self.checkpoint).init_network()
        self._bind_network(net)

    def _bind_network(self, net) -> None:
        self._net = net
        raw = net.connectome.nodes.type[:]
        self.types = np.array([t.decode() if isinstance(t, bytes) else str(t) for t in raw])
        u = np.asarray(net.connectome.nodes.u[:], dtype=np.float32)
        v = np.asarray(net.connectome.nodes.v[:], dtype=np.float32)
        for t in HANDOFF_FLYVIS_TYPES:
            idx = np.flatnonzero(self.types == t)
            self.type_index[t] = idx
            self.uv[t] = np.stack([u[idx], v[idx]], axis=1)

    def reset(self) -> None:
        if self._net is None:
            self._state = None
            return
        self._state = self._net.steady_state(1.0, self.dt, 1)

    def simulate_movie(self, movie: np.ndarray) -> tuple[Dict[str, np.ndarray], float]:
        """Batch-simulate a (T, 721) hex movie. Returns ({type: (T, n_cols)}, ms/frame)."""
        import time
        import torch

        movie = np.asarray(movie, dtype=np.float32)
        if movie.ndim != 2 or movie.shape[1] != N_HEX:
            raise ValueError(f"movie must be (T, {N_HEX}), got {movie.shape}")
        if self._state is None:
            self.reset()
        x = torch.as_tensor(movie[None, :, None, :])
        t0 = time.perf_counter()
        with torch.no_grad():
            act = self._net.simulate(x, self.dt, initial_state=self._state, as_states=False)
        ms = 1000.0 * (time.perf_counter() - t0) / max(movie.shape[0], 1)
        act = np.asarray(act.detach().cpu().numpy())
        # (batch, T, n) or (T, n)
        if act.ndim == 3:
            act = act[0]
        out = {t: act[:, idx] for t, idx in self.type_index.items() if idx.size}
        return out, float(ms)

    def step_pair_batched(self, hex_L, hex_R, state_L, state_R):
        """One flyvis forward for both eyes (batch=2). Returns vL, vR, sL, sR."""
        import torch

        fL = np.asarray(hex_L, dtype=np.float32).reshape(-1)
        fR = np.asarray(hex_R, dtype=np.float32).reshape(-1)
        if fL.size != N_HEX or fR.size != N_HEX:
            raise ValueError(f"hex frames must have {N_HEX} cells")
        if state_L is None or state_R is None:
            raise RuntimeError("batched step needs both eye states")
        # Stack *nodes/edges only* on dim 0, then rebuild sources/targets as
        # RefTensors. Concatenating already-dereferenced sources.activity
        # along the last axis makes weight (n_edges,) vs activity (2 n_edges).
        stacked = stack_flyvis_states(state_L, state_R, self._net)
        x = torch.as_tensor(np.stack([fL, fR], axis=0)[:, None, None, :], dtype=torch.float32)
        with torch.inference_mode():
            states = self._net.simulate(
                x, self.dt, initial_state=stacked, as_states=True
            )
        if isinstance(states, (list, tuple)):
            last = states[-1]
        else:
            last = states
        sL, sR = split_flyvis_state(last, self._net)
        act = _activity_from_state(last)
        if act is None:
            raise RuntimeError("flyvis batched state has no nodes.activity")
        act = np.asarray(act)
        if act.ndim == 1:
            raise RuntimeError("batched activity is not (2, n)")
        if act.shape[0] != 2:
            raise RuntimeError(
                f"batched activity shape {act.shape} is not (2, n_nodes); "
                "refusing to flatten (that was the weight vs 2×activity bug)"
            )
        def _volts(row):
            return {t: row[idx] for t, idx in self.type_index.items() if idx.size}
        return _volts(act[0].reshape(-1)), _volts(act[1].reshape(-1)), sL, sR

    def step(self, hex_frame: np.ndarray) -> Dict[str, np.ndarray]:
        """Advance one frame, keeping network state. ``hex_frame`` is (721,) in [0, 1]."""
        import torch

        frame = np.asarray(hex_frame, dtype=np.float32).reshape(-1)
        if frame.size != N_HEX:
            raise ValueError(f"hex frame must have {N_HEX} cells, got {frame.size}")
        if self._state is None:
            self.reset()
        x = torch.as_tensor(frame.reshape(1, 1, 1, -1), dtype=torch.float32)
        with torch.inference_mode():
            states = self._net.simulate(
                x, self.dt, initial_state=self._state, as_states=True
            )
        if isinstance(states, (list, tuple)):
            self._state = states[-1]
            activity = _activity_from_state(states[-1])
        else:
            self._state = states
            activity = _activity_from_state(states)
        if activity is None:
            raise RuntimeError("flyvis state has no nodes.activity; cannot stream")
        activity = np.asarray(activity).reshape(-1)
        return {t: activity[idx] for t, idx in self.type_index.items() if idx.size}

    def calibrate(
        self,
        conditions: Optional[Dict[str, np.ndarray]] = None,
        pre: float = 1.0,
    ) -> FlyvisCalibration:
        """Gray + loom + gratings + expansion → per-type V_gray and p99.9."""
        if conditions is None:
            conditions = calibration_conditions(pre=pre)
        nb = int(round(pre / self.dt))
        traces: Dict[str, Dict[str, np.ndarray]] = {t: {} for t in HANDOFF_FLYVIS_TYPES}
        for name, movie in conditions.items():
            self.reset()
            acts, _ = self.simulate_movie(movie)
            for t, arr in acts.items():
                traces[t][name] = arr
        V_gray, p99 = {}, {}
        for t in HANDOFF_FLYVIS_TYPES:
            if "gray" not in traces[t]:
                continue
            base = traces[t]["gray"][nb - 30 : nb].mean(axis=0)
            pos = [np.maximum(traces[t][k] - base, 0.0).ravel() for k in traces[t]]
            p = float(np.percentile(np.concatenate(pos), 99.9)) if pos else 1.0
            V_gray[t] = base.astype(np.float32)
            p99[t] = p
        return FlyvisCalibration(V_gray=V_gray, p99=p99, uv=self.uv, rmax=RMAX)


def _raw_group(state, group: str):
    """``nodes`` / ``edges`` mapping without AutoDeref-dereferencing RefTensors."""
    if state is None:
        return None
    if isinstance(state, dict):
        return dict.get(state, group)
    return getattr(state, group, None)


def _raw_tensor(group, key: str):
    import torch
    from flyvis.utils.tensor_utils import RefTensor

    if group is None:
        return None
    if isinstance(group, dict):
        val = dict.get(group, key)
    else:
        val = getattr(group, key, None)
    if isinstance(val, RefTensor):
        val = val.values
    if isinstance(val, torch.Tensor) and val.ndim == 1:
        val = val.unsqueeze(0)
    return val


def stack_node_tensors(state_L, state_R):
    """Batch-cat ``nodes`` / ``edges`` on dim 0. Does not touch sources/targets."""
    import torch
    from flyvis.utils.tensor_utils import AutoDeref

    out = AutoDeref(nodes=AutoDeref(), edges=AutoDeref())
    for group in ("nodes", "edges"):
        a = _raw_group(state_L, group)
        b = _raw_group(state_R, group)
        if a is None or b is None:
            continue
        keys = list(a.keys()) if hasattr(a, "keys") else []
        for k in keys:
            va, vb = _raw_tensor(a, k), _raw_tensor(b, k)
            if not isinstance(va, torch.Tensor) or not isinstance(vb, torch.Tensor):
                continue
            if va.shape[0] < 1 or vb.shape[0] < 1:
                raise ValueError(f"{group}.{k} empty batch: {tuple(va.shape)}")
            out[group][k] = torch.cat([va[:1], vb[:1]], dim=0)
    return out


def split_node_tensors(state):
    """Split a batch-2 ``nodes`` / ``edges`` state into two batch-1 states."""
    import torch
    from flyvis.utils.tensor_utils import AutoDeref

    left = AutoDeref(nodes=AutoDeref(), edges=AutoDeref())
    right = AutoDeref(nodes=AutoDeref(), edges=AutoDeref())
    for group in ("nodes", "edges"):
        g = _raw_group(state, group)
        if g is None:
            continue
        keys = list(g.keys()) if hasattr(g, "keys") else []
        for k in keys:
            v = _raw_tensor(g, k)
            if not isinstance(v, torch.Tensor):
                continue
            if v.shape[0] != 2:
                raise ValueError(f"{group}.{k} expected batch 2, got {tuple(v.shape)}")
            left[group][k] = v[0:1].contiguous()
            right[group][k] = v[1:2].contiguous()
    return left, right


def stack_flyvis_states(state_L, state_R, net):
    """Batch=2 flyvis state with sources/targets rebuilt as RefTensors."""
    raw = stack_node_tensors(state_L, state_R)
    if hasattr(net, "_state_api"):
        return net._state_api(raw)
    return raw


def split_flyvis_state(state, net):
    left, right = split_node_tensors(state)
    if hasattr(net, "_state_api"):
        return net._state_api(left), net._state_api(right)
    return left, right


# Kept for tests that imported the old walk-everything stack.
def _stack_autoderef(a, b):
    return stack_node_tensors(a, b)


def _split_autoderef(state):
    return split_node_tensors(state)


def _activity_from_state(state) -> Optional[np.ndarray]:
    try:
        act = state.nodes.activity
        if hasattr(act, "detach"):
            act = act.detach().cpu().numpy()
        return np.asarray(act)
    except Exception:
        return None


@dataclass
class HandoffPlan:
    """Precomputed nearest-column map so each frame is O(n) indexing."""

    idx: np.ndarray
    types: np.ndarray
    cols: np.ndarray
    v_gray: np.ndarray
    p99: np.ndarray
    rmax: float = RMAX
    # Contiguous [start, end) per flyvis type — avoids np.unique + object masks.
    type_ranges: Dict[str, tuple] = field(default_factory=dict)


def compile_handoff_plan(
    hmap: HandoffMap,
    calib: FlyvisCalibration,
    side: str,
    remap: Optional[np.ndarray] = None,
) -> HandoffPlan:
    idx_all, typ_all, col_all, vg_all, p_all = [], [], [], [], []
    type_ranges: Dict[str, tuple] = {}
    off = 0
    for t in HANDOFF_FLYVIS_TYPES:
        key = f"{t}|{side}"
        if key not in hmap.index or t not in calib.uv or t not in calib.V_gray:
            continue
        cidx = hmap.index[key]
        uv = hmap.uv[key]
        if cidx.size == 0:
            continue
        col = nearest_flyvis_columns(uv, calib.uv[t])
        V_gray = calib.V_gray[t]
        vg = V_gray[col] if np.ndim(V_gray) else np.full(col.shape, float(V_gray))
        if remap is not None:
            mapped = remap[cidx]
            ok = mapped >= 0
            cidx, col, vg = mapped[ok], col[ok], vg[ok]
        n = int(len(cidx))
        if n == 0:
            continue
        type_ranges[t] = (off, off + n)
        off += n
        idx_all.append(cidx)
        typ_all.append(np.full(n, t, dtype=object))
        col_all.append(col)
        vg_all.append(np.asarray(vg, dtype=np.float32))
        p_all.append(np.full(n, float(calib.p99[t]), dtype=np.float32))
    if not idx_all:
        return HandoffPlan(
            idx=np.zeros(0, dtype=np.int64),
            types=np.zeros(0, dtype=object),
            cols=np.zeros(0, dtype=np.int64),
            v_gray=np.zeros(0, dtype=np.float32),
            p99=np.zeros(0, dtype=np.float32),
            rmax=calib.rmax,
            type_ranges={},
        )
    return HandoffPlan(
        idx=np.concatenate(idx_all).astype(np.int64),
        types=np.concatenate(typ_all),
        cols=np.concatenate(col_all).astype(np.int64),
        v_gray=np.concatenate(vg_all),
        p99=np.concatenate(p_all),
        rmax=float(calib.rmax),
        type_ranges=type_ranges,
    )


# T4/T5 drive LPLC2/LC4/DNp01. Calibration p99 comes from high-contrast
# hex movies; a compact room object sits well below that, so GF waits
# until ~80°. A modest type gain restores biological 40–60° sensitivity
# without a geometry oracle. Wide-field HS is subtracted in the controller.
# Identity: amplifying T4/T5 to chase 40–60° also amplifies room flow
# (LPLC2 0.43 in no-object yaw0) and false-escapes. Closed-loop LC4/LPLC2
# stay near 0 until ~60–70° of disk on this hex/FOV; report that window.
EXPANSION_HANDOFF_GAIN = {
    "T4a": 1.0,
    "T4b": 1.0,
    "T4c": 1.0,
    "T4d": 1.0,
    "T5a": 1.0,
    "T5b": 1.0,
    "T5c": 1.0,
    "T5d": 1.0,
}


def apply_handoff_plan(
    plan: HandoffPlan,
    voltages: Dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    if plan.idx.size == 0:
        return plan.idx, np.zeros(0, dtype=np.float32)
    V = np.zeros(plan.idx.size, dtype=np.float32)
    ranges = plan.type_ranges
    if ranges:
        items = ranges.items()
    else:
        items = ((t, None) for t in np.unique(plan.types))
    for t, sl in items:
        if t not in voltages:
            continue
        src = np.asarray(voltages[t], dtype=np.float32)
        if sl is None:
            m = plan.types == t
            V[m] = src[plan.cols[m]]
        else:
            a, b = sl
            V[a:b] = src[plan.cols[a:b]]
    rates = (plan.rmax * np.clip(np.maximum(V - plan.v_gray, 0.0) / (plan.p99 + 1e-6), 0.0, 1.0)).astype(
        np.float32
    )
    if ranges:
        for t, (a, b) in ranges.items():
            g = EXPANSION_HANDOFF_GAIN.get(t)
            if g is not None and g != 1.0:
                rates[a:b] *= float(g)
    return plan.idx, rates


def handoff_rates_for_frame(
    voltages: Dict[str, np.ndarray],
    calib: FlyvisCalibration,
    hmap: HandoffMap,
    side: str,
    remap: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (core_indices, rates) for one eye / frame.

    ``remap`` maps full-connectome indices → core indices (−1 if dropped).
    Prefer ``compile_handoff_plan`` + ``apply_handoff_plan`` in tight loops.
    """
    return apply_handoff_plan(compile_handoff_plan(hmap, calib, side, remap=remap), voltages)


def halt_name_mask(values) -> np.ndarray:
    """True where a type / subclass / extra annotation mentions a haltere."""
    arr = np.asarray(values, dtype=object).reshape(-1)
    return np.fromiter(("halt" in str(v).lower() for v in arr), dtype=bool, count=arr.size)


def haltere_indices(connectome: ConnectomeData, remap: Optional[np.ndarray] = None) -> np.ndarray:
    """Search type, subclass, superclass, and extras for ``halt`` (not exact 'haltere')."""
    mask = halt_name_mask(connectome.subclass)
    mask |= halt_name_mask(connectome.typ)
    extras = getattr(connectome, "extras", None) or {}
    for key in extras:
        extra = np.asarray(extras[key], dtype=object)
        if extra.shape == mask.shape:
            mask |= halt_name_mask(extra)
    idx = np.flatnonzero(mask)
    if remap is not None:
        mapped = remap[idx]
        return mapped[mapped >= 0]
    return idx


def default_hex_uv_for_type(n: int, extent: int = HEX_EXTENT) -> np.ndarray:
    """Synthetic flyvis (u, v) lattice (tests without flyvis)."""
    u, v = get_hex_coords(extent)
    uv = np.stack([u, v], axis=1).astype(np.float32)
    if n <= len(uv):
        return uv[:n]
    reps = int(np.ceil(n / len(uv)))
    return np.tile(uv, (reps, 1))[:n]
