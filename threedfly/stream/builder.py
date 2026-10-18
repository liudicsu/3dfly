"""Build protocol-v1 frames from closed-loop sim state.

``FrameBuilder.accumulate`` is called every sim step (cheap: an in-place
``np.maximum``). ``build`` is called only when a display frame is due and
encodes the *window maximum* of the rates since the last frame, so a brief
DNp01 burst between two 30 Hz display frames still shows up.
"""

from __future__ import annotations

import re
from dataclasses import fields
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from threedfly.stream.protocol import (
    ACTIVITY_CAP,
    ACTIVITY_FLOOR,
    POSE_FIELDS,
    PROTOCOL_NAME,
    STATS_FIELDS,
    Section,
    encode_frame,
    encode_hello,
    quantize_linear,
    quantize_log1p,
    section_names,
)

# Cell-type groups streamed in the CELLTYPE section, eye → wing order.
CELLTYPE_GROUPS: Tuple[Tuple[str, str], ...] = (
    ("Tm", r"^Tm"),
    ("T2/T3", r"^T[23]"),
    ("T4a", r"^T4a$"),
    ("T4b", r"^T4b$"),
    ("T4c", r"^T4c$"),
    ("T4d", r"^T4d$"),
    ("T5a", r"^T5a$"),
    ("T5b", r"^T5b$"),
    ("T5c", r"^T5c$"),
    ("T5d", r"^T5d$"),
    ("LPi", r"^LPi"),
    ("LPLC1", r"^LPLC1$"),
    ("LPLC2", r"^LPLC2$"),
    ("LPLC4", r"^LPLC4$"),
    ("LC4", r"^LC4$"),
    ("HSE", r"^HSE$"),
    ("DNp01", r"^DNp01$"),
    ("DNp03", r"^DNp03$"),
    ("DNa02", r"^DNa02$"),
    ("DNp15", r"^(DNp15|DNHS1)$"),
    ("DNg02", r"^DNg02"),
    ("TTMn", r"^TTMn$"),
    ("PSI", r"^PSI$"),
    ("SApp", r"^SApp$"),
)

DN_FIELDS: Tuple[str, ...] = (
    "dnp01_L", "dnp01_R", "dnp03_L", "dnp03_R", "dna02_L", "dna02_R",
    "dnp15_L", "dnp15_R", "hse_L", "hse_R", "dng02_L", "dng02_R", "dng02_amp",
    "lplc2_L", "lplc2_R", "lc4_L", "lc4_R", "lplc2_p90_L", "lplc2_p90_R",
    "lc4_p90_L", "lc4_p90_R",
)
CONTROL_FIELDS: Tuple[str, ...] = ("fwd", "up", "roll", "pitch", "yaw")


def _eye_layout(n_hex: int) -> Optional[dict]:
    try:
        from threedfly.brain.eyes import hex_body_azel_deg

        out = {}
        for side in "LR":
            az, el = hex_body_azel_deg(side)
            if len(az) != n_hex:
                return None
            out[side] = {
                "az_deg": [round(float(v), 3) for v in az],
                "el_deg": [round(float(v), 3) for v in el],
            }
        return out
    except Exception:
        return None


class FrameBuilder:
    def __init__(
        self,
        types: Optional[Sequence[str]] = None,
        sides: Optional[Sequence[str]] = None,
        body_ids: Optional[Sequence[int]] = None,
        n_neurons: Optional[int] = None,
        n_hex: int = 721,
        groups: Sequence[Tuple[str, str]] = CELLTYPE_GROUPS,
        floor: float = ACTIVITY_FLOOR,
        cap: float = ACTIVITY_CAP,
        include_eye_layout: bool = True,
    ):
        if types is None and n_neurons is None:
            raise ValueError("need types or n_neurons")
        self.types = None if types is None else np.asarray([str(t) for t in types], dtype=object)
        self.sides = None if sides is None else np.asarray([str(s) for s in sides], dtype=object)
        self.body_ids = None if body_ids is None else np.asarray(body_ids, dtype=np.int64)
        self.n = int(len(self.types) if self.types is not None else n_neurons)
        self.n_hex = int(n_hex)
        self.floor = float(floor)
        self.cap = float(cap)
        self.include_eye_layout = include_eye_layout
        self.group_names: List[str] = []
        self.group_index: List[np.ndarray] = []
        if self.types is not None:
            for name, pat in groups:
                rx = re.compile(pat)
                idx = np.flatnonzero([bool(rx.match(t)) for t in self.types]).astype(np.int64)
                if idx.size:
                    self.group_names.append(name)
                    self.group_index.append(idx)
        self._rmax = np.zeros(self.n, dtype=np.float32)
        self._window = 0
        self.frame_index = 0
        self.last_bytes = 0

    # ---------------------------------------------------------------- meta
    def meta(self) -> dict:
        m: dict = {
            "protocol": PROTOCOL_NAME,
            "sections": section_names(),
            "n_neurons": self.n,
            "neuron_order": "coreTplus RateCore index (ClosedLoopBrain.core.r)",
            "pose_fields": list(POSE_FIELDS),
            "dn_names": list(DN_FIELDS),
            "celltype_names": list(self.group_names),
            "celltype_sizes": [int(i.size) for i in self.group_index],
            "control_names": list(CONTROL_FIELDS),
            "stats_fields": list(STATS_FIELDS),
            "activity": {
                "dtype": "uint8",
                "scheme": "log1p",
                "floor": self.floor,
                "cap": self.cap,
                "decode": "r = floor * expm1(q / 255 * log1p(cap / floor))",
                "reduction": "max over sim steps since previous frame",
            },
            "eye": {"dtype": "uint8", "n_hex": self.n_hex, "scheme": "linear", "min": 0.0, "max": 1.0},
            "units": {"rates": "Hz", "pose": "m, quaternion wxyz, m/s, rad", "time": "s"},
        }
        if self.body_ids is not None:
            import hashlib

            m["body_ids_sha1"] = hashlib.sha1(self.body_ids.astype("<i8").tobytes()).hexdigest()
        if self.include_eye_layout:
            lay = _eye_layout(self.n_hex)
            if lay is not None:
                m["eye_layout"] = lay
        return m

    def hello(self) -> bytes:
        return encode_hello(self.meta())

    # --------------------------------------------------------------- frames
    def accumulate(self, rates: np.ndarray) -> None:
        r = np.asarray(rates, dtype=np.float32).ravel()
        if r.size != self.n:
            raise ValueError(f"rates size {r.size} != n_neurons {self.n}")
        np.maximum(self._rmax, r, out=self._rmax)
        self._window += 1

    def _readout_vec(self, readout) -> np.ndarray:
        if readout is None:
            return np.zeros(len(DN_FIELDS), dtype=np.float32)
        get = readout.get if isinstance(readout, dict) else (lambda k, d=0.0: getattr(readout, k, d))
        return np.array([float(get(k, 0.0)) for k in DN_FIELDS], dtype=np.float32)

    def build(
        self,
        sim_time: float,
        readout=None,
        position=None,
        orientation=None,
        velocity=None,
        wing_angles=None,
        hex_L=None,
        hex_R=None,
        control=None,
        rates: Optional[np.ndarray] = None,
    ) -> bytes:
        """Encode one frame. ``rates`` (if given) is folded into the window first."""
        if rates is not None:
            self.accumulate(rates)
        r = self._rmax
        pose = np.zeros(len(POSE_FIELDS), dtype=np.float32)
        pose[3] = 1.0
        if position is not None:
            pose[0:3] = np.asarray(position, dtype=np.float32).ravel()[:3]
        if orientation is not None:
            pose[3:7] = np.asarray(orientation, dtype=np.float32).ravel()[:4]
        if velocity is not None:
            pose[7:10] = np.asarray(velocity, dtype=np.float32).ravel()[:3]
        if wing_angles is not None:
            w = np.asarray(wing_angles, dtype=np.float32).ravel()[:2]
            pose[10 : 10 + w.size] = w
        n_act = int(np.count_nonzero(r > 0.01))
        stats = np.array(
            [float(r.mean()), float(r.max()), n_act / max(self.n, 1), float(n_act), float(self._window)],
            dtype=np.float32,
        )
        sections: Dict[int, np.ndarray] = {
            Section.POSE: pose,
            Section.DN: self._readout_vec(readout),
            Section.CELLTYPE: np.array([float(r[i].mean()) for i in self.group_index], dtype=np.float32),
            Section.ACTIVITY: quantize_log1p(r, self.floor, self.cap),
            Section.STATS: stats,
        }
        if hex_L is not None:
            sections[Section.EYE_L] = quantize_linear(np.asarray(hex_L).ravel()[: self.n_hex])
        if hex_R is not None:
            sections[Section.EYE_R] = quantize_linear(np.asarray(hex_R).ravel()[: self.n_hex])
        if control is not None:
            c = np.zeros(len(CONTROL_FIELDS), dtype=np.float32)
            cc = np.asarray(control, dtype=np.float32).ravel()[: len(CONTROL_FIELDS)]
            c[: cc.size] = cc
            sections[Section.CONTROL] = c
        buf = encode_frame(self.frame_index, sim_time, sections)
        self.frame_index += 1
        self._rmax.fill(0.0)
        self._window = 0
        self.last_bytes = len(buf)
        return buf
