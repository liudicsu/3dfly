"""3dfly brain-stream wire format, version 1.

All integers/floats little-endian. One websocket binary message = one
message. Layout::

    message header (32 B)
      0  4s  magic  b"3DFY"
      4  u16 version            (PROTOCOL_VERSION = 1)
      6  u8  msg_type           (0 HELLO, 1 FRAME)
      7  u8  n_sections
      8  u32 frame_index        (0 for HELLO)
     12  u32 total_length       (bytes, including this header)
     16  f64 sim_time  [s]
     24  f64 wall_time [unix s]
    then n_sections × section:
      0  u16 section_id         (Section enum; unknown ids MUST be skipped)
      2  u8  dtype              (DType enum)
      3  u8  flags              (reserved, 0)
      4  u32 payload_len        (bytes, unpadded)
      8  payload, zero-padded to a multiple of 8 bytes

Every payload therefore starts on an 8-byte boundary, so a JS client can
build ``Float32Array``/``Uint8Array`` views on the message buffer without
copying. HELLO carries one ``META_JSON`` section (UTF-8 JSON) describing
names, sizes and quantisation of every FRAME section; clients must read it
before decoding frames. New sections can be added without a version bump;
changing an existing section's meaning bumps ``PROTOCOL_VERSION``.
"""

from __future__ import annotations

import json
import struct
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, Optional, Tuple

import numpy as np

MAGIC = b"3DFY"
PROTOCOL_VERSION = 1
PROTOCOL_NAME = "3dfly-brain-stream"

_HEADER = struct.Struct("<4sHBBIIdd")  # 32 bytes
_SECTION = struct.Struct("<HBBI")  # 8 bytes
HEADER_SIZE = _HEADER.size
SECTION_HEADER_SIZE = _SECTION.size
ALIGN = 8


class MsgType(IntEnum):
    HELLO = 0
    FRAME = 1


class Section(IntEnum):
    META_JSON = 0  # u8: UTF-8 JSON (HELLO only)
    POSE = 1  # f32[12]: pos xyz, quat wxyz, vel xyz, wing L/R angle
    DN = 2  # f32[len(dn_names)]: DN readouts (Hz)
    CELLTYPE = 3  # f32[len(celltype_names)]: mean rate per cell-type group (Hz)
    ACTIVITY = 4  # u8[n_neurons]: log1p-quantised rate, coreTplus core order
    EYE_L = 5  # u8[n_hex]: left-eye hex luminance 0..1 → 0..255
    EYE_R = 6  # u8[n_hex]
    CONTROL = 7  # f32[len(control_names)]: flight command
    STATS = 8  # f32[len(stats_names)]: population stats


class DType(IntEnum):
    U8 = 1
    U16 = 2
    F32 = 3
    F64 = 4
    I32 = 5
    F16 = 6


_NP = {
    DType.U8: np.dtype("<u1"),
    DType.U16: np.dtype("<u2"),
    DType.F32: np.dtype("<f4"),
    DType.F64: np.dtype("<f8"),
    DType.I32: np.dtype("<i4"),
    DType.F16: np.dtype("<f2"),
}
_KIND = {(v.kind, v.itemsize): k for k, v in _NP.items()}

POSE_FIELDS = ("x", "y", "z", "qw", "qx", "qy", "qz", "vx", "vy", "vz", "wing_L", "wing_R")
STATS_FIELDS = ("mean_rate", "max_rate", "active_fraction", "n_active", "window_steps")

# ---------------------------------------------------------------- quantisers
ACTIVITY_FLOOR = 1e-3  # Hz
ACTIVITY_CAP = 200.0  # Hz; RateCore caps at 50 Hz but clamped handoff/haltere inputs can exceed it


def quantize_log1p(r: np.ndarray, floor: float = ACTIVITY_FLOOR, cap: float = ACTIVITY_CAP) -> np.ndarray:
    """uint8 code = round(255·log1p(r/floor)/log1p(cap/floor)). 0 ↔ silent."""
    r = np.clip(np.asarray(r, dtype=np.float32), 0.0, cap)
    k = 255.0 / np.log1p(cap / floor)
    return np.rint(np.log1p(r / floor) * k).astype(np.uint8)


def dequantize_log1p(q: np.ndarray, floor: float = ACTIVITY_FLOOR, cap: float = ACTIVITY_CAP) -> np.ndarray:
    k = np.log1p(cap / floor) / 255.0
    return (floor * np.expm1(np.asarray(q, dtype=np.float32) * k)).astype(np.float32)


def quantize_linear(x: np.ndarray, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    x = (np.asarray(x, dtype=np.float32) - lo) / max(hi - lo, 1e-12)
    return np.rint(np.clip(x, 0.0, 1.0) * 255.0).astype(np.uint8)


def dequantize_linear(q: np.ndarray, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    return (lo + (hi - lo) * np.asarray(q, dtype=np.float32) / 255.0).astype(np.float32)


# ------------------------------------------------------------------ messages
@dataclass
class Frame:
    msg_type: int
    version: int
    frame_index: int
    sim_time: float
    wall_time: float
    sections: Dict[int, np.ndarray] = field(default_factory=dict)
    nbytes: int = 0

    @property
    def meta(self) -> Optional[dict]:
        raw = self.sections.get(Section.META_JSON)
        if raw is None:
            return None
        return json.loads(bytes(raw).decode("utf-8"))


def _pad(n: int) -> int:
    return (-n) % ALIGN


def encode_message(
    msg_type: int,
    sections: Dict[int, np.ndarray],
    frame_index: int = 0,
    sim_time: float = 0.0,
    wall_time: Optional[float] = None,
) -> bytes:
    parts = []
    total = HEADER_SIZE
    for sid, arr in sections.items():
        a = np.ascontiguousarray(arr)
        code = _KIND.get((a.dtype.kind, a.dtype.itemsize))
        if code is None:
            raise TypeError(f"unsupported dtype {a.dtype} for section {sid}")
        a = a.astype(_NP[code], copy=False)
        payload = a.tobytes()
        parts.append(_SECTION.pack(int(sid), int(code), 0, len(payload)))
        parts.append(payload)
        pad = _pad(len(payload))
        if pad:
            parts.append(b"\0" * pad)
        total += SECTION_HEADER_SIZE + len(payload) + pad
    head = _HEADER.pack(
        MAGIC,
        PROTOCOL_VERSION,
        int(msg_type),
        len(sections),
        int(frame_index) & 0xFFFFFFFF,
        total,
        float(sim_time),
        float(time.time() if wall_time is None else wall_time),
    )
    return b"".join([head] + parts)


def encode_hello(meta: dict) -> bytes:
    body = dict(meta)
    body.setdefault("protocol", PROTOCOL_NAME)
    body["version"] = PROTOCOL_VERSION
    raw = np.frombuffer(json.dumps(body, separators=(",", ":")).encode("utf-8"), dtype=np.uint8)
    return encode_message(MsgType.HELLO, {Section.META_JSON: raw})


def encode_frame(
    frame_index: int,
    sim_time: float,
    sections: Dict[int, np.ndarray],
    wall_time: Optional[float] = None,
) -> bytes:
    return encode_message(MsgType.FRAME, sections, frame_index, sim_time, wall_time)


class ProtocolError(ValueError):
    pass


def decode_message(buf, copy: bool = False) -> Frame:
    mv = memoryview(buf)
    if len(mv) < HEADER_SIZE:
        raise ProtocolError("short message")
    magic, ver, mtype, nsec, idx, total, t_sim, t_wall = _HEADER.unpack_from(mv, 0)
    if magic != MAGIC:
        raise ProtocolError(f"bad magic {magic!r}")
    if ver != PROTOCOL_VERSION:
        raise ProtocolError(f"unsupported version {ver} (have {PROTOCOL_VERSION})")
    if total != len(mv):
        raise ProtocolError(f"length mismatch header={total} actual={len(mv)}")
    off = HEADER_SIZE
    sections: Dict[int, np.ndarray] = {}
    for _ in range(nsec):
        sid, code, _flags, plen = _SECTION.unpack_from(mv, off)
        off += SECTION_HEADER_SIZE
        if off + plen > total:
            raise ProtocolError("section overruns message")
        dt = _NP.get(code)
        if dt is not None:
            arr = np.frombuffer(mv[off : off + plen], dtype=dt)
            sections[int(sid)] = arr.copy() if copy else arr
        # unknown dtype: skip (forward compatible)
        off += plen + _pad(plen)
    return Frame(
        msg_type=int(mtype),
        version=int(ver),
        frame_index=int(idx),
        sim_time=float(t_sim),
        wall_time=float(t_wall),
        sections=sections,
        nbytes=int(total),
    )


def section_names() -> Dict[str, str]:
    return {str(int(s)): s.name for s in Section}


def decode_frame_values(frame: Frame, meta: dict) -> dict:
    """Human-readable dict of a FRAME using the HELLO ``meta``."""
    s = frame.sections
    out: dict = {"frame": frame.frame_index, "t": frame.sim_time}
    if Section.POSE in s:
        out["pose"] = dict(zip(meta.get("pose_fields", POSE_FIELDS), map(float, s[Section.POSE])))
    if Section.DN in s:
        out["dn"] = dict(zip(meta["dn_names"], map(float, s[Section.DN])))
    if Section.CELLTYPE in s:
        out["celltype"] = dict(zip(meta["celltype_names"], map(float, s[Section.CELLTYPE])))
    if Section.CONTROL in s:
        out["control"] = dict(zip(meta.get("control_names", []), map(float, s[Section.CONTROL])))
    if Section.STATS in s:
        out["stats"] = dict(zip(meta.get("stats_fields", STATS_FIELDS), map(float, s[Section.STATS])))
    if Section.ACTIVITY in s:
        q = meta["activity"]
        r = dequantize_log1p(s[Section.ACTIVITY], q["floor"], q["cap"])
        out["activity"] = r
    for key, sid in (("eye_L", Section.EYE_L), ("eye_R", Section.EYE_R)):
        if sid in s:
            e = meta["eye"]
            out[key] = dequantize_linear(s[sid], e["min"], e["max"])
    return out


def header_tuple(buf) -> Tuple:
    return _HEADER.unpack_from(memoryview(buf), 0)
