"""Tiny synchronous Python client for the brain stream (tests / CLI)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

from websockets.sync.client import connect

from threedfly.stream.protocol import Frame, MsgType, decode_frame_values, decode_message


@dataclass
class ClientReport:
    meta: dict
    frames: List[Frame] = field(default_factory=list)
    bytes_payload: int = 0
    t_first: float = 0.0
    t_last: float = 0.0
    decode_ms_total: float = 0.0
    hello_raw: bytes = b""
    raw: List[bytes] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.frames)

    @property
    def fps(self) -> float:
        dt = self.t_last - self.t_first
        return (self.n - 1) / dt if self.n > 1 and dt > 0 else 0.0

    @property
    def bytes_per_s(self) -> float:
        dt = self.t_last - self.t_first
        return self.bytes_payload * (self.n - 1) / max(self.n, 1) / dt if dt > 0 else 0.0

    def latest_values(self) -> Optional[dict]:
        return decode_frame_values(self.frames[-1], self.meta) if self.frames else None


def receive(
    url: str,
    max_frames: int = 30,
    duration: float = 10.0,
    timeout: float = 10.0,
    compression: Optional[str] = "deflate",
    keep_raw: bool = False,
) -> ClientReport:
    """Connect, read HELLO, then frames until ``max_frames`` or ``duration``."""
    with connect(url, max_size=None, compression=compression, open_timeout=timeout) as ws:
        hello_raw = ws.recv(timeout=timeout)
        hello = decode_message(hello_raw)
        if hello.msg_type != MsgType.HELLO:
            raise RuntimeError("first message is not HELLO")
        rep = ClientReport(meta=hello.meta, hello_raw=bytes(hello_raw) if keep_raw else b"")
        t_end = time.perf_counter() + duration
        while rep.n < max_frames and time.perf_counter() < t_end:
            try:
                raw = ws.recv(timeout=max(0.05, t_end - time.perf_counter()))
            except TimeoutError:
                break
            now = time.perf_counter()
            t0 = time.perf_counter()
            fr = decode_message(raw, copy=True)
            rep.decode_ms_total += 1000 * (time.perf_counter() - t0)
            if fr.msg_type != MsgType.FRAME:
                continue
            if not rep.frames:
                rep.t_first = now
            rep.t_last = now
            rep.frames.append(fr)
            if keep_raw:
                rep.raw.append(bytes(raw))
            rep.bytes_payload += fr.nbytes
        return rep


def write_recording(path, hello_raw: bytes, frames_raw: List[bytes]) -> None:
    """Length-prefixed (u32 LE) concatenation: HELLO then FRAMEs."""
    import struct

    with open(path, "wb") as fh:
        for b in [hello_raw] + list(frames_raw):
            fh.write(struct.pack("<I", len(b)))
            fh.write(b)


def read_recording(path):
    import struct

    data = open(path, "rb").read()
    out, off = [], 0
    while off < len(data):
        (n,) = struct.unpack_from("<I", data, off)
        off += 4
        out.append(data[off : off + n])
        off += n
    return out[0], out[1:]
