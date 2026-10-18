#!/usr/bin/env python
"""Measure protocol-v1 throughput independent of sim speed.

Replays a recording made with ``scripts/stream_client.py --dump`` (real
closed-loop frames) through ``BrainStreamServer`` at several target rates
and measures what a Python client receives. Also times
``FrameBuilder.build`` for the full coreTplus size.

    python scripts/stream_bench.py --recording output/ui_verify/pr2/live_frames.bin
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from threedfly.stream.builder import FrameBuilder  # noqa: E402
from threedfly.stream.client import read_recording, receive  # noqa: E402
from threedfly.stream.protocol import decode_frame_values, decode_message  # noqa: E402
from threedfly.stream.server import BrainStreamServer  # noqa: E402


def bench_rate(hello, frames, hz, seconds, compression):
    srv = BrainStreamServer(host="127.0.0.1", port=0, max_hz=hz, compression=compression).start()
    srv.set_hello(hello)
    stop = threading.Event()

    def produce():
        i = 0
        while not stop.is_set():
            if srv.wants_frame():
                srv.publish(frames[i % len(frames)])
                i += 1
            time.sleep(0.0005)

    th = threading.Thread(target=produce, daemon=True)
    th.start()
    try:
        rep = receive(srv.url, max_frames=10**9, duration=seconds, compression=compression)
    finally:
        stop.set()
        th.join()
        srv.stop()
    return {
        "target_hz": hz,
        "compression": compression or "none",
        "frames": rep.n,
        "fps": round(rep.fps, 2),
        "payload_kib_per_s": round(rep.bytes_per_s / 1024, 1),
        "decode_ms": round(rep.decode_ms_total / max(rep.n, 1), 3),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recording", required=True)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--rates", default="10,30,60,120")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    hello, frames = read_recording(a.recording)
    meta = decode_message(hello).meta
    import zlib

    sizes = [len(f) for f in frames]
    zs = [len(zlib.compress(f, 6)) for f in frames]
    # encode cost on real rates (dequantised from the recording)
    vals = decode_frame_values(decode_message(frames[-1]), meta)
    fb = FrameBuilder(n_neurons=meta["n_neurons"], include_eye_layout=False)
    r = vals["activity"]
    t0 = time.perf_counter()
    for _ in range(50):
        fb.accumulate(r)
    acc_ms = 1000 * (time.perf_counter() - t0) / 50
    t0 = time.perf_counter()
    for _ in range(50):
        fb.build(0.0, rates=r, hex_L=vals.get("eye_L"), hex_R=vals.get("eye_R"), control=np.zeros(5))
    build_ms = 1000 * (time.perf_counter() - t0) / 50
    out = {
        "recording_frames": len(frames),
        "n_neurons": meta["n_neurons"],
        "frame_bytes_mean": float(np.mean(sizes)),
        "zlib6_bytes_mean": float(np.mean(zs)),
        "zlib_ratio": round(float(np.mean(zs) / np.mean(sizes)), 3),
        "accumulate_ms": round(acc_ms, 3),
        "build_encode_ms": round(build_ms, 3),
        "runs": [],
    }
    for comp in ("deflate", None):
        for hz in [float(x) for x in a.rates.split(",")]:
            res = bench_rate(hello, frames, hz, a.seconds, comp)
            print(res, flush=True)
            out["runs"].append(res)
    print(json.dumps(out, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
