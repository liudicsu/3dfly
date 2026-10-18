#!/usr/bin/env python
"""Tiny brain-stream client: connect, decode protocol-v1 frames, report.

    python scripts/stream_client.py --url ws://localhost:8765 --seconds 20 \
        --json output/ui_verify/pr2/client_report.json

Prints the HELLO summary, the decoded values of the latest frame (pose, DN
readouts, cell-type means, activity stats, eye maps) and measured frame
rate / payload bandwidth. ``zlib`` size of the frames is reported as an
estimate of the permessage-deflate wire size.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from threedfly.stream.client import receive  # noqa: E402
from threedfly.stream.protocol import Section, decode_frame_values  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://localhost:8765")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--max-frames", type=int, default=100000)
    ap.add_argument("--json", default=None)
    ap.add_argument("--no-compression", action="store_true")
    ap.add_argument("--dump", default=None, help="write received HELLO+frames (u32-length-prefixed)")
    a = ap.parse_args()

    t0 = time.time()
    rep = receive(
        a.url,
        max_frames=a.max_frames,
        duration=a.seconds,
        compression=None if a.no_compression else "deflate",
        keep_raw=bool(a.dump),
    )
    if a.dump:
        from threedfly.stream.client import write_recording

        write_recording(a.dump, rep.hello_raw, rep.raw)
    meta = rep.meta
    print(f"HELLO protocol={meta['protocol']} v{meta['version']} n_neurons={meta['n_neurons']} "
          f"dn={len(meta['dn_names'])} celltypes={len(meta['celltype_names'])} "
          f"activity={meta['activity']['scheme']} floor={meta['activity']['floor']} cap={meta['activity']['cap']}")
    if not rep.frames:
        print("no frames received")
        return 1
    sizes = np.array([f.nbytes for f in rep.frames])
    # zlib on re-encoded raw frames ~ permessage-deflate (per-message, no context takeover diff)
    from threedfly.stream.protocol import encode_message

    z = []
    for f in rep.frames[-min(len(rep.frames), 50):]:
        raw = encode_message(f.msg_type, f.sections, f.frame_index, f.sim_time, f.wall_time)
        z.append(len(zlib.compress(raw, 6)))
    z = np.array(z)
    v = decode_frame_values(rep.frames[-1], meta)
    act = v.get("activity")
    sim_span = rep.frames[-1].sim_time - rep.frames[0].sim_time
    summary = {
        "url": a.url,
        "frames": rep.n,
        "elapsed_wall_s": round(time.time() - t0, 2),
        "fps": round(rep.fps, 2),
        "payload_bytes_per_frame_mean": float(sizes.mean()),
        "payload_kib_per_s": round(rep.bytes_per_s / 1024, 1),
        "zlib_bytes_per_frame_mean": float(z.mean()),
        "zlib_kib_per_s_est": round(rep.fps * z.mean() / 1024, 1),
        "decode_ms_per_frame": round(rep.decode_ms_total / rep.n, 3),
        "sim_time_span_s": round(sim_span, 3),
        "realtime_factor": round(sim_span / max(rep.t_last - rep.t_first, 1e-9), 3),
        "latest": {
            "frame": v["frame"],
            "t": round(v["t"], 3),
            "pose": {k: round(x, 4) for k, x in v.get("pose", {}).items()},
            "dn": {k: float(f"{x:.4g}") for k, x in v.get("dn", {}).items()},
            "celltype": {k: float(f"{x:.4g}") for k, x in v.get("celltype", {}).items()},
            "control": {k: round(x, 4) for k, x in v.get("control", {}).items()},
            "stats": v.get("stats", {}),
            "activity_nonzero": int(np.count_nonzero(act)) if act is not None else 0,
            "activity_max_hz": float(act.max()) if act is not None else 0.0,
            "activity_mean_hz": float(act.mean()) if act is not None else 0.0,
            "eye_L_mean": float(v["eye_L"].mean()) if "eye_L" in v else None,
            "eye_R_mean": float(v["eye_R"].mean()) if "eye_R" in v else None,
        },
    }
    print(json.dumps(summary, indent=1))
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
