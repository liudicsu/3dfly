#!/usr/bin/env python3
"""Replay a protocol-v1 recording through BrainStreamServer (for UI verify / demos).

    python scripts/stream_replay.py output/ui_verify/pr2/live_frames.bin --port 8767 --hz 10

Serves the inspector at ``http://localhost:PORT/`` and the three.js brain panel
at ``http://localhost:PORT/brain``. Loops the recording until Ctrl+C.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv=None) -> int:
    from threedfly.stream.client import read_recording
    from threedfly.stream.server import BrainStreamServer

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("recording", type=Path, help="Length-prefixed HELLO+FRAME dump")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8767)
    p.add_argument("--hz", type=float, default=10.0, help="Replay rate (loops)")
    p.add_argument("--once", action="store_true", help="Play once then idle")
    args = p.parse_args(argv)

    hello, frames = read_recording(args.recording)
    if not frames:
        print("ERROR: recording has no frames", file=sys.stderr)
        return 2
    srv = BrainStreamServer(host=args.host, port=args.port, max_hz=max(args.hz, 1.0)).start()
    srv.set_hello(hello)
    print(f"replaying {len(frames)} frames @ {args.hz} Hz → {srv.url}")
    print(f"  inspector: http://localhost:{srv.port}/")
    print(f"  brain panel: http://localhost:{srv.port}/brain")
    dt = 1.0 / args.hz if args.hz > 0 else 0.0
    try:
        while True:
            for fr in frames:
                # publish even without clients so the latest frame is ready
                srv._latest = bytes(fr)  # noqa: SLF001 — warm cache
                if srv.n_clients > 0 or True:
                    srv.publish(fr)
                time.sleep(dt)
            if args.once:
                print("done (idle); Ctrl+C to stop")
                while True:
                    time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        srv.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
