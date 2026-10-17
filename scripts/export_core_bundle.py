#!/usr/bin/env python3
"""Export a compact coreTplus bundle for closed-loop runs without the full connectome.

On the verification box (full MaleCNS + annotations):

    python scripts/export_core_bundle.py
    python scripts/export_core_bundle.py --output data/core_bundle.npz

Writes ``data/core_bundle.npz`` (target < 50 MB): signed coreTplus subgraph
(plus any ``halt*`` afferents so the closed loop can clamp them), types/side/hex,
remapped flyvis handoff map, DN/halt indices, and ``rho_full_reference``.

``ClosedLoopBrain.build`` loads this file when ``data/full_connectome.npz`` is
absent. Flyvis + ``data/cache/flyvis_cal.npz`` are still required for the eyes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv=None) -> int:
    from threedfly.brain.constants import connectome_available
    from threedfly.brain.core_bundle import core_bundle_path, export_closed_loop_bundle

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="", help="Destination npz (default data/core_bundle.npz)")
    p.add_argument("--data-dir", default="", help="Directory with full_connectome.npz")
    args = p.parse_args(argv)

    data_dir = args.data_dir or None
    if not connectome_available(data_dir):
        print(
            "ERROR: export needs data/full_connectome.npz (and the MaleCNS feathers).\n"
            "Run this on the verification box, then upload the npz.",
            file=sys.stderr,
        )
        return 2
    dest = Path(args.output) if args.output else core_bundle_path(data_dir)
    print(f"Exporting core bundle → {dest}", flush=True)
    path = export_closed_loop_bundle(data_dir, dest)
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"wrote {path} ({size_mb:.1f} MB)", flush=True)
    if size_mb > 50:
        print("WARNING: bundle exceeds 50 MB target", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
