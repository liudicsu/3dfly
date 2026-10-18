#!/usr/bin/env python3
"""Export coreTplus neuron positions / cell-type ids for the three.js brain panel.

On the verification box (full MaleCNS annotations + core bundle):

    python scripts/export_brain_assets.py
    python scripts/export_brain_assets.py --output-dir assets/brain

Writes ``assets/brain/coreTplus_v1.npz`` + ``meta.json`` (and optional region
shell GLBs). The ``body_ids_sha1`` in the asset MUST match the protocol-v1
HELLO hash (see ``docs/stream_protocol.md``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv=None) -> int:
    from threedfly.stream.builder import FrameBuilder
    from threedfly.viz.brain_assets import (
        ASSET_NAME,
        META_NAME,
        assert_matches_hello,
        build_from_connectome,
        default_asset_dir,
        load_asset,
        maybe_decimate_shells,
        save_asset,
    )

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default="", help="Default: assets/brain")
    p.add_argument(
        "--core-bundle",
        default="",
        help="core_bundle.npz (default: data/core_bundle.npz)",
    )
    p.add_argument(
        "--annotations",
        default="",
        help="full_connectome.annotations.pkl",
    )
    p.add_argument(
        "--shells",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Copy/decimate CB + OL region shell GLBs (default: on)",
    )
    p.add_argument(
        "--verify-only",
        action="store_true",
        help="Load existing asset and check hash vs FrameBuilder HELLO",
    )
    args = p.parse_args(argv)

    out_dir = Path(args.output_dir) if args.output_dir else default_asset_dir(ROOT)
    core_path = Path(args.core_bundle) if args.core_bundle else ROOT / "data" / "core_bundle.npz"
    ann_path = (
        Path(args.annotations)
        if args.annotations
        else ROOT / "data" / "full_connectome.annotations.pkl"
    )

    if args.verify_only:
        asset = load_asset(out_dir / ASSET_NAME)
        import numpy as np

        # Rebuild HELLO hash from the same body_ids the asset was built for.
        # Prefer live core bundle when present.
        if not core_path.is_file():
            print(f"ERROR: need {core_path} to verify against FrameBuilder", file=sys.stderr)
            return 2
        b = np.load(core_path, allow_pickle=True)
        fb = FrameBuilder(types=b["typ"], sides=b["side"], body_ids=b["body_ids"], include_eye_layout=False)
        assert_matches_hello(asset, fb.meta())
        meta = json.loads((out_dir / META_NAME).read_text())
        print(
            f"OK  n={asset.n_neurons}  sha1={asset.body_ids_sha1}  "
            f"with_soma={int(asset.has_soma.sum())}  "
            f"npz={ (out_dir/ASSET_NAME).stat().st_size } B  "
            f"types={len(asset.type_names)}  keys={list(asset.key_idx)}"
        )
        print(f"meta keys: {sorted(meta.keys())}")
        return 0

    if not core_path.is_file():
        print(f"ERROR: missing {core_path}", file=sys.stderr)
        return 2
    if not ann_path.is_file():
        print(f"ERROR: missing {ann_path}", file=sys.stderr)
        return 2

    import numpy as np

    b = np.load(core_path, allow_pickle=True)
    print(f"Building brain asset from {core_path} ({len(b['body_ids'])} neurons)…", flush=True)
    asset = build_from_connectome(
        body_ids=b["body_ids"],
        types=b["typ"],
        sides=b["side"],
        annotations_path=ann_path,
    )

    shells = None
    if args.shells:
        cache = ROOT / "output" / "research_ui" / "cache"
        shells = maybe_decimate_shells(cache, out_dir)
        if shells:
            print(f"  shells: { {k: str(v.name) for k,v in shells.items()} }", flush=True)
        else:
            print("  shells: none (cache/prototype missing)", flush=True)

    npz = save_asset(asset, out_dir, shells=shells)
    # Verify against FrameBuilder HELLO
    fb = FrameBuilder(types=b["typ"], sides=b["side"], body_ids=b["body_ids"], include_eye_layout=False)
    assert_matches_hello(asset, fb.meta())
    # Round-trip load
    loaded = load_asset(npz)
    assert_matches_hello(loaded, fb.meta())

    size = npz.stat().st_size
    print(
        f"wrote {npz} ({size/1024:.1f} KiB)  sha1={asset.body_ids_sha1}\n"
        f"  n={asset.n_neurons}  with_soma={int(asset.has_soma.sum())} "
        f"({100*asset.has_soma.mean():.1f}%)  types={len(asset.type_names)}  "
        f"key_groups={len(asset.key_idx)}"
    )
    print(f"  meta → {out_dir / META_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
