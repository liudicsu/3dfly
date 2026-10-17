#!/usr/bin/env python3
"""Optionally fetch the full flybody menagerie tree (meshes + fruitfly.xml).

The default 3dfly model vendors a subset of flybody meshes (Apache-2.0) under
`assets/fly_meshes/`. This script sparse-clones the DeepMind menagerie copy into
`assets/flybody/full/` (gitignored, ~140 MB).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/google-deepmind/mujoco_menagerie.git"
DEST = Path(__file__).resolve().parent.parent / "assets" / "flybody" / "full"


def main() -> int:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if (DEST / "fruitfly.xml").exists():
        print(f"Already present: {DEST}")
        return 0
    tmp = DEST.parent / "_clone"
    if tmp.exists():
        subprocess.check_call(["rm", "-rf", str(tmp)])
    print(f"Sparse-cloning flybody from {REPO} ...")
    subprocess.check_call(
        ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", REPO, str(tmp)]
    )
    subprocess.check_call(["git", "-C", str(tmp), "sparse-checkout", "set", "flybody"])
    subprocess.check_call(["rm", "-rf", str(DEST)])
    subprocess.check_call(["mv", str(tmp / "flybody"), str(DEST)])
    subprocess.check_call(["rm", "-rf", str(tmp)])
    print(f"Fetched flybody to {DEST}")
    print("License: Apache-2.0 (see THIRD_PARTY.md). Default 3dfly MJCF still uses assets/fly_meshes/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
