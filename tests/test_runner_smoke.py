"""Smoke test: biological-mode run_simulation with synthetic annotations."""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse


def _write_synthetic_connectome(tmp_path: Path) -> Path:
    n = 48
    rng = np.random.RandomState(0)
    adj = sparse.random(n, n, density=0.08, format="csr", dtype=np.float32, random_state=rng)
    body_ids = np.arange(1, n + 1, dtype=np.int64)
    types = []
    sides = []
    catalog = (
        ["R1"] * 6
        + ["T4a"] * 4
        + ["T5a"] * 4
        + ["LPLC2"] * 2
        + ["LC4"] * 2
        + ["DNp01"] * 2
        + ["DNg02_a"] * 6
        + ["DNa01"] * 2
        + ["DNa02"] * 2
        + ["DNb01"] * 4
    )
    catalog += ["other"] * (n - len(catalog))
    for i in range(n):
        types.append(catalog[i])
        sides.append("L" if i % 2 == 0 else "R")
    annotations = pd.DataFrame({
        "bodyId": body_ids,
        "type": types,
        "celltype": types,
        "somaSide": sides,
        "rootSide": sides,
    })
    npz_path = tmp_path / "synthetic_bio.npz"
    np.savez_compressed(
        npz_path,
        body_ids=body_ids,
        adj_data=adj.data,
        adj_indices=adj.indices,
        adj_indptr=adj.indptr,
        adj_shape=np.array(adj.shape),
        n_neurons=n,
        n_connections=adj.nnz,
        is_full_brain=False,
        min_synapses=1,
    )
    with open(npz_path.with_suffix(".annotations.pkl"), "wb") as f:
        pickle.dump(annotations, f)
    return npz_path


def test_run_simulation_biological_mode_smoke(tmp_path):
    """run_simulation must complete a few biological-mode steps without crashing."""
    pytest.importorskip("open3d")
    from threedfly.runner import run_simulation

    subgraph = _write_synthetic_connectome(tmp_path)
    run_simulation(
        subgraph_path=str(subgraph),
        n_steps=3,
        viz_mode="none",
        save_output_dir=str(tmp_path / "out"),
        simulator_type="rate",
        seed=0,
        warmup_steps=2,
        enable_soft_recovery=True,
        use_core_tplus=False,
    )
