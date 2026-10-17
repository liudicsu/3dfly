"""Tests for connectome loading and simulation."""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from pathlib import Path
import tempfile

from threedfly.connectome import ConnectomeLoader, BrainSimulator
from threedfly.connectome.simulator import RateSimulator


def test_brain_simulator_init():
    """Test BrainSimulator initialization."""
    n = 100
    adj = sparse.random(n, n, density=0.1, format='csr')
    
    sim = BrainSimulator(adj, dt=0.001, tau=0.01)
    
    assert sim.n_neurons == n
    assert sim.v.shape == (n,)
    assert sim.spikes.shape == (n,)


def test_brain_simulator_step():
    """Test BrainSimulator step."""
    n = 50
    adj = sparse.random(n, n, density=0.1, format='csr')
    
    sim = BrainSimulator(adj, dt=0.001)
    
    # Set input
    input_vec = np.random.rand(10)
    sim.set_input(input_vec)
    
    # Step
    spikes = sim.step()
    
    assert spikes.shape == (n,)
    assert spikes.dtype == bool


def test_brain_simulator_firing_rates():
    """Test firing rate computation."""
    n = 50
    adj = sparse.random(n, n, density=0.1, format='csr')
    
    sim = BrainSimulator(adj, dt=0.001)
    
    # Run for a few steps
    for _ in range(100):
        input_vec = np.random.rand(n) * 0.5
        sim.set_input(input_vec)
        sim.step()
    
    rates = sim.get_firing_rates(window=50)
    
    assert rates.shape == (n,)
    assert np.all(rates >= 0)


def test_rate_simulator():
    """Test rate-based simulator."""
    n = 50
    adj = sparse.random(n, n, density=0.1, format='csr')
    
    sim = RateSimulator(adj, dt=0.01, activation='relu')
    
    # Set input and step
    input_vec = np.random.rand(10)
    sim.set_input(input_vec)
    
    rates = sim.step()
    
    assert rates.shape == (n,)
    assert np.all(rates >= 0)


def test_subgraph_save_load():
    """Test subgraph saving and loading."""
    # Create synthetic subgraph
    n = 100
    body_ids = np.arange(n)
    adj = sparse.random(n, n, density=0.05, format='csr')
    
    subgraph = {
        "body_ids": body_ids,
        "body_id_to_idx": {bid: idx for idx, bid in enumerate(body_ids)},
        "adjacency": adj,
        "n_neurons": n,
        "n_connections": adj.nnz,
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_subgraph.npz"
        
        # Simulate saving
        np.savez_compressed(
            path,
            body_ids=subgraph["body_ids"],
            adj_data=subgraph["adjacency"].data,
            adj_indices=subgraph["adjacency"].indices,
            adj_indptr=subgraph["adjacency"].indptr,
            adj_shape=subgraph["adjacency"].shape,
            n_neurons=subgraph["n_neurons"],
            n_connections=subgraph["n_connections"],
        )
        
        # Load
        loaded = ConnectomeLoader.load_subgraph(path)
        
        assert loaded["n_neurons"] == n
        assert loaded["n_connections"] == adj.nnz
        assert loaded["adjacency"].shape == (n, n)
        assert np.array_equal(loaded["body_ids"], body_ids)


def test_population_activity():
    """Test population activity statistics."""
    n = 50
    adj = sparse.random(n, n, density=0.1, format='csr')
    
    sim = BrainSimulator(adj)
    
    # Set strong input
    sim.set_input(np.ones(n) * 2.0)
    sim.step()
    
    stats = sim.get_population_activity()
    
    assert "mean_voltage" in stats
    assert "spike_count" in stats
    assert "active_fraction" in stats
    assert stats["active_fraction"] >= 0
    assert stats["active_fraction"] <= 1


def test_full_connectome_structure():
    """Test full connectome data structure (using synthetic data)."""
    # Create synthetic "full connectome" data
    n = 1000
    body_ids = np.arange(10000, 10000 + n)  # Use high IDs like real data
    adj = sparse.random(n, n, density=0.01, format='csr', dtype=np.float32)
    
    # Create annotations dataframe
    annotations = pd.DataFrame({
        "bodyId": body_ids,
        "type": [f"neuron_{i % 10}" for i in range(n)],
    })
    
    subgraph = {
        "body_ids": body_ids,
        "body_id_to_idx": {bid: idx for idx, bid in enumerate(body_ids)},
        "adjacency": adj,
        "annotations": annotations,
        "n_neurons": n,
        "n_connections": adj.nnz,
        "is_full_brain": True,
        "min_synapses": 3,
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_full_connectome.npz"
        
        # Save
        np.savez_compressed(
            path,
            body_ids=subgraph["body_ids"],
            adj_data=subgraph["adjacency"].data,
            adj_indices=subgraph["adjacency"].indices,
            adj_indptr=subgraph["adjacency"].indptr,
            adj_shape=subgraph["adjacency"].shape,
            n_neurons=subgraph["n_neurons"],
            n_connections=subgraph["n_connections"],
            is_full_brain=subgraph["is_full_brain"],
            min_synapses=subgraph["min_synapses"],
        )
        
        # Load
        loaded = ConnectomeLoader.load_subgraph(path)
        
        assert loaded["n_neurons"] == n
        assert loaded["n_connections"] == adj.nnz
        assert loaded["is_full_brain"] is True
        assert loaded["min_synapses"] == 3
        assert loaded["adjacency"].shape == (n, n)
        assert loaded["adjacency"].dtype == np.float32
        assert loaded["adjacency"].indices.dtype == np.int32


def test_large_sparse_simulator():
    """Test that simulator can handle large sparse matrices."""
    # Test with a larger graph similar to full connectome scale
    n = 10000
    density = 0.001  # Very sparse, like real connectome
    adj = sparse.random(n, n, density=density, format='csr', dtype=np.float32)
    
    # Test rate simulator (more efficient for large graphs)
    sim = RateSimulator(adj, dt=0.01)
    
    # Set input to small subset
    input_vec = np.random.rand(100)
    sim.set_input(input_vec)
    
    # Should be able to step without issues
    rates = sim.step()
    
    assert rates.shape == (n,)
    assert np.all(rates >= 0)
    assert np.all(rates <= sim.max_rate)


def test_rate_simulator_ei_signs():
    """Test E/I sign application in rate simulator."""
    # Create simple 3-neuron network: 0→1, 0→2
    # 0 is excitatory (+1), 1 is inhibitory (-1), 2 is unclear (+1)
    adjacency = sparse.csr_matrix([
        [0, 1, 1],  # 0 → 1, 2
        [0, 0, 0],
        [0, 0, 0],
    ], dtype=float)
    
    transmitter_signs = {0: 1.0, 1: -1.0, 2: 1.0}
    
    # Without E/I signs
    sim_no_ei = RateSimulator(adjacency, apply_ei_signs=False, transmitter_signs=transmitter_signs)
    # Should have positive weights
    assert sim_no_ei.adj[0, 1] > 0
    assert sim_no_ei.adj[0, 2] > 0
    
    # With E/I signs
    sim_with_ei = RateSimulator(adjacency, apply_ei_signs=True, transmitter_signs=transmitter_signs)
    # 0→1 should be positive (excitatory presynaptic)
    # 0→2 should be positive (excitatory presynaptic)
    assert sim_with_ei.adj[0, 1] > 0
    assert sim_with_ei.adj[0, 2] > 0


def test_inhibitory_presynaptic_row_is_negative():
    """Presynaptic GABA/glutamate/histamine rows must flip sign."""
    adjacency = sparse.csr_matrix([
        [0, 0, 0],
        [0, 0, 2.0],  # neuron 1 → 2
        [0, 0, 0],
    ], dtype=float)
    transmitter_signs = {0: 1.0, 1: -1.0, 2: 1.0}
    sim = RateSimulator(adjacency, apply_ei_signs=True, transmitter_signs=transmitter_signs)
    assert sim.adj[1, 2] < 0


def test_apply_ei_signs_is_vectorized_and_fast():
    """sparse.diags @ adj must finish quickly on a large sparse graph."""
    import time
    from threedfly.connectome.simulator import apply_ei_signs

    n = 20000
    adj = sparse.random(n, n, density=0.0005, format="csr", dtype=np.float32)
    signs = np.ones(n, dtype=np.float32)
    signs[::3] = -1.0
    t0 = time.perf_counter()
    signed = apply_ei_signs(adj, signs)
    elapsed = time.perf_counter() - t0
    assert elapsed < 1.0, f"apply_ei_signs took {elapsed:.2f}s, expected < 1s"
    assert signed.shape == adj.shape
    assert signed.nnz == adj.nnz


def test_lif_simulator_ei_signs():
    """Test E/I sign application in LIF simulator."""
    adjacency = sparse.csr_matrix([
        [0, 1, 1],
        [0, 0, 0],
        [0, 0, 0],
    ], dtype=float)
    
    transmitter_signs = {0: 1.0, 1: -1.0, 2: 1.0}
    
    # Without E/I signs
    sim_no_ei = BrainSimulator(adjacency, apply_ei_signs=False, transmitter_signs=transmitter_signs)
    assert sim_no_ei.adj[0, 1] > 0
    
    # With E/I signs
    sim_with_ei = BrainSimulator(adjacency, apply_ei_signs=True, transmitter_signs=transmitter_signs)
    assert sim_with_ei.adj[0, 1] > 0


def test_amplify_incoming_scales_positive_synapses_to_targets():
    """Incoming gain to DNp01 must strengthen existing positive 1-hop weights."""
    from threedfly.connectome.simulator import amplify_incoming_to_targets

    adj = sparse.csr_matrix(np.array([
        [0.0, 0.0, 0.2],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ], dtype=np.float32))
    out = amplify_incoming_to_targets(adj, targets=np.array([2]), gain=5.0)
    assert float(out[0, 2]) == pytest.approx(1.0)


def test_diagnose_looming_reports_dnp01_rise():
    """Diagnostic helper injects looming into LPLC2/LC4 and reports DNp01 vs rest."""
    from threedfly.connectome.simulator import diagnose_looming_to_dnp01
    from tests.test_control import _synthetic_bio_annotations
    from threedfly.control.controller import BiologicalFlightController

    n, annotations, body_ids, body_id_to_idx = _synthetic_bio_annotations()
    controller = BiologicalFlightController(
        n_neurons=n,
        annotations=annotations,
        body_ids=body_ids,
        body_id_to_idx=body_id_to_idx,
        warmup_steps=3,
    )
    adj = sparse.csr_matrix((n, n), dtype=np.float32).tolil()
    loom = np.concatenate([
        controller.input_mapping["looming_left"],
        controller.input_mapping["looming_right"],
    ])
    dnp = controller.output_mapping["DNp01"]
    for pre in loom:
        for post in dnp:
            adj[int(pre), int(post)] = 2.0
    report = diagnose_looming_to_dnp01(
        adjacency=adj.tocsr(),
        looming_indices=loom,
        dnp01_indices=dnp,
        n_rest=6,
        n_drive=12,
        looming_current=1.0,
    )
    assert report["n_one_hop"] > 0
    assert report["dnp01_driven"] > report["dnp01_rest"] + 0.05


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
