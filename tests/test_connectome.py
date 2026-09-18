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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
