"""Create a minimal demo subgraph without requiring full download."""

import numpy as np
from scipy import sparse
from pathlib import Path


def create_demo_subgraph(output_path: str = "data/demo_subgraph.npz", n_neurons: int = 500):
    """
    Create a small synthetic subgraph for demo purposes.
    
    This allows the demo to run without downloading 1.1GB of data.
    The synthetic graph has realistic properties but is not actual connectome data.
    """
    print(f"Creating demo subgraph with {n_neurons} neurons...")
    
    # Synthetic body IDs
    body_ids = np.arange(1000000, 1000000 + n_neurons)
    
    # Create sparse connectivity with realistic properties
    # Visual system typically has ~1-5% connectivity
    connectivity = 0.03
    n_connections = int(n_neurons * n_neurons * connectivity)
    
    # Generate random connections
    pre_indices = np.random.randint(0, n_neurons, n_connections)
    post_indices = np.random.randint(0, n_neurons, n_connections)
    
    # Weights follow log-normal distribution (realistic for synapses)
    weights = np.random.lognormal(mean=2.0, sigma=1.0, size=n_connections)
    weights = np.clip(weights, 1, 100)
    
    # Create sparse matrix
    adjacency = sparse.coo_matrix(
        (weights, (pre_indices, post_indices)),
        shape=(n_neurons, n_neurons)
    )
    adjacency = adjacency.tocsr()
    
    # Remove duplicates by summing
    adjacency.sum_duplicates()
    
    print(f"Created graph: {n_neurons} neurons, {adjacency.nnz} connections")
    print(f"Sparsity: {100 * adjacency.nnz / (n_neurons ** 2):.3f}%")
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez_compressed(
        output_path,
        body_ids=body_ids,
        adj_data=adjacency.data,
        adj_indices=adjacency.indices,
        adj_indptr=adjacency.indptr,
        adj_shape=adjacency.shape,
        n_neurons=n_neurons,
        n_connections=adjacency.nnz,
    )
    
    print(f"✓ Saved demo subgraph: {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    create_demo_subgraph()
