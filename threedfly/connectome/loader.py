"""Load and extract visual-flight subgraph from MaleCNS connectome."""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import sparse
from typing import Optional, Dict, Tuple
import pickle


# Visual system neuron types relevant for vision and flight
VISUAL_TYPES = {
    # Photoreceptors and early visual processing
    "photoreceptor",
    "R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8",  # Photoreceptor subtypes
    "lamina",
    "L1", "L2", "L3", "L4", "L5",  # Lamina neurons
    "medulla",
    "Mi", "Tm", "T4", "T5",  # Medulla neurons (motion detection)
    "lobula",
    "LC", "LPLC",  # Lobula complex
    "lobula_plate",
    "LPi",  # Lobula plate intrinsic
    # Visual projection neurons
    "VPN",
    "LC4", "LC6", "LC9", "LC10", "LC11", "LC15", "LC16", "LC17", "LC18", "LC20", "LC21", "LC22", "LC24", "LC25", "LC26",
    "LPLC1", "LPLC2", "LPLC4",
    # Descending neurons (motor control)
    "DN",
    "DNa", "DNb", "DNp",  # Descending neuron subtypes
    # Optic glomeruli and higher visual areas
    "OG",
    "optic_glomeruli",
    # Central complex (navigation)
    "FB", "EB", "PB", "NO",  # Fan-shaped body, ellipsoid body, protocerebral bridge, noduli
}


class ConnectomeLoader:
    """Load and process MaleCNS connectome data."""
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize loader.
        
        Args:
            data_dir: Directory containing downloaded connectome data
        """
        if data_dir is None:
            data_dir = Path(__file__).parent.parent.parent / "data" / "connectome-raw"
        self.data_dir = Path(data_dir)
        
        self.annotations: Optional[pd.DataFrame] = None
        self.weights: Optional[pd.DataFrame] = None
        self.subgraph: Optional[Dict] = None
    
    def load_annotations(self) -> pd.DataFrame:
        """Load body annotations."""
        path = self.data_dir / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
        if not path.exists():
            raise FileNotFoundError(
                f"Annotations file not found: {path}\n"
                "Run: threedfly download"
            )
        
        print(f"Loading annotations from {path}...")
        self.annotations = pd.read_feather(path)
        print(f"✓ Loaded {len(self.annotations)} neurons")
        return self.annotations
    
    def load_weights(self, min_weight: Optional[int] = None) -> pd.DataFrame:
        """
        Load connectome weights (full 1.1GB file).
        
        Args:
            min_weight: If specified, filter connections to only those >= this weight
                       (saves memory by filtering before loading into pandas)
        """
        path = self.data_dir / "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
        if not path.exists():
            raise FileNotFoundError(
                f"Weights file not found: {path}\n"
                "Run: threedfly download"
            )
        
        print(f"Loading weights from {path}...")
        
        if min_weight is not None and min_weight > 1:
            # Use pyarrow to filter before loading into pandas (memory efficient)
            print(f"  Filtering for weight >= {min_weight} (memory efficient)...")
            import pyarrow.feather as feather
            import pyarrow.compute as pc
            
            table = feather.read_table(path)
            print(f"  Loaded {len(table):,} total connections")
            
            # Filter using pyarrow (more efficient than pandas)
            mask = pc.greater_equal(table['weight'], min_weight)
            filtered_table = table.filter(mask)
            print(f"  Filtered to {len(filtered_table):,} connections")
            
            self.weights = filtered_table.to_pandas()
        else:
            self.weights = pd.read_feather(path)
            print(f"✓ Loaded {len(self.weights):,} connections")
        
        return self.weights
    
    def extract_subgraph(
        self,
        visual_types: Optional[set] = None,
        max_neurons: Optional[int] = None
    ) -> Dict:
        """
        Extract visual-flight subgraph.
        
        Args:
            visual_types: Set of neuron types to include
            max_neurons: Maximum number of neurons (for demo/testing)
        
        Returns:
            Dictionary with subgraph data
        """
        if self.annotations is None:
            self.load_annotations()
        if self.weights is None:
            self.load_weights()
        
        if visual_types is None:
            visual_types = VISUAL_TYPES
        
        print("\nExtracting visual-flight subgraph...")
        
        # Filter neurons by type (check multiple type columns)
        type_columns = [col for col in self.annotations.columns if "type" in col.lower()]
        
        selected_mask = np.zeros(len(self.annotations), dtype=bool)
        for col in type_columns:
            if col in self.annotations.columns:
                for vtype in visual_types:
                    mask = self.annotations[col].astype(str).str.contains(
                        vtype, case=False, na=False
                    )
                    selected_mask |= mask
        
        # Fallback: if no type matches, select by neuropil region
        if selected_mask.sum() == 0:
            print("No type matches found, using neuropil-based selection...")
            neuropil_col = None
            for col in self.annotations.columns:
                if "neuropil" in col.lower() or "region" in col.lower():
                    neuropil_col = col
                    break
            
            if neuropil_col:
                visual_regions = ["ME", "LO", "LOP", "LA", "OG"]  # Medulla, Lobula, etc.
                for region in visual_regions:
                    mask = self.annotations[neuropil_col].astype(str).str.contains(
                        region, case=False, na=False
                    )
                    selected_mask |= mask
        
        # If still no matches, take first N neurons as demo
        if selected_mask.sum() == 0:
            print("Warning: No visual neurons found by type/region.")
            print("Creating demo subgraph with first 1000 neurons...")
            selected_mask[:1000] = True
        
        selected_neurons = self.annotations[selected_mask].copy()
        
        if max_neurons is not None and len(selected_neurons) > max_neurons:
            print(f"Limiting to {max_neurons} neurons for demo...")
            selected_neurons = selected_neurons.iloc[:max_neurons]
        
        body_ids = set(selected_neurons["bodyId"].values)
        print(f"Selected {len(body_ids)} neurons")
        
        # Filter connections
        print("Filtering connections...")
        conn_mask = (
            self.weights["body_pre"].isin(body_ids) &
            self.weights["body_post"].isin(body_ids)
        )
        selected_connections = self.weights[conn_mask].copy()
        print(f"Selected {len(selected_connections)} connections")
        
        # Build sparse adjacency matrix
        print("Building adjacency matrix...")
        body_id_to_idx = {bid: idx for idx, bid in enumerate(sorted(body_ids))}
        n_neurons = len(body_id_to_idx)
        
        pre_idx = selected_connections["body_pre"].map(body_id_to_idx).values
        post_idx = selected_connections["body_post"].map(body_id_to_idx).values
        weights = selected_connections["weight"].values
        
        adjacency = sparse.csr_matrix(
            (weights, (pre_idx, post_idx)),
            shape=(n_neurons, n_neurons)
        )
        
        # Store subgraph
        self.subgraph = {
            "body_ids": np.array(sorted(body_ids)),
            "body_id_to_idx": body_id_to_idx,
            "adjacency": adjacency,
            "annotations": selected_neurons,
            "n_neurons": n_neurons,
            "n_connections": adjacency.nnz,
        }
        
        print(f"✓ Subgraph: {n_neurons} neurons, {adjacency.nnz} connections")
        print(f"  Sparsity: {100 * adjacency.nnz / (n_neurons ** 2):.3f}%")
        
        return self.subgraph
    
    def save_subgraph(self, output_path: Path) -> None:
        """Save subgraph to compressed npz file."""
        if self.subgraph is None:
            raise ValueError("No subgraph extracted. Call extract_subgraph() or build_full_connectome() first.")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving subgraph to {output_path}...")
        
        # Save sparse matrix and metadata
        np.savez_compressed(
            output_path,
            body_ids=self.subgraph["body_ids"],
            adj_data=self.subgraph["adjacency"].data,
            adj_indices=self.subgraph["adjacency"].indices,
            adj_indptr=self.subgraph["adjacency"].indptr,
            adj_shape=self.subgraph["adjacency"].shape,
            n_neurons=self.subgraph["n_neurons"],
            n_connections=self.subgraph["n_connections"],
            is_full_brain=self.subgraph.get("is_full_brain", False),
            min_synapses=self.subgraph.get("min_synapses", 1),
        )
        
        print(f"✓ Saved ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")
    
    def build_full_connectome(self, min_synapses: int = 1) -> Dict:
        """
        Build full-brain connectome from all neurons in the dataset.
        
        Args:
            min_synapses: Minimum connection weight to include (default: 1)
                         Higher values (3-5) significantly reduce memory usage
        
        Returns:
            Dictionary with full connectome data
        """
        if self.annotations is None:
            self.load_annotations()
        
        # Get set of valid annotated body IDs first
        print("Identifying valid annotated neurons...")
        valid_body_ids = set(self.annotations["bodyId"].values)
        print(f"  Found {len(valid_body_ids):,} annotated neurons")
        
        # Load weights with filtering (memory efficient)
        if self.weights is None:
            self.load_weights(min_weight=min_synapses)
        elif min_synapses > 1:
            # Re-filter if already loaded
            print(f"Filtering connections (min_synapses={min_synapses})...")
            self.weights = self.weights[self.weights["weight"] >= min_synapses].copy()
            print(f"  Kept {len(self.weights):,} connections")
        
        print("\nBuilding full-brain connectome...")
        
        # Filter connections to only those between annotated neurons
        print("Filtering connections to annotated neurons...")
        conn_mask = (
            self.weights["body_pre"].isin(valid_body_ids) &
            self.weights["body_post"].isin(valid_body_ids)
        )
        filtered_weights = self.weights[conn_mask].copy()
        print(f"  Kept {len(filtered_weights):,} connections between annotated neurons")
        
        # Get unique body IDs from filtered connections
        print("Finding neurons with connections...")
        all_body_ids_pre = set(filtered_weights["body_pre"].unique())
        all_body_ids_post = set(filtered_weights["body_post"].unique())
        all_body_ids = all_body_ids_pre | all_body_ids_post
        
        print(f"  Pre-synaptic: {len(all_body_ids_pre):,}")
        print(f"  Post-synaptic: {len(all_body_ids_post):,}")
        print(f"  Total: {len(all_body_ids):,}")
        
        # Build body_id to index mapping (sorted for consistency)
        print("Building index mapping...")
        body_id_to_idx = {bid: idx for idx, bid in enumerate(sorted(all_body_ids))}
        n_neurons = len(body_id_to_idx)
        
        print(f"Building sparse adjacency matrix for {n_neurons:,} neurons...")
        
        # Map body IDs to indices (use int32 to save memory)
        print("  Mapping indices...")
        pre_idx = filtered_weights["body_pre"].map(body_id_to_idx).values.astype(np.int32)
        post_idx = filtered_weights["body_post"].map(body_id_to_idx).values.astype(np.int32)
        weights = filtered_weights["weight"].values.astype(np.float32)
        
        print("  Creating sparse matrix...")
        # Build sparse adjacency matrix
        adjacency = sparse.csr_matrix(
            (weights, (pre_idx, post_idx)),
            shape=(n_neurons, n_neurons),
            dtype=np.float32
        )
        
        # Free memory
        del pre_idx, post_idx, weights, filtered_weights
        
        # Filter body_ids to only those with connections
        connected_body_ids = np.array(sorted(all_body_ids), dtype=np.int64)
        
        # Get annotations for connected neurons
        print("Extracting annotations for connected neurons...")
        annotations_mask = self.annotations["bodyId"].isin(all_body_ids)
        connected_annotations = self.annotations[annotations_mask].copy()
        
        print(f"  Matched {len(connected_annotations):,} neurons with annotations")
        
        # Store full connectome
        self.subgraph = {
            "body_ids": connected_body_ids,
            "body_id_to_idx": body_id_to_idx,
            "adjacency": adjacency,
            "annotations": connected_annotations,
            "n_neurons": n_neurons,
            "n_connections": adjacency.nnz,
            "is_full_brain": True,  # Flag to indicate this is the full brain
            "min_synapses": min_synapses,  # Record filtering threshold
        }
        
        mem_mb = (adjacency.data.nbytes + adjacency.indices.nbytes + adjacency.indptr.nbytes) / 1024**2
        sparsity = 100 * adjacency.nnz / (n_neurons ** 2)
        
        print(f"\n✓ Full connectome built successfully!")
        print(f"  Neurons: {n_neurons:,}")
        print(f"  Connections: {adjacency.nnz:,}")
        print(f"  Min synapses threshold: {min_synapses}")
        print(f"  Sparsity: {sparsity:.4f}%")
        print(f"  Memory (sparse matrix): ~{mem_mb:.1f} MB")
        
        return self.subgraph
    
    @staticmethod
    def load_subgraph(path: Path) -> Dict:
        """Load subgraph from npz file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Subgraph not found: {path}")
        
        print(f"Loading subgraph from {path}...")
        data = np.load(path)
        
        adjacency = sparse.csr_matrix(
            (data["adj_data"], data["adj_indices"], data["adj_indptr"]),
            shape=tuple(data["adj_shape"])
        )
        
        body_ids = data["body_ids"]
        body_id_to_idx = {bid: idx for idx, bid in enumerate(body_ids)}
        
        is_full_brain = bool(data.get("is_full_brain", False))
        min_synapses = int(data.get("min_synapses", 1))
        
        subgraph = {
            "body_ids": body_ids,
            "body_id_to_idx": body_id_to_idx,
            "adjacency": adjacency,
            "n_neurons": int(data["n_neurons"]),
            "n_connections": int(data["n_connections"]),
            "is_full_brain": is_full_brain,
            "min_synapses": min_synapses,
        }
        
        print(f"✓ Loaded: {subgraph['n_neurons']:,} neurons, {subgraph['n_connections']:,} connections")
        if is_full_brain:
            print(f"  (Full-brain connectome, min_synapses={min_synapses})")
        
        return subgraph
