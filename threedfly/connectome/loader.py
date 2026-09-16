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
    
    def load_weights(self) -> pd.DataFrame:
        """Load connectome weights (full 1.1GB file)."""
        path = self.data_dir / "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
        if not path.exists():
            raise FileNotFoundError(
                f"Weights file not found: {path}\n"
                "Run: threedfly download"
            )
        
        print(f"Loading weights from {path} (this may take a minute)...")
        self.weights = pd.read_feather(path)
        print(f"✓ Loaded {len(self.weights)} connections")
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
            self.weights["bodyId_pre"].isin(body_ids) &
            self.weights["bodyId_post"].isin(body_ids)
        )
        selected_connections = self.weights[conn_mask].copy()
        print(f"Selected {len(selected_connections)} connections")
        
        # Build sparse adjacency matrix
        print("Building adjacency matrix...")
        body_id_to_idx = {bid: idx for idx, bid in enumerate(sorted(body_ids))}
        n_neurons = len(body_id_to_idx)
        
        pre_idx = selected_connections["bodyId_pre"].map(body_id_to_idx).values
        post_idx = selected_connections["bodyId_post"].map(body_id_to_idx).values
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
            raise ValueError("No subgraph extracted. Call extract_subgraph() first.")
        
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
        )
        
        print(f"✓ Saved ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")
    
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
        
        subgraph = {
            "body_ids": body_ids,
            "body_id_to_idx": body_id_to_idx,
            "adjacency": adjacency,
            "n_neurons": int(data["n_neurons"]),
            "n_connections": int(data["n_connections"]),
        }
        
        print(f"✓ Loaded: {subgraph['n_neurons']} neurons, {subgraph['n_connections']} connections")
        
        return subgraph
