"""Helper functions for selecting neurons by type from annotations."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Set


def find_neurons_by_type(
    annotations: pd.DataFrame,
    body_ids: np.ndarray,
    body_id_to_idx: Dict[int, int],
    neuron_types: Set[str],
    case_sensitive: bool = False
) -> np.ndarray:
    """
    Find neuron indices by type annotation.
    
    Args:
        annotations: DataFrame with bodyId and type columns
        body_ids: Array of body IDs in the connectome
        body_id_to_idx: Mapping from body ID to index
        neuron_types: Set of neuron type strings to search for
        case_sensitive: Whether to use case-sensitive matching
    
    Returns:
        Array of neuron indices matching any of the types
    """
    # Filter annotations to only those in our connectome
    valid_annotations = annotations[annotations["bodyId"].isin(body_ids)].copy()
    
    # Search across type columns
    type_columns = ["type", "superclass", "class", "subclass", "instance"]
    matching_body_ids = set()
    
    for col in type_columns:
        if col not in valid_annotations.columns:
            continue
        
        for neuron_type in neuron_types:
            if case_sensitive:
                mask = valid_annotations[col].astype(str).str.contains(
                    neuron_type, case=True, na=False, regex=False
                )
            else:
                mask = valid_annotations[col].astype(str).str.contains(
                    neuron_type, case=False, na=False, regex=False
                )
            
            matching_ids = valid_annotations.loc[mask, "bodyId"].values
            matching_body_ids.update(matching_ids)
    
    # Convert body IDs to indices
    indices = np.array([body_id_to_idx[bid] for bid in matching_body_ids 
                       if bid in body_id_to_idx], dtype=np.int32)
    
    return indices


def find_visual_neurons(
    annotations: pd.DataFrame,
    body_ids: np.ndarray,
    body_id_to_idx: Dict[int, int]
) -> Dict[str, np.ndarray]:
    """
    Find visual system neurons by type.
    
    Returns dictionary with keys:
        - "photoreceptor": R1-R8 photoreceptors
        - "lamina": L1-L5 lamina neurons
        - "medulla": Medulla neurons (Mi, Tm, T4, T5)
        - "lobula": Lobula neurons (LC, LPLC)
        - "all_visual": All visual neurons combined
    """
    visual_types = {
        "photoreceptor": {"photoreceptor", "R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8"},
        "lamina": {"lamina", "L1", "L2", "L3", "L4", "L5"},
        "medulla": {"medulla", "Mi", "Tm", "T4", "T5"},
        "lobula": {"lobula", "LC", "LPLC", "LPi"},
    }
    
    result = {}
    all_visual_indices = []
    
    for category, types in visual_types.items():
        indices = find_neurons_by_type(annotations, body_ids, body_id_to_idx, types)
        result[category] = indices
        all_visual_indices.extend(indices)
    
    result["all_visual"] = np.unique(np.array(all_visual_indices, dtype=np.int32))
    
    return result


def find_descending_neurons(
    annotations: pd.DataFrame,
    body_ids: np.ndarray,
    body_id_to_idx: Dict[int, int]
) -> Dict[str, np.ndarray]:
    """
    Find descending neurons (motor output) by type.
    
    Returns dictionary with keys:
        - "DN": All descending neurons
        - "DNa": Descending neuron type a
        - "DNb": Descending neuron type b
        - "DNp": Descending neuron type p
    """
    dn_types = {
        "DN": {"DN", "descending"},
        "DNa": {"DNa"},
        "DNb": {"DNb"},
        "DNp": {"DNp"},
    }
    
    result = {}
    
    for category, types in dn_types.items():
        indices = find_neurons_by_type(annotations, body_ids, body_id_to_idx, types)
        result[category] = indices
    
    return result


def find_central_complex_neurons(
    annotations: pd.DataFrame,
    body_ids: np.ndarray,
    body_id_to_idx: Dict[int, int]
) -> Dict[str, np.ndarray]:
    """
    Find central complex neurons (navigation circuits).
    
    Returns dictionary with keys:
        - "FB": Fan-shaped body
        - "EB": Ellipsoid body
        - "PB": Protocerebral bridge
        - "NO": Noduli
        - "all_central_complex": All CX neurons combined
    """
    cx_types = {
        "FB": {"FB", "fan-shaped body", "fan shaped body"},
        "EB": {"EB", "ellipsoid body"},
        "PB": {"PB", "protocerebral bridge"},
        "NO": {"NO", "noduli"},
    }
    
    result = {}
    all_cx_indices = []
    
    for category, types in cx_types.items():
        indices = find_neurons_by_type(annotations, body_ids, body_id_to_idx, types)
        result[category] = indices
        all_cx_indices.extend(indices)
    
    result["all_central_complex"] = np.unique(np.array(all_cx_indices, dtype=np.int32))
    
    return result


def get_neuron_selection_summary(
    annotations: pd.DataFrame,
    body_ids: np.ndarray,
    body_id_to_idx: Dict[int, int]
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Get a comprehensive summary of neuron selections by functional category.
    
    Returns:
        Dictionary with keys "visual", "descending", "central_complex"
    """
    return {
        "visual": find_visual_neurons(annotations, body_ids, body_id_to_idx),
        "descending": find_descending_neurons(annotations, body_ids, body_id_to_idx),
        "central_complex": find_central_complex_neurons(annotations, body_ids, body_id_to_idx),
    }


def print_neuron_selection_report(selection: Dict[str, Dict[str, np.ndarray]]) -> None:
    """Print a readable report of neuron selections."""
    print("\n" + "=" * 70)
    print("Neuron Selection Report (by type)")
    print("=" * 70)
    
    for category, subcategories in selection.items():
        print(f"\n{category.upper()}:")
        for name, indices in subcategories.items():
            print(f"  {name:20s}: {len(indices):6,} neurons")
    
    print("=" * 70)
