"""Helper functions for selecting neurons by type from annotations."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Set
from pathlib import Path


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


def load_neurotransmitter_data(feather_path: str) -> Optional[pd.DataFrame]:
    """
    Load neurotransmitter data from feather file.
    
    Args:
        feather_path: Path to body-neurotransmitters-*.feather
    
    Returns:
        DataFrame with columns: body, consensus_nt, celltype_predicted_nt
        Returns None if file doesn't exist
    """
    if not Path(feather_path).exists():
        return None
    
    try:
        return pd.read_feather(feather_path)
    except Exception as e:
        print(f"Warning: Failed to load neurotransmitter data: {e}")
        return None


def get_neuron_transmitters(
    body_ids: np.ndarray,
    nt_data: Optional[pd.DataFrame]
) -> Dict[int, str]:
    """
    Map body IDs to neurotransmitter types.
    
    Args:
        body_ids: Array of body IDs to look up
        nt_data: Neurotransmitter dataframe (or None)
    
    Returns:
        Dict mapping body_id → transmitter string (lowercase)
        Falls back to celltype_predicted_nt if consensus_nt is 'unclear'
        Returns 'unclear' if not found
    """
    if nt_data is None:
        return {int(bid): "unclear" for bid in body_ids}
    
    result = {}
    nt_lookup = nt_data.set_index("body")
    
    for bid in body_ids:
        if bid in nt_lookup.index:
            row = nt_lookup.loc[bid]
            consensus = str(row.get("consensus_nt", "unclear")).lower().strip()
            if consensus == "unclear" or consensus == "":
                # Fallback to predicted
                predicted = str(row.get("celltype_predicted_nt", "unclear")).lower().strip()
                result[int(bid)] = predicted
            else:
                result[int(bid)] = consensus
        else:
            result[int(bid)] = "unclear"
    
    return result


def create_fixed_input_assignment(
    population_size: int,
    feature_size: int,
    seed: int = 42
) -> np.ndarray:
    """
    Create fixed seeded assignment of input features to neuron population.
    
    Spreads features evenly across population with deterministic randomization.
    
    Args:
        population_size: Number of neurons in population
        feature_size: Number of input features to spread
        seed: Random seed for reproducibility
    
    Returns:
        Array of neuron indices (length = feature_size)
    """
    rng = np.random.RandomState(seed)
    if population_size == 0:
        return np.array([], dtype=np.int32)
    return rng.choice(population_size, size=feature_size, replace=True).astype(np.int32)


def _get_side_mask(annotations: pd.DataFrame, side: str) -> pd.Series:
    """Get boolean mask for neurons on specified side (L/R)."""
    # Check somaSide first, fallback to rootSide if missing/unclear
    if "somaSide" in annotations.columns and "rootSide" in annotations.columns:
        soma = annotations["somaSide"].fillna("?")
        root = annotations["rootSide"].fillna("?")
        # Use somaSide if available, otherwise rootSide
        effective_side = soma.where(soma.isin(["L", "R"]), root)
        return effective_side == side
    elif "somaSide" in annotations.columns:
        return annotations["somaSide"] == side
    elif "rootSide" in annotations.columns:
        return annotations["rootSide"] == side
    else:
        # No side information, return all False
        return pd.Series([False] * len(annotations))


def select_input_neurons_for_luminance(
    annotations: pd.DataFrame,
    body_ids: np.ndarray,
    body_id_to_idx: Dict[int, int],
    side: str
) -> np.ndarray:
    """
    Select R1-R6 photoreceptor neurons for luminance input.
    
    Args:
        annotations: Annotation DataFrame
        body_ids: Body IDs in connectome
        body_id_to_idx: Body ID to index mapping
        side: "L" or "R"
    
    Returns:
        Array of neuron indices for R1-R6 on specified side
    """
    # Find R1-R6 neurons
    valid_annotations = annotations[annotations["bodyId"].isin(body_ids)].copy()
    
    # Search for R1-R6 in type or celltype columns
    type_col = "celltype" if "celltype" in valid_annotations.columns else "type"
    r_mask = valid_annotations[type_col].astype(str).str.contains(
        r"^R[1-6]", case=False, na=False, regex=True
    )
    
    # Filter by side
    side_mask = _get_side_mask(valid_annotations, side)
    
    # Combine masks
    combined_mask = r_mask & side_mask
    matching_body_ids = valid_annotations.loc[combined_mask, "bodyId"].values
    
    # Convert to indices
    indices = np.array([body_id_to_idx[bid] for bid in matching_body_ids
                       if bid in body_id_to_idx], dtype=np.int32)
    
    return indices


def select_input_neurons_for_optic_flow(
    annotations: pd.DataFrame,
    body_ids: np.ndarray,
    body_id_to_idx: Dict[int, int],
    subtype: str,
    side: str
) -> np.ndarray:
    """
    Select T4/T5 neurons for optic flow input.
    
    Args:
        annotations: Annotation DataFrame
        body_ids: Body IDs in connectome
        body_id_to_idx: Body ID to index mapping
        subtype: "a", "b", "c", or "d"
        side: "L" or "R"
    
    Returns:
        Array of neuron indices for T4/T5 of specified subtype and side
    """
    valid_annotations = annotations[annotations["bodyId"].isin(body_ids)].copy()
    
    # Search for T4/T5 + subtype
    type_col = "celltype" if "celltype" in valid_annotations.columns else "type"
    t_mask = valid_annotations[type_col].astype(str).str.contains(
        f"T[45]{subtype}", case=False, na=False, regex=True
    )
    
    # Filter by side
    side_mask = _get_side_mask(valid_annotations, side)
    
    # Combine masks
    combined_mask = t_mask & side_mask
    matching_body_ids = valid_annotations.loc[combined_mask, "bodyId"].values
    
    # Convert to indices
    indices = np.array([body_id_to_idx[bid] for bid in matching_body_ids
                       if bid in body_id_to_idx], dtype=np.int32)
    
    return indices


def select_input_neurons_for_looming(
    annotations: pd.DataFrame,
    body_ids: np.ndarray,
    body_id_to_idx: Dict[int, int],
    side: str
) -> np.ndarray:
    """
    Select LPLC2 + LC4 neurons for looming input.
    
    Args:
        annotations: Annotation DataFrame
        body_ids: Body IDs in connectome
        body_id_to_idx: Body ID to index mapping
        side: "L" or "R"
    
    Returns:
        Array of neuron indices for LPLC2 + LC4 on specified side
    """
    valid_annotations = annotations[annotations["bodyId"].isin(body_ids)].copy()
    
    # Search for LPLC2 or LC4
    type_col = "celltype" if "celltype" in valid_annotations.columns else "type"
    loom_mask = valid_annotations[type_col].astype(str).str.contains(
        r"(LPLC2|LC4)", case=False, na=False, regex=True
    )
    
    # Filter by side
    side_mask = _get_side_mask(valid_annotations, side)
    
    # Combine masks
    combined_mask = loom_mask & side_mask
    matching_body_ids = valid_annotations.loc[combined_mask, "bodyId"].values
    
    # Convert to indices
    indices = np.array([body_id_to_idx[bid] for bid in matching_body_ids
                       if bid in body_id_to_idx], dtype=np.int32)
    
    return indices


def select_output_neurons_for_motor(
    annotations: pd.DataFrame,
    body_ids: np.ndarray,
    body_id_to_idx: Dict[int, int]
) -> Dict[str, np.ndarray]:
    """
    Select descending neurons for motor readout.
    
    Maps specific DN types to flight behaviors:
    - DNp01: escape/sharp avoidance
    - DNg02: thrust/altitude
    - DNa01/DNa02: yaw (left-right difference)
    - DNb: additional motor pathways
    
    Args:
        annotations: Annotation DataFrame
        body_ids: Body IDs in connectome
        body_id_to_idx: Body ID to index mapping
    
    Returns:
        Dict with keys "DNp01", "DNg02", "DNa01", "DNa02", "DNb"
        Each value is array of neuron indices
    """
    valid_annotations = annotations[annotations["bodyId"].isin(body_ids)].copy()
    type_col = "celltype" if "celltype" in valid_annotations.columns else "type"
    
    result = {}
    
    # DNp01 - escape/sharp avoidance
    dnp01_mask = valid_annotations[type_col].astype(str).str.contains(
        r"DNp01", case=False, na=False, regex=True
    )
    dnp01_bodies = valid_annotations.loc[dnp01_mask, "bodyId"].values
    result["DNp01"] = np.array([body_id_to_idx[bid] for bid in dnp01_bodies
                                if bid in body_id_to_idx], dtype=np.int32)
    
    # DNg02 - thrust/altitude (all variants a-g)
    dng02_mask = valid_annotations[type_col].astype(str).str.contains(
        r"DNg02", case=False, na=False, regex=True
    )
    dng02_bodies = valid_annotations.loc[dng02_mask, "bodyId"].values
    result["DNg02"] = np.array([body_id_to_idx[bid] for bid in dng02_bodies
                                if bid in body_id_to_idx], dtype=np.int32)
    
    # DNa01 - yaw component
    dna01_mask = valid_annotations[type_col].astype(str).str.contains(
        r"DNa01", case=False, na=False, regex=True
    )
    dna01_bodies = valid_annotations.loc[dna01_mask, "bodyId"].values
    result["DNa01"] = np.array([body_id_to_idx[bid] for bid in dna01_bodies
                                if bid in body_id_to_idx], dtype=np.int32)
    
    # DNa02 - yaw component
    dna02_mask = valid_annotations[type_col].astype(str).str.contains(
        r"DNa02", case=False, na=False, regex=True
    )
    dna02_bodies = valid_annotations.loc[dna02_mask, "bodyId"].values
    result["DNa02"] = np.array([body_id_to_idx[bid] for bid in dna02_bodies
                                if bid in body_id_to_idx], dtype=np.int32)
    
    # DNb - additional motor pathways (DNb01-06)
    dnb_mask = valid_annotations[type_col].astype(str).str.contains(
        r"DNb\d", case=False, na=False, regex=True
    )
    dnb_bodies = valid_annotations.loc[dnb_mask, "bodyId"].values
    result["DNb"] = np.array([body_id_to_idx[bid] for bid in dnb_bodies
                              if bid in body_id_to_idx], dtype=np.int32)
    
    return result
