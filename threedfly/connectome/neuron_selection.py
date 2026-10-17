"""Helper functions for selecting neurons by type from annotations."""

import re
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path

_TYPE_SIDE_SUFFIX = re.compile(r"[_-]([LRlr])(?:$|[^A-Za-z0-9])")


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


DEFAULT_INHIBITORY_TRANSMITTERS = ("gaba", "glutamate", "histamine")


def transmitter_to_sign(
    transmitter: str,
    inhibitory_transmitters=None,
    default_sign: float = 1.0,
) -> float:
    """
    Map a neurotransmitter name to a synaptic sign.

    Default: acetylcholine and others/unclear → +1;
    GABA, glutamate, histamine → -1. Inhibitory set is configurable.
    """
    inhibitory = {
        str(name).lower().strip()
        for name in (inhibitory_transmitters or DEFAULT_INHIBITORY_TRANSMITTERS)
    }
    name = str(transmitter).lower().strip()
    if name in inhibitory:
        return -1.0
    return float(default_sign)


def build_transmitter_sign_vector(
    n_neurons: int,
    body_ids: np.ndarray,
    body_id_to_idx: Dict[int, int],
    nt_data: Optional[pd.DataFrame],
    inhibitory_transmitters=None,
    default_sign: float = 1.0,
) -> np.ndarray:
    """Build length-n_neurons sign vector (+1/-1) from neurotransmitter tables."""
    signs = np.full(n_neurons, default_sign, dtype=np.float32)
    if nt_data is None:
        return signs
    transmitters = get_neuron_transmitters(body_ids, nt_data)
    for bid, nt in transmitters.items():
        idx = body_id_to_idx.get(int(bid))
        if idx is not None:
            signs[idx] = transmitter_to_sign(
                nt, inhibitory_transmitters=inhibitory_transmitters, default_sign=default_sign
            )
    return signs


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


def _annotation_type_mask(annotations: pd.DataFrame, pattern: str) -> pd.Series:
    """Match `pattern` against celltype and/or type columns."""
    mask = pd.Series(False, index=annotations.index)
    for col in ("celltype", "type"):
        if col in annotations.columns:
            mask = mask | annotations[col].astype(str).str.contains(
                pattern, case=False, na=False, regex=True
            )
    return mask


def _bodies_to_indices(
    body_ids: np.ndarray,
    body_id_to_idx: Dict[int, int],
) -> np.ndarray:
    return np.array(
        [body_id_to_idx[bid] for bid in body_ids if bid in body_id_to_idx],
        dtype=np.int32,
    )


def _canonical_side_label(val) -> str:
    """Map somaSide/rootSide/type-suffix labels to L, R, or ?."""
    if val is None:
        return "?"
    try:
        if pd.isna(val):
            return "?"
    except (TypeError, ValueError):
        pass
    s = str(val).strip().upper()
    if s in ("L", "LEFT", "LHS", "LH"):
        return "L"
    if s in ("R", "RIGHT", "RHS", "RH"):
        return "R"
    return "?"


def _type_suffix_sides(annotations: pd.DataFrame) -> pd.Series:
    sides = pd.Series(["?"] * len(annotations), index=annotations.index)
    for col in ("celltype", "type"):
        if col not in annotations.columns:
            continue
        extracted = annotations[col].astype(str).str.extract(_TYPE_SIDE_SUFFIX, expand=False)
        mapped = extracted.map(lambda x: _canonical_side_label(x) if pd.notna(x) else "?")
        missing = ~sides.isin(["L", "R"])
        sides = sides.where(~missing, mapped)
    return sides


# Sensory cells: soma in the optic lobe. Descending neurons: laterality of the
# descending tract (rootSide) — somaSide often clusters on one side of the CNS.
_SENSORY_SIDE_PRIORITY = ("somaSide", "rootSide", "suffix")
_MOTOR_SIDE_PRIORITY = ("rootSide", "somaSide", "suffix")


def _effective_side_series(
    annotations: pd.DataFrame,
    priority=None,
) -> pd.Series:
    """Resolve L/R from the first source in ``priority`` that is L or R."""
    order = tuple(priority) if priority is not None else _SENSORY_SIDE_PRIORITY
    sides = pd.Series(["?"] * len(annotations), index=annotations.index)
    for src in order:
        if src == "suffix":
            cand = _type_suffix_sides(annotations)
        elif src in annotations.columns:
            cand = annotations[src].map(_canonical_side_label)
        else:
            continue
        missing = ~sides.isin(["L", "R"])
        sides = sides.where(~missing, cand)
    return sides


def _get_side_mask(
    annotations: pd.DataFrame,
    side: str,
    priority=None,
) -> pd.Series:
    """Get boolean mask for neurons on specified side (L/R)."""
    want = _canonical_side_label(side)
    if want not in ("L", "R"):
        return pd.Series([False] * len(annotations), index=annotations.index)
    return _effective_side_series(annotations, priority=priority) == want


def _fill_unlabeled_sides(
    all_idx: np.ndarray,
    left_idx: np.ndarray,
    right_idx: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """If a DN type exists but one side is empty, split unlabeled cells across L/R."""
    all_i = np.asarray(all_idx, dtype=np.int32)
    left = np.asarray(left_idx, dtype=np.int32)
    right = np.asarray(right_idx, dtype=np.int32)
    labeled = set(int(i) for i in left) | set(int(i) for i in right)
    unlabeled = [int(i) for i in all_i if int(i) not in labeled]
    if not unlabeled:
        return left, right
    if left.size == 0 and right.size == 0:
        if len(unlabeled) >= 2:
            return (
                np.asarray(unlabeled[0::2], dtype=np.int32),
                np.asarray(unlabeled[1::2], dtype=np.int32),
            )
        return left, right
    if left.size == 0:
        return np.asarray(unlabeled, dtype=np.int32), right
    if right.size == 0:
        return left, np.asarray(unlabeled, dtype=np.int32)
    return left, right


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
    valid_annotations = annotations[annotations["bodyId"].isin(body_ids)].copy()
    side_mask = _get_side_mask(valid_annotations, side)
    r_mask = _annotation_type_mask(valid_annotations, r"^(?:R[1-6]|R1-R6)(?:$|[^0-9A-Za-z])")
    matching_body_ids = valid_annotations.loc[r_mask & side_mask, "bodyId"].values
    indices = _bodies_to_indices(matching_body_ids, body_id_to_idx)
    if indices.size > 0:
        return indices

    # Fallback: L1-L5 by somaSide/rootSide when R1-R6 are absent
    l_mask = _annotation_type_mask(valid_annotations, r"^L[1-5](?:$|[^0-9A-Za-z])")
    matching_body_ids = valid_annotations.loc[l_mask & side_mask, "bodyId"].values
    return _bodies_to_indices(matching_body_ids, body_id_to_idx)


def select_input_neurons_for_optic_flow(
    annotations: pd.DataFrame,
    body_ids: np.ndarray,
    body_id_to_idx: Dict[int, int],
    subtype: str,
    side: str,
    cell_class: str = "T4",
) -> np.ndarray:
    """
    Select T4 or T5 neurons for optic flow input.

    T4 and T5 are selected separately with anchored regexes so T4a ≠ T5a.
    """
    valid_annotations = annotations[annotations["bodyId"].isin(body_ids)].copy()
    prefix = str(cell_class).upper()
    pattern = rf"^{prefix}{subtype}(?:$|[^A-Za-z0-9])"
    t_mask = _annotation_type_mask(valid_annotations, pattern)
    side_mask = _get_side_mask(valid_annotations, side)
    matching_body_ids = valid_annotations.loc[t_mask & side_mask, "bodyId"].values
    return _bodies_to_indices(matching_body_ids, body_id_to_idx)


# Ipsilateral looming detectors. LPLC2 → DNp01; LC4/LPLC1/LPLC4 → DNp03
# (LPLC2 has essentially no synapses onto DNp03 in MaleCNS).
LOOMING_TYPE_PATTERN = r"^(?:LPLC[124]|LC4)$"
LOOMING_TYPE_NAMES = ("LPLC1", "LPLC2", "LPLC4", "LC4")


def select_input_neurons_for_looming(
    annotations: pd.DataFrame,
    body_ids: np.ndarray,
    body_id_to_idx: Dict[int, int],
    side: Optional[str] = None,
) -> np.ndarray:
    """
    Select ipsilateral looming detectors: LPLC1, LPLC2, LPLC4, and LC4.
    
    Args:
        annotations: Annotation DataFrame
        body_ids: Body IDs in connectome
        body_id_to_idx: Body ID to index mapping
        side: "L" or "R", or None for both sides
    
    Returns:
        Array of neuron indices for looming cells on specified side
    """
    valid_annotations = annotations[annotations["bodyId"].isin(body_ids)].copy()
    loom_mask = _annotation_type_mask(valid_annotations, LOOMING_TYPE_PATTERN)
    if side is None:
        matching_body_ids = valid_annotations.loc[loom_mask, "bodyId"].values
        return _bodies_to_indices(matching_body_ids, body_id_to_idx)
    side_mask = _get_side_mask(valid_annotations, side)
    matching_body_ids = valid_annotations.loc[loom_mask & side_mask, "bodyId"].values
    return _bodies_to_indices(matching_body_ids, body_id_to_idx)


def looming_type_composition(
    annotations: pd.DataFrame,
    body_ids: np.ndarray,
    indices: np.ndarray,
) -> Dict[str, int]:
    """Count LPLC1/LPLC2/LPLC4/LC4 among mapped looming indices."""
    counts = {name: 0 for name in LOOMING_TYPE_NAMES}
    if annotations is None or indices is None:
        return counts
    body_ids = np.asarray(body_ids)
    idx = np.asarray(indices, dtype=np.int32)
    idx = idx[(idx >= 0) & (idx < body_ids.size)]
    if idx.size == 0:
        return counts
    selected = set(int(b) for b in body_ids[idx])
    valid = annotations[annotations["bodyId"].isin(selected)].copy()
    for name in LOOMING_TYPE_NAMES:
        mask = _annotation_type_mask(valid, rf"^{name}$")
        counts[name] = int(mask.sum())
    return counts


def select_output_neurons_for_motor(
    annotations: pd.DataFrame,
    body_ids: np.ndarray,
    body_id_to_idx: Dict[int, int]
) -> Dict[str, np.ndarray]:
    """
    Select descending neurons for motor readout.
    
    Maps specific DN types to flight behaviors:
    - DNp01: escape/sharp avoidance (giant fiber)
    - DNp03: looming-evoked flight saccades (yaw away from threat)
    - DNg02: thrust/altitude
    - DNa01/DNa02: yaw (left-right difference)
    - DNa15 / DNb01: downstream saccade-network outputs of DNp03
    - DNb: additional motor pathways
    
    Args:
        annotations: Annotation DataFrame
        body_ids: Body IDs in connectome
        body_id_to_idx: Body ID to index mapping
    
    Returns:
        Dict with keys "DNp01", "DNp03", "DNg02", "DNa01", "DNa02", "DNb"
        plus side splits "DNg02_L/R", "DNa01_L/R", "DNa02_L/R", "DNp03_L/R",
        "DNa15_L/R", "DNb01_L/R" when sides exist.
    """
    valid_annotations = annotations[annotations["bodyId"].isin(body_ids)].copy()

    def _select(pattern: str) -> np.ndarray:
        mask = _annotation_type_mask(valid_annotations, pattern)
        return _bodies_to_indices(
            valid_annotations.loc[mask, "bodyId"].values, body_id_to_idx
        )

    def _select_side(pattern: str, side: str) -> np.ndarray:
        mask = _annotation_type_mask(valid_annotations, pattern) & _get_side_mask(
            valid_annotations, side, priority=_MOTOR_SIDE_PRIORITY
        )
        return _bodies_to_indices(
            valid_annotations.loc[mask, "bodyId"].values, body_id_to_idx
        )

    result = {
        "DNp01": _select(r"^DNp01(?:$|[^A-Za-z0-9])"),
        "DNp03": _select(r"^DNp03(?:$|[^A-Za-z0-9])"),
        "DNg02": _select(r"^DNg02"),
        "DNa01": _select(r"^DNa01(?:$|[^A-Za-z0-9])"),
        "DNa02": _select(r"^DNa02(?:$|[^A-Za-z0-9])"),
        "DNa15": _select(r"^DNa15(?:$|[^A-Za-z0-9])"),
        "DNb": _select(r"^DNb\d"),
        "DNb01": _select(r"^DNb01(?:$|[^A-Za-z0-9])"),
        "DNg02_L": _select_side(r"^DNg02", "L"),
        "DNg02_R": _select_side(r"^DNg02", "R"),
        "DNa01_L": _select_side(r"^DNa01(?:$|[^A-Za-z0-9])", "L"),
        "DNa01_R": _select_side(r"^DNa01(?:$|[^A-Za-z0-9])", "R"),
        "DNa02_L": _select_side(r"^DNa02(?:$|[^A-Za-z0-9])", "L"),
        "DNa02_R": _select_side(r"^DNa02(?:$|[^A-Za-z0-9])", "R"),
        "DNa15_L": _select_side(r"^DNa15(?:$|[^A-Za-z0-9])", "L"),
        "DNa15_R": _select_side(r"^DNa15(?:$|[^A-Za-z0-9])", "R"),
        "DNp03_L": _select_side(r"^DNp03(?:$|[^A-Za-z0-9])", "L"),
        "DNp03_R": _select_side(r"^DNp03(?:$|[^A-Za-z0-9])", "R"),
        "DNb01_L": _select_side(r"^DNb01(?:$|[^A-Za-z0-9])", "L"),
        "DNb01_R": _select_side(r"^DNb01(?:$|[^A-Za-z0-9])", "R"),
    }
    for pooled, left_key, right_key in (
        ("DNp03", "DNp03_L", "DNp03_R"),
        ("DNa02", "DNa02_L", "DNa02_R"),
        ("DNg02", "DNg02_L", "DNg02_R"),
        ("DNa01", "DNa01_L", "DNa01_R"),
        ("DNa15", "DNa15_L", "DNa15_R"),
        ("DNb01", "DNb01_L", "DNb01_R"),
    ):
        left, right = _fill_unlabeled_sides(
            result[pooled], result[left_key], result[right_key]
        )
        result[left_key] = left
        result[right_key] = right
    return result


def describe_side_assignment(
    annotations: pd.DataFrame,
    body_ids: np.ndarray,
    body_id_to_idx: Dict[int, int],
    indices: np.ndarray,
) -> List[Dict]:
    """Return index, bodyId, somaSide, rootSide for mapped cells (debug laterality)."""
    body_ids = np.asarray(body_ids)
    rows = []
    for i in np.asarray(indices, dtype=np.int32):
        i = int(i)
        bid = int(body_ids[i]) if 0 <= i < body_ids.size else None
        soma = root = "?"
        if bid is not None and annotations is not None and "bodyId" in annotations.columns:
            hit = annotations.loc[annotations["bodyId"] == bid]
            if len(hit):
                rec = hit.iloc[0]
                if "somaSide" in hit.columns:
                    soma = str(rec["somaSide"])
                if "rootSide" in hit.columns:
                    root = str(rec["rootSide"])
        rows.append(
            {"index": i, "bodyId": bid, "somaSide": soma, "rootSide": root}
        )
    return rows


# Columns that would carry MaleCNS optic-lobe hex / column coordinates if present.
_HEX_COLUMN_CANDIDATES = (
    ("hexX", "hexY"),
    ("hex_x", "hex_y"),
    ("column_x", "column_y"),
    ("ommatidia_x", "ommatidia_y"),
    ("retinotopy_x", "retinotopy_y"),
)


def find_optic_lobe_hex_columns(annotations: pd.DataFrame) -> Optional[tuple]:
    """
    Return (x_col, y_col) if annotations include optic-lobe hex coordinates.

    MaleCNS v1.0 body-annotation tables used here do not include these fields;
    callers should fall back to the synthetic hex lattice in `threedfly.vision.hex_eye`.
    """
    if annotations is None or len(annotations.columns) == 0:
        return None
    lower = {str(c).lower(): c for c in annotations.columns}
    for x_name, y_name in _HEX_COLUMN_CANDIDATES:
        if x_name.lower() in lower and y_name.lower() in lower:
            return (lower[x_name.lower()], lower[y_name.lower()])
    return None


def select_haltere_afferent_neurons(
    annotations: pd.DataFrame,
    body_ids: np.ndarray,
    body_id_to_idx: Dict[int, int],
) -> Dict[str, object]:
    """
    Map body angular velocity onto annotated haltere afferents.

    Search order (logged):
      1. type/celltype/instance containing "haltere"
      2. If none, empty mapping — **fallback is engineered angular-rate damping**
         in the physics step (`SAFETY_HALTERE_KD`). DNp cells are *not* used as
         a silent substitute; that would contaminate the escape pathway.
    """
    valid_annotations = annotations[annotations["bodyId"].isin(body_ids)].copy()
    type_columns = ["type", "celltype", "instance", "class", "subclass"]
    mask = pd.Series(False, index=valid_annotations.index)
    for col in type_columns:
        if col not in valid_annotations.columns:
            continue
        mask = mask | valid_annotations[col].astype(str).str.contains(
            "haltere", case=False, na=False, regex=False
        )
    matching = valid_annotations.loc[mask]
    left_mask = _get_side_mask(matching, "L") if len(matching) else matching.index
    right_mask = _get_side_mask(matching, "R") if len(matching) else matching.index
    left_ids = matching.loc[left_mask, "bodyId"].values if len(matching) else np.array([])
    right_ids = matching.loc[right_mask, "bodyId"].values if len(matching) else np.array([])
    # Unsided hits go to both pools so a type-only annotation still drives input.
    if len(matching):
        sided = np.concatenate([np.asarray(left_ids), np.asarray(right_ids)])
        unsided = matching.loc[~matching["bodyId"].isin(sided), "bodyId"].values
        if len(unsided):
            left_ids = np.concatenate([left_ids, unsided])
            right_ids = np.concatenate([right_ids, unsided])
    left = _bodies_to_indices(left_ids, body_id_to_idx)
    right = _bodies_to_indices(right_ids, body_id_to_idx)
    all_idx = np.unique(np.concatenate([left, right]).astype(np.int32)) if (left.size + right.size) else np.array([], dtype=np.int32)
    fallback = "engineered_damping" if all_idx.size == 0 else "annotation"
    print(
        f"Haltere afferents: L={left.size}, R={right.size}, all={all_idx.size} "
        f"(search='haltere' in type/celltype/instance; fallback={fallback})"
    )
    return {
        "left": left,
        "right": right,
        "all": all_idx,
        "fallback": fallback,
        "n_left": int(left.size),
        "n_right": int(right.size),
    }
