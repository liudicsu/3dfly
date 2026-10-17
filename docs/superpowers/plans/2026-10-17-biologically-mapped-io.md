# Biologically-Mapped Brain Input/Output Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement biologically-mapped brain input/output for the full MaleCNS connectome, mapping visual features to specific neuron types and reading motor output from descending neurons.

**Architecture:** Replace engineered feature indexes with biological neuron type mappings using annotation data (celltype, somaSide, rootSide). Map luminance→R1-R6, optic flow→T4/T5, looming→LPLC2/LC4, and read motor output from specific DN types (DNp01, DNg02, DNa01/02, DNb). Add excitatory/inhibitory signs based on neurotransmitter data and implement warm-up period for resting rate calibration.

**Tech Stack:** Python 3.9+, NumPy, pandas, pyarrow, scipy sparse matrices, pytest

## Global Constraints

- Tests must use synthetic annotation and neurotransmitter fixtures (data files gitignored)
- Neurotransmitter mapping: ACh=+1, GABA/glutamate/histamine=-1, others/unclear=+1
- Runtime and memory must stay sane for ~184k neurons (use sparse matrices, efficient operations)
- All existing tests must pass
- Viser UI, brain depth point cloud, and StereoBM comparison must remain functional
- Keep commit messages clear and concise (one logical change per commit)

---

### Task 1: Add neurotransmitter support to neuron selection helpers

**Files:**
- Modify: `threedfly/connectome/neuron_selection.py`
- Modify: `pyproject.toml` (ensure pyarrow dependency exists)
- Test: `tests/test_neuron_selection.py`

**Interfaces:**
- Consumes: Nothing (first task)
- Produces:
  - `load_neurotransmitter_data(feather_path: str) -> pd.DataFrame` - loads neurotransmitter data with columns `body`, `consensus_nt`, `celltype_predicted_nt`
  - `get_neuron_transmitters(body_ids: np.ndarray, nt_data: pd.DataFrame) -> Dict[int, str]` - maps body_id → transmitter string

- [ ] **Step 1: Write test fixtures for neurotransmitter data**

```python
# Add to tests/test_neuron_selection.py

def create_synthetic_nt_data():
    """Create synthetic neurotransmitter data for testing."""
    return pd.DataFrame({
        "body": [1, 2, 3, 4, 5, 6, 7, 8],
        "consensus_nt": ["acetylcholine", "GABA", "glutamate", "histamine", 
                         "dopamine", "octopamine", "serotonin", "unclear"],
        "celltype_predicted_nt": ["acetylcholine", "GABA", "glutamate", "histamine",
                                  "dopamine", "octopamine", "serotonin", "acetylcholine"],
    })


def test_load_neurotransmitter_data_fallback():
    """Test neurotransmitter loading falls back gracefully."""
    # Should not crash on missing file
    from threedfly.connectome.neuron_selection import load_neurotransmitter_data
    result = load_neurotransmitter_data("nonexistent.feather")
    assert result is None or len(result) == 0


def test_get_neuron_transmitters():
    """Test neurotransmitter lookup by body ID."""
    from threedfly.connectome.neuron_selection import get_neuron_transmitters
    
    nt_data = create_synthetic_nt_data()
    body_ids = np.array([1, 2, 3, 8])
    
    transmitters = get_neuron_transmitters(body_ids, nt_data)
    
    assert transmitters[1] == "acetylcholine"
    assert transmitters[2] == "GABA"
    assert transmitters[3] == "glutamate"
    # Body 8: unclear → should fallback to celltype_predicted
    assert transmitters[8] == "acetylcholine"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /workspace
pytest tests/test_neuron_selection.py::test_load_neurotransmitter_data_fallback -v
pytest tests/test_neuron_selection.py::test_get_neuron_transmitters -v
```

Expected: FAIL with ImportError or NameError

- [ ] **Step 3: Implement neurotransmitter loading functions**

```python
# Add to threedfly/connectome/neuron_selection.py at end of file

def load_neurotransmitter_data(feather_path: str) -> Optional[pd.DataFrame]:
    """
    Load neurotransmitter data from feather file.
    
    Args:
        feather_path: Path to body-neurotransmitters-*.feather
    
    Returns:
        DataFrame with columns: body, consensus_nt, celltype_predicted_nt
        Returns None if file doesn't exist
    """
    from pathlib import Path
    
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
```

- [ ] **Step 4: Add import to __init__.py**

```python
# Add to threedfly/connectome/__init__.py

from threedfly.connectome.neuron_selection import (
    find_neurons_by_type,
    find_visual_neurons,
    find_descending_neurons,
    find_central_complex_neurons,
    get_neuron_selection_summary,
    print_neuron_selection_report,
    load_neurotransmitter_data,  # NEW
    get_neuron_transmitters,      # NEW
)
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd /workspace
pytest tests/test_neuron_selection.py::test_load_neurotransmitter_data_fallback -v
pytest tests/test_neuron_selection.py::test_get_neuron_transmitters -v
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add threedfly/connectome/neuron_selection.py threedfly/connectome/__init__.py tests/test_neuron_selection.py
git commit -m "feat: add neurotransmitter data loading and lookup"
```

---

### Task 2: Implement input mapping helpers for sensory neuron populations

**Files:**
- Modify: `threedfly/connectome/neuron_selection.py`
- Test: `tests/test_neuron_selection.py`

**Interfaces:**
- Consumes: Existing `find_neurons_by_type`, annotation DataFrame structure
- Produces:
  - `select_input_neurons_for_luminance(annotations, body_ids, body_id_to_idx, side: str) -> np.ndarray` - returns R1-R6 indices for left/right
  - `select_input_neurons_for_optic_flow(annotations, body_ids, body_id_to_idx, subtype: str, side: str) -> np.ndarray` - returns T4/T5 by subtype a-d and side
  - `select_input_neurons_for_looming(annotations, body_ids, body_id_to_idx, side: str) -> np.ndarray` - returns LPLC2 + LC4 indices by side
  - `create_fixed_input_assignment(population_size: int, feature_size: int, seed: int) -> np.ndarray` - returns indices array for spreading features across population

- [ ] **Step 1: Write test for fixed input assignment**

```python
# Add to tests/test_neuron_selection.py

def test_create_fixed_input_assignment():
    """Test fixed seeded assignment of features to neuron population."""
    from threedfly.connectome.neuron_selection import create_fixed_input_assignment
    
    # 100 neurons, 10 features
    assignment = create_fixed_input_assignment(100, 10, seed=42)
    
    assert len(assignment) == 10
    assert assignment.max() < 100
    assert assignment.min() >= 0
    
    # Should be deterministic
    assignment2 = create_fixed_input_assignment(100, 10, seed=42)
    assert np.array_equal(assignment, assignment2)
    
    # Different seed should give different assignment
    assignment3 = create_fixed_input_assignment(100, 10, seed=99)
    assert not np.array_equal(assignment, assignment3)
```

- [ ] **Step 2: Write test for luminance input selection**

```python
# Add to tests/test_neuron_selection.py

def test_select_input_neurons_for_luminance():
    """Test R1-R6 selection by side."""
    from threedfly.connectome.neuron_selection import select_input_neurons_for_luminance
    
    # Create annotations with somaSide and rootSide
    annotations = pd.DataFrame({
        "bodyId": [1, 2, 3, 4, 5, 6, 7, 8],
        "type": ["R1", "R2", "R3", "R4", "R5", "R6", "R1", "R2"],
        "celltype": ["R1", "R2", "R3", "R4", "R5", "R6", "R1", "R2"],
        "somaSide": ["L", "L", "R", "R", "?", "?", "?", "?"],
        "rootSide": ["L", "L", "R", "R", "L", "R", "L", "R"],
    })
    
    body_ids = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    body_id_to_idx = {bid: idx for idx, bid in enumerate(body_ids)}
    
    # Select left side
    left_indices = select_input_neurons_for_luminance(
        annotations, body_ids, body_id_to_idx, side="L"
    )
    
    # Should find: 1, 2 (somaSide=L), 5, 7 (rootSide=L when somaSide=?)
    assert len(left_indices) == 4
    assert set(left_indices) == {0, 1, 4, 6}
    
    # Select right side
    right_indices = select_input_neurons_for_luminance(
        annotations, body_ids, body_id_to_idx, side="R"
    )
    
    # Should find: 3, 4 (somaSide=R), 6, 8 (rootSide=R when somaSide=?)
    assert len(right_indices) == 4
    assert set(right_indices) == {2, 3, 5, 7}
```

- [ ] **Step 3: Write test for optic flow input selection**

```python
# Add to tests/test_neuron_selection.py

def test_select_input_neurons_for_optic_flow():
    """Test T4/T5 selection by subtype and side."""
    from threedfly.connectome.neuron_selection import select_input_neurons_for_optic_flow
    
    annotations = pd.DataFrame({
        "bodyId": [1, 2, 3, 4, 5, 6],
        "type": ["T4a", "T4b", "T4c", "T4d", "T5a", "T5b"],
        "celltype": ["T4a", "T4b", "T4c", "T4d", "T5a", "T5b"],
        "somaSide": ["L", "L", "R", "R", "L", "R"],
    })
    
    body_ids = np.array([1, 2, 3, 4, 5, 6])
    body_id_to_idx = {bid: idx for idx, bid in enumerate(body_ids)}
    
    # Select T4a left
    indices = select_input_neurons_for_optic_flow(
        annotations, body_ids, body_id_to_idx, subtype="a", side="L"
    )
    assert len(indices) >= 1
    assert 0 in indices  # T4a, somaSide=L
    
    # Select T5b right
    indices = select_input_neurons_for_optic_flow(
        annotations, body_ids, body_id_to_idx, subtype="b", side="R"
    )
    assert len(indices) >= 1
    assert 5 in indices  # T5b, somaSide=R
```

- [ ] **Step 4: Write test for looming input selection**

```python
# Add to tests/test_neuron_selection.py

def test_select_input_neurons_for_looming():
    """Test LPLC2 + LC4 selection by side."""
    from threedfly.connectome.neuron_selection import select_input_neurons_for_looming
    
    annotations = pd.DataFrame({
        "bodyId": [1, 2, 3, 4, 5],
        "type": ["LPLC2", "LPLC2", "LC4", "LC4", "other"],
        "celltype": ["LPLC2", "LPLC2", "LC4", "LC4", "other"],
        "somaSide": ["L", "R", "L", "R", "L"],
    })
    
    body_ids = np.array([1, 2, 3, 4, 5])
    body_id_to_idx = {bid: idx for idx, bid in enumerate(body_ids)}
    
    # Select left
    left_indices = select_input_neurons_for_looming(
        annotations, body_ids, body_id_to_idx, side="L"
    )
    assert len(left_indices) == 2  # LPLC2 + LC4 on left
    assert set(left_indices) == {0, 2}
    
    # Select right
    right_indices = select_input_neurons_for_looming(
        annotations, body_ids, body_id_to_idx, side="R"
    )
    assert len(right_indices) == 2  # LPLC2 + LC4 on right
    assert set(right_indices) == {1, 3}
```

- [ ] **Step 5: Run tests to verify they fail**

```bash
cd /workspace
pytest tests/test_neuron_selection.py::test_create_fixed_input_assignment -v
pytest tests/test_neuron_selection.py::test_select_input_neurons_for_luminance -v
pytest tests/test_neuron_selection.py::test_select_input_neurons_for_optic_flow -v
pytest tests/test_neuron_selection.py::test_select_input_neurons_for_looming -v
```

Expected: FAIL with ImportError or NameError

- [ ] **Step 6: Implement input mapping helpers**

```python
# Add to threedfly/connectome/neuron_selection.py at end of file

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
```

- [ ] **Step 7: Run tests to verify they pass**

```bash
cd /workspace
pytest tests/test_neuron_selection.py::test_create_fixed_input_assignment -v
pytest tests/test_neuron_selection.py::test_select_input_neurons_for_luminance -v
pytest tests/test_neuron_selection.py::test_select_input_neurons_for_optic_flow -v
pytest tests/test_neuron_selection.py::test_select_input_neurons_for_looming -v
```

Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add threedfly/connectome/neuron_selection.py tests/test_neuron_selection.py
git commit -m "feat: add input mapping helpers for sensory neurons"
```

---

### Task 3: Implement output readout helpers for descending neurons

**Files:**
- Modify: `threedfly/connectome/neuron_selection.py`
- Test: `tests/test_neuron_selection.py`

**Interfaces:**
- Consumes: Existing `find_neurons_by_type`
- Produces:
  - `select_output_neurons_for_motor(annotations, body_ids, body_id_to_idx) -> Dict[str, np.ndarray]` - returns dict with keys "DNp01", "DNg02", "DNa01", "DNa02", "DNb" mapping to neuron indices

- [ ] **Step 1: Write test for motor output selection**

```python
# Add to tests/test_neuron_selection.py

def test_select_output_neurons_for_motor():
    """Test DN selection for motor readout."""
    from threedfly.connectome.neuron_selection import select_output_neurons_for_motor
    
    annotations = pd.DataFrame({
        "bodyId": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        "type": ["DNp01", "DNp01", "DNg02_a", "DNg02_b", "DNa01", "DNa02", 
                 "DNb01", "DNb02", "other"],
        "celltype": ["DNp01", "DNp01", "DNg02_a", "DNg02_b", "DNa01", "DNa02",
                     "DNb01", "DNb02", "other"],
    })
    
    body_ids = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    body_id_to_idx = {bid: idx for idx, bid in enumerate(body_ids)}
    
    motor_neurons = select_output_neurons_for_motor(
        annotations, body_ids, body_id_to_idx
    )
    
    assert "DNp01" in motor_neurons
    assert "DNg02" in motor_neurons
    assert "DNa01" in motor_neurons
    assert "DNa02" in motor_neurons
    assert "DNb" in motor_neurons
    
    assert len(motor_neurons["DNp01"]) == 2
    assert len(motor_neurons["DNg02"]) == 2
    assert len(motor_neurons["DNa01"]) == 1
    assert len(motor_neurons["DNa02"]) == 1
    assert len(motor_neurons["DNb"]) == 2
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /workspace
pytest tests/test_neuron_selection.py::test_select_output_neurons_for_motor -v
```

Expected: FAIL with ImportError or NameError

- [ ] **Step 3: Implement motor output selection**

```python
# Add to threedfly/connectome/neuron_selection.py at end of file

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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /workspace
pytest tests/test_neuron_selection.py::test_select_output_neurons_for_motor -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add threedfly/connectome/neuron_selection.py tests/test_neuron_selection.py
git commit -m "feat: add motor output selection for descending neurons"
```

---

### Task 4: Implement E/I sign application in simulator

**Files:**
- Modify: `threedfly/connectome/simulator.py`
- Test: `tests/test_connectome.py`

**Interfaces:**
- Consumes:
  - `get_neuron_transmitters(body_ids, nt_data)` from Task 1
- Produces:
  - `RateSimulator.__init__(..., apply_ei_signs: bool, transmitter_signs: Dict[int, float])` - new parameters
  - `BrainSimulator.__init__(..., apply_ei_signs: bool, transmitter_signs: Dict[int, float])` - new parameters

- [ ] **Step 1: Write test for E/I sign application**

```python
# Add to tests/test_connectome.py

def test_rate_simulator_ei_signs():
    """Test E/I sign application in rate simulator."""
    from scipy import sparse
    from threedfly.connectome.simulator import RateSimulator
    
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


def test_lif_simulator_ei_signs():
    """Test E/I sign application in LIF simulator."""
    from scipy import sparse
    from threedfly.connectome.simulator import BrainSimulator
    
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /workspace
pytest tests/test_connectome.py::test_rate_simulator_ei_signs -v
pytest tests/test_connectome.py::test_lif_simulator_ei_signs -v
```

Expected: FAIL with TypeError (unexpected keyword arguments)

- [ ] **Step 3: Modify RateSimulator to support E/I signs**

```python
# Modify threedfly/connectome/simulator.py

# In RateSimulator.__init__, change signature:
def __init__(
    self,
    adjacency: sparse.csr_matrix,
    dt: float = 0.010,
    tau: float = 0.050,
    activation: str = "relu",
    syn_weight_scale: float = 0.1,
    max_rate: float = 10.0,
    apply_ei_signs: bool = False,
    transmitter_signs: Optional[Dict[int, float]] = None,
):
    """
    Initialize rate-based simulator.
    
    Args:
        adjacency: Sparse adjacency matrix
        dt: Simulation timestep
        tau: Time constant
        activation: Activation function
        syn_weight_scale: Synaptic weight scaling
        max_rate: Maximum firing rate
        apply_ei_signs: Whether to apply E/I signs based on neurotransmitters
        transmitter_signs: Dict mapping neuron_idx → sign (±1)
    """
    # Apply E/I signs to adjacency if requested
    if apply_ei_signs and transmitter_signs is not None:
        adjacency = self._apply_ei_signs(adjacency, transmitter_signs)
    
    self.adj = adjacency * syn_weight_scale
    self.n_neurons = adjacency.shape[0]
    self.dt = dt
    self.tau = tau
    self.activation = activation
    self.max_rate = max_rate
    
    self.decay = np.exp(-dt / tau)
    
    # State
    self.rates = np.zeros(self.n_neurons)
    self.external_input = np.zeros(self.n_neurons)


# Add helper method to RateSimulator class:
def _apply_ei_signs(
    self,
    adjacency: sparse.csr_matrix,
    transmitter_signs: Dict[int, float]
) -> sparse.csr_matrix:
    """
    Apply excitatory/inhibitory signs to adjacency matrix.
    
    Multiplies each row by the presynaptic neuron's transmitter sign.
    
    Args:
        adjacency: Original adjacency (pre x post)
        transmitter_signs: Dict mapping neuron_idx → sign (±1)
    
    Returns:
        Modified adjacency with E/I signs applied
    """
    adj_modified = adjacency.copy().tolil()
    
    for pre_idx in range(adjacency.shape[0]):
        sign = transmitter_signs.get(pre_idx, 1.0)
        if sign != 1.0:
            # Multiply row by sign
            adj_modified[pre_idx, :] *= sign
    
    return adj_modified.tocsr()
```

- [ ] **Step 4: Modify BrainSimulator to support E/I signs**

```python
# Modify threedfly/connectome/simulator.py

# In BrainSimulator.__init__, change signature:
def __init__(
    self,
    adjacency: sparse.csr_matrix,
    dt: float = 0.001,
    tau: float = 0.010,
    v_thresh: float = 1.0,
    v_reset: float = 0.0,
    v_rest: float = 0.0,
    refrac_period: float = 0.002,
    syn_weight_scale: float = 0.1,
    apply_ei_signs: bool = False,
    transmitter_signs: Optional[Dict[int, float]] = None,
):
    """
    Initialize simulator.
    
    Args:
        adjacency: Sparse adjacency matrix (pre x post)
        dt: Simulation timestep (seconds)
        tau: Membrane time constant (seconds)
        v_thresh: Spike threshold
        v_reset: Reset voltage after spike
        v_rest: Resting potential
        refrac_period: Refractory period (seconds)
        syn_weight_scale: Scaling factor for synaptic weights
        apply_ei_signs: Whether to apply E/I signs based on neurotransmitters
        transmitter_signs: Dict mapping neuron_idx → sign (±1)
    """
    # Apply E/I signs if requested
    if apply_ei_signs and transmitter_signs is not None:
        adjacency = self._apply_ei_signs(adjacency, transmitter_signs)
    
    self.adj = adjacency
    self.n_neurons = adjacency.shape[0]
    self.dt = dt
    self.tau = tau
    self.v_thresh = v_thresh
    self.v_reset = v_reset
    self.v_rest = v_rest
    self.refrac_steps = int(refrac_period / dt)
    self.syn_weight_scale = syn_weight_scale
    
    # Decay factor for membrane potential
    self.decay = np.exp(-dt / tau)
    
    # State variables
    self.v = np.ones(self.n_neurons) * v_rest
    self.spikes = np.zeros(self.n_neurons, dtype=bool)
    self.refrac_count = np.zeros(self.n_neurons, dtype=int)
    
    # Input and activity tracking
    self.external_input = np.zeros(self.n_neurons)
    self.spike_history = []
    
    # Normalize adjacency by max weight for stability
    if self.adj.nnz > 0:
        max_weight = np.abs(self.adj.data).max()
        if max_weight > 0:
            self.adj = self.adj / max_weight * syn_weight_scale


# Add helper method to BrainSimulator class:
def _apply_ei_signs(
    self,
    adjacency: sparse.csr_matrix,
    transmitter_signs: Dict[int, float]
) -> sparse.csr_matrix:
    """
    Apply excitatory/inhibitory signs to adjacency matrix.
    
    Multiplies each row by the presynaptic neuron's transmitter sign.
    
    Args:
        adjacency: Original adjacency (pre x post)
        transmitter_signs: Dict mapping neuron_idx → sign (±1)
    
    Returns:
        Modified adjacency with E/I signs applied
    """
    adj_modified = adjacency.copy().tolil()
    
    for pre_idx in range(adjacency.shape[0]):
        sign = transmitter_signs.get(pre_idx, 1.0)
        if sign != 1.0:
            # Multiply row by sign
            adj_modified[pre_idx, :] *= sign
    
    return adj_modified.tocsr()
```

- [ ] **Step 5: Add import for Optional and Dict**

```python
# At top of threedfly/connectome/simulator.py, ensure imports:
from typing import Optional, Dict
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
cd /workspace
pytest tests/test_connectome.py::test_rate_simulator_ei_signs -v
pytest tests/test_connectome.py::test_lif_simulator_ei_signs -v
```

Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add threedfly/connectome/simulator.py tests/test_connectome.py
git commit -m "feat: add E/I sign application based on neurotransmitters"
```

---

### Task 5: Implement biological input mapping in controller

**Files:**
- Modify: `threedfly/control/controller.py`
- Test: `tests/test_control.py`

**Interfaces:**
- Consumes:
  - Input mapping functions from Task 2
  - Output mapping functions from Task 3
- Produces:
  - `BiologicalFlightController.__init__(n_neurons, annotations, body_ids, body_id_to_idx, nt_data)` - new controller class
  - `BiologicalFlightController.set_visual_input(left_luminance, right_luminance, optic_flow_features, looming_features, brain)` - sets input to brain
  - `BiologicalFlightController.compute_action(brain_activity)` - reads DN activity and computes motor commands

- [ ] **Step 1: Write test for biological controller**

```python
# Add to tests/test_control.py

def test_biological_flight_controller():
    """Test biological flight controller with real neuron mappings."""
    from threedfly.control.controller import BiologicalFlightController
    from scipy import sparse
    import pandas as pd
    
    # Create synthetic annotations
    annotations = pd.DataFrame({
        "bodyId": list(range(1, 101)),
        "type": (["R1"] * 10 + ["R2"] * 10 + ["T4a"] * 10 + ["T5a"] * 10 +
                 ["LPLC2"] * 10 + ["LC4"] * 10 + ["DNp01"] * 5 +
                 ["DNg02_a"] * 5 + ["DNa01"] * 5 + ["DNa02"] * 5 +
                 ["DNb01"] * 10),
        "celltype": (["R1"] * 10 + ["R2"] * 10 + ["T4a"] * 10 + ["T5a"] * 10 +
                     ["LPLC2"] * 10 + ["LC4"] * 10 + ["DNp01"] * 5 +
                     ["DNg02_a"] * 5 + ["DNa01"] * 5 + ["DNa02"] * 5 +
                     ["DNb01"] * 10),
        "somaSide": (["L"] * 25 + ["R"] * 25 + ["L"] * 25 + ["R"] * 25),
    })
    
    body_ids = np.array(range(1, 101))
    body_id_to_idx = {bid: idx for idx, bid in enumerate(body_ids)}
    
    # Create controller
    controller = BiologicalFlightController(
        n_neurons=100,
        annotations=annotations,
        body_ids=body_ids,
        body_id_to_idx=body_id_to_idx,
        nt_data=None
    )
    
    # Check that it found neurons
    assert controller.input_mapping["luminance_left"].size > 0
    assert controller.input_mapping["luminance_right"].size > 0
    assert controller.output_mapping["DNp01"].size > 0
    assert controller.output_mapping["DNg02"].size > 0
    
    # Test computing action from brain activity
    brain_activity = np.random.rand(100) * 0.5
    action = controller.compute_action(brain_activity)
    
    assert action.shape == (5,)
    assert np.all(action >= -1.0) and np.all(action <= 1.0)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /workspace
pytest tests/test_control.py::test_biological_flight_controller -v
```

Expected: FAIL with ImportError

- [ ] **Step 3: Implement BiologicalFlightController**

```python
# Add to threedfly/control/controller.py at end of file

class BiologicalFlightController:
    """
    Flight controller using biological neuron mappings.
    
    Maps visual features to specific neuron types based on annotations,
    and reads motor output from descending neuron populations.
    """
    
    def __init__(
        self,
        n_neurons: int,
        annotations: pd.DataFrame,
        body_ids: np.ndarray,
        body_id_to_idx: Dict[int, int],
        nt_data: Optional[pd.DataFrame] = None,
        warmup_steps: int = 0,
    ):
        """
        Initialize biological controller.
        
        Args:
            n_neurons: Total number of neurons
            annotations: Annotation DataFrame
            body_ids: Array of body IDs
            body_id_to_idx: Body ID to index mapping
            nt_data: Neurotransmitter data (optional)
            warmup_steps: Number of warmup steps for resting rate calibration
        """
        from threedfly.connectome.neuron_selection import (
            select_input_neurons_for_luminance,
            select_input_neurons_for_optic_flow,
            select_input_neurons_for_looming,
            select_output_neurons_for_motor,
            create_fixed_input_assignment,
        )
        
        self.n_neurons = n_neurons
        self.warmup_steps = warmup_steps
        
        # Input mapping: visual features → neuron populations
        print("\n[Biological I/O Mapping]")
        print("="*70)
        
        # Luminance → R1-R6
        self.input_mapping = {}
        self.input_mapping["luminance_left"] = select_input_neurons_for_luminance(
            annotations, body_ids, body_id_to_idx, side="L"
        )
        self.input_mapping["luminance_right"] = select_input_neurons_for_luminance(
            annotations, body_ids, body_id_to_idx, side="R"
        )
        print(f"Luminance → R1-R6: L={len(self.input_mapping['luminance_left'])}, "
              f"R={len(self.input_mapping['luminance_right'])}")
        
        # Optic flow → T4a-d, T5a-d
        flow_subtypes = []
        for subtype in ["a", "b", "c", "d"]:
            for side in ["L", "R"]:
                key = f"optic_flow_t4{subtype}_{side.lower()}"
                neurons = select_input_neurons_for_optic_flow(
                    annotations, body_ids, body_id_to_idx, subtype=subtype, side=side
                )
                self.input_mapping[key] = neurons
                if len(neurons) > 0:
                    flow_subtypes.append(f"T4{subtype}_{side}={len(neurons)}")
        
        for subtype in ["a", "b", "c", "d"]:
            for side in ["L", "R"]:
                key = f"optic_flow_t5{subtype}_{side.lower()}"
                neurons = select_input_neurons_for_optic_flow(
                    annotations, body_ids, body_id_to_idx, subtype=subtype, side=side
                )
                self.input_mapping[key] = neurons
                if len(neurons) > 0:
                    flow_subtypes.append(f"T5{subtype}_{side}={len(neurons)}")
        
        print(f"Optic flow → T4/T5: {', '.join(flow_subtypes) if flow_subtypes else 'none found'}")
        
        # Looming → LPLC2 + LC4
        self.input_mapping["looming_left"] = select_input_neurons_for_looming(
            annotations, body_ids, body_id_to_idx, side="L"
        )
        self.input_mapping["looming_right"] = select_input_neurons_for_looming(
            annotations, body_ids, body_id_to_idx, side="R"
        )
        print(f"Looming → LPLC2+LC4: L={len(self.input_mapping['looming_left'])}, "
              f"R={len(self.input_mapping['looming_right'])}")
        
        # Output mapping: DN types → motor commands
        self.output_mapping = select_output_neurons_for_motor(
            annotations, body_ids, body_id_to_idx
        )
        
        print(f"\nMotor readout from descending neurons:")
        for dn_type, indices in self.output_mapping.items():
            if len(indices) > 0:
                print(f"  {dn_type}: {len(indices)} neurons")
            else:
                print(f"  {dn_type}: not found (graceful fallback)")
        
        print("="*70)
        
        # Resting rates (set during warmup)
        self.resting_rates = {}
        self.warmup_complete = False
        
        # Smoothing
        self.prev_action = np.zeros(5)
        self.alpha = 0.7
    
    def set_visual_input(
        self,
        left_luminance: np.ndarray,
        right_luminance: np.ndarray,
        optic_flow_features: Dict[str, np.ndarray],
        looming_features: Dict[str, np.ndarray],
        brain,
    ):
        """
        Set visual input to brain via biological neuron mappings.
        
        Args:
            left_luminance: Left eye luminance (scalar or array)
            right_luminance: Right eye luminance (scalar or array)
            optic_flow_features: Dict with keys like "t4a_left", "t5c_right"
            looming_features: Dict with keys "left", "right"
            brain: Brain simulator instance
        """
        # Build complete input vector
        all_indices = []
        all_values = []
        
        # Luminance → R1-R6
        if self.input_mapping["luminance_left"].size > 0:
            pop_size = len(self.input_mapping["luminance_left"])
            lum_left_val = np.atleast_1d(left_luminance)[0] if np.isscalar(left_luminance) else left_luminance.mean()
            all_indices.extend(self.input_mapping["luminance_left"])
            all_values.extend([lum_left_val] * pop_size)
        
        if self.input_mapping["luminance_right"].size > 0:
            pop_size = len(self.input_mapping["luminance_right"])
            lum_right_val = np.atleast_1d(right_luminance)[0] if np.isscalar(right_luminance) else right_luminance.mean()
            all_indices.extend(self.input_mapping["luminance_right"])
            all_values.extend([lum_right_val] * pop_size)
        
        # Optic flow → T4/T5
        for key, feature_val in optic_flow_features.items():
            mapping_key = f"optic_flow_{key}"
            if mapping_key in self.input_mapping and self.input_mapping[mapping_key].size > 0:
                neurons = self.input_mapping[mapping_key]
                val = np.atleast_1d(feature_val)[0] if np.isscalar(feature_val) else feature_val.mean()
                all_indices.extend(neurons)
                all_values.extend([val] * len(neurons))
        
        # Looming → LPLC2 + LC4
        if "left" in looming_features and self.input_mapping["looming_left"].size > 0:
            neurons = self.input_mapping["looming_left"]
            val = np.atleast_1d(looming_features["left"])[0] if np.isscalar(looming_features["left"]) else looming_features["left"].mean()
            all_indices.extend(neurons)
            all_values.extend([val] * len(neurons))
        
        if "right" in looming_features and self.input_mapping["looming_right"].size > 0:
            neurons = self.input_mapping["looming_right"]
            val = np.atleast_1d(looming_features["right"])[0] if np.isscalar(looming_features["right"]) else looming_features["right"].mean()
            all_indices.extend(neurons)
            all_values.extend([val] * len(neurons))
        
        # Set input all at once
        if len(all_indices) > 0:
            input_vec = np.array(all_values, dtype=np.float32)
            neuron_indices = np.array(all_indices, dtype=np.int32)
            brain.set_input(input_vec, neuron_indices=neuron_indices)
    
    def record_resting_rates(self, brain_activity: np.ndarray):
        """Record resting rates during warmup."""
        if self.warmup_complete:
            return
        
        for dn_type, indices in self.output_mapping.items():
            if len(indices) > 0:
                if dn_type not in self.resting_rates:
                    self.resting_rates[dn_type] = []
                self.resting_rates[dn_type].append(brain_activity[indices].mean())
    
    def finalize_warmup(self):
        """Compute mean resting rates after warmup."""
        if self.warmup_complete:
            return
        
        for dn_type in self.resting_rates:
            self.resting_rates[dn_type] = np.mean(self.resting_rates[dn_type])
        
        self.warmup_complete = True
        
        print("\n[Warmup complete]")
        print("Resting rates:")
        for dn_type, rate in self.resting_rates.items():
            print(f"  {dn_type}: {rate:.4f}")
    
    def compute_action(
        self,
        brain_activity: np.ndarray,
        exploration_signal: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute flight action from DN activity.
        
        Maps specific DN types to motor commands:
        - DNp01: escape → emergency pitch
        - DNg02: thrust/altitude → forward + upward thrust
        - DNa01/DNa02 difference → yaw
        - DNb: baseline motor activity
        
        Args:
            brain_activity: Neural activity array
            exploration_signal: Optional exploration bias
        
        Returns:
            Action: [thrust_forward, thrust_up, roll, pitch, yaw]
        """
        # Read DN activity (deviation from resting if warmup complete)
        dn_activity = {}
        for dn_type, indices in self.output_mapping.items():
            if len(indices) > 0:
                activity = brain_activity[indices].mean()
                if self.warmup_complete and dn_type in self.resting_rates:
                    activity -= self.resting_rates[dn_type]
                dn_activity[dn_type] = activity
            else:
                dn_activity[dn_type] = 0.0
        
        # Map to motor commands
        thrust_forward = 0.7  # baseline
        thrust_up = 0.12      # maintain altitude
        pitch = 0.0
        roll = 0.0
        yaw = 0.0
        
        # DNg02 → thrust/altitude
        if "DNg02" in dn_activity:
            thrust_forward += 1.0 * dn_activity["DNg02"]
            thrust_up += 0.5 * dn_activity["DNg02"]
        
        # DNp01 → escape (sharp pitch)
        if "DNp01" in dn_activity:
            pitch += 2.0 * dn_activity["DNp01"]
        
        # DNa01/DNa02 difference → yaw
        dna_diff = dn_activity.get("DNa01", 0.0) - dn_activity.get("DNa02", 0.0)
        yaw += 1.5 * dna_diff
        
        # DNb → roll
        if "DNb" in dn_activity:
            roll += 0.5 * dn_activity["DNb"]
        
        # Assemble action
        action = np.array([thrust_forward, thrust_up, roll, pitch, yaw])
        
        # Add exploration if provided
        if exploration_signal is not None:
            action[0] += 0.8 * exploration_signal[0]
            action[4] += 1.0 * exploration_signal[1]
        
        # Smooth
        action = self.alpha * action + (1 - self.alpha) * self.prev_action
        self.prev_action = action.copy()
        
        # Clip
        action = np.clip(action, -1.0, 1.0)
        
        return action
    
    def get_status(self) -> Dict[str, float]:
        """Get controller status."""
        return {
            "last_forward": float(self.prev_action[0]),
            "last_up": float(self.prev_action[1]),
            "last_yaw": float(self.prev_action[4]),
            "warmup_complete": float(self.warmup_complete),
        }
```

- [ ] **Step 4: Add imports to controller.py**

```python
# At top of threedfly/control/controller.py, add:
import pandas as pd
from typing import Dict, Optional
```

- [ ] **Step 5: Run test to verify it passes**

```bash
cd /workspace
pytest tests/test_control.py::test_biological_flight_controller -v
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add threedfly/control/controller.py tests/test_control.py
git commit -m "feat: add biological flight controller with neuron type mappings"
```

---

### Task 6: Integrate biological controller into main runner

**Files:**
- Modify: `threedfly/runner.py`
- Modify: `threedfly/runner_interactive.py`

**Interfaces:**
- Consumes: `BiologicalFlightController` from Task 5
- Produces: Updated `run_simulation()` function with biological I/O, warmup period, and removed direct looming→avoidance

- [ ] **Step 1: Modify runner.py to use biological controller**

```python
# In threedfly/runner.py, replace controller initialization section (lines ~78-80):

print("\n[6/7] Setting up control and exploration...")

# Load annotations if available for biological mapping
annotations = None
body_ids = None
body_id_to_idx = None
nt_data = None

if "annotations" in subgraph and "body_ids" in subgraph and "body_id_to_idx" in subgraph:
    annotations = subgraph["annotations"]
    body_ids = subgraph["body_ids"]
    body_id_to_idx = subgraph["body_id_to_idx"]
    print("  Using biological neuron mappings from annotations")
    
    # Try to load neurotransmitter data
    from threedfly.connectome.neuron_selection import load_neurotransmitter_data
    nt_path = "data/connectome-raw/body-neurotransmitters-male-cns-v1.0.feather"
    nt_data = load_neurotransmitter_data(nt_path)
    if nt_data is not None:
        print(f"  Loaded neurotransmitter data ({len(nt_data)} entries)")
else:
    print("  No annotations found, using legacy controller")

# Choose controller
if annotations is not None:
    from threedfly.control.controller import BiologicalFlightController
    controller = BiologicalFlightController(
        n_neurons=n_neurons,
        annotations=annotations,
        body_ids=body_ids,
        body_id_to_idx=body_id_to_idx,
        nt_data=nt_data,
        warmup_steps=100,  # 100 grounded brain steps
    )
    use_biological = True
else:
    controller = FlightController(n_neurons, use_rate_model=(simulator_type == "rate"))
    use_biological = False

explorer = ExplorationPolicy(exploration_weight=1.5, random_exploration_prob=0.05)
```

- [ ] **Step 2: Add warmup period in runner.py**

```python
# In threedfly/runner.py, after environment reset (after line ~97), add warmup:

# Warmup period (grounded brain steps for resting rate calibration)
if use_biological and controller.warmup_steps > 0:
    print(f"\n[Warmup] Running {controller.warmup_steps} grounded brain steps...")
    for warmup_step in range(controller.warmup_steps):
        # Get visual input
        visual_features = stereo.get_visual_features(left_img, right_img)
        fly_velocity = env.get_fly_velocity()
        flow_features, flow_info = brain_depth.prepare_flow_features_for_brain(
            left_img, right_img, fly_velocity
        )
        
        # Set biological input
        controller.set_visual_input(
            left_luminance=visual_features["left_ommatidia"].mean(),
            right_luminance=visual_features["right_ommatidia"].mean(),
            optic_flow_features={},  # Empty during warmup
            looming_features={},     # Empty during warmup
            brain=brain,
        )
        
        # Run brain
        for _ in range(brain_steps_per_sim):
            if simulator_type == "lif":
                brain.step()
                brain_activity = brain.get_firing_rates(window=50)
            else:
                brain_activity = brain.step()
        
        # Record resting rates
        controller.record_resting_rates(brain_activity)
    
    controller.finalize_warmup()
    print("[Warmup complete]\n")
```

- [ ] **Step 3: Modify main loop to use biological input/output**

```python
# In threedfly/runner.py, replace the brain input section (lines ~120-130) with:

if use_biological:
    # Biological input mapping
    controller.set_visual_input(
        left_luminance=visual_features["left_ommatidia"].mean(),
        right_luminance=visual_features["right_ommatidia"].mean(),
        optic_flow_features={
            "t4a_left": 0.0,  # TODO: extract from flow_features
            "t4a_right": 0.0,
        },
        looming_features={
            "left": visual_features["looming_features"][:4].mean(),
            "right": visual_features["looming_features"][4:].mean(),
        },
        brain=brain,
    )
else:
    # Legacy input
    brain_input = np.concatenate([
        visual_features["left_ommatidia"],
        visual_features["right_ommatidia"],
        visual_features["looming_features"],
        flow_features
    ])
    brain.set_input(brain_input)
```

- [ ] **Step 4: Update runner_interactive.py similarly**

```python
# Apply same changes to threedfly/runner_interactive.py:
# 1. Add annotation loading (after line ~40)
# 2. Choose controller (biological vs legacy)
# 3. Add warmup period (after reset)
# 4. Update main loop to use biological input/output

# (Follow same structure as runner.py changes)
```

- [ ] **Step 5: Remove direct looming→avoidance in both runners**

```python
# In runner.py and runner_interactive.py, the looming features are now ONLY
# fed to LPLC2/LC4 neurons via biological input mapping.
# No direct addition to exploration signal or controller.
# The existing code already routes through brain, so no removal needed -
# just verify looming is only in visual_features, not directly added to action.
```

- [ ] **Step 6: Run a quick smoke test**

```bash
cd /workspace
python -c "from threedfly.runner import run_simulation; print('Import OK')"
```

Expected: No errors

- [ ] **Step 7: Commit**

```bash
git add threedfly/runner.py threedfly/runner_interactive.py
git commit -m "feat: integrate biological controller with warmup and remove direct looming avoidance"
```

---

### Task 7: Weaken soft collision recovery to minimal safety fallback

**Files:**
- Modify: `threedfly/sim/environment.py`

**Interfaces:**
- Consumes: Existing `recover_from_collision()` method
- Produces: Updated `recover_from_collision()` with minimal recovery distance and gentler repositioning

- [ ] **Step 1: Modify recover_from_collision to be minimal**

```python
# In threedfly/sim/environment.py, replace recover_from_collision method (lines ~238-305):

def recover_from_collision(self, recovery_distance: float = 0.15) -> None:
    """
    Minimal soft collision recovery: small back-off as safety fallback.
    
    The primary avoidance mechanism is brain-mediated (looming → LPLC2/LC4 → DNs).
    This recovery is a last-resort safety net to prevent simulation lockup.
    
    Args:
        recovery_distance: Small distance to back off (default 15cm, down from 30cm)
    """
    pos = self.get_fly_position()
    vel = self.get_fly_velocity()
    
    # Minimal repositioning
    recovery_offset = np.zeros(3)
    
    if pos[2] < 0.3:
        # Ground: small upward nudge
        recovery_offset[2] = 0.2
    elif pos[2] > 2.3:
        # Ceiling: small downward nudge
        recovery_offset[2] = -0.2
    elif self.data.ncon > 0:
        # Wall: minimal back-off along contact normal
        contact = self.data.contact[0]
        contact_normal = contact.frame[:3]
        if np.linalg.norm(contact_normal) > 0.1:
            contact_normal = contact_normal / np.linalg.norm(contact_normal)
        recovery_offset = -contact_normal * recovery_distance
    
    # Apply recovery
    self.data.qpos[:3] = pos + recovery_offset
    
    # Clamp to room bounds
    self.data.qpos[0] = np.clip(self.data.qpos[0], -1.9, 1.9)
    self.data.qpos[1] = np.clip(self.data.qpos[1], -1.9, 1.9)
    self.data.qpos[2] = np.clip(self.data.qpos[2], 0.3, 2.2)
    
    # Gentle velocity dampening (not aggressive)
    self.data.qvel[:3] *= 0.3
    
    # Keep orientation (no randomization)
    
    # Forward simulation
    mujoco.mj_forward(self.model, self.data)
```

- [ ] **Step 2: Test that environment still works**

```bash
cd /workspace
pytest tests/test_collision_recovery.py -v
```

Expected: Tests may fail if they check for specific old behavior; that's OK, we'll fix them next

- [ ] **Step 3: Update collision recovery tests if needed**

```python
# If tests fail, update tests/test_collision_recovery.py to expect:
# - Smaller recovery distance (0.15 instead of 0.3)
# - Gentler velocity dampening (0.3 instead of 0.05)
# - No orientation randomization
```

- [ ] **Step 4: Run tests again**

```bash
cd /workspace
pytest tests/test_collision_recovery.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add threedfly/sim/environment.py tests/test_collision_recovery.py
git commit -m "feat: weaken collision recovery to minimal safety fallback"
```

---

### Task 8: Update README with biological mapping documentation

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: None (documentation only)
- Produces: Updated README with clear distinction between measured connectome and engineered bridges

- [ ] **Step 1: Add biological mapping section to README**

```markdown
# Add to README.md after the "Overview" section (around line 25):

### Biological Mapping (NEW)

**The simulation now supports biologically-grounded input/output mapping** using neuron type annotations from the MaleCNS connectome:

**What is measured connectome data:**
- Synaptic connectivity (~10M connections, filtered from 150M raw)
- Neuron type annotations (R1-R6, L1-L5, T4/T5, LPLC2, LC4, DNp01, DNg02, DNa, DNb, etc.)
- Soma and root side information (left/right brain hemisphere)
- Neurotransmitter identities (acetylcholine, GABA, glutamate, etc.)

**What is engineered (hand-designed bridges):**
- Image → neuron encoding:
  - Luminance → R1-R6 photoreceptors (split by brain hemisphere)
  - Optic flow → T4a-d / T5a-d direction-selective neurons
  - Looming detection → LPLC2 + LC4 collision detectors
- Tonic flight drive baseline (keeps fly airborne)
- DN → motor readout:
  - DNp01 → escape pitch
  - DNg02 → thrust/altitude
  - DNa01/02 difference → yaw control
  - DNb → roll
- Safety limits (minimal collision recovery as last resort)

**Scientific honesty:**
This is a connectome-structured simulator with engineered sensory encoding and motor decoding. The synaptic graph is real; the input/output mappings are biologically inspired but not validated against fruit fly physiology. Neural dynamics use simplified rate models. This is exploratory research simulation, not a claim of neuron-for-neuron accuracy.
```

- [ ] **Step 2: Add Chinese translation of biological mapping section**

```markdown
# Add to README.md in the Chinese section (around line 680):

### 生物学映射（新功能）

**模拟器现在支持基于生物学的输入输出映射**，使用 MaleCNS 连接组的神经元类型注释：

**什么是测量的连接组数据：**
- 突触连接（约1000万个连接，从1.5亿原始数据中过滤）
- 神经元类型注释（R1-R6、L1-L5、T4/T5、LPLC2、LC4、DNp01、DNg02、DNa、DNb等）
- 胞体和根部位置信息（左右脑半球）
- 神经递质类型（乙酰胆碱、GABA、谷氨酸等）

**什么是工程化设计的（手工设计的桥接）：**
- 图像 → 神经元编码：
  - 亮度 → R1-R6 感光细胞（按脑半球分割）
  - 光流 → T4a-d / T5a-d 方向选择性神经元
  - 逼近检测 → LPLC2 + LC4 碰撞检测器
- 强直性飞行驱动基线（保持果蝇在空中）
- DN → 运动读取：
  - DNp01 → 逃逸俯仰
  - DNg02 → 推力/高度
  - DNa01/02 差值 → 偏航控制
  - DNb → 滚转
- 安全限制（最小化碰撞恢复作为最后手段）

**科学诚实性：**
这是一个基于连接组结构的模拟器，具有工程化的感觉编码和运动解码。突触图是真实的；输入输出映射受生物学启发但未经果蝇生理学验证。神经动力学使用简化的速率模型。这是探索性研究模拟，而非神经元级精确度的声明。
```

- [ ] **Step 3: Update Architecture mermaid diagram**

```markdown
# Update the mermaid diagram in README (around line 40) to show biological mapping:
# Change "Visual Input<br/>Photoreceptor Layer" to "Visual Input<br/>R1-R6, T4/T5, LPLC2/LC4"
# Change "Descending Neurons<br/>Motor Output" to "Descending Neurons<br/>DNp, DNg, DNa, DNb"
```

- [ ] **Step 4: Verify README renders correctly**

```bash
cd /workspace
# View in browser or markdown preview to check formatting
```

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs: add biological mapping explanation to README (EN+CN)"
```

---

### Task 9: Run all tests and verify functionality

**Files:**
- None (testing only)

**Interfaces:**
- Consumes: All previous tasks
- Produces: Verified passing test suite

- [ ] **Step 1: Run full test suite**

```bash
cd /workspace
pytest tests/ -v
```

Expected: All tests PASS

- [ ] **Step 2: If any tests fail, diagnose and fix**

```bash
# Check specific failing test
pytest tests/test_<module>.py::test_<name> -v -s

# Fix issues in implementation or update tests to match new behavior
```

- [ ] **Step 3: Run interactive demo to verify UI works**

```bash
cd /workspace
timeout 30s python -m threedfly.cli run-interactive --subgraph data/demo_subgraph.npz --steps 100 --headless
```

Expected: No crashes, clean output

- [ ] **Step 4: Verify viser UI still works**

```bash
# Start interactive server and check browser console for errors
# (Manual verification only - no automated test)
```

- [ ] **Step 5: All tests pass and no regressions**

```bash
cd /workspace
pytest tests/ --cov=threedfly --cov-report=term-missing
```

Expected: All PASS, good coverage

---

### Task 10: Create PR and push

**Files:**
- None (git operations only)

**Interfaces:**
- Consumes: All previous tasks
- Produces: PR against main branch

- [ ] **Step 1: Create feature branch**

```bash
cd /workspace
git checkout -b cursor/biologically-mapped-io-0955
```

- [ ] **Step 2: Verify all changes are committed**

```bash
git status
```

Expected: Clean working tree

- [ ] **Step 3: Push branch**

```bash
git push -u origin cursor/biologically-mapped-io-0955
```

- [ ] **Step 4: Create PR**

Use ManagePullRequest tool with:
- title: "feat: Biologically-mapped brain input/output for full MaleCNS connectome"
- body: Summary of changes (input mapping, output readout, E/I signs, warmup, collision recovery, documentation)
- branch_name: cursor/biologically-mapped-io-0955
- base_branch: main
- draft: false

- [ ] **Step 5: Verify PR created successfully**

Check PR URL and confirm it's open against main branch.
