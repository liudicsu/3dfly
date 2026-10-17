"""Tests for neuron selection by type."""

import numpy as np
import pandas as pd
import pytest

from threedfly.connectome.neuron_selection import (
    find_neurons_by_type,
    find_visual_neurons,
    find_descending_neurons,
    find_central_complex_neurons,
    get_neuron_selection_summary,
    load_neurotransmitter_data,
    get_neuron_transmitters,
)


def create_synthetic_nt_data():
    """Create synthetic neurotransmitter data for testing."""
    return pd.DataFrame({
        "body": [1, 2, 3, 4, 5, 6, 7, 8],
        "consensus_nt": ["acetylcholine", "GABA", "glutamate", "histamine", 
                         "dopamine", "octopamine", "serotonin", "unclear"],
        "celltype_predicted_nt": ["acetylcholine", "GABA", "glutamate", "histamine",
                                  "dopamine", "octopamine", "serotonin", "acetylcholine"],
    })


def test_find_neurons_by_type():
    """Test basic neuron type search."""
    # Create synthetic annotations
    annotations = pd.DataFrame({
        "bodyId": [1, 2, 3, 4, 5],
        "type": ["photoreceptor_R1", "lamina_L1", "DN", "medulla_Mi1", "LC4"],
        "class": ["visual", "visual", "motor", "visual", "visual"],
    })
    
    body_ids = np.array([1, 2, 3, 4, 5])
    body_id_to_idx = {bid: idx for idx, bid in enumerate(body_ids)}
    
    # Search for photoreceptors
    indices = find_neurons_by_type(
        annotations, body_ids, body_id_to_idx, {"photoreceptor"}
    )
    assert len(indices) == 1
    assert indices[0] == 0  # Body ID 1
    
    # Search for visual (should find 4)
    indices = find_neurons_by_type(
        annotations, body_ids, body_id_to_idx, {"visual"}
    )
    assert len(indices) == 4
    
    # Search for DN
    indices = find_neurons_by_type(
        annotations, body_ids, body_id_to_idx, {"DN"}
    )
    assert len(indices) == 1
    assert indices[0] == 2


def test_find_visual_neurons():
    """Test visual neuron selection."""
    annotations = pd.DataFrame({
        "bodyId": [1, 2, 3, 4, 5, 6],
        "type": ["R1", "L1", "Mi1", "Tm1", "LC4", "other"],
    })
    
    body_ids = np.array([1, 2, 3, 4, 5, 6])
    body_id_to_idx = {bid: idx for idx, bid in enumerate(body_ids)}
    
    visual = find_visual_neurons(annotations, body_ids, body_id_to_idx)
    
    assert "photoreceptor" in visual
    assert "lamina" in visual
    assert "medulla" in visual
    assert "lobula" in visual
    assert "all_visual" in visual
    
    # Check counts
    assert len(visual["photoreceptor"]) >= 1  # R1
    assert len(visual["lamina"]) >= 1  # L1
    assert len(visual["medulla"]) >= 2  # Mi1, Tm1
    assert len(visual["lobula"]) >= 1  # LC4
    assert len(visual["all_visual"]) >= 5  # All except "other"


def test_find_descending_neurons():
    """Test descending neuron selection."""
    annotations = pd.DataFrame({
        "bodyId": [1, 2, 3, 4, 5],
        "type": ["DN", "DNa02", "DNb01", "DNp01", "other"],
    })
    
    body_ids = np.array([1, 2, 3, 4, 5])
    body_id_to_idx = {bid: idx for idx, bid in enumerate(body_ids)}
    
    descending = find_descending_neurons(annotations, body_ids, body_id_to_idx)
    
    assert "DN" in descending
    assert "DNa" in descending
    assert "DNb" in descending
    assert "DNp" in descending
    
    # All should be found
    assert len(descending["DN"]) >= 1
    assert len(descending["DNa"]) >= 1
    assert len(descending["DNb"]) >= 1
    assert len(descending["DNp"]) >= 1


def test_find_central_complex_neurons():
    """Test central complex neuron selection."""
    annotations = pd.DataFrame({
        "bodyId": [1, 2, 3, 4, 5],
        "type": ["FB_neuron", "EB_neuron", "PB_neuron", "NO_neuron", "other"],
    })
    
    body_ids = np.array([1, 2, 3, 4, 5])
    body_id_to_idx = {bid: idx for idx, bid in enumerate(body_ids)}
    
    cx = find_central_complex_neurons(annotations, body_ids, body_id_to_idx)
    
    assert "FB" in cx
    assert "EB" in cx
    assert "PB" in cx
    assert "NO" in cx
    assert "all_central_complex" in cx
    
    assert len(cx["FB"]) >= 1
    assert len(cx["EB"]) >= 1
    assert len(cx["PB"]) >= 1
    assert len(cx["NO"]) >= 1
    assert len(cx["all_central_complex"]) >= 4


def test_get_neuron_selection_summary():
    """Test comprehensive neuron selection."""
    annotations = pd.DataFrame({
        "bodyId": [1, 2, 3, 4, 5],
        "type": ["R1", "DN", "FB_neuron", "Mi1", "LC4"],
    })
    
    body_ids = np.array([1, 2, 3, 4, 5])
    body_id_to_idx = {bid: idx for idx, bid in enumerate(body_ids)}
    
    selection = get_neuron_selection_summary(annotations, body_ids, body_id_to_idx)
    
    assert "visual" in selection
    assert "descending" in selection
    assert "central_complex" in selection
    
    # Each category should have subcategories
    assert len(selection["visual"]) > 0
    assert len(selection["descending"]) > 0
    assert len(selection["central_complex"]) > 0


def test_case_sensitivity():
    """Test case-insensitive matching (default behavior)."""
    annotations = pd.DataFrame({
        "bodyId": [1, 2, 3],
        "type": ["Photoreceptor", "photoreceptor", "PHOTORECEPTOR"],
    })
    
    body_ids = np.array([1, 2, 3])
    body_id_to_idx = {bid: idx for idx, bid in enumerate(body_ids)}
    
    # Case-insensitive (default)
    indices = find_neurons_by_type(
        annotations, body_ids, body_id_to_idx, {"photoreceptor"}, case_sensitive=False
    )
    assert len(indices) == 3  # Should find all three
    
    # Case-sensitive
    indices = find_neurons_by_type(
        annotations, body_ids, body_id_to_idx, {"photoreceptor"}, case_sensitive=True
    )
    assert len(indices) == 1  # Should find only exact match


def test_empty_results():
    """Test handling of no matches."""
    annotations = pd.DataFrame({
        "bodyId": [1, 2, 3],
        "type": ["other1", "other2", "other3"],
    })
    
    body_ids = np.array([1, 2, 3])
    body_id_to_idx = {bid: idx for idx, bid in enumerate(body_ids)}
    
    indices = find_neurons_by_type(
        annotations, body_ids, body_id_to_idx, {"nonexistent"}
    )
    assert len(indices) == 0


def test_load_neurotransmitter_data_fallback():
    """Test neurotransmitter loading falls back gracefully."""
    # Should not crash on missing file
    result = load_neurotransmitter_data("nonexistent.feather")
    assert result is None or len(result) == 0


def test_get_neuron_transmitters():
    """Test neurotransmitter lookup by body ID."""
    nt_data = create_synthetic_nt_data()
    body_ids = np.array([1, 2, 3, 8])
    
    transmitters = get_neuron_transmitters(body_ids, nt_data)
    
    assert transmitters[1] == "acetylcholine"
    assert transmitters[2] == "gaba"
    assert transmitters[3] == "glutamate"
    # Body 8: unclear → should fallback to celltype_predicted
    assert transmitters[8] == "acetylcholine"


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
        annotations, body_ids, body_id_to_idx, subtype="a", side="L", cell_class="T4"
    )
    assert len(indices) >= 1
    assert 0 in indices  # T4a, somaSide=L
    assert 4 not in indices  # T5a must not match T4a
    
    # Select T5b right
    indices = select_input_neurons_for_optic_flow(
        annotations, body_ids, body_id_to_idx, subtype="b", side="R", cell_class="T5"
    )
    assert len(indices) >= 1
    assert 5 in indices  # T5b, somaSide=R


def test_t4_and_t5_populations_are_distinct():
    """T4a and T5a must not collapse into the same population."""
    from threedfly.connectome.neuron_selection import select_input_neurons_for_optic_flow

    annotations = pd.DataFrame({
        "bodyId": [1, 2, 3, 4],
        "type": ["T4a", "T5a", "T4a", "T5a"],
        "celltype": ["T4a", "T5a", "T4a", "T5a"],
        "somaSide": ["L", "L", "R", "R"],
    })
    body_ids = np.array([1, 2, 3, 4])
    body_id_to_idx = {bid: idx for idx, bid in enumerate(body_ids)}

    t4_left = select_input_neurons_for_optic_flow(
        annotations, body_ids, body_id_to_idx, subtype="a", side="L", cell_class="T4"
    )
    t5_left = select_input_neurons_for_optic_flow(
        annotations, body_ids, body_id_to_idx, subtype="a", side="L", cell_class="T5"
    )
    assert set(t4_left) == {0}
    assert set(t5_left) == {1}
    assert set(t4_left).isdisjoint(set(t5_left))


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


def test_looming_does_not_overmatch_similar_types():
    """LPLC4 / LC40 / LC43 must not be selected as LPLC2+LC4."""
    from threedfly.connectome.neuron_selection import select_input_neurons_for_looming

    annotations = pd.DataFrame({
        "bodyId": [1, 2, 3, 4, 5, 6, 7, 8],
        "type": ["LPLC2", "LC4", "LPLC4", "LC40", "LC43", "LC41", "LC46b", "LC44"],
        "celltype": ["LPLC2", "LC4", "LPLC4", "LC40", "LC43", "LC41", "LC46b", "LC44"],
        "somaSide": ["L"] * 8,
    })
    body_ids = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    body_id_to_idx = {bid: idx for idx, bid in enumerate(body_ids)}

    left = select_input_neurons_for_looming(
        annotations, body_ids, body_id_to_idx, side="L"
    )
    assert set(left) == {0, 1}


def test_transmitter_to_sign_mapping():
    """ACh +1; GABA/glutamate/histamine -1; others +1 (configurable)."""
    from threedfly.connectome.neuron_selection import transmitter_to_sign

    assert transmitter_to_sign("acetylcholine") == 1.0
    assert transmitter_to_sign("GABA") == -1.0
    assert transmitter_to_sign("glutamate") == -1.0
    assert transmitter_to_sign("histamine") == -1.0
    assert transmitter_to_sign("dopamine") == 1.0
    assert transmitter_to_sign("unclear") == 1.0
    assert transmitter_to_sign("glutamate", inhibitory_transmitters=("gaba",)) == 1.0


def test_side_mask_accepts_left_right_aliases():
    """somaSide/rootSide may be left/right/LHS/RHS rather than L/R."""
    from threedfly.connectome.neuron_selection import select_input_neurons_for_looming

    annotations = pd.DataFrame({
        "bodyId": [1, 2, 3, 4],
        "type": ["LPLC2", "LPLC2", "LC4", "LC4"],
        "celltype": ["LPLC2", "LPLC2", "LC4", "LC4"],
        "somaSide": ["left", "RIGHT", "LHS", "rhs"],
        "rootSide": ["?", "?", "?", "?"],
    })
    body_ids = np.array([1, 2, 3, 4])
    mapping = {int(b): i for i, b in enumerate(body_ids)}
    left = select_input_neurons_for_looming(annotations, body_ids, mapping, side="L")
    right = select_input_neurons_for_looming(annotations, body_ids, mapping, side="R")
    assert set(left) == {0, 2}
    assert set(right) == {1, 3}


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
