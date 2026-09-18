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
)


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
