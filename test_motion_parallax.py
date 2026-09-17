#!/usr/bin/env python3
"""Test motion parallax depth estimation without requiring MuJoCo rendering."""

import numpy as np
import sys
sys.path.insert(0, '/workspace/3dfly')

from scipy import sparse
from threedfly.vision.brain_depth import BrainDepthEstimator

def test_brain_depth_estimator():
    """Test the new motion parallax-based BrainDepthEstimator."""
    print("\n" + "="*70)
    print("Testing Motion Parallax Brain Depth Estimator")
    print("="*70)
    
    # Create a small test connectome
    n_neurons = 500
    np.random.seed(42)
    
    # Sparse random adjacency matrix
    density = 0.1
    data = np.random.randn(int(n_neurons * n_neurons * density))
    rows = np.random.randint(0, n_neurons, len(data))
    cols = np.random.randint(0, n_neurons, len(data))
    adjacency = sparse.csr_matrix((data, (rows, cols)), shape=(n_neurons, n_neurons))
    
    print(f"\n1. Created test connectome: {n_neurons} neurons")
    print(f"   Adjacency shape: {adjacency.shape}, nnz: {adjacency.nnz}")
    
    # Initialize brain depth estimator with motion parallax
    brain_depth = BrainDepthEstimator(
        n_neurons=n_neurons,
        adjacency=adjacency,
        use_rate_model=True,
        flow_grid_size=(12, 16)
    )
    
    print(f"\n2. Initialized BrainDepthEstimator")
    print(f"   Visual neurons: {brain_depth.n_visual}")
    print(f"   Flow grid: {brain_depth.flow_grid_h}x{brain_depth.flow_grid_w}")
    print(f"   Flow features: {brain_depth.n_flow_features}")
    print(f"   OLD approach used 8x6 = 48 cells")
    print(f"   NEW approach uses 12x16 = 192 cells (4x resolution!)")
    
    # Test flow feature preparation
    print("\n3. Testing optical flow computation...")
    
    # Create synthetic images (frame 1)
    h, w = 120, 160
    left_img1 = (np.random.rand(h, w, 3) * 255).astype(np.uint8)
    right_img1 = (np.random.rand(h, w, 3) * 255).astype(np.uint8)
    
    # Velocity (flying forward and to the right)
    velocity = np.array([0.5, 0.2, 0.0])
    
    # First frame (should return zero features)
    flow_features1, flow_info1 = brain_depth.prepare_flow_features_for_brain(
        left_img1, right_img1, velocity
    )
    
    print(f"   Frame 1: {flow_features1.shape}, has_flow={flow_info1['has_flow']}")
    assert flow_info1['has_flow'] == False
    assert flow_features1.shape == (brain_depth.n_flow_features,)
    
    # Create synthetic images (frame 2 - slightly shifted)
    left_img2 = np.roll(left_img1, shift=(2, 3), axis=(0, 1))
    right_img2 = np.roll(right_img1, shift=(2, 3), axis=(0, 1))
    
    flow_features2, flow_info2 = brain_depth.prepare_flow_features_for_brain(
        left_img2, right_img2, velocity
    )
    
    print(f"   Frame 2: {flow_features2.shape}, has_flow={flow_info2['has_flow']}")
    print(f"   Flow magnitude: {flow_info2['flow_magnitude']:.2f} pixels")
    assert flow_info2['has_flow'] == True
    assert flow_features2.shape == (brain_depth.n_flow_features,)
    
    # Test depth estimation from brain activity
    print("\n4. Testing depth estimation from brain activity...")
    
    # Simulate brain activity (random for testing)
    brain_activity = np.random.rand(n_neurons) * 0.5
    
    # Estimate depth
    depth_result = brain_depth.estimate_depth_from_activity(
        brain_activity,
        stereo_features={'flow_magnitude': flow_info2['flow_magnitude']},
        velocity=velocity
    )
    
    print(f"   Depth map shape: {depth_result['depth_map'].shape}")
    print(f"   Mean depth: {depth_result['mean_depth']:.2f} m")
    
    assert depth_result['depth_map'].shape == (12, 16)
    assert depth_result['confidence'].shape == (12, 16)
    
    print("\n" + "="*70)
    print("✅ All core tests passed!")
    print("="*70)
    
    return True

if __name__ == "__main__":
    success = test_brain_depth_estimator()
    sys.exit(0 if success else 1)
