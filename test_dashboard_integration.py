#!/usr/bin/env python3
"""Integration test for combined 3D view + dashboard panels."""

import numpy as np
import time
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing

from threedfly.viz.interactive import InteractiveVisualizer

def test_dashboard_integration():
    """Test that InteractiveVisualizer creates both 3D view and dashboard."""
    print("\n" + "=" * 70)
    print("Testing Interactive Visualizer with Dashboard Integration")
    print("=" * 70)
    
    # Create visualizer
    print("\n[1/3] Creating visualizer...")
    viz = InteractiveVisualizer(host="127.0.0.1", port=8081)
    print("  ✓ Visualizer created")
    
    # Check that matplotlib figure exists
    print("\n[2/3] Checking dashboard components...")
    assert viz.fig is not None, "Matplotlib figure not created"
    assert viz.axes is not None, "Axes not created"
    assert viz.axes.shape == (2, 3), f"Expected (2,3) axes, got {viz.axes.shape}"
    print("  ✓ Dashboard has 6 panels (2x3 grid)")
    
    # Test update with sample data
    print("\n[3/3] Testing update with sample data...")
    sample_img = np.random.randint(0, 255, (48, 64, 3), dtype=np.uint8)
    sample_position = np.array([1.0, 2.0, 3.0])
    sample_brain = np.random.rand(500) * 0.5
    sample_action = np.array([0.5, 0.3, -0.2, 0.1, -0.1])
    
    viz.update(
        fly_position=sample_position,
        left_img=sample_img,
        right_img=sample_img,
        brain_activity=sample_brain,
        control_action=sample_action,
        brain_stats={"mean_rate": 0.15, "active_fraction": 0.6},
        control_stats={"thrust": 0.5},
    )
    print("  ✓ Update successful")
    
    # Verify data was stored
    assert viz.left_img is not None, "Left image not stored"
    assert viz.right_img is not None, "Right image not stored"
    assert viz.brain_activity is not None, "Brain activity not stored"
    assert viz.control_action is not None, "Control action not stored"
    assert len(viz.trajectory_points) == 1, "Trajectory not updated"
    print("  ✓ Data stored correctly")
    
    # Cleanup
    viz.close()
    print("\n" + "=" * 70)
    print("✅ All tests passed!")
    print("=" * 70)
    print("\nThe interactive visualizer correctly integrates:")
    print("  1. 3D god's-eye view (viser server)")
    print("  2. Dashboard panels (matplotlib 2x3 grid)")
    print("  3. Real-time updates for both views")
    print()

if __name__ == "__main__":
    test_dashboard_integration()
