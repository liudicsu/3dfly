"""Test script with 2000 steps."""

import numpy as np
from pathlib import Path
from threedfly.runner import run_simulation

trajectory_points = []

def capture_trajectory():
    """Capture trajectory during simulation."""
    from threedfly import sim
    original_step_func = sim.FlyEnvironment.step
    
    def step_with_capture(self, action):
        result = original_step_func(self, action)
        pos = self.get_fly_position()
        trajectory_points.append(pos.copy())
        return result
    
    sim.FlyEnvironment.step = step_with_capture

print("\n" + "="*70)
print("Testing horizontal trajectory coverage (seed 42, 2000 steps)")
print("="*70 + "\n")

capture_trajectory()

try:
    run_simulation(
        subgraph_path="data/demo_subgraph.npz",
        n_steps=2000,
        viz_mode="none",
        save_output_dir="output_trajectory_2000",
        simulator_type="rate",
        seed=42,
        max_collisions=100,
        enable_soft_recovery=True,
    )
except Exception as e:
    print(f"Simulation error: {e}")

if len(trajectory_points) > 0:
    traj = np.array(trajectory_points)
    
    print("\n" + "="*70)
    print("TRAJECTORY ANALYSIS (2000 STEPS)")
    print("="*70)
    print(f"\nTotal trajectory points: {len(traj)}")
    print(f"\nPosition ranges:")
    print(f"  X: [{traj[:, 0].min():.3f}, {traj[:, 0].max():.3f}] m  (ΔX = {traj[:, 0].max() - traj[:, 0].min():.3f} m)")
    print(f"  Y: [{traj[:, 1].min():.3f}, {traj[:, 1].max():.3f}] m  (ΔY = {traj[:, 1].max() - traj[:, 1].min():.3f} m)")
    print(f"  Z: [{traj[:, 2].min():.3f}, {traj[:, 2].max():.3f}] m  (ΔZ = {traj[:, 2].max() - traj[:, 2].min():.3f} m)")
    
    diffs = np.diff(traj, axis=0)
    path_length = np.sum(np.linalg.norm(diffs, axis=1))
    horizontal_diffs = diffs[:, :2]
    horizontal_path = np.sum(np.linalg.norm(horizontal_diffs, axis=1))
    vertical_path = np.sum(np.abs(diffs[:, 2]))
    
    print(f"\nPath lengths:")
    print(f"  Total path: {path_length:.3f} m")
    print(f"  Horizontal (XY) path: {horizontal_path:.3f} m ({100*horizontal_path/path_length:.1f}%)")
    print(f"  Vertical (Z) path: {vertical_path:.3f} m ({100*vertical_path/path_length:.1f}%)")
    
    dx = traj[:, 0].max() - traj[:, 0].min()
    dy = traj[:, 1].max() - traj[:, 1].min()
    dz = traj[:, 2].max() - traj[:, 2].min()
    
    print(f"\n{'='*70}")
    print("SUCCESS CRITERIA CHECK")
    print("="*70)
    
    success_count = 0
    
    print(f"\n1. Horizontal coverage (ΔX or ΔY > 1.0m):")
    if dx > 1.0 or dy > 1.0:
        print(f"   ✓ PASS: ΔX={dx:.2f}m, ΔY={dy:.2f}m")
        success_count += 1
    else:
        print(f"   ✗ FAIL: ΔX={dx:.2f}m, ΔY={dy:.2f}m (need >1.0m)")
    
    horizontal_disp = np.sqrt(dx**2 + dy**2)
    print(f"\n2. Not a point (horizontal displacement > 0.5m):")
    if horizontal_disp > 0.5:
        print(f"   ✓ PASS: {horizontal_disp:.2f}m")
        success_count += 1
    else:
        print(f"   ✗ FAIL: {horizontal_disp:.2f}m")
    
    print(f"\n3. Reduced vertical bouncing (ΔZ < 2.0m):")
    if dz < 2.0:
        print(f"   ✓ PASS: ΔZ={dz:.2f}m")
        success_count += 1
    else:
        print(f"   ⚠ WARNING: ΔZ={dz:.2f}m")
    
    print(f"\n{'='*70}")
    print(f"PASSED {success_count}/3 criteria")
    print("="*70 + "\n")
    
    np.save("output_trajectory_2000/trajectory_seed42.npy", traj)
    print(f"Trajectory saved to: output_trajectory_2000/trajectory_seed42.npy\n")
