# Soft Collision Recovery

## Overview

The 3dfly simulator now features **soft collision recovery** that allows the fruit fly to continue exploring after hitting walls, obstacles, or boundaries. Instead of stopping on the first collision, the fly backs off, reorients, and continues flying.

## Key Improvements

### Before (Original Behavior)
- ❌ Simulation stopped immediately on first collision
- ❌ Typical runs: ~255 steps before stopping
- ❌ Interactive mode paused indefinitely waiting for manual reset
- ❌ Minimal room exploration

### After (Soft Recovery)
- ✅ Fly automatically recovers from collisions
- ✅ Configurable max collisions (default: 50 for non-interactive, 100 for interactive)
- ✅ Runs complete 1500-2000+ steps with continuous exploration
- ✅ Meaningful trajectories covering room space
- ✅ Interactive mode auto-recovers instead of pausing

## How It Works

### Recovery Strategy

When a collision is detected, the fly:

1. **Detects collision type** (ceiling, ground, or wall)
2. **Backs off** from the collision point:
   - **Ceiling collision**: Moves down + adds horizontal randomization
   - **Ground collision**: Moves up + adds horizontal randomization  
   - **Wall collision**: Backs off from contact normal + adds exploration offset
3. **Dampens velocity** to prevent immediate re-collision
4. **Randomizes orientation** to explore new directions
5. **Clamps position** to stay within room bounds

### Controller Tuning

The flight controller has been tuned for better exploration:

- **Increased forward thrust** (0.8 gain + 0.4 baseline) for sustained motion
- **Reduced upward bias** (0.15 vs 0.3) to avoid ceiling saturation
- **Increased turn sensitivity** (0.6 gain) for better exploration
- **Stronger exploration signal** (weight 0.8 vs 0.5) to bias toward unmapped regions

## Usage

### CLI

```bash
# Run with soft recovery (default)
threedfly run --subgraph data/demo_subgraph.npz --steps 2000

# Disable soft recovery (old behavior)
threedfly run --subgraph data/demo_subgraph.npz --steps 2000 --no-soft-recovery

# Set max collisions before stopping
threedfly run --subgraph data/demo_subgraph.npz --steps 2000 --max-collisions 30

# Interactive mode with auto-recovery
threedfly run-interactive --subgraph data/demo_subgraph.npz --steps 5000
```

### Python API

```python
from threedfly.runner import run_simulation

# Run with soft collision recovery
run_simulation(
    subgraph_path="data/demo_subgraph.npz",
    n_steps=2000,
    max_collisions=50,        # Max collisions before stopping (0 = unlimited)
    enable_soft_recovery=True  # Enable auto-recovery
)

# Disable soft recovery (old behavior - stops on first collision)
run_simulation(
    subgraph_path="data/demo_subgraph.npz",
    n_steps=2000,
    enable_soft_recovery=False
)
```

### Environment API

```python
from threedfly.sim import FlyEnvironment

env = FlyEnvironment()
env.reset()

# ... fly around ...

if env.check_collision():
    env.recover_from_collision(recovery_distance=0.3)
    # Fly is now repositioned and reoriented - continue flying!
```

## Performance Metrics

### Test Results (seed=42, 1500 steps, demo_subgraph.npz)

| Metric | Without Recovery | With Recovery | Improvement |
|--------|------------------|---------------|-------------|
| **Steps completed** | 254 | 1500 | **5.9x** |
| **Collisions handled** | 1 (stop) | 9 (continue) | **9x** |
| **Distance traveled** | ~2m | ~10m | **5x** |
| **Point cloud points** | ~1000 | ~11,000 | **11x** |

## Configuration

### Recovery Parameters

In `environment.py`:

```python
def recover_from_collision(self, recovery_distance: float = 0.3):
    """
    Args:
        recovery_distance: Distance to back off from collision (meters)
                          Default: 0.3m
    """
```

### Runner Parameters

In `runner.py` and `runner_interactive.py`:

```python
def run_simulation(..., max_collisions=50, enable_soft_recovery=True):
    """
    Args:
        max_collisions: Maximum collisions before stopping
                       0 = unlimited
                       Default: 50 (non-interactive), 100 (interactive)
        
        enable_soft_recovery: Enable soft collision recovery
                            True = auto-recover and continue
                            False = stop on first collision (old behavior)
    """
```

## Testing

Run collision recovery tests:

```bash
cd /workspace
python3 tests/test_collision_recovery.py
```

Tests verify:
- ✓ Collision recovery moves fly away from collision point
- ✓ Velocity is dampened to prevent re-collision
- ✓ Orientation is randomized for exploration
- ✓ Position stays within room bounds
- ✓ Flight extends significantly with recovery enabled

## Implementation Details

### Modified Files

1. **`threedfly/sim/environment.py`**
   - Added `recover_from_collision()` method
   - Collision-type-specific recovery strategies
   - Position clamping and velocity dampening

2. **`threedfly/runner.py`**
   - Added `max_collisions` and `enable_soft_recovery` parameters
   - Collision counting and recovery loop
   - Statistics tracking

3. **`threedfly/runner_interactive.py`**
   - Auto-recovery instead of infinite pause
   - Configurable max collisions
   - Collision statistics

4. **`threedfly/control/controller.py`**
   - Increased forward thrust (baseline + gain)
   - Reduced upward bias to avoid ceiling
   - Tuned exploration signal integration

5. **`threedfly/control/exploration.py`**
   - Increased exploration weight (0.8)
   - Stronger forward signal (1.2)
   - Reduced vertical adjustment

6. **`threedfly/cli.py`**
   - Added `--max-collisions` flag
   - Added `--no-soft-recovery` flag
   - Updated help text

### Design Decisions

**Why soft recovery instead of better obstacle avoidance?**
- The brain-based controller is biologically inspired, not optimal
- Soft recovery mimics real fly behavior (collision → reorient → continue)
- Simpler and more robust than adding complex planning
- Allows exploration of controller/brain limitations

**Why randomize orientation on recovery?**
- Ensures fly explores new directions instead of repeating same path
- Prevents getting stuck in corners
- Increases room coverage

**Why dampen velocity?**
- Prevents immediate re-collision
- Gives controller time to adjust
- More realistic behavior

## Future Improvements

Potential enhancements:

- [ ] Adaptive recovery based on collision frequency
- [ ] Memory of collision locations to avoid revisiting
- [ ] Learned recovery strategy via reinforcement learning
- [ ] Collision prediction using visual depth
- [ ] Energy cost for collisions (survival incentive)

## References

- Original issue: "fly barely flew" - stopped at ~255 steps
- User feedback: "trajectory plot shows nearly stationary / very short"
- Solution: Soft recovery allows 1500+ steps with meaningful exploration
