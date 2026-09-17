# Unified Web Interface Documentation

## Overview

The 3dfly interactive visualization now provides a **single-page web interface** that combines:
1. Interactive 3D god's-eye view (viser)
2. Dashboard panels (matplotlib-generated)

**Everything in one browser tab at http://localhost:8080** — no separate windows!

## Architecture

### Before
- **viser server**: 3D view in browser
- **matplotlib window**: Separate OS window with dashboard panels

### After
- **viser server with embedded dashboard**: Single web page containing:
  - 3D scene (room, fly, trajectory, point cloud)
  - GUI sidebar with controls (play/pause/step/reset/speed)
  - Dashboard panels folder with matplotlib-generated image

## Implementation Details

### Key Changes in `InteractiveVisualizer`

1. **Added matplotlib with Agg backend** (headless rendering):
```python
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
```

2. **Dashboard panel generation** (`_render_dashboard_panels()`):
   - Creates 2x3 matplotlib figure with all panels
   - Converts figure to PNG image in memory (BytesIO)
   - Displays image in viser GUI folder using `server.gui.add_image()`

3. **Dashboard layout**:
```
┌─────────────┬─────────────┬─────────────┐
│  Left Eye   │  Right Eye  │ Trajectory  │
│   (image)   │   (image)   │ (top view)  │
├─────────────┼─────────────┼─────────────┤
│    Brain    │   Flight    │   System    │
│  Activity   │  Commands   │   Status    │
│   (bars)    │   (bars)    │   (text)    │
└─────────────┴─────────────┴─────────────┘
```

### Dashboard Panel Contents

1. **Left Eye**: Stereo camera left eye view (64x48 RGB)
2. **Right Eye**: Stereo camera right eye view (64x48 RGB)
3. **Trajectory**: XY plot of flight path with current position marker
4. **Brain Activity**: Bar chart showing sampled neuron activity (200 neurons)
5. **Flight Commands**: Bar chart of 5 control signals (Fwd, Up, Roll, Pitch, Yaw)
6. **System Status**: Text display with position, map statistics, brain stats

### Update Flow

```
Simulation Step
    ↓
viz.update(fly_position, left_img, right_img, brain_activity, ...)
    ↓
_render_fly()           → Update 3D fly mesh
_render_trajectory()    → Update 3D trajectory line
_render_point_cloud()   → Update 3D point cloud
_render_dashboard_panels() → Generate matplotlib figure → Convert to image → Display in GUI
_render_status()        → Update compact status text
```

### Performance

- Dashboard updated every 5 simulation steps (configurable via `viz_update_interval`)
- Matplotlib figure generation: ~50-100ms
- Image conversion (PNG): ~20-50ms
- Total overhead per update: ~100-150ms

## Usage

### Running the Unified Interface

```bash
# Start interactive simulation
threedfly run-interactive --subgraph data/demo_subgraph.npz --steps 5000

# Open browser to http://localhost:8080
# You'll see:
#   - 3D view in main viewport
#   - Controls in left sidebar
#   - Dashboard panels in collapsible "Dashboard Panels" folder
```

### Programmatic Usage

```python
from threedfly.runner_interactive import run_interactive_simulation

run_interactive_simulation(
    subgraph_path="data/demo_subgraph.npz",
    max_steps=5000,
    save_output_dir="output",
    simulator_type="rate",
    seed=42,
    host="0.0.0.0",
    port=8080,
)
```

## Benefits

1. **Single browser tab**: No window management
2. **Consistent interface**: Everything in one place
3. **Web-accessible**: Works on any device with browser
4. **Better for remote work**: Forward one port instead of X11/VNC
5. **Screenshot-friendly**: Capture everything at once

## Technical Notes

### Why Agg Backend?

The Agg (Anti-Grain Geometry) backend allows matplotlib to render plots without a display:
- No X11/Wayland required
- Faster rendering (no GUI overhead)
- Direct to memory (BytesIO buffer)
- Perfect for server environments

### Why Not Real-time Plot Updates?

We generate static images instead of live plots because:
- viser doesn't support embedding interactive matplotlib widgets
- Image approach is simpler and more portable
- Update rate (every 5 steps) is sufficient for monitoring
- Smaller payload over network

### Alternative Approaches Considered

1. **Plotly/Dash**: Would require separate web server and iframe embedding
2. **Three.js overlays**: Complex to maintain, mixing rendering contexts
3. **WebGL plots**: Browser compatibility issues, added complexity
4. **Real-time canvas streaming**: Higher bandwidth, lower quality

Current approach (matplotlib → PNG → viser GUI) provides best balance of simplicity, quality, and performance.

## Future Enhancements

Possible improvements:
- [ ] Configurable dashboard layout (user-selectable panels)
- [ ] Higher update rate option (every step)
- [ ] Panel zoom/focus mode (expand one panel)
- [ ] Export dashboard as separate image file
- [ ] Dark mode support
- [ ] Custom panel content via plugin system

## Migration Guide

### For Users

No changes needed! Just run:
```bash
threedfly run-interactive ...
```

### For Developers

If you extended `InteractiveVisualizer`:

**Old pattern** (separate camera views in 3D scene):
```python
self.server.scene.add_image("/world/left_eye_view", ...)
```

**New pattern** (dashboard panels in GUI):
```python
# Dashboard panels are auto-generated from:
# - self.left_img
# - self.right_img
# - self.brain_activity
# - self.control_action
# - self.trajectory_points
# - self.map_stats
```

## Troubleshooting

### Dashboard not updating
- Check that `viz_update_interval` is set (default: 5)
- Verify matplotlib Agg backend: `matplotlib.get_backend()` should return `'agg'`

### Image quality issues
- Increase DPI in `fig.savefig(buf, format='png', dpi=100, ...)`
- Default 100 DPI is good balance; 150-200 for higher quality

### Performance issues
- Increase `viz_update_interval` (update less frequently)
- Reduce figure size: `fig = plt.figure(figsize=(12, 7))` instead of `(14, 8)`
- Sample fewer brain neurons for display

## References

- [viser documentation](https://github.com/nerfstudio-project/viser)
- [Matplotlib backends](https://matplotlib.org/stable/users/explain/backends.html)
- [3dfly repository](https://github.com/liudicsu/3dfly)
