"""Interactive 3D visualization using viser with embedded dashboard panels."""

import numpy as np
import viser
import time
from typing import Optional, Dict, List
import threading
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
from pathlib import Path
import trimesh


class InteractiveVisualizer:
    """
    Interactive 3D god's-eye visualization for fly exploration.
    
    Single-page unified interface with:
    - 3D god's-eye view: MuJoCo environment (room walls, obstacles), fruit fly body, 
      flight trajectory, and 3D point cloud reconstruction
    - Dashboard panels: Live stereo camera views, trajectory plot, brain activity,
      flight commands, and system status
    
    Controls:
    - Orbit/pan/zoom the 3D scene
    - Play/pause simulation
    - Step through frames
    - Reset simulation
    - Adjust visualization speed
    """
    
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8080,
    ):
        """
        Initialize interactive visualizer.
        
        Args:
            host: Server host address
            port: Server port
        """
        self.server = viser.ViserServer(host=host, port=port)
        print(f"\n🌐 Interactive 3D visualization server started!")
        print(f"   Open your browser to: http://localhost:{port}")
        print(f"   Use mouse to orbit/pan/zoom the 3D scene")
        
        # State
        self.fly_position = np.zeros(3)
        self.fly_orientation = np.array([1.0, 0.0, 0.0, 0.0])  # quaternion
        self.trajectory_points: List[np.ndarray] = []
        self.point_cloud_points: Optional[np.ndarray] = None
        self.point_cloud_colors: Optional[np.ndarray] = None
        
        self.left_img: Optional[np.ndarray] = None
        self.right_img: Optional[np.ndarray] = None
        
        self.brain_activity: Optional[np.ndarray] = None
        self.control_action: Optional[np.ndarray] = None
        self.brain_stats: Dict = {}
        self.control_stats: Dict = {}
        self.map_stats: Dict = {}
        
        # Control state
        self.paused = False
        self.should_step = False
        self.should_reset = False
        
        # Visualization handles (3D scene)
        self._fly_handle = None
        self._trajectory_handle = None
        self._point_cloud_handle = None
        
        # Dashboard panel handles (GUI images)
        self._dashboard_folder = None
        self._panel_left_eye = None
        self._panel_right_eye = None
        self._panel_trajectory = None
        self._panel_brain = None
        self._panel_commands = None
        self._panel_status = None
        
        # Setup scene
        self._setup_environment()
        self._setup_ui_controls()
        self._setup_dashboard_panels()
        
    def _setup_environment(self):
        """Setup static environment geometry (room, obstacles)."""
        # Coordinate frame at origin
        self.server.scene.add_frame(
            "/world",
            wxyz=(1.0, 0.0, 0.0, 0.0),
            position=(0.0, 0.0, 0.0),
            show_axes=True,
            axes_length=1.0,
            axes_radius=0.02,
        )
        
        # Ground plane (grid)
        self.server.scene.add_grid(
            "/world/ground",
            width=20.0,
            height=20.0,
            width_segments=40,
            height_segments=40,
            plane="xz",
            cell_color=(200, 200, 220),
            cell_thickness=1.0,
            cell_size=0.5,
        )
        
        # Room walls (simplified boxes)
        wall_thickness = 0.2
        wall_height = 4.0
        room_size = 10.0
        
        # North wall (positive X)
        self.server.scene.add_box(
            "/world/wall_north",
            color=(200, 200, 230),
            dimensions=(wall_thickness, room_size, wall_height),
            position=(room_size/2, 0.0, wall_height/2),
        )
        
        # South wall (negative X)
        self.server.scene.add_box(
            "/world/wall_south",
            color=(200, 230, 200),
            dimensions=(wall_thickness, room_size, wall_height),
            position=(-room_size/2, 0.0, wall_height/2),
        )
        
        # East wall (positive Y)
        self.server.scene.add_box(
            "/world/wall_east",
            color=(230, 200, 200),
            dimensions=(room_size, wall_thickness, wall_height),
            position=(0.0, room_size/2, wall_height/2),
        )
        
        # West wall (negative Y)
        self.server.scene.add_box(
            "/world/wall_west",
            color=(230, 230, 200),
            dimensions=(room_size, wall_thickness, wall_height),
            position=(0.0, -room_size/2, wall_height/2),
        )
        
        # Obstacles
        self.server.scene.add_box(
            "/world/obstacle_1",
            color=(153, 76, 51),
            dimensions=(0.6, 0.6, 1.0),
            position=(2.0, 2.0, 0.5),
        )
        
        self.server.scene.add_icosphere(
            "/world/obstacle_2",
            color=(102, 128, 102),
            radius=0.4,
            position=(-2.0, -2.0, 0.4),
        )
        
        self.server.scene.add_icosphere(
            "/world/obstacle_3",
            color=(102, 102, 153),
            radius=0.4,
            position=(-1.0, 3.0, 0.4),
        )
        
    def _setup_ui_controls(self):
        """Setup UI control panel."""
        # Playback controls
        self.play_button = self.server.gui.add_button(
            "▶️ Play",
            hint="Start/pause simulation"
        )
        
        @self.play_button.on_click
        def _(_):
            self.paused = not self.paused
            if self.paused:
                self.play_button.name = "▶️ Play"
            else:
                self.play_button.name = "⏸️ Pause"
        
        self.step_button = self.server.gui.add_button(
            "⏭️ Step",
            hint="Step one frame"
        )
        
        @self.step_button.on_click
        def _(_):
            self.should_step = True
        
        self.reset_button = self.server.gui.add_button(
            "🔄 Reset",
            hint="Reset simulation"
        )
        
        @self.reset_button.on_click
        def _(_):
            self.should_reset = True
        
        # Visualization options
        self.show_trajectory = self.server.gui.add_checkbox(
            "Show Trajectory",
            initial_value=True,
            hint="Display flight path in 3D view"
        )
        
        self.show_point_cloud = self.server.gui.add_checkbox(
            "Show Point Cloud",
            initial_value=True,
            hint="Display reconstructed 3D map"
        )
        
        # Speed control
        self.speed_slider = self.server.gui.add_slider(
            "Visualization Speed",
            min=0.1,
            max=2.0,
            step=0.1,
            initial_value=1.0,
            hint="Adjust playback speed"
        )
        
        # Info text (will be updated each frame)
        self._status_text_handle = self.server.gui.add_text(
            "Status",
            initial_value="Initializing...",
            disabled=True,
        )
    
    def _setup_dashboard_panels(self):
        """Setup dashboard panel folder for images."""
        # Create a collapsible folder for dashboard
        self._dashboard_folder = self.server.gui.add_folder("Dashboard Panels")
        
        # We'll add dashboard images here (updated in _render_dashboard_panels)
        
    def update(
        self,
        fly_position: Optional[np.ndarray] = None,
        fly_orientation: Optional[np.ndarray] = None,
        left_img: Optional[np.ndarray] = None,
        right_img: Optional[np.ndarray] = None,
        brain_activity: Optional[np.ndarray] = None,
        control_action: Optional[np.ndarray] = None,
        point_cloud_mapper = None,
        brain_stats: Optional[Dict] = None,
        control_stats: Optional[Dict] = None,
    ):
        """
        Update visualization with new data.
        
        Args:
            fly_position: Current position (x, y, z)
            fly_orientation: Quaternion (w, x, y, z)
            left_img: Left eye image
            right_img: Right eye image
            brain_activity: Brain activity vector
            control_action: Control commands
            point_cloud_mapper: PointCloudMapper instance
            brain_stats: Brain statistics dict
            control_stats: Control statistics dict
        """
        # Update fly position
        if fly_position is not None:
            self.fly_position = fly_position
            self.trajectory_points.append(fly_position.copy())
            
        if fly_orientation is not None:
            self.fly_orientation = fly_orientation
            
        # Update images
        if left_img is not None:
            self.left_img = left_img
        if right_img is not None:
            self.right_img = right_img
        
        # Update brain and control data
        if brain_activity is not None:
            self.brain_activity = brain_activity
        if control_action is not None:
            self.control_action = control_action
            
        # Update stats
        if brain_stats is not None:
            self.brain_stats = brain_stats
        if control_stats is not None:
            self.control_stats = control_stats
            
        # Get point cloud data
        if point_cloud_mapper is not None:
            # Use smaller voxel size for visualization (denser cloud)
            if len(point_cloud_mapper.global_cloud.points) > 0:
                # Show denser cloud - use 1/3 of the mapper's voxel size
                viz_voxel_size = point_cloud_mapper.voxel_size / 3.0
                downsampled = point_cloud_mapper.global_cloud.voxel_down_sample(voxel_size=viz_voxel_size)
                if len(downsampled.points) > 0:
                    self.point_cloud_points = np.asarray(downsampled.points)
                    self.point_cloud_colors = np.asarray(downsampled.colors)
            self.map_stats = point_cloud_mapper.get_statistics()
            
        # Render updates
        self._render_fly()
        self._render_trajectory()
        self._render_point_cloud()
        self._render_dashboard_panels()
        self._render_status()
        
    def _load_fly_mesh(self):
        """Load and combine fruit fly meshes into a single mesh."""
        assets_dir = Path(__file__).parent.parent.parent / "assets" / "fly_meshes"
        
        # Load main body parts
        try:
            # Scale factor to match MuJoCo (0.01) and adjust for Viser
            scale = 0.01
            
            # Load thorax (main body)
            thorax = trimesh.load(assets_dir / "thorax_body.obj", force="mesh")
            thorax.apply_scale(scale)
            
            # Load head
            head = trimesh.load(assets_dir / "head_body.obj", force="mesh")
            head.apply_scale(scale)
            head.apply_translation([0.012, 0, 0])
            
            # Load abdomen
            abdomen = trimesh.load(assets_dir / "abdomen_1_body.obj", force="mesh")
            abdomen.apply_scale(scale)
            abdomen.apply_translation([-0.012, 0, -0.002])
            
            # Load wings
            wing_left = trimesh.load(assets_dir / "wing_left_membrane.obj", force="mesh")
            wing_left.apply_scale(scale)
            wing_left.apply_translation([-0.002, 0.008, 0.002])
            
            wing_right = trimesh.load(assets_dir / "wing_right_membrane.obj", force="mesh")
            wing_right.apply_scale(scale)
            wing_right.apply_translation([-0.002, -0.008, 0.002])
            
            # Combine all parts
            combined = trimesh.util.concatenate([thorax, head, abdomen, wing_left, wing_right])
            
            # Set consistent color (brown for body)
            combined.visual.vertex_colors = np.array([76, 51, 25, 255], dtype=np.uint8)
            
            return combined
            
        except Exception as e:
            print(f"Warning: Could not load fly mesh: {e}")
            print("Falling back to simple sphere representation")
            return None
    
    def _render_fly(self):
        """Render fruit fly body using realistic mesh."""
        # Remove old fly
        if self._fly_handle is not None:
            self._fly_handle.remove()
        
        # Try to load and display realistic mesh
        if not hasattr(self, '_fly_mesh'):
            self._fly_mesh = self._load_fly_mesh()
        
        if self._fly_mesh is not None:
            # Use mesh representation
            self._fly_handle = self.server.scene.add_mesh_trimesh(
                "/world/fly",
                mesh=self._fly_mesh,
                position=tuple(self.fly_position),
                wxyz=tuple(self.fly_orientation),
            )
        else:
            # Fallback to simple sphere
            self._fly_handle = self.server.scene.add_icosphere(
                "/world/fly",
                radius=0.015,
                color=(76, 51, 25),  # Brown
                position=tuple(self.fly_position),
                wxyz=tuple(self.fly_orientation),
            )
        
        # Add a small directional indicator (forward vector)
        # Convert quaternion to forward direction
        qw, qx, qy, qz = self.fly_orientation
        # Forward direction (approximate, x-axis in body frame)
        forward = np.array([
            1 - 2*(qy**2 + qz**2),
            2*(qx*qy + qw*qz),
            2*(qx*qz - qw*qy)
        ])
        forward = forward / (np.linalg.norm(forward) + 1e-8)
        
        arrow_end = self.fly_position + forward * 0.05
        
        self.server.scene.add_spline_catmull_rom(
            "/world/fly_direction",
            positions=np.array([self.fly_position, arrow_end]),
            color=(255, 0, 0),
            line_width=3.0,
            segments=2,
        )
        
    def _render_trajectory(self):
        """Render flight trajectory."""
        if not self.show_trajectory.value or len(self.trajectory_points) < 2:
            if self._trajectory_handle is not None:
                self._trajectory_handle.remove()
                self._trajectory_handle = None
            return
            
        # Remove old trajectory
        if self._trajectory_handle is not None:
            self._trajectory_handle.remove()
            
        # Draw trajectory as spline
        positions = np.array(self.trajectory_points)
        self._trajectory_handle = self.server.scene.add_spline_catmull_rom(
            "/world/trajectory",
            positions=positions,
            color=(50, 100, 255),
            line_width=2.0,
            segments=max(2, len(positions) // 10),
        )
        
    def _render_point_cloud(self):
        """Render 3D point cloud."""
        if not self.show_point_cloud.value:
            if self._point_cloud_handle is not None:
                self._point_cloud_handle.remove()
                self._point_cloud_handle = None
            return
            
        if self.point_cloud_points is None or len(self.point_cloud_points) == 0:
            return
            
        # Remove old cloud
        if self._point_cloud_handle is not None:
            self._point_cloud_handle.remove()
            
        # Add new point cloud
        colors_uint8 = (self.point_cloud_colors * 255).astype(np.uint8)
        self._point_cloud_handle = self.server.scene.add_point_cloud(
            "/world/point_cloud",
            points=self.point_cloud_points,
            colors=colors_uint8,
            point_size=0.02,
            point_shape="circle",
        )
        
    def _render_dashboard_panels(self):
        """Render dashboard panels as matplotlib images in viser GUI."""
        # Create matplotlib figure with all dashboard panels
        fig = plt.figure(figsize=(14, 8))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # Row 0: Camera views and trajectory
        ax_left = fig.add_subplot(gs[0, 0])
        ax_right = fig.add_subplot(gs[0, 1])
        ax_traj = fig.add_subplot(gs[0, 2])
        
        # Row 1: Brain activity, control commands, and status
        ax_brain = fig.add_subplot(gs[1, 0])
        ax_control = fig.add_subplot(gs[1, 1])
        ax_status = fig.add_subplot(gs[1, 2])
        
        # Left eye view
        ax_left.set_title("Left Eye", fontweight='bold', fontsize=10)
        if self.left_img is not None:
            ax_left.imshow(self.left_img)
        ax_left.axis('off')
        
        # Right eye view
        ax_right.set_title("Right Eye", fontweight='bold', fontsize=10)
        if self.right_img is not None:
            ax_right.imshow(self.right_img)
        ax_right.axis('off')
        
        # Flight trajectory (top view)
        ax_traj.set_title("Flight Trajectory (Top View)", fontweight='bold', fontsize=10)
        if len(self.trajectory_points) > 1:
            traj = np.array(self.trajectory_points)
            ax_traj.plot(traj[:, 0], traj[:, 1], 'b-', alpha=0.5, linewidth=1)
            ax_traj.scatter(traj[-1, 0], traj[-1, 1], c='r', s=50, marker='o', zorder=5)
        ax_traj.set_xlabel("X (m)", fontsize=9)
        ax_traj.set_ylabel("Y (m)", fontsize=9)
        ax_traj.set_xlim(-5, 5)
        ax_traj.set_ylim(-5, 5)
        ax_traj.grid(True, alpha=0.3)
        ax_traj.tick_params(labelsize=8)
        
        # Brain activity
        ax_brain.set_title("Brain Activity", fontweight='bold', fontsize=10)
        if self.brain_activity is not None:
            # Sample for visualization (show subset)
            n_sample = min(200, len(self.brain_activity))
            indices = np.linspace(0, len(self.brain_activity)-1, n_sample, dtype=int)
            ax_brain.bar(indices, self.brain_activity[indices], width=1, alpha=0.7, color='steelblue')
        ax_brain.set_xlabel("Neuron Index", fontsize=9)
        ax_brain.set_ylabel("Activity", fontsize=9)
        ax_brain.tick_params(labelsize=8)
        ax_brain.grid(True, alpha=0.3, axis='y')
        
        # Flight commands
        ax_control.set_title("Flight Commands", fontweight='bold', fontsize=10)
        if self.control_action is not None:
            labels = ["Fwd", "Up", "Roll", "Pitch", "Yaw"]
            colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
            ax_control.bar(labels, self.control_action, color=colors, alpha=0.7)
            ax_control.axhline(0, color='k', linewidth=0.5)
        ax_control.set_ylabel("Command Value", fontsize=9)
        ax_control.set_ylim(-1, 1)
        ax_control.tick_params(labelsize=8)
        ax_control.grid(True, alpha=0.3, axis='y')
        
        # System status
        ax_status.set_title("System Status", fontweight='bold', fontsize=10)
        ax_status.axis('off')
        
        status_text = f"Position: ({self.fly_position[0]:.2f}, {self.fly_position[1]:.2f}, {self.fly_position[2]:.2f}) m\n"
        status_text += f"Trajectory: {len(self.trajectory_points)} points\n\n"
        
        if self.map_stats:
            status_text += "Mapping:\n"
            status_text += f"  Points: {self.map_stats.get('total_points', 0)}\n"
            status_text += f"  Voxels: {self.map_stats.get('occupied_voxels', 0)}\n"
            status_text += f"  Explored: {100*self.map_stats.get('exploration_ratio', 0):.1f}%\n\n"
        
        if self.brain_stats:
            status_text += "Brain:\n"
            for i, (key, val) in enumerate(self.brain_stats.items()):
                if i >= 4:  # Limit to first 4 stats
                    break
                status_text += f"  {key}: {val:.3f}\n"
        
        ax_status.text(0.05, 0.95, status_text, transform=ax_status.transAxes,
                      verticalalignment='top', fontfamily='monospace', fontsize=9)
        
        # Convert matplotlib figure to image
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        dashboard_img = np.array(Image.open(buf).convert('RGB'))
        plt.close(fig)
        
        # Display in viser GUI folder
        with self._dashboard_folder:
            if self._panel_left_eye is not None:
                self._panel_left_eye.remove()
            self._panel_left_eye = self.server.gui.add_image(
                dashboard_img,
                label="Dashboard Panels",
            )
    
    def _render_camera_views(self):
        """Legacy method - camera views now in dashboard panels."""
        pass
            
    def _render_status(self):
        """Render compact status text in UI."""
        status_lines = []
        status_lines.append(f"Pos: ({self.fly_position[0]:.2f}, {self.fly_position[1]:.2f}, {self.fly_position[2]:.2f}) m")
        status_lines.append(f"Trajectory: {len(self.trajectory_points)} pts")
        
        if self.map_stats:
            status_lines.append(f"Map: {self.map_stats.get('total_points', 0)} pts, "
                              f"{100*self.map_stats.get('exploration_ratio', 0):.1f}% explored")
        
        self._status_text_handle.value = " | ".join(status_lines)
        
    def check_controls(self) -> Dict[str, bool]:
        """
        Check control state.
        
        Returns:
            Dict with control flags: paused, should_step, should_reset
        """
        controls = {
            "paused": self.paused,
            "should_step": self.should_step,
            "should_reset": self.should_reset,
            "speed": self.speed_slider.value,
        }
        
        # Reset step and reset flags after reading
        if self.should_step:
            self.should_step = False
        if self.should_reset:
            self.should_reset = False
            
        return controls
        
    def reset_trajectory(self):
        """Clear trajectory visualization."""
        self.trajectory_points = []
        if self._trajectory_handle is not None:
            self._trajectory_handle.remove()
            self._trajectory_handle = None
            
    def close(self):
        """Close the visualization server."""
        print("\n🛑 Closing interactive visualization server...")
