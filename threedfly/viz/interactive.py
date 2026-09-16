"""Interactive 3D visualization using viser."""

import numpy as np
import viser
import time
from typing import Optional, Dict, List
import threading


class InteractiveVisualizer:
    """
    Interactive 3D god's-eye visualization for fly exploration.
    
    Shows:
    - MuJoCo environment (room walls, obstacles)
    - Fruit fly body and position
    - Flight trajectory
    - 3D point cloud reconstruction
    - Live camera views
    - Brain and control status
    
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
        
        self.brain_stats: Dict = {}
        self.control_stats: Dict = {}
        self.map_stats: Dict = {}
        
        # Control state
        self.paused = False
        self.should_step = False
        self.should_reset = False
        
        # Visualization handles
        self._fly_handle = None
        self._trajectory_handle = None
        self._point_cloud_handle = None
        self._left_img_handle = None
        self._right_img_handle = None
        self._status_text_handle = None
        
        # Setup scene
        self._setup_environment()
        self._setup_ui_controls()
        
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
            hint="Display flight path"
        )
        
        self.show_point_cloud = self.server.gui.add_checkbox(
            "Show Point Cloud",
            initial_value=True,
            hint="Display reconstructed 3D map"
        )
        
        self.show_cameras = self.server.gui.add_checkbox(
            "Show Eye Views",
            initial_value=True,
            hint="Display stereo camera images"
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
            
        # Update stats
        if brain_stats is not None:
            self.brain_stats = brain_stats
        if control_stats is not None:
            self.control_stats = control_stats
            
        # Get point cloud data
        if point_cloud_mapper is not None:
            downsampled = point_cloud_mapper.downsample()
            if len(downsampled.points) > 0:
                self.point_cloud_points = np.asarray(downsampled.points)
                self.point_cloud_colors = np.asarray(downsampled.colors)
            self.map_stats = point_cloud_mapper.get_statistics()
            
        # Render updates
        self._render_fly()
        self._render_trajectory()
        self._render_point_cloud()
        self._render_camera_views()
        self._render_status()
        
    def _render_fly(self):
        """Render fruit fly body."""
        # Remove old fly
        if self._fly_handle is not None:
            self._fly_handle.remove()
            
        # Fly body (ellipsoid approximated as sphere)
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
        
    def _render_camera_views(self):
        """Render stereo camera views as overlaid images."""
        if not self.show_cameras.value:
            if self._left_img_handle is not None:
                self._left_img_handle.remove()
                self._left_img_handle = None
            if self._right_img_handle is not None:
                self._right_img_handle.remove()
                self._right_img_handle = None
            return
            
        # Left eye
        if self.left_img is not None:
            if self._left_img_handle is not None:
                self._left_img_handle.remove()
            self._left_img_handle = self.server.scene.add_image(
                "/world/left_eye_view",
                image=self.left_img,
                render_width=0.3,
                render_height=0.225,
                position=(self.fly_position[0] - 0.5, self.fly_position[1] - 0.3, self.fly_position[2] + 0.5),
                wxyz=(1.0, 0.0, 0.0, 0.0),
            )
            
        # Right eye
        if self.right_img is not None:
            if self._right_img_handle is not None:
                self._right_img_handle.remove()
            self._right_img_handle = self.server.scene.add_image(
                "/world/right_eye_view",
                image=self.right_img,
                render_width=0.3,
                render_height=0.225,
                position=(self.fly_position[0] - 0.5, self.fly_position[1] + 0.3, self.fly_position[2] + 0.5),
                wxyz=(1.0, 0.0, 0.0, 0.0),
            )
            
    def _render_status(self):
        """Render status text in UI."""
        status_lines = []
        status_lines.append(f"Position: ({self.fly_position[0]:.2f}, {self.fly_position[1]:.2f}, {self.fly_position[2]:.2f}) m")
        status_lines.append(f"Trajectory: {len(self.trajectory_points)} points")
        
        if self.map_stats:
            status_lines.append("")
            status_lines.append(f"Point Cloud: {self.map_stats.get('total_points', 0)} points")
            status_lines.append(f"Occupied Voxels: {self.map_stats.get('occupied_voxels', 0)}")
            status_lines.append(f"Exploration: {100*self.map_stats.get('exploration_ratio', 0):.1f}%")
            
        if self.brain_stats:
            status_lines.append("")
            status_lines.append("Brain:")
            for key, val in list(self.brain_stats.items())[:3]:  # Show first 3
                status_lines.append(f"  {key}: {val:.3f}")
                
        self._status_text_handle.value = "\n".join(status_lines)
        
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
