"""Interactive 3D visualization for fly exploration."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import open3d as o3d
from typing import Optional, Dict
import threading
import time


class Visualizer:
    """
    Real-time visualization of fly flight and point cloud mapping.
    
    Displays:
    - MuJoCo environment view
    - Stereo camera views
    - Accumulated 3D point cloud
    - Status HUD (position, brain activity, exploration)
    """
    
    def __init__(
        self,
        mode: str = "matplotlib",  # "matplotlib" or "open3d" or "both"
        window_size: tuple = (12, 8),
    ):
        """
        Initialize visualizer.
        
        Args:
            mode: Visualization mode
            window_size: Figure size for matplotlib
        """
        self.mode = mode
        self.window_size = window_size
        
        # State
        self.left_img = None
        self.right_img = None
        self.point_cloud = None
        self.fly_position = np.zeros(3)
        self.fly_trajectory = []
        self.brain_stats = {}
        self.control_stats = {}
        self.map_stats = {}
        
        # Matplotlib figure
        if mode in ["matplotlib", "both"]:
            self.fig, self.axes = plt.subplots(3, 3, figsize=window_size)
            self.fig.suptitle("3dfly: Connectome-Driven Exploration", fontsize=14, fontweight='bold')
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            
            # Initialize plots
            self._init_matplotlib_plots()
        
        # Open3D visualizer
        if mode in ["open3d", "both"]:
            self.o3d_vis = o3d.visualization.Visualizer()
            self.o3d_vis.create_window(window_name="3dfly Point Cloud", width=800, height=600)
            self.o3d_cloud = o3d.geometry.PointCloud()
            self.o3d_vis.add_geometry(self.o3d_cloud)
            
            # Coordinate frame
            self.coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
            self.o3d_vis.add_geometry(self.coord_frame)
    
    def _init_matplotlib_plots(self):
        """Initialize matplotlib subplot layout."""
        # Row 0: Camera views and top-view trajectory
        self.ax_left = self.axes[0, 0]
        self.ax_right = self.axes[0, 1]
        self.ax_traj_top = self.axes[0, 2]
        
        self.ax_left.set_title("Left Eye")
        self.ax_left.axis('off')
        self.ax_right.set_title("Right Eye")
        self.ax_right.axis('off')
        self.ax_traj_top.set_title("Trajectory (Top View: XY)")
        self.ax_traj_top.set_xlabel("X (m)")
        self.ax_traj_top.set_ylabel("Y (m)")
        self.ax_traj_top.set_xlim(-5, 5)
        self.ax_traj_top.set_ylim(-5, 5)
        self.ax_traj_top.grid(True, alpha=0.3)
        self.ax_traj_top.set_aspect('equal')
        
        # Row 1: Side-view trajectory, brain activity, control
        self.ax_traj_side = self.axes[1, 0]
        self.ax_brain = self.axes[1, 1]
        self.ax_control = self.axes[1, 2]
        
        self.ax_traj_side.set_title("Trajectory (Side View: XZ)")
        self.ax_traj_side.set_xlabel("X (m)")
        self.ax_traj_side.set_ylabel("Z (m)")
        self.ax_traj_side.set_xlim(-5, 5)
        self.ax_traj_side.set_ylim(0, 3)
        self.ax_traj_side.grid(True, alpha=0.3)
        
        self.ax_brain.set_title("Brain Activity")
        self.ax_brain.set_xlabel("Neuron Index")
        self.ax_brain.set_ylabel("Activity")
        
        self.ax_control.set_title("Flight Commands")
        self.ax_control.set_ylabel("Command Value")
        self.ax_control.set_ylim(-1, 1)
        self.ax_control.grid(True, alpha=0.3)
        
        # Row 2: Status and additional info
        self.ax_status = self.axes[2, 0]
        self.ax_depth = self.axes[2, 1]
        self.ax_looming = self.axes[2, 2]
        
        self.ax_status.set_title("System Status")
        self.ax_status.axis('off')
        
        self.ax_depth.set_title("Brain Depth Estimate")
        self.ax_depth.axis('off')
        
        self.ax_looming.set_title("Looming/Proximity Signals")
        self.ax_looming.set_ylim(0, 1)
        self.ax_looming.grid(True, alpha=0.3)
    
    def update(
        self,
        left_img: Optional[np.ndarray] = None,
        right_img: Optional[np.ndarray] = None,
        fly_position: Optional[np.ndarray] = None,
        brain_activity: Optional[np.ndarray] = None,
        control_action: Optional[np.ndarray] = None,
        point_cloud_mapper = None,
        brain_stats: Optional[Dict] = None,
        control_stats: Optional[Dict] = None,
        looming_features: Optional[np.ndarray] = None,
    ):
        """
        Update visualization with new data.
        
        Args:
            left_img: Left eye image
            right_img: Right eye image
            fly_position: Current position
            brain_activity: Brain activity vector
            control_action: Control commands
            point_cloud_mapper: PointCloudMapper instance
            brain_stats: Brain statistics
            control_stats: Control statistics
            looming_features: Obstacle proximity signals (8 sectors)
        """
        # Store state
        if left_img is not None:
            self.left_img = left_img
        if right_img is not None:
            self.right_img = right_img
        if fly_position is not None:
            self.fly_position = fly_position
            self.fly_trajectory.append(fly_position.copy())
        if brain_stats is not None:
            self.brain_stats = brain_stats
        if control_stats is not None:
            self.control_stats = control_stats
        
        # Get point cloud
        if point_cloud_mapper is not None:
            self.point_cloud = point_cloud_mapper.downsample()
            self.map_stats = point_cloud_mapper.get_statistics()
        
        # Update matplotlib
        if self.mode in ["matplotlib", "both"]:
            self._update_matplotlib(brain_activity, control_action, looming_features)
        
        # Update Open3D
        if self.mode in ["open3d", "both"]:
            self._update_open3d()
    
    def _update_matplotlib(self, brain_activity, control_action, looming_features):
        """Update matplotlib plots."""
        # Left eye
        if self.left_img is not None:
            self.ax_left.clear()
            self.ax_left.imshow(self.left_img)
            self.ax_left.set_title("Left Eye")
            self.ax_left.axis('off')
        
        # Right eye
        if self.right_img is not None:
            self.ax_right.clear()
            self.ax_right.imshow(self.right_img)
            self.ax_right.set_title("Right Eye")
            self.ax_right.axis('off')
        
        # Top-view trajectory (XY)
        if len(self.fly_trajectory) > 1:
            traj = np.array(self.fly_trajectory)
            self.ax_traj_top.clear()
            self.ax_traj_top.plot(traj[:, 0], traj[:, 1], 'b-', alpha=0.5, linewidth=1)
            self.ax_traj_top.scatter(traj[-1, 0], traj[-1, 1], c='r', s=100, marker='o', zorder=5)
            self.ax_traj_top.set_title("Trajectory (Top View: XY)")
            self.ax_traj_top.set_xlabel("X (m)")
            self.ax_traj_top.set_ylabel("Y (m)")
            self.ax_traj_top.set_xlim(-5, 5)
            self.ax_traj_top.set_ylim(-5, 5)
            self.ax_traj_top.grid(True, alpha=0.3)
            self.ax_traj_top.set_aspect('equal')
            
            # Show trajectory range
            dx = traj[:, 0].max() - traj[:, 0].min()
            dy = traj[:, 1].max() - traj[:, 1].min()
            self.ax_traj_top.text(0.02, 0.98, f"ΔX={dx:.2f}m\nΔY={dy:.2f}m", 
                                  transform=self.ax_traj_top.transAxes, 
                                  verticalalignment='top', fontsize=8,
                                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Side-view trajectory (XZ)
        if len(self.fly_trajectory) > 1:
            traj = np.array(self.fly_trajectory)
            self.ax_traj_side.clear()
            self.ax_traj_side.plot(traj[:, 0], traj[:, 2], 'g-', alpha=0.5, linewidth=1)
            self.ax_traj_side.scatter(traj[-1, 0], traj[-1, 2], c='r', s=100, marker='o', zorder=5)
            self.ax_traj_side.set_title("Trajectory (Side View: XZ)")
            self.ax_traj_side.set_xlabel("X (m)")
            self.ax_traj_side.set_ylabel("Z (m)")
            self.ax_traj_side.set_xlim(-5, 5)
            self.ax_traj_side.set_ylim(0, 3)
            self.ax_traj_side.grid(True, alpha=0.3)
            
            # Show vertical range
            dz = traj[:, 2].max() - traj[:, 2].min()
            self.ax_traj_side.text(0.02, 0.98, f"ΔZ={dz:.2f}m", 
                                   transform=self.ax_traj_side.transAxes, 
                                   verticalalignment='top', fontsize=8,
                                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        
        # Brain activity
        if brain_activity is not None:
            self.ax_brain.clear()
            # Sample for visualization
            n_sample = min(200, len(brain_activity))
            indices = np.linspace(0, len(brain_activity)-1, n_sample, dtype=int)
            self.ax_brain.bar(indices, brain_activity[indices], width=1, alpha=0.7)
            self.ax_brain.set_title("Brain Activity (sampled)")
            self.ax_brain.set_xlabel("Neuron Index")
            self.ax_brain.set_ylabel("Activity")
        
        # Control commands
        if control_action is not None:
            self.ax_control.clear()
            labels = ["Fwd", "Up", "Roll", "Pitch", "Yaw"]
            colors = ['b', 'g', 'r', 'orange', 'purple']
            self.ax_control.bar(labels, control_action, color=colors, alpha=0.7)
            self.ax_control.set_title("Flight Commands")
            self.ax_control.set_ylabel("Command Value")
            self.ax_control.set_ylim(-1, 1)
            self.ax_control.axhline(0, color='k', linewidth=0.5)
            self.ax_control.grid(True, alpha=0.3, axis='y')
        
        # Looming/proximity features
        if looming_features is not None:
            self.ax_looming.clear()
            sector_labels = ["Fwd", "FL", "L", "BL", "Back", "BR", "R", "FR"]
            colors = ['red' if loom > 0.5 else 'orange' if loom > 0.3 else 'yellow' 
                     for loom in looming_features]
            self.ax_looming.bar(sector_labels, looming_features, color=colors, alpha=0.7)
            self.ax_looming.set_title("Looming/Proximity (Brain Input)")
            self.ax_looming.set_ylabel("Proximity")
            self.ax_looming.set_ylim(0, 1)
            self.ax_looming.axhline(0.3, color='orange', linestyle='--', linewidth=0.5, alpha=0.5)
            self.ax_looming.axhline(0.5, color='red', linestyle='--', linewidth=0.5, alpha=0.5)
            self.ax_looming.grid(True, alpha=0.3, axis='y')
        
        # Status text
        self.ax_status.clear()
        self.ax_status.axis('off')
        
        status_text = f"Position: ({self.fly_position[0]:.2f}, {self.fly_position[1]:.2f}, {self.fly_position[2]:.2f}) m\n"
        status_text += f"Trajectory points: {len(self.fly_trajectory)}\n\n"
        
        if self.brain_stats:
            status_text += "Brain:\n"
            for key, val in self.brain_stats.items():
                status_text += f"  {key}: {val:.3f}\n"
            status_text += "\n"
        
        if self.map_stats:
            status_text += "Mapping:\n"
            status_text += f"  Total points: {self.map_stats.get('total_points', 0)}\n"
            status_text += f"  Occupied voxels: {self.map_stats.get('occupied_voxels', 0)}\n"
            status_text += f"  Exploration: {100*self.map_stats.get('exploration_ratio', 0):.1f}%\n"
        
        self.ax_status.text(0.05, 0.95, status_text, transform=self.ax_status.transAxes,
                           verticalalignment='top', fontfamily='monospace', fontsize=9)
        
        plt.pause(0.001)
    
    def _update_open3d(self):
        """Update Open3D point cloud."""
        if self.point_cloud is not None and len(self.point_cloud.points) > 0:
            self.o3d_cloud.points = self.point_cloud.points
            self.o3d_cloud.colors = self.point_cloud.colors
            self.o3d_vis.update_geometry(self.o3d_cloud)
        
        self.o3d_vis.poll_events()
        self.o3d_vis.update_renderer()
    
    def show(self, block: bool = True):
        """Show visualization (blocking or non-blocking)."""
        if self.mode in ["matplotlib", "both"]:
            if block:
                plt.show()
            else:
                plt.ion()
                plt.show()
    
    def close(self):
        """Close visualization."""
        if self.mode in ["matplotlib", "both"]:
            plt.close(self.fig)
        
        if self.mode in ["open3d", "both"]:
            self.o3d_vis.destroy_window()
    
    def save_figure(self, filename: str):
        """Save current matplotlib figure."""
        if self.mode in ["matplotlib", "both"]:
            self.fig.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"✓ Saved figure: {filename}")
