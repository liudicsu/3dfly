"""MuJoCo environment for fruit fly flight."""

import numpy as np
import mujoco
from typing import Tuple, Optional, Dict
import tempfile
from pathlib import Path

from threedfly.sim.mjcf import get_fly_mjcf


class FlyEnvironment:
    """
    MuJoCo environment for fruit fly 3D flight.
    
    Provides stereo camera views, applies control commands,
    and tracks fly state.
    """
    
    def __init__(
        self,
        render_mode: Optional[str] = "rgb_array",
        camera_width: int = 64,
        camera_height: int = 48,
    ):
        """
        Initialize environment.
        
        Args:
            render_mode: "rgb_array" or "human" or None (headless)
            camera_width: Eye camera width in pixels
            camera_height: Eye camera height in pixels
        """
        self.render_mode = render_mode
        self.camera_width = camera_width
        self.camera_height = camera_height
        
        # Load MJCF model
        mjcf_xml = get_fly_mjcf()
        self.model = mujoco.MjModel.from_xml_string(mjcf_xml)
        self.data = mujoco.MjData(self.model)
        
        # Camera IDs
        self.left_eye_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "left_eye")
        self.right_eye_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "right_eye")
        
        # Renderer for cameras
        if render_mode is not None:
            self.renderer = mujoco.Renderer(self.model, camera_height, camera_width)
        else:
            self.renderer = None
        
        # Control bounds
        self.thrust_max = 0.01
        self.torque_max = 0.001
        
        # State tracking
        self.time = 0.0
        self.step_count = 0
        
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reset environment.
        
        Returns:
            (left_eye_image, right_eye_image) as uint8 arrays
        """
        if seed is not None:
            np.random.seed(seed)
        
        mujoco.mj_resetData(self.model, self.data)
        
        # Randomize initial position slightly
        self.data.qpos[0] = np.random.uniform(-0.5, 0.5)  # x
        self.data.qpos[1] = np.random.uniform(-0.5, 0.5)  # y
        self.data.qpos[2] = np.random.uniform(1.0, 2.0)   # z (height)
        
        # Random initial orientation
        angle = np.random.uniform(0, 2 * np.pi)
        self.data.qpos[3:7] = [np.cos(angle/2), 0, 0, np.sin(angle/2)]  # quaternion
        
        mujoco.mj_forward(self.model, self.data)
        
        self.time = 0.0
        self.step_count = 0
        
        return self.get_eye_images()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Step simulation.
        
        Args:
            action: [thrust_forward, thrust_up, torque_roll, torque_pitch, torque_yaw]
        
        Returns:
            (left_eye_image, right_eye_image, info_dict)
        """
        # Clip and apply control
        action = np.clip(action, -1.0, 1.0)
        
        self.data.ctrl[0] = action[0] * self.thrust_max  # forward thrust
        self.data.ctrl[1] = action[1] * self.thrust_max  # upward thrust
        self.data.ctrl[2] = action[2] * self.torque_max  # roll
        self.data.ctrl[3] = action[3] * self.torque_max  # pitch
        self.data.ctrl[4] = action[4] * self.torque_max  # yaw
        
        # Step physics
        mujoco.mj_step(self.model, self.data)
        
        self.time = self.data.time
        self.step_count += 1
        
        # Get observations
        left_img, right_img = self.get_eye_images()
        
        # Info
        info = {
            "position": self.get_fly_position(),
            "velocity": self.get_fly_velocity(),
            "orientation": self.get_fly_orientation(),
        }
        
        return left_img, right_img, info
    
    def get_eye_images(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Render stereo camera views.
        
        Returns:
            (left_eye_image, right_eye_image) as uint8 RGB arrays (H, W, 3)
        """
        if self.renderer is None:
            # Headless mode: return dummy images
            shape = (self.camera_height, self.camera_width, 3)
            return np.zeros(shape, dtype=np.uint8), np.zeros(shape, dtype=np.uint8)
        
        # Render left eye
        self.renderer.update_scene(self.data, camera=self.left_eye_id)
        left_img = self.renderer.render()
        
        # Render right eye
        self.renderer.update_scene(self.data, camera=self.right_eye_id)
        right_img = self.renderer.render()
        
        return left_img.copy(), right_img.copy()
    
    def get_fly_position(self) -> np.ndarray:
        """Get fly body position (x, y, z)."""
        return self.data.qpos[:3].copy()
    
    def get_fly_velocity(self) -> np.ndarray:
        """Get fly body velocity."""
        return self.data.qvel[:3].copy()
    
    def get_fly_orientation(self) -> np.ndarray:
        """Get fly orientation quaternion (w, x, y, z)."""
        return self.data.qpos[3:7].copy()
    
    def get_camera_poses(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get left and right camera poses in world coordinates.
        
        Returns:
            (left_pose, right_pose) as 4x4 transformation matrices
        """
        mujoco.mj_forward(self.model, self.data)
        
        # Get fly body pose
        fly_pos = self.get_fly_position()
        fly_quat = self.get_fly_orientation()  # (w, x, y, z)
        
        # Convert quaternion to rotation matrix
        fly_rot = self._quat_to_rotation_matrix(fly_quat)
        
        # Camera positions in fly body frame (forward, sideways, up)
        # Cameras point forward (+x in fly frame)
        left_cam_offset = np.array([0.015, 0.006, 0.002])   # forward, left, up
        right_cam_offset = np.array([0.015, -0.006, 0.002])  # forward, right, up
        
        # Transform camera positions to world frame
        left_pos_world = fly_pos + fly_rot @ left_cam_offset
        right_pos_world = fly_pos + fly_rot @ right_cam_offset
        
        # Build 4x4 transformation matrices
        # Cameras inherit fly's rotation (pointing forward in fly's body frame)
        left_pose = np.eye(4)
        left_pose[:3, :3] = fly_rot
        left_pose[:3, 3] = left_pos_world
        
        right_pose = np.eye(4)
        right_pose[:3, :3] = fly_rot
        right_pose[:3, 3] = right_pos_world
        
        return left_pose, right_pose
    
    def _quat_to_rotation_matrix(self, quat: np.ndarray) -> np.ndarray:
        """
        Convert quaternion to 3x3 rotation matrix.
        
        Args:
            quat: Quaternion (w, x, y, z)
        
        Returns:
            3x3 rotation matrix
        """
        w, x, y, z = quat
        
        # Rotation matrix from quaternion
        R = np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
        ])
        
        return R
    
    def check_collision(self) -> bool:
        """Check if fly has collided with environment."""
        # Simple check: z position too low or velocity too high
        pos = self.get_fly_position()
        vel = self.get_fly_velocity()
        
        if pos[2] < 0.1:  # Too close to ground
            return True
        
        if np.linalg.norm(vel) > 5.0:  # Too fast (unlikely for fly)
            return True
        
        # Check contacts
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if contact.geom1 == 0 or contact.geom2 == 0:  # thorax geom
                return True
        
        return False
    
    def close(self):
        """Close renderer."""
        if self.renderer is not None:
            self.renderer.close()
