"""MuJoCo environment for fruit fly flight."""

import numpy as np
import mujoco
from typing import Tuple, Optional, Dict
import tempfile
from pathlib import Path

from threedfly.sim.mjcf import get_fly_mjcf
from threedfly.sim.wings import stroke_averaged_forces, visual_wing_angles

# Engineered aero: linear+quadratic drag sized for ~1 m/s cruise, not a CFD model.
DRAG_LINEAR = 0.008       # N·s/m
DRAG_QUADRATIC = 0.014    # N·s²/m²
# Weak engineered altitude hold around a mid-room hover height.
HOVER_Z = 1.5
ALTITUDE_HOLD_KP = 0.025  # N/m
ALTITUDE_HOLD_KD = 0.012  # N·s/m
# Room walls at ±5 m, height ≈ 4 m; clamp only for real penetrations.
ROOM_XY_LIMIT = 4.85
ROOM_Z_MIN = 0.25
ROOM_Z_MAX = 3.70
# Visual-only wingbeat (stroke-averaged forces, not a 200 Hz aero sim).
WINGBEAT_HZ = 200.0
# Stroke-difference torques sized for the ~1.5 g, 2.5 cm scaled body.
WING_YAW_GAIN = 8e-6      # N·m per unit (L−R) stroke
WING_ROLL_GAIN = 3e-6     # N·m per unit (L−R) stroke
# Minimal engineered angular-rate damping kept as a safety net. Primary
# haltere path is the connectome mapping (see select_haltere_afferent_neurons).
SAFETY_HALTERE_KD = 2e-6  # N·m / (rad/s)


class FlyEnvironment:
    """
    MuJoCo environment for fruit fly 3D flight.
    
    Provides stereo camera views, applies control commands,
    and tracks fly state.
    """
    
    def __init__(
        self,
        render_mode: Optional[str] = "rgb_array",
        camera_width: int = 160,
        camera_height: int = 120,
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
        
        # Control bounds in SI units (gainprm=1). Max lift ~3x body weight (~0.016 N).
        self.thrust_max = 0.05
        self.torque_max = 5e-6
        self.lift_max = self.thrust_max  # peak combined wing lift; hover = weight/lift_max
        # Minimal engineered damping (safety). Brain haltere afferents are the primary path.
        self.haltere_kd = SAFETY_HALTERE_KD
        self.drag_linear = DRAG_LINEAR
        self.drag_quadratic = DRAG_QUADRATIC
        self.hover_z = HOVER_Z
        self.altitude_hold_kp = ALTITUDE_HOLD_KP
        self.altitude_hold_kd = ALTITUDE_HOLD_KD
        self.wing_yaw_gain = WING_YAW_GAIN
        self.wing_roll_gain = WING_ROLL_GAIN
        
        self._fly_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "fly"
        )
        self._fly_geom_ids = self._collect_fly_geom_ids()
        self._wing_left_jnt = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "wing_left_stroke"
        )
        self._wing_right_jnt = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "wing_right_stroke"
        )
        self._wing_left_qposadr = int(self.model.jnt_qposadr[self._wing_left_jnt])
        self._wing_right_qposadr = int(self.model.jnt_qposadr[self._wing_right_jnt])
        # Align physics with the 20 ms brain/control tick (MuJoCo dt = 2 ms → 10 substeps).
        self.control_dt = 0.02
        self.n_substeps = max(1, int(round(self.control_dt / float(self.model.opt.timestep))))
        
        # State tracking
        self.time = 0.0
        self.step_count = 0
        self._wing_phase = 0.0
        self.last_wing_strokes = np.array([0.0, 0.0], dtype=np.float64)
        self.last_wing_angles = np.array([0.0, 0.0], dtype=np.float64)

    def _collect_fly_geom_ids(self) -> set:
        """Geoms whose body is the fly or a descendant of the fly body."""
        fly_bodies = set()
        for bid in range(self.model.nbody):
            b = bid
            while b > 0:
                if b == self._fly_body_id:
                    fly_bodies.add(bid)
                    break
                parent = int(self.model.body_parentid[b])
                if parent == b:
                    break
                b = parent
        return {
            gid
            for gid in range(self.model.ngeom)
            if int(self.model.geom_bodyid[gid]) in fly_bodies
        }

    @property
    def hover_action(self) -> float:
        """Normalized up-thrust that balances weight (lift = mg at rest)."""
        mass = float(self.model.body_subtreemass[self._fly_body_id])
        weight = mass * abs(float(self.model.opt.gravity[2]))
        return float(np.clip(weight / max(self.thrust_max, 1e-12), 0.0, 1.0))
        
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
        self._wing_phase = 0.0
        
        return self.get_eye_images()
    
    def step(
        self,
        action: np.ndarray,
        wing_strokes: Optional[Tuple[float, float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Step simulation.
        
        Args:
            action: [thrust_forward, thrust_up, torque_roll, torque_pitch, torque_yaw]
            wing_strokes: Optional (left, right) stroke amplitudes in [0, 1].
                When omitted, strokes are derived from thrust_up and yaw so the
                legacy 5-vector API still flies.
        
        Returns:
            (left_eye_image, right_eye_image, info_dict)
        """
        # Clip and apply control for n_substeps so one control tick = 20 ms of physics.
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        if wing_strokes is None:
            mean = float(np.clip(action[1], 0.0, 1.0))
            diff = float(np.clip(action[4], -1.0, 1.0))
            stroke_l = float(np.clip(mean + 0.5 * diff, 0.0, 1.0))
            stroke_r = float(np.clip(mean - 0.5 * diff, 0.0, 1.0))
        else:
            stroke_l = float(np.clip(wing_strokes[0], 0.0, 1.0))
            stroke_r = float(np.clip(wing_strokes[1], 0.0, 1.0))
        self.last_wing_strokes = np.array([stroke_l, stroke_r], dtype=np.float64)

        for _ in range(self.n_substeps):
            # Actuators are kept in the MJCF for tests/API; forces are applied below.
            self.data.ctrl[:] = 0.0
            self.data.xfrc_applied[:] = 0.0
            self.data.qfrc_applied[:] = 0.0

            wing_force_body, wing_torque_body = stroke_averaged_forces(
                stroke_l,
                stroke_r,
                lift_max=self.lift_max,
                yaw_gain=self.wing_yaw_gain,
                roll_gain=self.wing_roll_gain,
            )
            # Forward thrust and residual DN torques stay in the body frame.
            wing_force_body[0] += float(action[0]) * self.thrust_max
            wing_torque_body[0] += float(action[2]) * self.torque_max
            wing_torque_body[1] += float(action[3]) * self.torque_max
            wing_torque_body[2] += float(action[4]) * self.torque_max

            rot = self._quat_to_rotation_matrix(self.get_fly_orientation())
            force = rot @ wing_force_body
            torque = rot @ wing_torque_body

            # Haltere-like safety damping on body angular rate (engineered residual).
            angvel = self.data.qvel[3:6]
            torque = torque - rot @ (self.haltere_kd * np.asarray(angvel, dtype=np.float64))

            # Aerodynamic drag F = -b v - c |v| v, plus a weak engineered altitude hold.
            vel = self.data.qvel[:3]
            speed = float(np.linalg.norm(vel))
            drag = -self.drag_linear * vel
            if speed > 1e-8:
                drag = drag - self.drag_quadratic * speed * vel
            z_err = self.hover_z - float(self.data.qpos[2])
            alt = self.altitude_hold_kp * z_err - self.altitude_hold_kd * float(vel[2])
            force = force + np.array([drag[0], drag[1], drag[2] + alt], dtype=np.float64)

            point = self.data.subtree_com[self._fly_body_id].copy()
            mujoco.mj_applyFT(
                self.model,
                self.data,
                force,
                torque,
                point,
                self._fly_body_id,
                self.data.qfrc_applied,
            )
            mujoco.mj_step(self.model, self.data)
            self._animate_wings(stroke_l, stroke_r)

        self.time = self.data.time
        self.step_count += 1
        
        # Get observations
        left_img, right_img = self.get_eye_images()
        
        # Info
        info = {
            "position": self.get_fly_position(),
            "velocity": self.get_fly_velocity(),
            "orientation": self.get_fly_orientation(),
            "wing_strokes": self.last_wing_strokes.copy(),
            "wing_angles": self.last_wing_angles.copy(),
        }
        
        return left_img, right_img, info

    def _animate_wings(self, stroke_l: float, stroke_r: float) -> None:
        """Overwrite wing-hinge qpos with a visual 200 Hz flap (not simulated aero)."""
        dt = float(self.model.opt.timestep)
        self._wing_phase += 2.0 * np.pi * WINGBEAT_HZ * dt
        left_ang, right_ang = visual_wing_angles(self._wing_phase, stroke_l, stroke_r)
        self.data.qpos[self._wing_left_qposadr] = left_ang
        self.data.qpos[self._wing_right_qposadr] = right_ang
        self.last_wing_angles = np.array([left_ang, right_ang], dtype=np.float64)
        mujoco.mj_forward(self.model, self.data)
    
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

    def get_fly_angular_velocity(self) -> np.ndarray:
        """Body-frame angular velocity (rad/s) from the freejoint."""
        return self.data.qvel[3:6].copy()

    def get_wing_angles(self) -> np.ndarray:
        """Current visual wing hinge angles (left, right) in radians."""
        return np.array(
            [
                float(self.data.qpos[self._wing_left_qposadr]),
                float(self.data.qpos[self._wing_right_qposadr]),
            ],
            dtype=np.float64,
        )
    
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
        """True only for ground proximity or fly-vs-environment contacts.

        Overspeed is not a collision (drag limits terminal speed instead).
        Fly geoms are identified by body ancestry, not geom index 0 (the floor).
        """
        pos = self.get_fly_position()
        if pos[2] < 0.1:
            return True

        fly_geoms = self._fly_geom_ids
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            g1_fly = int(contact.geom1) in fly_geoms
            g2_fly = int(contact.geom2) in fly_geoms
            if g1_fly != g2_fly:
                return True

        return False
    
    def recover_from_collision(self, recovery_distance: float = 0.15) -> None:
        """
        Minimal soft collision recovery: small back-off as safety fallback.
        
        The primary avoidance mechanism is brain-mediated (looming → LPLC2/LC4 → DNs).
        This recovery is a last-resort safety net to prevent simulation lockup.
        
        Args:
            recovery_distance: Small distance to back off (default 15cm, down from 30cm)
        """
        pos = self.get_fly_position()
        
        # Minimal deterministic nudge — never a random teleport
        recovery_offset = np.zeros(3)
        
        if pos[2] < ROOM_Z_MIN:
            recovery_offset[2] = recovery_distance
        elif pos[2] > ROOM_Z_MAX:
            recovery_offset[2] = -recovery_distance
        elif self.data.ncon > 0:
            contact = None
            for i in range(self.data.ncon):
                c = self.data.contact[i]
                g1_fly = int(c.geom1) in self._fly_geom_ids
                g2_fly = int(c.geom2) in self._fly_geom_ids
                if g1_fly != g2_fly:
                    contact = c
                    break
            if contact is None:
                contact = self.data.contact[0]
            contact_normal = contact.frame[:3]
            if np.linalg.norm(contact_normal) > 0.1:
                contact_normal = contact_normal / np.linalg.norm(contact_normal)
            recovery_offset = -contact_normal * recovery_distance
        
        offset_norm = np.linalg.norm(recovery_offset)
        if offset_norm > recovery_distance and offset_norm > 0:
            recovery_offset = recovery_offset * (recovery_distance / offset_norm)
        
        print(
            f"[collision-recovery] fired pos=({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}) "
            f"offset=({recovery_offset[0]:.3f},{recovery_offset[1]:.3f},{recovery_offset[2]:.3f}) "
            f"ncon={self.data.ncon}"
        )
        
        self.data.qpos[:3] = pos + recovery_offset
        # Clamp only at actual room bounds minus a small margin, never an inner ±1.9 box.
        self.data.qpos[0] = np.clip(self.data.qpos[0], -ROOM_XY_LIMIT, ROOM_XY_LIMIT)
        self.data.qpos[1] = np.clip(self.data.qpos[1], -ROOM_XY_LIMIT, ROOM_XY_LIMIT)
        self.data.qpos[2] = np.clip(self.data.qpos[2], ROOM_Z_MIN, ROOM_Z_MAX)
        self.data.qvel[:3] *= 0.3
        
        mujoco.mj_forward(self.model, self.data)
    
    def close(self):
        """Close renderer."""
        if self.renderer is not None:
            self.renderer.close()
