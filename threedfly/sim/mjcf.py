"""Generate MJCF (MuJoCo XML) models for fly and environment.

The fly is a flybody-derived (Apache-2.0) fruit-fly model: head, thorax,
abdomen, wings, halteres, and six legs with flybody mesh assets and a
thorax-heavy mass distribution. See `threedfly/sim/scale.py` and THIRD_PARTY.md
for license, citation, and the 10× length / ~1000× mass scaling into the 10 m
room.

Wings have hinge joints (visual flapping). Legs are welded in a retracted
flight pose. Aerodynamic lift is applied in Python (stroke-averaged), not via
a 200 Hz wingbeat.
"""

from pathlib import Path

# Get path to assets directory
ASSETS_DIR = Path(__file__).parent.parent.parent / "assets" / "fly_meshes"

# OBJ millimetres → metres (10× life-size). Keep this literal so tests can exec this file.
MESH_SCALE = 0.01
_S = f"{MESH_SCALE} {MESH_SCALE} {MESH_SCALE}"

FLY_MJCF = """
<mujoco model="fruit_fly">
  <compiler angle="radian" coordinate="local" meshdir="{meshdir}"/>
  
  <option timestep="0.002" gravity="0 0 -9.81"/>
  
  <asset>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512" 
             rgb1="0.08 0.08 0.10" rgb2="0.75 0.72 0.65"/>
    <material name="grid_mat" texture="grid" texrepeat="16 16" reflectance="0.08"/>
    
    <!-- High-contrast ~20 cm checkers: LPLC2/LC4/DNp03 need expanding
         edges, not a flat beige field. texrepeat on a 10 m × 4 m face. -->
    <texture name="wall_check" type="2d" builtin="checker" width="256" height="256"
             rgb1="0.02 0.02 0.04" rgb2="0.95 0.93 0.88"/>
    <material name="wall_mat" texture="wall_check" texrepeat="48 20" reflectance="0.04"/>
    <texture name="obs_check" type="2d" builtin="checker" width="128" height="128"
             rgb1="0.05 0.04 0.04" rgb2="0.85 0.25 0.18"/>
    <material name="obs_mat" texture="obs_check" texrepeat="6 6" reflectance="0.04"/>
    
    <!-- flybody mesh assets (Apache-2.0). Scale: millimetres × 0.01 → metres (10× life-size). -->
    <mesh name="thorax_mesh" file="thorax_body.obj" scale=""" + f'"{_S}"' + """/>
    <mesh name="head_mesh" file="head_body.obj" scale=""" + f'"{_S}"' + """/>
    <mesh name="abdomen_mesh" file="abdomen_1_body.obj" scale=""" + f'"{_S}"' + """/>
    <mesh name="wing_left_mesh" file="wing_left_membrane.obj" scale=""" + f'"{_S}"' + """/>
    <mesh name="wing_right_mesh" file="wing_right_membrane.obj" scale=""" + f'"{_S}"' + """/>
    <mesh name="leg_femur_mesh" file="femur_T2_left_body.obj" scale=""" + f'"{_S}"' + """/>
    <mesh name="leg_tibia_mesh" file="tibia_T2_left_body.obj" scale=""" + f'"{_S}"' + """/>
    <mesh name="leg_femur_r_mesh" file="femur_T2_right_body.obj" scale=""" + f'"{_S}"' + """/>
    <mesh name="leg_tibia_r_mesh" file="tibia_T2_right_body.obj" scale=""" + f'"{_S}"' + """/>
    <mesh name="haltere_left_mesh" file="haltere_left_body.obj" scale=""" + f'"{_S}"' + """/>
    <mesh name="haltere_right_mesh" file="haltere_right_body.obj" scale=""" + f'"{_S}"' + """/>
    
    <material name="fly_body_mat" rgba="0.25 0.15 0.08 1"/>
    <material name="fly_wing_mat" rgba="0.7 0.7 0.8 0.4"/>
    <material name="fly_haltere_mat" rgba="0.35 0.25 0.12 1"/>
  </asset>
  
  <worldbody>
    <!-- Ground plane (geom 0; collision tests rely on this). -->
    <geom name="floor" type="plane" size="10 10 0.1" material="grid_mat"/>
    
    <!-- Room walls -->
    <body name="wall_north" pos="5 0 2">
      <geom type="box" size="0.1 5 2" material="wall_mat"/>
    </body>
    <body name="wall_south" pos="-5 0 2">
      <geom type="box" size="0.1 5 2" material="wall_mat"/>
    </body>
    <body name="wall_east" pos="0 5 2">
      <geom type="box" size="5 0.1 2" material="wall_mat"/>
    </body>
    <body name="wall_west" pos="0 -5 2">
      <geom type="box" size="5 0.1 2" material="wall_mat"/>
    </body>
    
    <!-- Some obstacles for exploration -->
    <body name="obstacle_1" pos="2 2 0.5">
      <geom type="box" size="0.3 0.3 0.5" material="obs_mat"/>
    </body>
    <body name="obstacle_2" pos="-2 -2 0.3">
      <geom type="cylinder" size="0.2 0.3" material="obs_mat"/>
    </body>
    <body name="obstacle_3" pos="-1 3 0.4">
      <geom type="sphere" size="0.4" material="obs_mat"/>
    </body>
    
    <!-- Fruit fly: flybody topology (head / thorax / abdomen / wings / legs / halteres).
         Masses keep flybody's thorax-heavy ratios, scaled to ~1.5 g total (see scale.py). -->
    <body name="fly" pos="0 0 1.5">
      <freejoint name="fly_joint"/>
      <!-- Body-frame control site at subtree COM so residual torques do not induce tumble. -->
      <site name="thorax_ctrl" pos="-0.00112645 -0.00015024 0.01175987" size="0.001" rgba="1 0 0 0"/>
      
      <geom name="thorax" type="mesh" mesh="thorax_mesh" 
            material="fly_body_mat" mass="0.001"/>
      
      <!-- Head with stereo cameras (neck hinge, stiff). -->
      <body name="head" pos="0.012 0 0">
        <joint name="neck_pitch" type="hinge" axis="0 1 0" range="-0.3 0.3"
               stiffness="0.05" damping="0.01"/>
        <geom name="head_geom" type="mesh" mesh="head_mesh" 
              material="fly_body_mat" mass="0.0002" contype="0" conaffinity="0"/>
        
        <!-- 120° FOV, 30° outward cant. A 90° forward FOV misses the ±Y
             walls (bearing 90°) so DNp03 never saw a true side loom. -->
        <camera name="left_eye" pos="0.003 0.006 0.002"
                xyaxes="0.500 -0.866 0.000  0.296 0.171 0.940"
                fovy="120" mode="fixed"/>
        <!-- Right X is mirrored (+Y). Y must also flip so X×Y stays forward
             (MuJoCo looks along −Z). That inverts the image; stereo_to_hex
             mirrors V so flyvis still sees upright flow. -->
        <camera name="right_eye" pos="0.003 -0.006 0.002"
                xyaxes="0.500 0.866 0.000  -0.296 0.171 -0.940"
                fovy="120" mode="fixed"/>
      </body>
      
      <!-- Wings: stroke hinge for visual flapping; aero force is applied in Python. -->
      <body name="left_wing" pos="-0.002 0.008 0.002">
        <joint name="wing_left_stroke" type="hinge" axis="1 0 0" range="-1.2 1.2"
               damping="0.002" stiffness="0.0"/>
        <site name="left_wing_aero" pos="0 0.006 0" size="0.001" rgba="0 1 0 0"/>
        <geom name="left_wing_geom" type="mesh" mesh="wing_left_mesh" 
              material="fly_wing_mat" mass="0.00005" contype="0" conaffinity="0"/>
      </body>
      <body name="right_wing" pos="-0.002 -0.008 0.002">
        <joint name="wing_right_stroke" type="hinge" axis="1 0 0" range="-1.2 1.2"
               damping="0.002" stiffness="0.0"/>
        <site name="right_wing_aero" pos="0 -0.006 0" size="0.001" rgba="0 1 0 0"/>
        <geom name="right_wing_geom" type="mesh" mesh="wing_right_mesh" 
              material="fly_wing_mat" mass="0.00005" contype="0" conaffinity="0"/>
      </body>
      
      <!-- Abdomen (pitch hinge, stiff). -->
      <body name="abdomen" pos="-0.012 0 -0.002">
        <joint name="abdomen_pitch" type="hinge" axis="0 1 0" range="-0.2 0.2"
               stiffness="0.08" damping="0.02"/>
        <geom name="abdomen_geom" type="mesh" mesh="abdomen_mesh" 
              material="fly_body_mat" mass="0.0003" contype="0" conaffinity="0"/>
      </body>
      
      <!-- Halteres (flybody meshes). Angular rate is read in Python, not from these joints. -->
      <body name="haltere_left" pos="-0.005 0.005 -0.001">
        <geom name="haltere_left_geom" type="mesh" mesh="haltere_left_mesh"
              material="fly_haltere_mat" mass="0.000005" contype="0" conaffinity="0"/>
      </body>
      <body name="haltere_right" pos="-0.005 -0.005 -0.001">
        <geom name="haltere_right_geom" type="mesh" mesh="haltere_right_mesh"
              material="fly_haltere_mat" mass="0.000005" contype="0" conaffinity="0"/>
      </body>
      
      <!-- Six legs, retracted flight pose, welded (no active joints). T2 meshes reused. -->
      <body name="left_leg_T1" pos="0.006 0.005 -0.002" euler="0.4 -1.1 0.5">
        <geom name="left_femur_T1" type="mesh" mesh="leg_femur_mesh" 
              material="fly_body_mat" mass="0.000008" contype="0" conaffinity="0"/>
        <body name="left_tibia_T1" pos="0 0.006 -0.004" euler="0.2 1.2 0">
          <geom name="left_tibia_T1_geom" type="mesh" mesh="leg_tibia_mesh" 
                material="fly_body_mat" mass="0.000006" contype="0" conaffinity="0"/>
        </body>
      </body>
      <body name="left_leg_T2" pos="0.001 0.006 -0.003" euler="0.2 -0.9 0.2">
        <geom name="left_femur_T2" type="mesh" mesh="leg_femur_mesh" 
              material="fly_body_mat" mass="0.000008" contype="0" conaffinity="0"/>
        <body name="left_tibia_T2" pos="0 0.006 -0.004" euler="0 1.1 0">
          <geom name="left_tibia_T2_geom" type="mesh" mesh="leg_tibia_mesh" 
                material="fly_body_mat" mass="0.000006" contype="0" conaffinity="0"/>
        </body>
      </body>
      <body name="left_leg_T3" pos="-0.005 0.005 -0.002" euler="-0.3 -0.8 -0.4">
        <geom name="left_femur_T3" type="mesh" mesh="leg_femur_mesh" 
              material="fly_body_mat" mass="0.000008" contype="0" conaffinity="0"/>
        <body name="left_tibia_T3" pos="0 0.006 -0.004" euler="0 1.0 0">
          <geom name="left_tibia_T3_geom" type="mesh" mesh="leg_tibia_mesh" 
                material="fly_body_mat" mass="0.000006" contype="0" conaffinity="0"/>
        </body>
      </body>
      <body name="right_leg_T1" pos="0.006 -0.005 -0.002" euler="-0.4 -1.1 -0.5">
        <geom name="right_femur_T1" type="mesh" mesh="leg_femur_r_mesh" 
              material="fly_body_mat" mass="0.000008" contype="0" conaffinity="0"/>
        <body name="right_tibia_T1" pos="0 -0.006 -0.004" euler="-0.2 1.2 0">
          <geom name="right_tibia_T1_geom" type="mesh" mesh="leg_tibia_r_mesh" 
                material="fly_body_mat" mass="0.000006" contype="0" conaffinity="0"/>
        </body>
      </body>
      <body name="right_leg_T2" pos="0.001 -0.006 -0.003" euler="-0.2 -0.9 -0.2">
        <geom name="right_femur_T2" type="mesh" mesh="leg_femur_r_mesh" 
              material="fly_body_mat" mass="0.000008" contype="0" conaffinity="0"/>
        <body name="right_tibia_T2" pos="0 -0.006 -0.004" euler="0 1.1 0">
          <geom name="right_tibia_T2_geom" type="mesh" mesh="leg_tibia_r_mesh" 
                material="fly_body_mat" mass="0.000006" contype="0" conaffinity="0"/>
        </body>
      </body>
      <body name="right_leg_T3" pos="-0.005 -0.005 -0.002" euler="0.3 -0.8 0.4">
        <geom name="right_femur_T3" type="mesh" mesh="leg_femur_r_mesh" 
              material="fly_body_mat" mass="0.000008" contype="0" conaffinity="0"/>
        <body name="right_tibia_T3" pos="0 -0.006 -0.004" euler="0 1.0 0">
          <geom name="right_tibia_T3_geom" type="mesh" mesh="leg_tibia_r_mesh" 
                material="fly_body_mat" mass="0.000006" contype="0" conaffinity="0"/>
        </body>
      </body>
      
      <!-- Legacy aliases kept so older visualisers that look for left_leg_front still find a body. -->
      <body name="left_leg_front" pos="0.005 0.006 -0.003" euler="0 -0.3 0">
        <geom name="left_femur_front" type="mesh" mesh="leg_femur_mesh" 
              material="fly_body_mat" mass="0.000001" contype="0" conaffinity="0" rgba="0.25 0.15 0.08 0"/>
      </body>
      <body name="right_leg_front" pos="0.005 -0.006 -0.003" euler="0 -0.3 0">
        <geom name="right_femur_front" type="mesh" mesh="leg_femur_r_mesh" 
              material="fly_body_mat" mass="0.000001" contype="0" conaffinity="0" rgba="0.25 0.15 0.08 0"/>
      </body>
    </body>
  </worldbody>
  
  <actuator>
    <!-- Kept for API/tests (gainprm=1, SI units). Flight forces are applied in Python
         via the wing model; ctrl is left at 0 so actuators do not double-count lift. -->
    <general name="thrust_forward" gear="1 0 0 0 0 0" site="thorax_ctrl" gainprm="1 0 0"/>
    <general name="thrust_up" gear="0 0 1 0 0 0" site="thorax_ctrl" gainprm="1 0 0"/>
    <general name="torque_roll" gear="0 0 0 1 0 0" site="thorax_ctrl" gainprm="1 0 0"/>
    <general name="torque_pitch" gear="0 0 0 0 1 0" site="thorax_ctrl" gainprm="1 0 0"/>
    <general name="torque_yaw" gear="0 0 0 0 0 1" site="thorax_ctrl" gainprm="1 0 0"/>
  </actuator>
  
  <sensor>
    <framepos name="fly_pos" objtype="body" objname="fly"/>
    <framequat name="fly_quat" objtype="body" objname="fly"/>
    <framelinvel name="fly_vel" objtype="body" objname="fly"/>
    <frameangvel name="fly_angvel" objtype="body" objname="fly"/>
  </sensor>
</mujoco>
"""


def get_fly_mjcf() -> str:
    """Get MJCF XML string for fly environment."""
    meshdir = str(ASSETS_DIR.absolute())
    return FLY_MJCF.format(meshdir=meshdir)


# Match Drosophila-ish frontolateral coverage: each eye cants out 30°
# (left toward +Y, right toward −Y) with a 20° downward tilt and 120° fovy.
EYE_CANT_DEG = 30.0
EYE_DOWN_DEG = 20.0
EYE_FOVY_DEG = 120.0


def eye_xyaxes(side: str, cant_deg: float = EYE_CANT_DEG, down_deg: float = EYE_DOWN_DEG):
    """6-float MuJoCo xyaxes for an outward-canted eye.

    Image-right is nasal on both sides. The right eye's Y axis is flipped so
    look stays forward (the image is inverted; ``stereo_to_hex`` V-flips).
    """
    import numpy as np

    if side not in ("L", "R"):
        raise ValueError(f"side must be L or R, got {side!r}")
    a = np.deg2rad(float(cant_deg))
    d = np.deg2rad(float(down_deg))
    look = np.array(
        [np.cos(d) * np.cos(a), np.cos(d) * np.sin(a), -np.sin(d)],
        dtype=float,
    )
    look = look / float(np.linalg.norm(look))
    z = -look
    x = np.cross(np.array([0.0, 0.0, 1.0]), z)
    x = x / float(np.linalg.norm(x))
    y = np.cross(z, x)
    y = y / float(np.linalg.norm(y))
    if side == "R":
        x = np.array([x[0], -x[1], x[2]], dtype=float)
        y = np.array([-y[0], y[1], -y[2]], dtype=float)
    return [float(v) for v in np.concatenate([x, y])]


def camera_look_direction(xyaxes) -> "object":
    """World-frame look direction. MuJoCo cameras look along −Z; Z = X × Y."""
    import numpy as np

    a = np.asarray(xyaxes, dtype=float).reshape(6)
    z = np.cross(a[:3], a[3:])
    look = -z
    nrm = float(np.linalg.norm(look))
    return look / nrm if nrm > 0 else look


def parse_eye_xyaxes(xml: str, name: str):
    """Read the 6 ``xyaxes`` floats for ``camera name=...`` from an MJCF string."""
    import re

    m = re.search(
        rf'<camera name="{re.escape(name)}"[^>]*xyaxes="([^"]+)"',
        xml,
    )
    if m is None:
        raise ValueError(f"camera {name} xyaxes not found")
    return [float(x) for x in m.group(1).split()]
