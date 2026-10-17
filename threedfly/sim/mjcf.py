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

# Absolute near clipping distance (m). MuJoCo's map.znear is a fraction of
# model.stat.extent (~20 m for the room), so the default 0.01 clips a 2.5 cm fly.
FLY_ZNEAR_M = 0.0008

# Anatomical attachments in the fly/thorax frame (metres, after MESH_SCALE).
# Flybody meshes use +X anterior, +Y right, +Z dorsal; named "left" parts sit at -Y.
WING_LEFT_HINGE = (0.000485, -0.004619, 0.013267)
WING_RIGHT_HINGE = (0.000485, 0.004619, 0.013267)
COXA_T1_LEFT = (0.002696, -0.002257, 0.008379)
COXA_T2_LEFT = (0.000479, -0.002645, 0.007537)
COXA_T3_LEFT = (-0.003954, -0.002366, 0.009764)
COXA_T1_RIGHT = (0.002696, 0.002257, 0.008379)
COXA_T2_RIGHT = (0.000479, 0.002645, 0.007537)
COXA_T3_RIGHT = (-0.003954, 0.002366, 0.009764)


def _fmt(vec) -> str:
    return f"{vec[0]:.6f} {vec[1]:.6f} {vec[2]:.6f}"


def _neg(vec):
    return (-vec[0], -vec[1], -vec[2])


def _leg_xml(
    body_name: str,
    coxa,
    femur_mesh: str,
    femur_geom: str,
    tibia_name: str,
    tibia_mesh: str,
    tibia_geom: str,
    euler: str,
) -> str:
    """Weld a T2 femur/tibia mesh at a thorax coxa, rotating about that coxa."""
    geom_pos = _fmt(_neg(COXA_T2_LEFT if coxa[1] < 0 else COXA_T2_RIGHT))
    return f"""
      <body name="{body_name}" pos="{_fmt(coxa)}" euler="{euler}">
        <geom name="{femur_geom}" type="mesh" mesh="{femur_mesh}"
              material="fly_body_mat" mass="0.000008" contype="0" conaffinity="0" pos="{geom_pos}"/>
        <body name="{tibia_name}" pos="0 0 0">
          <geom name="{tibia_geom}" type="mesh" mesh="{tibia_mesh}"
                material="fly_body_mat" mass="0.000006" contype="0" conaffinity="0" pos="{geom_pos}"/>
        </body>
      </body>"""


_WLH = _fmt(WING_LEFT_HINGE)
_WRH = _fmt(WING_RIGHT_HINGE)
_WLG = _fmt(_neg(WING_LEFT_HINGE))
_WRG = _fmt(_neg(WING_RIGHT_HINGE))

_LEGS = (
    _leg_xml("left_leg_T1", COXA_T1_LEFT, "leg_femur_mesh", "left_femur_T1",
             "left_tibia_T1", "leg_tibia_mesh", "left_tibia_T1_geom", "0.45 0.10 0.25")
    + _leg_xml("left_leg_T2", COXA_T2_LEFT, "leg_femur_mesh", "left_femur_T2",
               "left_tibia_T2", "leg_tibia_mesh", "left_tibia_T2_geom", "0.30 0.05 0.15")
    + _leg_xml("left_leg_T3", COXA_T3_LEFT, "leg_femur_mesh", "left_femur_T3",
               "left_tibia_T3", "leg_tibia_mesh", "left_tibia_T3_geom", "0.20 -0.10 -0.20")
    + _leg_xml("right_leg_T1", COXA_T1_RIGHT, "leg_femur_r_mesh", "right_femur_T1",
               "right_tibia_T1", "leg_tibia_r_mesh", "right_tibia_T1_geom", "-0.45 0.10 -0.25")
    + _leg_xml("right_leg_T2", COXA_T2_RIGHT, "leg_femur_r_mesh", "right_femur_T2",
               "right_tibia_T2", "leg_tibia_r_mesh", "right_tibia_T2_geom", "-0.30 0.05 -0.15")
    + _leg_xml("right_leg_T3", COXA_T3_RIGHT, "leg_femur_r_mesh", "right_femur_T3",
               "right_tibia_T3", "leg_tibia_r_mesh", "right_tibia_T3_geom", "-0.20 -0.10 0.20")
)

FLY_MJCF = """
<mujoco model="fruit_fly">
  <compiler angle="radian" coordinate="local" meshdir="{meshdir}"/>
  
  <option timestep="0.002" gravity="0 0 -9.81"/>

  <visual>
    <global azimuth="140" elevation="-25"/>
    <map znear="0.00005" zfar="50"/>
  </visual>
  
  <asset>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512" 
             rgb1="0.2 0.3 0.4" rgb2="0.3 0.4 0.5"/>
    <material name="grid_mat" texture="grid" texrepeat="10 10" reflectance="0.1"/>
    
    <texture name="wall" type="2d" builtin="flat" width="512" height="512"
             rgb1="0.8 0.75 0.7"/>
    <material name="wall_mat" texture="wall" reflectance="0.2"/>
    
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
      <geom type="box" size="0.1 5 2" material="wall_mat" rgba="0.8 0.8 0.9 1"/>
    </body>
    <body name="wall_south" pos="-5 0 2">
      <geom type="box" size="0.1 5 2" material="wall_mat" rgba="0.8 0.9 0.8 1"/>
    </body>
    <body name="wall_east" pos="0 5 2">
      <geom type="box" size="5 0.1 2" material="wall_mat" rgba="0.9 0.8 0.8 1"/>
    </body>
    <body name="wall_west" pos="0 -5 2">
      <geom type="box" size="5 0.1 2" material="wall_mat" rgba="0.9 0.9 0.8 1"/>
    </body>
    
    <!-- Some obstacles for exploration -->
    <body name="obstacle_1" pos="2 2 0.5">
      <geom type="box" size="0.3 0.3 0.5" rgba="0.6 0.3 0.2 1"/>
    </body>
    <body name="obstacle_2" pos="-2 -2 0.3">
      <geom type="cylinder" size="0.2 0.3" rgba="0.3 0.5 0.3 1"/>
    </body>
    <body name="obstacle_3" pos="-1 3 0.4">
      <geom type="sphere" size="0.4" rgba="0.4 0.4 0.6 1"/>
    </body>
    
    <!-- Fruit fly: flybody topology (head / thorax / abdomen / wings / legs / halteres).
         Masses keep flybody's thorax-heavy ratios, scaled to ~1.5 g total (see scale.py). -->
    <body name="fly" pos="0 0 1.5">
      <freejoint name="fly_joint"/>
      <!-- Body-frame control site at subtree COM so residual torques do not induce tumble. -->
      <site name="thorax_ctrl" pos="-0.00112645 -0.00015024 0.01175987" size="0.001" rgba="1 0 0 0"/>
      
      <geom name="thorax" type="mesh" mesh="thorax_mesh" 
            material="fly_body_mat" mass="0.001"/>

      <!-- Close-up camera follows the fly (body-fixed, looking at the thorax). -->
      <camera name="fly_closeup" pos="0.055 -0.045 0.035"
              xyaxes="0.63 0.78 0  -0.27 0.22 0.94"
              fovy="50" mode="fixed"/>
      
      <!-- Head with stereo cameras (neck hinge, stiff). -->
      <body name="head" pos="0.012 0 0">
        <joint name="neck_pitch" type="hinge" axis="0 1 0" range="-0.3 0.3"
               stiffness="0.05" damping="0.01"/>
        <geom name="head_geom" type="mesh" mesh="head_mesh" 
              material="fly_body_mat" mass="0.0002" contype="0" conaffinity="0"/>
        
        <camera name="left_eye" pos="0.003 0.006 0.002" 
                xyaxes="0 -1 0  0.342 0 0.940"
                fovy="90" mode="fixed"/>
        <camera name="right_eye" pos="0.003 -0.006 0.002"
                xyaxes="0 1 0  0.342 0 0.940"
                fovy="90" mode="fixed"/>
      </body>
      
      <!-- Wings: hinge at the membrane base on the thorax. Geom pos puts the
           mesh back in the authored rest pose; L/R joint axes are mirrored. -->
      <body name="left_wing" pos=""" + f'"{_WLH}"' + """>
        <joint name="wing_left_stroke" type="hinge" axis="1 0 0" range="-1.2 1.2"
               damping="0.002" stiffness="0.0"/>
        <site name="left_wing_aero" pos="0 -0.012 0" size="0.001" rgba="0 1 0 0"/>
        <geom name="left_wing_geom" type="mesh" mesh="wing_left_mesh" 
              material="fly_wing_mat" mass="0.00005" contype="0" conaffinity="0" pos=""" + f'"{_WLG}"' + """/>
      </body>
      <body name="right_wing" pos=""" + f'"{_WRH}"' + """>
        <joint name="wing_right_stroke" type="hinge" axis="-1 0 0" range="-1.2 1.2"
               damping="0.002" stiffness="0.0"/>
        <site name="right_wing_aero" pos="0 0.012 0" size="0.001" rgba="0 1 0 0"/>
        <geom name="right_wing_geom" type="mesh" mesh="wing_right_mesh" 
              material="fly_wing_mat" mass="0.00005" contype="0" conaffinity="0" pos=""" + f'"{_WRG}"' + """/>
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
      
      <!-- Six legs, retracted flight pose, welded at thorax coxa (no active joints). -->
""" + _LEGS + """
      
      <!-- Legacy aliases kept so older visualisers that look for left_leg_front still find a body. -->
      <body name="left_leg_front" pos=""" + f'"{_fmt(COXA_T1_LEFT)}"' + """ euler="0 -0.3 0">
        <geom name="left_femur_front" type="mesh" mesh="leg_femur_mesh" 
              material="fly_body_mat" mass="0.000001" contype="0" conaffinity="0" rgba="0.25 0.15 0.08 0"
              pos=""" + f'"{_fmt(_neg(COXA_T2_LEFT))}"' + """/>
      </body>
      <body name="right_leg_front" pos=""" + f'"{_fmt(COXA_T1_RIGHT)}"' + """ euler="0 -0.3 0">
        <geom name="right_femur_front" type="mesh" mesh="leg_femur_r_mesh" 
              material="fly_body_mat" mass="0.000001" contype="0" conaffinity="0" rgba="0.25 0.15 0.08 0"
              pos=""" + f'"{_fmt(_neg(COXA_T2_RIGHT))}"' + """/>
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
