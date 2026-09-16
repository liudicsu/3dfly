"""Generate MJCF (MuJoCo XML) models for fly and environment."""


FLY_MJCF = """
<mujoco model="fruit_fly">
  <compiler angle="radian" coordinate="local"/>
  
  <option timestep="0.002" gravity="0 0 -9.81"/>
  
  <asset>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512" 
             rgb1="0.2 0.3 0.4" rgb2="0.3 0.4 0.5"/>
    <material name="grid_mat" texture="grid" texrepeat="10 10" reflectance="0.1"/>
    
    <texture name="wall" type="2d" builtin="flat" width="512" height="512"
             rgb1="0.8 0.75 0.7"/>
    <material name="wall_mat" texture="wall" reflectance="0.2"/>
  </asset>
  
  <worldbody>
    <!-- Ground plane -->
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
    
    <!-- Fruit fly body -->
    <body name="fly" pos="0 0 1.5">
      <freejoint name="fly_joint"/>
      
      <!-- Main body (thorax) -->
      <geom name="thorax" type="ellipsoid" size="0.01 0.008 0.006" 
            rgba="0.3 0.2 0.1 1" mass="0.001"/>
      
      <!-- Head with cameras -->
      <body name="head" pos="0.012 0 0">
        <geom name="head_geom" type="sphere" size="0.006" rgba="0.2 0.15 0.1 1" mass="0.0002"/>
        
        <!-- Left eye camera -->
        <camera name="left_eye" pos="0.003 0.006 0.002" 
                xyaxes="0 -1 0  0.342 0 0.940"
                fovy="90" mode="fixed"/>
        
        <!-- Right eye camera -->
        <camera name="right_eye" pos="0.003 -0.006 0.002"
                xyaxes="0 1 0  0.342 0 0.940"
                fovy="90" mode="fixed"/>
      </body>
      
      <!-- Wings (simplified) -->
      <body name="left_wing" pos="-0.002 0.008 0.002">
        <geom type="box" size="0.015 0.005 0.0005" rgba="0.7 0.7 0.8 0.3" mass="0.00005"/>
      </body>
      <body name="right_wing" pos="-0.002 -0.008 0.002">
        <geom type="box" size="0.015 0.005 0.0005" rgba="0.7 0.7 0.8 0.3" mass="0.00005"/>
      </body>
      
      <!-- Abdomen -->
      <body name="abdomen" pos="-0.012 0 -0.002">
        <geom type="capsule" size="0.003 0.008" rgba="0.4 0.3 0.1 1" mass="0.0003"/>
      </body>
    </body>
  </worldbody>
  
  <actuator>
    <!-- Flight control: forces and torques on the fly body -->
    <general name="thrust_forward" gear="1 0 0 0 0 0" joint="fly_joint" gainprm="0.0001 0 0"/>
    <general name="thrust_up" gear="0 0 1 0 0 0" joint="fly_joint" gainprm="0.0001 0 0"/>
    <general name="torque_roll" gear="0 0 0 1 0 0" joint="fly_joint" gainprm="0.00001 0 0"/>
    <general name="torque_pitch" gear="0 0 0 0 1 0" joint="fly_joint" gainprm="0.00001 0 0"/>
    <general name="torque_yaw" gear="0 0 0 0 0 1" joint="fly_joint" gainprm="0.00001 0 0"/>
  </actuator>
  
  <sensor>
    <!-- Fly position and orientation -->
    <framepos name="fly_pos" objtype="body" objname="fly"/>
    <framequat name="fly_quat" objtype="body" objname="fly"/>
    <framelinvel name="fly_vel" objtype="body" objname="fly"/>
    <frameangvel name="fly_angvel" objtype="body" objname="fly"/>
  </sensor>
</mujoco>
"""


def get_fly_mjcf() -> str:
    """Get MJCF XML string for fly environment."""
    return FLY_MJCF
