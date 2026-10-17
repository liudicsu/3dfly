"""Documented geometric and mass scaling from flybody CGS to the 3dfly room.

flybody (TuragaLab / DeepMind) is authored in centimetres with g = 981 cm/s²
and a ~2.5 mm adult *Drosophila* (~1 mg). 3dfly explores a 10 m room with
160×120 90° eye cameras, so a life-size fly is a sub-pixel speck.

Scaling used by the MJCF (see `threedfly/sim/mjcf.py`):

* **Length**: original flybody OBJ millimetres × 0.01 → metres.
  A 2.5 mm fly becomes a **2.5 cm** visual body (10× life-size) so stereo
  cameras still resolve walls and obstacles.
* **Mass**: flybody's relative distribution (thorax-heavy, light wings/legs)
  is kept, but the **total mass is ~1.5 g** rather than ~1 mg. That ~1000×
  mass scale keeps SI actuator/drag numbers well-conditioned in a metre-scale
  room (the same order as the previous free-flying 3dfly body).
* **Time / gravity**: SI, g = 9.81 m/s², 2 ms MuJoCo substeps, 20 ms control.
* **Legs**: folded into a retracted flight pose and welded (no active joints).
* **Wings**: hinge joints exist for visual flapping; aerodynamic force is the
  stroke-averaged model in `threedfly/sim/wings.py`, not a 200 Hz blade element
  simulation.
"""

# OBJ millimetres → metres (10× life-size visual fly).
MESH_SCALE = 0.01

# 10× linear scale relative to a 2.5 mm adult.
LIFE_SIZE_LINEAR = 10.0
