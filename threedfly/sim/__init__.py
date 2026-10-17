"""MuJoCo physics simulation environment."""

from threedfly.sim.environment import FlyEnvironment
from threedfly.sim.wings import stroke_averaged_forces

__all__ = ["FlyEnvironment", "stroke_averaged_forces"]
