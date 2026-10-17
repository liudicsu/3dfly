"""Map coreTplus DN readouts to the 5-vector flight action.

Escape is a DNp01 threshold event. Direction during the burst comes from
DNp03 (contralateral turn). Steering is DNa02 + DNp15 L−R. Thrust and
wing stroke amplitude are DNg02 mean(L, R) only — no DNg02 yaw.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from threedfly.brain.readouts import DNReadout
from threedfly.control.flight_params import FlightParams


def _tanh(x: float) -> float:
    return float(np.tanh(x))


@dataclass
class EscapeEvent:
    step: int
    dnp01: float
    threat_side: str  # "L" or "R" — looming / DNp03 side
    yaw: float
    engineered: bool = False


class DNFlightController:
    """Closed-loop motor mapping for the M1 brain stack."""

    def __init__(self, params: Optional[FlightParams] = None):
        self.params = params or FlightParams()
        self.prev_action = np.zeros(5, dtype=np.float64)
        self.last_wing_strokes = np.array(
            [self.params.hover_up, self.params.hover_up], dtype=np.float64
        )
        self.escape_timer = 0
        self.brain_escape_count = 0
        self.engineered_escape_count = 0
        self.dnp01_prev = 0.0
        self.steps = 0
        self.dnp01_latch_steps = 0
        self.escape_events: List[EscapeEvent] = []
        self._last_threat = "L"
        self._last_readout = DNReadout()

    def bind_hover(self, hover_action: float) -> None:
        self.params.hover_up = float(hover_action)
        self.last_wing_strokes[:] = self.params.hover_up

    def reset(self) -> None:
        self.prev_action[:] = 0.0
        self.last_wing_strokes[:] = self.params.hover_up
        self.escape_timer = 0
        self.brain_escape_count = 0
        self.engineered_escape_count = 0
        self.dnp01_prev = 0.0
        self.steps = 0
        self.dnp01_latch_steps = 0
        self.escape_events.clear()
        self._last_threat = "L"

    def _dng_dev(self, ro: DNReadout) -> float:
        p = self.params
        return _tanh((float(ro.dng02_amp) - p.dng02_rest_hz) / max(p.dng02_scale_hz, 1e-6))

    def _steer_yaw(self, ro: DNReadout) -> float:
        p = self.params
        yaw = p.dna02_yaw_gain * (ro.dna02_L - ro.dna02_R)
        yaw += p.dnp15_yaw_gain * (ro.dnp15_L - ro.dnp15_R)
        yaw -= p.hse_yaw_damp * (ro.hse_L - ro.hse_R)
        return float(yaw)

    def _dnp03_yaw(self, ro: DNReadout) -> float:
        """Left DNp03 → right turn (negative yaw); right DNp03 → left turn."""
        p = self.params
        delta = float(ro.dnp03_R - ro.dnp03_L)
        if abs(ro.dnp03_L) < p.dnp03_active_hz and abs(ro.dnp03_R) < p.dnp03_active_hz:
            return 0.0
        return float(p.dnp03_yaw_gain * np.tanh(delta / 8.0))

    def _threat_side(self, ro: DNReadout) -> str:
        return "L" if ro.dnp03_L >= ro.dnp03_R else "R"

    def compute(
        self,
        readout: DNReadout,
        looming_left: float = 0.0,
        looming_right: float = 0.0,
    ) -> np.ndarray:
        """Return action ``[forward, up, roll, pitch, yaw]``."""
        p = self.params
        self._last_readout = readout
        dnp01 = max(float(readout.dnp01_L), float(readout.dnp01_R))
        if dnp01 >= p.rate_cap_hz - p.latch_eps_hz:
            self.dnp01_latch_steps += 1

        brain_fire = dnp01 >= p.dnp01_escape_hz and self.dnp01_prev < p.dnp01_escape_hz
        engineered_fire = False
        if p.use_engineered_looming_residual and not brain_fire:
            loom = max(float(looming_left), float(looming_right))
            loom_z = _tanh(p.engineered_loom_gain * (loom - p.engineered_loom_threshold))
            engineered_fire = loom_z > 0.55 and dnp01 < p.dnp01_escape_hz

        if brain_fire or engineered_fire:
            self.escape_timer = int(p.escape_duration_steps)
            self._last_threat = self._threat_side(readout)
            if float(looming_left) > float(looming_right) + 1e-6:
                self._last_threat = "L"
            elif float(looming_right) > float(looming_left) + 1e-6:
                self._last_threat = "R"
            ev = EscapeEvent(
                step=self.steps,
                dnp01=dnp01,
                threat_side=self._last_threat,
                yaw=0.0,
                engineered=bool(engineered_fire and not brain_fire),
            )
            if brain_fire:
                self.brain_escape_count += 1
            else:
                self.engineered_escape_count += 1
            self.escape_events.append(ev)

        self.dnp01_prev = dnp01

        dng = self._dng_dev(readout)
        hover = float(p.hover_up)
        thrust_forward = p.forward_bias + p.dng02_forward_gain * dng
        thrust_up = hover + p.dng02_up_gain * dng
        # Equal L/R strokes — DNg02 is amplitude, not a yaw pair.
        stroke = float(np.clip(hover + p.dng02_stroke_gain * dng, 0.0, 1.0))
        self.last_wing_strokes = np.array([stroke, stroke], dtype=np.float64)

        yaw = self._steer_yaw(readout)
        pitch = 0.0
        if self.escape_timer > 0:
            pitch = p.escape_pitch
            thrust_forward = float(np.clip(thrust_forward + p.escape_forward, -1.0, 1.0))
            thrust_up = float(np.clip(thrust_up + p.escape_up, -1.0, 1.0))
            dodge = self._dnp03_yaw(readout)
            yaw = dodge
            if self.escape_events:
                self.escape_events[-1].yaw = dodge
            self.escape_timer -= 1

        action = np.array([thrust_forward, thrust_up, 0.0, pitch, yaw], dtype=np.float64)
        action = p.action_alpha * action + (1.0 - p.action_alpha) * self.prev_action
        action = np.clip(action, -p.action_clip, p.action_clip)
        self.prev_action = action.copy()
        self.steps += 1
        return action

    def compute_action(self, readout: DNReadout, **kwargs) -> np.ndarray:
        return self.compute(readout, **kwargs)

    @property
    def dnp01_latch_frac(self) -> float:
        if self.steps <= 0:
            return 0.0
        return float(self.dnp01_latch_steps) / float(self.steps)

    def get_status(self) -> Dict[str, float]:
        ro = self._last_readout
        return {
            "last_forward": float(self.prev_action[0]),
            "last_up": float(self.prev_action[1]),
            "last_yaw": float(self.prev_action[4]),
            "warmup_complete": 1.0,
            "brain_escapes": float(self.brain_escape_count),
            "engineered_escapes": float(self.engineered_escape_count),
            "dnp01": float(max(ro.dnp01_L, ro.dnp01_R)),
            "dnp03_L": float(ro.dnp03_L),
            "dnp03_R": float(ro.dnp03_R),
            "dng02_amp": float(ro.dng02_amp),
            "dnp01_latch_frac": self.dnp01_latch_frac,
            "wing_stroke_left": float(self.last_wing_strokes[0]),
            "wing_stroke_right": float(self.last_wing_strokes[1]),
        }
