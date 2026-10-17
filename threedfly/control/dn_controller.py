"""Map coreTplus DN readouts to the 5-vector flight action.

Escape is a DNp01 *rise* over a calibrated rest taken after flyvis/brain
warmup — not an absolute 12 Hz gate and not a moving EMA that eats slow
looms. Direction during the burst comes from baseline-subtracted DNp03
(contralateral turn). Steering is DNa02 + DNp15 L−R plus continuous DNp03,
each tanh-normalised to its M1 peak-delta scale. Thrust is DNg02 mean(L, R)
only — no DNg02 yaw.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
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
    threat_side: str
    yaw: float
    engineered: bool = False
    rise: float = 0.0


class DNFlightController:
    """Closed-loop motor mapping for the M1 brain stack."""

    def __init__(self, params: Optional[FlightParams] = None):
        self.params = params or FlightParams()
        self.prev_action = np.zeros(5, dtype=np.float64)
        self.last_wing_strokes = np.array(
            [self.params.hover_up, self.params.hover_up], dtype=np.float64
        )
        self.escape_timer = 0
        self.escape_cooldown = 0
        self.brain_escape_count = 0
        self.engineered_escape_count = 0
        self.dnp01_prev = 0.0
        self.dnp01_rest: Optional[float] = None
        self.steps = 0
        self.dnp01_latch_steps = 0
        self.escape_events: List[EscapeEvent] = []
        self._last_threat = ""
        self._last_readout = DNReadout()
        self._base: Optional[DNReadout] = None
        self._warmup: List[float] = []
        self.last_rise = 0.0
        self._escape_armed = True
        self._baseline_step: Optional[int] = None
        self._rise_hold = 0
        self._approach_threat = ""
        self._d01_hist: List[float] = []
        self._steep_climb = False
        self.last_yaw_terms: Dict[str, float] = {}

    def bind_hover(self, hover_action: float) -> None:
        self.params.hover_up = float(hover_action)
        self.last_wing_strokes[:] = self.params.hover_up

    def reset(self) -> None:
        self.prev_action[:] = 0.0
        self.last_wing_strokes[:] = self.params.hover_up
        self.escape_timer = 0
        self.escape_cooldown = 0
        self.brain_escape_count = 0
        self.engineered_escape_count = 0
        self.dnp01_prev = 0.0
        self.dnp01_rest = None
        self.steps = 0
        self.dnp01_latch_steps = 0
        self.escape_events.clear()
        self._last_threat = ""
        self._base = None
        self._warmup.clear()
        self.last_rise = 0.0
        self._escape_armed = True
        self._baseline_step = None
        self._rise_hold = 0
        self._approach_threat = ""
        self._d01_hist = []
        self._steep_climb = False
        self.last_yaw_terms = {}

    def capture_baseline(self, readout: DNReadout, dnp01_rest: Optional[float] = None) -> None:
        """Snapshot per-side DN rest after brain/flyvis pre-warm."""
        self._base = replace(readout)
        if dnp01_rest is None:
            self.dnp01_rest = max(float(readout.dnp01_L), float(readout.dnp01_R))
        else:
            self.dnp01_rest = float(dnp01_rest)
        self._warmup.clear()
        self._baseline_step = int(self.steps)
        self._rise_hold = 0
        self._escape_armed = True
        self._approach_threat = ""
        self._d01_hist = []
        self._steep_climb = False
        if self.params.use_engineered_eye_laterality:
            eye = float(readout.eye_L) - float(readout.eye_R)
            if abs(eye) >= float(self.params.eye_threat_min):
                self._approach_threat = "L" if eye > 0.0 else "R"

    def _centered(self, ro: DNReadout) -> DNReadout:
        b = self._base
        if b is None:
            return ro
        return DNReadout(
            dnp01_L=float(ro.dnp01_L) - float(b.dnp01_L),
            dnp01_R=float(ro.dnp01_R) - float(b.dnp01_R),
            dnp03_L=float(ro.dnp03_L) - float(b.dnp03_L),
            dnp03_R=float(ro.dnp03_R) - float(b.dnp03_R),
            dna02_L=float(ro.dna02_L) - float(b.dna02_L),
            dna02_R=float(ro.dna02_R) - float(b.dna02_R),
            dnp15_L=float(ro.dnp15_L) - float(b.dnp15_L),
            dnp15_R=float(ro.dnp15_R) - float(b.dnp15_R),
            hse_L=float(ro.hse_L) - float(b.hse_L),
            hse_R=float(ro.hse_R) - float(b.hse_R),
            dng02_L=float(ro.dng02_L),
            dng02_R=float(ro.dng02_R),
            dng02_amp=float(ro.dng02_amp),
            lplc2_L=float(ro.lplc2_L) - float(b.lplc2_L),
            lplc2_R=float(ro.lplc2_R) - float(b.lplc2_R),
            lplc1_L=float(ro.lplc1_L) - float(b.lplc1_L),
            lplc1_R=float(ro.lplc1_R) - float(b.lplc1_R),
            lc4_L=float(ro.lc4_L) - float(b.lc4_L),
            lc4_R=float(ro.lc4_R) - float(b.lc4_R),
            eye_L=float(ro.eye_L) - float(b.eye_L),
            eye_R=float(ro.eye_R) - float(b.eye_R),
        )

    def _dng_dev(self, ro: DNReadout) -> float:
        p = self.params
        return _tanh((float(ro.dng02_amp) - p.dng02_rest_hz) / max(p.dng02_scale_hz, 1e-6))

    def _norm_lr(self, left: float, right: float, scale: float, gain: float) -> float:
        return gain * _tanh((float(left) - float(right)) / max(float(scale), 1e-6))

    def _steer_terms(self, ro: DNReadout) -> Dict[str, float]:
        """Per-source yaw before the escape burst. Official path has eye=0."""
        p = self.params
        dna02 = self._norm_lr(ro.dna02_L, ro.dna02_R, p.dna02_scale_hz, p.dna02_yaw_gain)
        dnp15 = self._norm_lr(ro.dnp15_L, ro.dnp15_R, p.dnp15_scale_hz, p.dnp15_yaw_gain)
        hse = -self._norm_lr(ro.hse_L, ro.hse_R, p.hse_scale_hz, p.hse_yaw_damp)
        d03, vpn = self._loom_yaw_parts(ro)
        eye = 0.0
        if p.use_engineered_eye_laterality:
            eye = self._eye_yaw()
        return {
            "dna02": float(dna02),
            "dnp15": float(dnp15),
            "hse": float(hse),
            "dnp03": float(d03),
            "vpn_loom": float(vpn),
            "eye": float(eye),
            "escape_kick": 0.0,
        }

    def _steer_yaw(self, ro: DNReadout) -> float:
        t = self._steer_terms(ro)
        yaw = t["dna02"] + t["dnp15"] + t["hse"] + t["dnp03"] + t["vpn_loom"] + t["eye"]
        t["steer"] = float(yaw)
        self.last_yaw_terms = t
        return float(yaw)

    def _raw_eye_lr(self) -> float:
        raw = self._last_readout
        return float(raw.eye_L) - float(raw.eye_R)

    def _eye_yaw(self) -> float:
        """Opt-in pixel reflex. Off unless use_engineered_eye_laterality."""
        p = self.params
        return float(-p.eye_yaw_gain * _tanh(self._raw_eye_lr() / max(p.eye_scale, 1e-6)))

    def _loom_lr(self, ro: DNReadout) -> tuple[float, float]:
        """(DNp03 L−R, VPN L−R). Positive = left-side (ipsilateral) loom.

        Hex-swap and single-eye movies: LC4/LPLC2/DNp03 are ipsilateral.
        ``ro`` is already baseline-subtracted so residual DC does not steer.
        """
        p = self.params
        left = max(0.0, float(ro.dnp03_L) - p.dnp03_deadzone_hz)
        right = max(0.0, float(ro.dnp03_R) - p.dnp03_deadzone_hz)
        d03 = left - right
        vpn = (
            float(p.lplc2_threat_gain) * (float(ro.lplc2_L) - float(ro.lplc2_R))
            + float(p.lplc1_threat_gain) * (float(ro.lplc1_L) - float(ro.lplc1_R))
            + float(p.lc4_threat_gain) * (float(ro.lc4_L) - float(ro.lc4_R))
        )
        return float(d03), float(vpn)

    def _loom_yaw_parts(self, ro: DNReadout) -> tuple[float, float]:
        p = self.params
        d03, vpn = self._loom_lr(ro)
        scale = max(p.dnp03_scale_hz, 1e-6)
        # Ipsi loom → contralateral turn: +L → −yaw.
        return (
            float(p.dnp03_yaw_gain * _tanh((-d03) / scale)),
            float(p.dnp03_yaw_gain * _tanh((-vpn) / scale)),
        )

    def _dnp03_yaw(self, ro: DNReadout) -> float:
        d03, vpn = self._loom_yaw_parts(ro)
        return float(d03 + vpn)

    def _loom_laterality(self, ro: DNReadout) -> float:
        d03, vpn = self._loom_lr(ro)
        return d03 + vpn

    def _threat_side(self, ro: DNReadout) -> str:
        if self.params.use_engineered_eye_laterality:
            if self._approach_threat:
                return self._approach_threat
            eye = self._raw_eye_lr()
            if abs(eye) >= float(self.params.eye_threat_min):
                return "L" if eye > 0.0 else "R"
        d = self._loom_laterality(ro)
        if d > 1e-6:
            return "L"
        if d < -1e-6:
            return "R"
        return ""

    def _update_rest(self, dnp01: float) -> float:
        p = self.params
        warm_n = int(p.dnp01_warmup_steps)
        if self.dnp01_rest is None and self.steps < warm_n:
            self._warmup.append(float(dnp01))
            return 0.0
        if self.dnp01_rest is None:
            if self._warmup:
                tail = self._warmup[-min(8, len(self._warmup)) :]
                self.dnp01_rest = float(np.mean(tail))
            else:
                self.dnp01_rest = float(dnp01)
        rise = float(dnp01) - float(self.dnp01_rest)
        if (not p.dnp01_freeze_rest) and self.escape_timer <= 0:
            tau = max(float(p.dnp01_rest_tau_steps), 1.0)
            self.dnp01_rest += (float(dnp01) - float(self.dnp01_rest)) / tau
        return rise

    def compute(
        self,
        readout: DNReadout,
        looming_left: float = 0.0,
        looming_right: float = 0.0,
    ) -> np.ndarray:
        """Return action ``[forward, up, roll, pitch, yaw]``."""
        p = self.params
        self._last_readout = readout
        ro = self._centered(readout)
        dnp01_raw = max(float(readout.dnp01_L), float(readout.dnp01_R))
        if dnp01_raw >= p.rate_cap_hz - p.latch_eps_hz:
            self.dnp01_latch_steps += 1

        if self.dnp01_rest is not None:
            rise = dnp01_raw - float(self.dnp01_rest)
            if (not p.dnp01_freeze_rest) and self.escape_timer <= 0:
                tau = max(float(p.dnp01_rest_tau_steps), 1.0)
                self.dnp01_rest += (dnp01_raw - float(self.dnp01_rest)) / tau
        elif self._base is not None:
            rise = max(float(ro.dnp01_L), float(ro.dnp01_R))
            self.dnp01_rest = dnp01_raw
            rise = 0.0
        else:
            rise = self._update_rest(dnp01_raw)
        self.last_rise = float(rise)
        self._d01_hist.append(float(dnp01_raw))
        if len(self._d01_hist) > 8:
            self._d01_hist.pop(0)
        if len(self._d01_hist) >= 5:
            slope = (self._d01_hist[-1] - self._d01_hist[-5]) / 4.0
        elif len(self._d01_hist) >= 2:
            slope = self._d01_hist[-1] - self._d01_hist[-2]
        else:
            slope = float(rise)
        if slope >= float(getattr(p, "dnp01_slope_min", 0.0)):
            self._steep_climb = True
        if rise < float(p.dnp01_rearm_delta_hz):
            self._escape_armed = True
            self._rise_hold = 0
            self._steep_climb = False
        if rise >= p.dnp01_escape_delta_hz:
            self._rise_hold += 1
        else:
            self._rise_hold = 0
        warmed = self.steps >= int(p.dnp01_warmup_steps) or self._base is not None
        post_base = True
        hold_ok = True
        if self._base is not None:
            delay = int(p.dnp01_post_baseline_steps)
            if self._baseline_step is not None:
                post_base = (self.steps - self._baseline_step) >= delay
            hold_ok = self._rise_hold >= int(p.dnp01_escape_hold_steps)
        lat_ok = True
        if self._base is not None and not p.use_engineered_eye_laterality:
            lat_ok = abs(self._loom_laterality(ro)) >= float(p.dnp01_loom_lat_min)
        brain_fire = (
            warmed
            and post_base
            and hold_ok
            and lat_ok
            and self._steep_climb
            and self._escape_armed
            and self.escape_timer <= 0
            and self.escape_cooldown <= 0
            and rise >= p.dnp01_escape_delta_hz
        )
        engineered_fire = False
        if p.use_engineered_looming_residual and not brain_fire:
            loom = max(float(looming_left), float(looming_right))
            loom_z = _tanh(p.engineered_loom_gain * (loom - p.engineered_loom_threshold))
            engineered_fire = loom_z > 0.55 and warmed

        if brain_fire or engineered_fire:
            self.escape_timer = int(p.escape_duration_steps)
            extra = int(getattr(p, "escape_cooldown_extra", 4))
            self.escape_cooldown = int(p.escape_duration_steps) + extra
            if brain_fire:
                self._escape_armed = False
            self._last_threat = self._threat_side(ro)
            if engineered_fire and not brain_fire:
                if float(looming_left) > float(looming_right) + 1e-6:
                    self._last_threat = "L"
                elif float(looming_right) > float(looming_left) + 1e-6:
                    self._last_threat = "R"
            ev = EscapeEvent(
                step=self.steps,
                dnp01=dnp01_raw,
                threat_side=self._last_threat,
                yaw=0.0,
                engineered=bool(engineered_fire and not brain_fire),
                rise=float(rise),
            )
            if brain_fire:
                self.brain_escape_count += 1
            else:
                self.engineered_escape_count += 1
            self.escape_events.append(ev)

        self.dnp01_prev = dnp01_raw

        dng = self._dng_dev(readout)
        hover = float(p.hover_up)
        thrust_forward = p.forward_bias + p.dng02_forward_gain * dng
        thrust_up = hover + p.dng02_up_gain * dng
        stroke_mean = float(np.clip(hover + p.dng02_stroke_gain * dng, 0.0, 1.0))

        yaw = self._steer_yaw(ro)
        pitch = 0.0
        loom_amp = (
            max(0.0, float(ro.dnp03_L), float(ro.dnp03_R))
            + 0.35 * max(0.0, float(ro.lc4_L), float(ro.lc4_R))
            + 0.80 * max(0.0, float(rise))
        )
        brake = _tanh(loom_amp / max(float(getattr(p, "loom_brake_scale_hz", 0.25)), 1e-6))
        thrust_forward = float(np.clip(thrust_forward - float(getattr(p, "loom_brake_gain", 0.0)) * brake, -1.0, 1.0))
        pitch = float(getattr(p, "loom_pitch_gain", 0.0)) * brake
        self.last_yaw_terms["loom_brake"] = float(brake)
        if self.escape_timer > 0:
            pitch = p.escape_pitch
            thrust_forward = float(np.clip(thrust_forward + p.escape_forward, -1.0, 1.0))
            thrust_up = float(np.clip(thrust_up + p.escape_up, -1.0, 1.0))
            dodge = self._dnp03_yaw(ro)
            kick = 0.0
            if self._last_threat == "L":
                kick = -float(p.escape_yaw_kick)
                yaw = kick + 0.25 * dodge
            elif self._last_threat == "R":
                kick = float(p.escape_yaw_kick)
                yaw = kick + 0.25 * dodge
            else:
                yaw = yaw + p.escape_yaw_boost * dodge
            self.last_yaw_terms["escape_kick"] = float(kick)
            self.last_yaw_terms["steer"] = float(yaw)
            if self.escape_events:
                self.escape_events[-1].yaw = float(yaw)
            self.escape_timer -= 1
        if self.escape_cooldown > 0:
            self.escape_cooldown -= 1

        action = np.array([thrust_forward, thrust_up, 0.0, pitch, yaw], dtype=np.float64)
        action = p.action_alpha * action + (1.0 - p.action_alpha) * self.prev_action
        action = np.clip(action, -p.action_clip, p.action_clip)
        # +yaw (left turn) → L−R > 0. DNg02 only sets the bilateral mean.
        yaw_stroke = float(p.yaw_stroke_gain) * float(action[4])
        self.last_wing_strokes = np.array(
            [
                float(np.clip(stroke_mean + yaw_stroke, 0.0, 1.0)),
                float(np.clip(stroke_mean - yaw_stroke, 0.0, 1.0)),
            ],
            dtype=np.float64,
        )
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
            "warmup_complete": 1.0 if self.steps >= int(self.params.dnp01_warmup_steps) else 0.0,
            "brain_escapes": float(self.brain_escape_count),
            "engineered_escapes": float(self.engineered_escape_count),
            "dnp01": float(max(ro.dnp01_L, ro.dnp01_R)),
            "dnp01_rest": float(self.dnp01_rest or 0.0),
            "dnp01_rise": float(self.last_rise),
            "dnp03_L": float(ro.dnp03_L),
            "dnp03_R": float(ro.dnp03_R),
            "lplc2_L": float(ro.lplc2_L),
            "lplc2_R": float(ro.lplc2_R),
            "lc4_L": float(ro.lc4_L),
            "lc4_R": float(ro.lc4_R),
            "dng02_amp": float(ro.dng02_amp),
            "dnp01_latch_frac": self.dnp01_latch_frac,
            "wing_stroke_left": float(self.last_wing_strokes[0]),
            "wing_stroke_right": float(self.last_wing_strokes[1]),
        }
