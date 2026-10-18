"""Map coreTplus DN readouts to the 5-vector flight action.

Escape is a DNp01 *rise* over a calibrated *per-side* rest taken after
flyvis/brain warmup — rise = max(L−rest_L, R−rest_R), never
max(L,R) − shared rest (an L DC would hide a right-side loom).
Direction during the burst comes from baseline-subtracted DNp03
(contralateral turn). Steering is DNa02 + DNp15 L−R plus continuous DNp03,
each tanh-normalised to its M1 peak-delta scale. Thrust is DNg02 mean(L, R)
only — no DNg02 yaw. No sim object pose/radius is read here.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Optional

import numpy as np

from threedfly.brain.readouts import DNReadout
from threedfly.control.flight_params import FlightParams
from threedfly.control.gf_readout import pd_score, rise_velocity


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
        self.would_fire_count = 0
        self.dnp01_prev = 0.0
        self.dnp01_rest: Optional[float] = None
        self.dnp01_rest_L: Optional[float] = None
        self.dnp01_rest_R: Optional[float] = None
        self.steps = 0
        self.dnp01_latch_steps = 0
        self.escape_events: List[EscapeEvent] = []
        self._last_threat = ""
        self._last_readout = DNReadout()
        self._base: Optional[DNReadout] = None
        self._warmup: List[float] = []
        self._warmup_L: List[float] = []
        self._warmup_R: List[float] = []
        self.last_rise = 0.0
        self._escape_armed = True
        self._baseline_step: Optional[int] = None
        self._rise_hold = 0
        self._approach_threat = ""
        self._d01_hist: List[float] = []
        self._lat_hist: List[tuple] = []
        self._steep_climb = False
        self._steep_hist: List[bool] = []
        self._loom_threat = ""
        self._lat_sign_hist: List[int] = []
        self.last_lat_integrated = 0.0
        self.last_lat_now = 0.0
        self.last_lat_delta = 0.0
        self._lat_rest: Optional[float] = None
        self._peak_lc4_L = 0.0
        self._peak_lc4_R = 0.0
        self._vpn_hist: List[float] = []
        self._vpn_rest: Optional[float] = None
        self.last_vpn_amp = 0.0
        self.last_vpn_delta = 0.0
        self.last_vpn_slope = 0.0
        self._vpn_rising = False
        self.last_lock_step: Optional[int] = None
        self.last_yaw_terms: Dict[str, float] = {}
        self.last_gf_score = 0.0
        self._rise_hist: List[float] = []
        self._rise_L_hist: List[float] = []
        self._rise_R_hist: List[float] = []
        self._gf_integ = 0.0
        self._gf_integ_L = 0.0
        self._gf_integ_R = 0.0

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
        self.would_fire_count = 0
        self.dnp01_prev = 0.0
        self.dnp01_rest = None
        self.dnp01_rest_L = None
        self.dnp01_rest_R = None
        self.steps = 0
        self.dnp01_latch_steps = 0
        self.escape_events.clear()
        self._last_threat = ""
        self._base = None
        self._warmup.clear()
        self._warmup_L.clear()
        self._warmup_R.clear()
        self.last_rise = 0.0
        self._escape_armed = True
        self._baseline_step = None
        self._rise_hold = 0
        self._approach_threat = ""
        self._d01_hist = []
        self._lat_hist = []
        self._steep_climb = False
        self._steep_hist = []
        self._loom_threat = ""
        self._lat_sign_hist = []
        self.last_lat_integrated = 0.0
        self.last_lat_now = 0.0
        self.last_lat_delta = 0.0
        self._lat_rest = None
        self._peak_lc4_L = 0.0
        self._peak_lc4_R = 0.0
        self._vpn_hist = []
        self._vpn_rest = None
        self.last_vpn_amp = 0.0
        self.last_vpn_delta = 0.0
        self.last_vpn_slope = 0.0
        self._vpn_rising = False
        self.last_lock_step = None
        self.last_yaw_terms = {}
        self.last_gf_score = 0.0
        self._rise_hist = []
        self._rise_L_hist = []
        self._rise_R_hist = []
        self._gf_integ = 0.0
        self._gf_integ_L = 0.0
        self._gf_integ_R = 0.0

    def capture_baseline(self, readout: DNReadout, dnp01_rest: Optional[float] = None) -> None:
        """Snapshot per-side DN rest after brain/flyvis pre-warm."""
        self._base = replace(readout)
        if dnp01_rest is None:
            self.dnp01_rest_L = float(readout.dnp01_L)
            self.dnp01_rest_R = float(readout.dnp01_R)
        else:
            self.dnp01_rest_L = float(dnp01_rest)
            self.dnp01_rest_R = float(dnp01_rest)
        self.dnp01_rest = max(float(self.dnp01_rest_L), float(self.dnp01_rest_R))
        self._warmup.clear()
        self._warmup_L.clear()
        self._warmup_R.clear()
        self._baseline_step = int(self.steps)
        self._rise_hold = 0
        self._escape_armed = True
        self._approach_threat = ""
        self._d01_hist = []
        self._lat_hist = []
        self._steep_climb = False
        self._steep_hist = []
        self._loom_threat = ""
        self._lat_sign_hist = []
        self.last_lat_integrated = 0.0
        self.last_lat_now = 0.0
        self.last_lat_delta = 0.0
        self._lat_rest = None
        self._peak_lc4_L = 0.0
        self._peak_lc4_R = 0.0
        self._vpn_hist = []
        self._vpn_rest = None
        self.last_vpn_amp = 0.0
        self.last_vpn_delta = 0.0
        self.last_vpn_slope = 0.0
        self._vpn_rising = False
        self.last_lock_step = None
        self.last_gf_score = 0.0
        self._rise_hist = []
        self._rise_L_hist = []
        self._rise_R_hist = []
        self._gf_integ = 0.0
        self._gf_integ_L = 0.0
        self._gf_integ_R = 0.0
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
            lc4_p90_L=float(getattr(ro, "lc4_p90_L", 0.0)) - float(getattr(b, "lc4_p90_L", 0.0)),
            lc4_p90_R=float(getattr(ro, "lc4_p90_R", 0.0)) - float(getattr(b, "lc4_p90_R", 0.0)),
            lplc2_p90_L=float(getattr(ro, "lplc2_p90_L", 0.0)) - float(getattr(b, "lplc2_p90_L", 0.0)),
            lplc2_p90_R=float(getattr(ro, "lplc2_p90_R", 0.0)) - float(getattr(b, "lplc2_p90_R", 0.0)),
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

    def _vpn_sides(self, ro: DNReadout) -> tuple[float, float]:
        """LC4 / LPLC2 amplitudes. On camera frames these are nearly binocular."""
        return (
            max(0.0, float(ro.lc4_L), float(ro.lplc2_L)),
            max(0.0, float(ro.lc4_R), float(ro.lplc2_R)),
        )

    def _dnp01_laterality(self) -> tuple[float, float]:
        if self._last_readout is None:
            return 0.0, 0.0
        return self._dnp01_rises(self._last_readout)

    def _lateral_vpn_climb(self, ro: DNReadout) -> bool:
        left, right = self._vpn_sides(ro)
        return abs(float(left) - float(right)) >= 0.08

    def _bilateral_expansion(self, ro: Optional[DNReadout] = None) -> bool:
        """Wall: both GF sides climb together. A 35° loom is DNp01-lateral.

        Camera-path LC4/LPLC2 are nearly binocular (and can flip). Per-side
        DNp01 rise is the reliable laterality; bilateral means *not* that.
        """
        if ro is None:
            if self._last_readout is None:
                return False
            ro = self._centered(self._last_readout) if self._base is not None else self._last_readout
        rise_L, rise_R = self._dnp01_laterality()
        vpn_L, vpn_R = self._vpn_sides(ro)
        both_gf = min(float(rise_L), float(rise_R)) >= 0.08
        both_vpn = min(float(vpn_L), float(vpn_R)) >= 0.08
        # A huge side disc raises both GF; LC4/LPLC2 stay one-sided.
        # A wall raises both. Require both, and neither lateral.
        return (
            bool(both_gf)
            and bool(both_vpn)
            and not self._lateral_dnp01_climb()
            and not self._lateral_vpn_climb(ro)
        )

    def _steer_yaw(self, ro: DNReadout) -> float:
        t = self._steer_terms(ro)
        # Side loom: DNp01 laterality — freeze course, keep loom yaw.
        # Bilateral wall (both GF): freeze all yaw and brake.
        side = self.last_rise >= 0.08 and self._lateral_dnp01_climb()
        bino = self._bilateral_expansion(ro)
        if side or bino:
            t["dna02"] = 0.0
            t["dnp15"] = 0.0
            t["hse"] = 0.0
        if bino:
            t["dnp03"] = 0.0
            t["vpn_loom"] = 0.0
        yaw = t["dna02"] + t["dnp15"] + t["hse"] + t["dnp03"] + t["vpn_loom"] + t["eye"]
        if abs(yaw) < float(getattr(self.params, "steer_deadzone", 0.0)):
            yaw = 0.0
        t["steer"] = float(yaw)
        t["course_frozen"] = 1.0 if (side or bino) else 0.0
        t["bilateral_brake"] = 1.0 if bino else 0.0
        vpn_L, vpn_R = self._vpn_sides(ro)
        t["vpn_L"] = float(vpn_L)
        t["vpn_R"] = float(vpn_R)
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
        rise_L, rise_R = self._dnp01_laterality()
        if abs(float(rise_L) - float(rise_R)) >= 0.08:
            # Camera LC4 L−R can flip; per-side DNp01 rise does not.
            vpn = float(rise_L) - float(rise_R)
        raw = float(d03 + vpn)
        if abs(raw) < float(getattr(p, "loom_yaw_deadzone", 0.0)):
            return 0.0, 0.0
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

    def _vpn_amp(self, ro: DNReadout) -> float:
        """Mean LC4/LPLC2. p90 flickers in cruise and looks like expansion."""
        return max(
            0.0,
            float(ro.lc4_L),
            float(ro.lc4_R),
            float(ro.lplc2_L),
            float(ro.lplc2_R),
        )

    def _update_vpn(self, ro: DNReadout) -> None:
        """Looming VPNs expand; residual / self-motion DC is tracked as rest.

        Wide-field HSE predicts translation-driven LC4. The GF sees the
        residual (LPLC2/LC4 after HS), so cruise optic flow cannot expand.
        Expansion is a *run of rising samples*, not a one-frame step.
        """
        # Do not subtract |HSE|: HS also tracks the loom's wide-field
        # expansion (13→50 Hz here) and would zero LC4/LPLC2 until ~80°.
        amp = float(self._vpn_amp(ro))
        if self._vpn_rest is None:
            self._vpn_rest = 0.0 if amp >= 0.28 else amp
        elif not self._vpn_expanding():
            tau = max(float(getattr(self.params, "lat_rest_tau_steps", 25.0)), 1.0)
            self._vpn_rest += (amp - float(self._vpn_rest)) / tau
        self.last_vpn_amp = amp
        self.last_vpn_delta = amp - float(self._vpn_rest or 0.0)
        self._vpn_hist.append(amp)
        if len(self._vpn_hist) > 8:
            self._vpn_hist.pop(0)
        if len(self._vpn_hist) >= 4:
            diffs = np.diff(np.asarray(self._vpn_hist[-4:], dtype=float))
            self.last_vpn_slope = float(np.mean(diffs))
            floor = 0.5 * float(getattr(self.params, "vpn_slope_min", 0.005))
            self._vpn_rising = bool(
                np.all(diffs >= floor) and self.last_vpn_slope >= float(getattr(self.params, "vpn_slope_min", 0.005))
            )
        else:
            self.last_vpn_slope = 0.0
            self._vpn_rising = False

    def _vpn_expanding(self) -> bool:
        """True only for a monotonic LC4/LPLC2 rise, not a step or standing DC."""
        p = self.params
        dmin = float(getattr(p, "vpn_amp_min", 0.28)) * 0.35
        slope_min = float(getattr(p, "vpn_slope_min", 0.005))
        # A 3-step flicker is not a loom: also need a real amplitude rise.
        dneed = max(0.20, dmin)
        if self.last_vpn_delta >= 0.35 and self.last_vpn_amp >= 0.35:
            return True
        if getattr(self, "_vpn_rising", False) and self.last_vpn_delta >= dneed:
            return True
        return self.last_vpn_delta >= dneed and self.last_vpn_slope >= slope_min

    def _integrated_laterality(self, ro: DNReadout) -> float:
        """Rise-weighted L−R *above cruise rest*, not standing DC."""
        p = self.params
        now = float(self._loom_laterality(ro))
        self.last_lat_now = now
        expanding = self._vpn_expanding()
        if self.last_rise < 0.08 and not expanding:
            tau = max(float(getattr(p, "lat_rest_tau_steps", 25.0)), 1.0)
            if self._lat_rest is None:
                self._lat_rest = now
            else:
                self._lat_rest += (now - float(self._lat_rest)) / tau
        if self._lat_rest is None:
            self._lat_rest = 0.0
        delta = now - float(self._lat_rest)
        self.last_lat_delta = float(delta)
        n = max(1, int(getattr(p, "threat_integrate_steps", 8)))
        if self.last_rise < float(p.dnp01_rearm_delta_hz) and not expanding:
            self._lat_hist = []
            self._lat_sign_hist = []
            self._peak_lc4_L = 0.0
            self._peak_lc4_R = 0.0
            self._loom_threat = ""
            self.last_lock_step = None
        if self.last_rise >= 0.08 or expanding:
            self._peak_lc4_L = max(self._peak_lc4_L, max(0.0, float(ro.lc4_L)))
            self._peak_lc4_R = max(self._peak_lc4_R, max(0.0, float(ro.lc4_R)))
        w = max(float(self.last_rise), float(self.last_vpn_amp), 0.0) if (self.last_rise >= 0.08 or expanding) else 0.0
        self._lat_hist.append((delta, w))
        if len(self._lat_hist) > n:
            self._lat_hist.pop(0)
        den = float(sum(wt for _, wt in self._lat_hist))
        if den > 1e-6:
            integ = float(sum(lat * wt for lat, wt in self._lat_hist) / den)
        else:
            integ = float(np.mean([lat for lat, _ in self._lat_hist])) if self._lat_hist else delta
        self.last_lat_integrated = integ
        if self.last_rise >= 0.08 or expanding:
            sgn = 1 if delta > 1e-6 else (-1 if delta < -1e-6 else 0)
            self._lat_sign_hist.append(int(sgn))
            keep = max(4, int(getattr(p, "threat_lock_consistent", 4)))
            if len(self._lat_sign_hist) > keep + 2:
                self._lat_sign_hist.pop(0)
            n_lock = max(1, int(getattr(p, "threat_lock_consistent", 4)))
            tail = self._lat_sign_hist[-n_lock:]
            consistent = (
                len(tail) >= n_lock
                and tail[0] != 0
                and all(s == tail[0] for s in tail)
            )
            lock_rise = float(getattr(p, "threat_lock_rise_hz", 0.12))
            lat_min = float(p.dnp01_loom_lat_min)
            if (
                not self._loom_threat
                and consistent
                and (self.last_rise >= lock_rise or expanding)
                and abs(integ) >= lat_min
            ):
                # Lock from monocular VPNs. DNp01 / mixed DNp03 can flip
                # a 35° loom to the wrong side (GF is binocular).
                vpn_L, vpn_R = self._vpn_sides(ro)
                rise_L, rise_R = self._dnp01_laterality()
                if abs(float(rise_L) - float(rise_R)) >= 0.08:
                    self._loom_threat = "L" if float(rise_L) > float(rise_R) else "R"
                elif abs(float(vpn_L) - float(vpn_R)) >= 0.08:
                    self._loom_threat = "L" if float(vpn_L) > float(vpn_R) else "R"
                else:
                    self._loom_threat = "L" if integ > 0.0 else "R"
                self.last_lock_step = int(self.steps)
        return integ

    def _threat_side(self, ro: DNReadout) -> str:
        if self.params.use_engineered_eye_laterality:
            if self._approach_threat:
                return self._approach_threat
            eye = self._raw_eye_lr()
            if abs(eye) >= float(self.params.eye_threat_min):
                return "L" if eye > 0.0 else "R"
        if self._base is not None:
            rise_L, rise_R = self._dnp01_laterality()
            d01 = float(rise_L) - float(rise_R)
            if abs(d01) >= 0.08:
                return "L" if d01 > 0.0 else "R"
        vpn_L, vpn_R = self._vpn_sides(ro)
        if abs(float(vpn_L) - float(vpn_R)) >= 0.08:
            return "L" if float(vpn_L) > float(vpn_R) else "R"
        if self._loom_threat:
            return self._loom_threat
        d = float(self.last_lat_integrated)
        peak = float(self._peak_lc4_L) - float(self._peak_lc4_R)
        lat_min = float(self.params.dnp01_loom_lat_min)
        # Peak LC4 over the rise is the fallback when last-8 DNp03 is weak
        # or disagrees (binocular overlap / offset_x+0.8).
        if abs(peak) >= 0.08 and (abs(d) < 2.0 * lat_min or peak * d <= 0.0):
            d = peak
        if abs(d) < 1e-9:
            d = self._loom_laterality(ro)
        if d > 1e-6:
            return "L"
        if d < -1e-6:
            return "R"
        return ""

    def _side_rest(self, side: str) -> float:
        if side == "L" and self.dnp01_rest_L is not None:
            return float(self.dnp01_rest_L)
        if side == "R" and self.dnp01_rest_R is not None:
            return float(self.dnp01_rest_R)
        return float(self.dnp01_rest or 0.0)

    def _dnp01_rises(self, readout: DNReadout) -> tuple[float, float]:
        return (
            float(readout.dnp01_L) - self._side_rest("L"),
            float(readout.dnp01_R) - self._side_rest("R"),
        )

    def _lateral_dnp01_climb(self, readout: Optional[DNReadout] = None) -> bool:
        """True when one eye's DNp01 rise exceeds the other by the climb bar.

        Shared-max rest hid a contralateral loom behind an L DC. The same
        definition — per-side rise — tells a side loom from a bilateral
        wall: freeze course only when the climb is one-sided.
        """
        ro = readout if readout is not None else self._last_readout
        rise_L, rise_R = self._dnp01_rises(ro)
        return abs(float(rise_L) - float(rise_R)) >= 0.08

    def _update_rest(self, dnp01_L: float, dnp01_R: float) -> float:
        p = self.params
        warm_n = int(p.dnp01_warmup_steps)
        if self.dnp01_rest_L is None and self.steps < warm_n:
            self._warmup_L.append(float(dnp01_L))
            self._warmup_R.append(float(dnp01_R))
            self._warmup.append(max(float(dnp01_L), float(dnp01_R)))
            return 0.0
        if self.dnp01_rest_L is None:
            if self._warmup_L:
                n = min(8, len(self._warmup_L))
                self.dnp01_rest_L = float(np.mean(self._warmup_L[-n:]))
                self.dnp01_rest_R = float(np.mean(self._warmup_R[-n:]))
            else:
                self.dnp01_rest_L = float(dnp01_L)
                self.dnp01_rest_R = float(dnp01_R)
            self.dnp01_rest = max(float(self.dnp01_rest_L), float(self.dnp01_rest_R))
        rise = max(
            float(dnp01_L) - float(self.dnp01_rest_L),
            float(dnp01_R) - float(self.dnp01_rest_R),
        )
        if (not p.dnp01_freeze_rest) and self.escape_timer <= 0:
            tau = max(float(p.dnp01_rest_tau_steps), 1.0)
            self.dnp01_rest_L += (float(dnp01_L) - float(self.dnp01_rest_L)) / tau
            self.dnp01_rest_R += (float(dnp01_R) - float(self.dnp01_rest_R)) / tau
            self.dnp01_rest = max(float(self.dnp01_rest_L), float(self.dnp01_rest_R))
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

        if self.dnp01_rest_L is not None and self.dnp01_rest_R is not None:
            rise_L, rise_R = self._dnp01_rises(readout)
            rise = max(float(rise_L), float(rise_R))
            if (not p.dnp01_freeze_rest) and self.escape_timer <= 0:
                tau = max(float(p.dnp01_rest_tau_steps), 1.0)
                self.dnp01_rest_L += (float(readout.dnp01_L) - float(self.dnp01_rest_L)) / tau
                self.dnp01_rest_R += (float(readout.dnp01_R) - float(self.dnp01_rest_R)) / tau
                self.dnp01_rest = max(float(self.dnp01_rest_L), float(self.dnp01_rest_R))
        elif self._base is not None:
            self.dnp01_rest_L = float(self._base.dnp01_L)
            self.dnp01_rest_R = float(self._base.dnp01_R)
            self.dnp01_rest = max(float(self.dnp01_rest_L), float(self.dnp01_rest_R))
            rise_L, rise_R = self._dnp01_rises(readout)
            rise = max(float(rise_L), float(rise_R))
        else:
            rise = self._update_rest(float(readout.dnp01_L), float(readout.dnp01_R))
            if self.dnp01_rest_L is not None and self.dnp01_rest_R is not None:
                rise_L, rise_R = self._dnp01_rises(readout)
            else:
                rise_L = rise_R = float(rise)
        self.last_rise = float(rise)
        dt_vis = float(getattr(p, "visual_dt", 0.02))
        win_s = float(getattr(p, "gf_vel_window_s", 0.04))
        self._rise_hist.append(float(rise))
        self._rise_L_hist.append(float(rise_L))
        self._rise_R_hist.append(float(rise_R))
        if len(self._rise_hist) > 16:
            self._rise_hist = self._rise_hist[-16:]
            self._rise_L_hist = self._rise_L_hist[-16:]
            self._rise_R_hist = self._rise_R_hist[-16:]
        vel = rise_velocity(self._rise_hist, dt_vis, win_s)
        vel_L = rise_velocity(self._rise_L_hist, dt_vis, win_s)
        vel_R = rise_velocity(self._rise_R_hist, dt_vis, win_s)
        k_gf = float(getattr(p, "gf_k", 0.0))
        inst = pd_score(rise, vel, k_gf)
        inst_L = pd_score(rise_L, vel_L, k_gf)
        inst_R = pd_score(rise_R, vel_R, k_gf)
        form = str(getattr(p, "gf_form", "pd"))
        if form == "integrate":
            if rise < float(p.dnp01_rearm_delta_hz):
                self._gf_integ = 0.0
                self._gf_integ_L = 0.0
                self._gf_integ_R = 0.0
            else:
                self._gf_integ += max(0.0, inst) * dt_vis
                self._gf_integ_L += max(0.0, inst_L) * dt_vis
                self._gf_integ_R += max(0.0, inst_R) * dt_vis
            score = float(self._gf_integ)
            score_L = float(self._gf_integ_L)
            score_R = float(self._gf_integ_R)
        else:
            score = inst
            score_L = inst_L
            score_R = inst_R
        self.last_gf_score = float(score)
        self._d01_hist.append(float(dnp01_raw))
        if len(self._d01_hist) > 8:
            self._d01_hist.pop(0)
        if len(self._d01_hist) >= 5:
            slope = (self._d01_hist[-1] - self._d01_hist[-5]) / 4.0
        elif len(self._d01_hist) >= 2:
            slope = self._d01_hist[-1] - self._d01_hist[-2]
        else:
            slope = float(rise)
        # Sticky steep during this climb (rise ≥ 0.08). Open-space false
        # fires are blocked by laterality, not by forgetting a real onset.
        self._update_vpn(ro)
        expanding = self._vpn_expanding()
        vpn_ok = True if self._base is None else expanding
        if self.params.use_engineered_eye_laterality:
            vpn_ok = True
        steep_now = slope >= float(getattr(p, "dnp01_slope_min", 0.0))
        if steep_now and rise >= 0.08:
            self._steep_climb = True
        if expanding and rise >= float(getattr(p, "dnp01_vpn_delta_hz", 0.12)):
            self._steep_climb = True
        fire_bar = float(getattr(p, "gf_threshold", p.dnp01_escape_delta_hz))
        if rise < float(p.dnp01_rearm_delta_hz) and not expanding:
            self._escape_armed = True
            self._rise_hold = 0
            self._steep_climb = False
            self._steep_hist = []
        # Hold counts time above the early GF bar so a climb that only
        # becomes "expanding" on frame 4 still has a 4-step hold.
        hold_bar = float(getattr(p, "dnp01_vpn_delta_hz", 0.10))
        if rise >= hold_bar:
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
        lat_int = self._integrated_laterality(ro)
        lat_ok = True
        if self._base is not None and not p.use_engineered_eye_laterality:
            lat_min = float(p.dnp01_loom_lat_min)
            # Residual L DC is in lat_rest; only a laterality *rise* counts.
            same = (self.last_lat_delta * lat_int) > 0.0 or abs(self.last_lat_delta) < 0.5 * lat_min
            lat_ok = abs(lat_int) >= lat_min and same and vpn_ok
        # Frontal GF: both giant fibers crossing the fire bar, with
        # LPLC2 on both sides, is a takeoff/climb — not a laterality-
        # gated side dodge. Translation LC4 (HSE-matched) is not LPLC2.
        # Brake without this fire is not an escape.
        both_gf = False
        if self.dnp01_rest_L is not None and self.dnp01_rest_R is not None:
            both_gf = min(float(score_L), float(score_R)) >= fire_bar
            lplc_L = float(readout.lplc2_L)
            lplc_R = float(readout.lplc2_R)
            if self._base is not None:
                lplc_L -= float(self._base.lplc2_L)
                lplc_R -= float(self._base.lplc2_R)
            both_lplc = min(lplc_L, lplc_R) >= 0.08
            if both_gf and both_lplc:
                lat_ok = True
        would_fire = (
            warmed
            and post_base
            and hold_ok
            and lat_ok
            and self._steep_climb
            and self._escape_armed
            and score >= fire_bar
            and vpn_ok
        )
        if would_fire:
            self.would_fire_count += 1
        brain_fire = (
            would_fire
            and self.escape_timer <= 0
            and self.escape_cooldown <= 0
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
            "would_fire": float(self.would_fire_count),
            "lat_integrated": float(self.last_lat_integrated),
            "lat_delta": float(self.last_lat_delta),
            "lat_rest": float(self._lat_rest or 0.0),
            "vpn_amp": float(self.last_vpn_amp),
            "vpn_slope": float(self.last_vpn_slope),
            "loom_threat": 1.0 if self._loom_threat == "L" else (-1.0 if self._loom_threat == "R" else 0.0),
            "peak_lc4_L": float(self._peak_lc4_L),
            "peak_lc4_R": float(self._peak_lc4_R),
            "dnp01": float(max(ro.dnp01_L, ro.dnp01_R)),
            "dnp01_rest": float(self.dnp01_rest or 0.0),
            "dnp01_rest_L": float(self.dnp01_rest_L or 0.0),
            "dnp01_rest_R": float(self.dnp01_rest_R or 0.0),
            "dnp01_rise": float(self.last_rise),
            "gf_score": float(self.last_gf_score),
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
