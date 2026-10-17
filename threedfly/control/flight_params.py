"""All engineered closed-loop flight parameters in one place.

These are calibration gains and thresholds on top of the M1 brain stack
(flyvis → handoff → coreTplus RateCore → DN readouts). They are not
connectome weights and they are not hand-coded escape *rules*: escape
fires only when DNp01 crosses ``dnp01_escape_hz``. Direction, steering,
and thrust are linear readouts of DNp03 / DNa02+DNp15 / DNg02.

Every number that is not a biological constant from M1 lives here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict


@dataclass
class FlightParams:
    """Calibration for ``DNFlightController``.

    Sign convention (body frame, matches MuJoCo / existing tests):
    positive yaw = left turn; negative yaw = right turn.
    DNp03 is ipsilateral-loom / contralateral-turn, so left DNp03 → right
    turn (negative yaw). DNg02 is a bilateral amplitude; there is no
    DNg02 L−R yaw term.
    """

    # --- Giant fiber / escape (DNp01) ---
    # Absolute Hz. Rest on coreTplus + full_reference scale is ~0; loom
    # peaks are tens of Hz. 12 Hz is above rest and well below the 50 Hz cap.
    dnp01_escape_hz: float = 12.0
    escape_duration_steps: int = 8  # 8 × 20 ms = 160 ms burst
    escape_pitch: float = 0.45  # climb / takeoff
    escape_forward: float = 0.30  # jump-forward component
    escape_up: float = 0.20

    # --- Escape / dodge direction (DNp03) ---
    # yaw = dnp03_yaw_gain * (R − L)  →  L high → negative yaw (right turn)
    dnp03_yaw_gain: float = 0.85
    dnp03_active_hz: float = 2.0  # ignore tiny L−R noise

    # --- Course steering (DNa02 + DNp15/DNHS1) ---
    # yaw += gain * (L − R); same sign as the pre-M2 DNa02 pair.
    dna02_yaw_gain: float = 0.55
    dnp15_yaw_gain: float = 0.20

    # --- Thrust / wing amplitude (DNg02 mean L+R only) ---
    # RateCore applies dng02_bias=5 Hz; deviations around that rest move thrust.
    dng02_rest_hz: float = 5.0
    dng02_scale_hz: float = 8.0
    dng02_forward_gain: float = 0.28
    dng02_up_gain: float = 0.12
    dng02_stroke_gain: float = 0.20
    forward_bias: float = 0.40  # cruise; engineered (not a DN)
    hover_up: float = 0.32  # overwritten from env.hover_action (weight-balance)

    # --- Optional HS / haltere-like rotational feedback ---
    # Compensatory: yaw -= gain * (HSE_L − HSE_R). Small; optomotor damp.
    hse_yaw_damp: float = 0.04
    # Haltere clamp on RateCore: HALTERE_RATE * (0.5 + 0.5*tanh(|ω|/ω0))
    haltere_omega_hz: float = 5.0  # rad/s scale for tanh

    # --- Engineered looming residual (OFF by default; ablation only) ---
    use_engineered_looming_residual: bool = False
    engineered_loom_threshold: float = 0.55
    engineered_loom_gain: float = 3.0

    # --- Dynamics / bookkeeping ---
    action_alpha: float = 0.55  # exponential smoother (1 = no memory)
    rate_cap_hz: float = 50.0
    latch_eps_hz: float = 1.0
    action_clip: float = 1.0

    notes: Dict[str, str] = field(default_factory=lambda: {
        "escape": "DNp01 threshold crossing only; residual is opt-in",
        "dnp03": "ipsilateral loom → contralateral turn (L→right/neg yaw)",
        "dng02": "mean(L,R) amplitude; NO L−R yaw",
        "hse": "optional small compensatory rotational damp",
        "hover_up": "copied from FlyEnvironment.hover_action at bind time",
    })

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d
