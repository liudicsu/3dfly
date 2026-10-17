"""All engineered closed-loop flight parameters in one place.

These are calibration gains and thresholds on top of the M1 brain stack
(flyvis → handoff → coreTplus RateCore → DN readouts). They are not
connectome weights and they are not hand-coded escape *rules*: escape
fires only when DNp01 rises above a *calibrated* rest (post-warmup /
pre-warm snapshot) by a margin taken from M1 loom vs rotation statistics.

M1 box (PR #17, ``full_reference`` 0.9/4085.83), peak *deltas*:
  loom_ipsi_DNp01 ≈ 2.17 Hz, rot_DNp01_max ≈ 0.15–0.33 Hz,
  DNp03 ipsi/contra ≈ 8.2×, DNa02 L−R ≈ 0.54 Hz, HSE L−R ≈ 20 Hz.
Closed-loop cruise DNp01 sits at ~1.5–3 Hz. Rest is frozen at the
post-warmup value so a slow loom (1.8 → 2.5) is not eaten by a moving EMA.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict


# Frozen M1 reference (Hz, peak deltas unless noted).
M1_DNP01_LOOM_IPSI_HZ = 2.17
M1_DNP01_ROT_MAX_HZ = 0.33
M1_DNA02_LR_HZ = 0.54
M1_HSE_LR_HZ = 20.0
M1_DNP03_T1_RATIO = 8.2


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
    dnp01_loom_hz: float = M1_DNP01_LOOM_IPSI_HZ
    dnp01_rot_hz: float = M1_DNP01_ROT_MAX_HZ
    # Frozen rest; 0.55 sits above rotation (0.33) and below a 0.7 Hz slow loom.
    dnp01_escape_delta_hz: float = 0.55
    dnp01_rest_tau_steps: float = 400.0  # unused while freeze_rest is True
    dnp01_freeze_rest: bool = True
    dnp01_warmup_steps: int = 40
    dnp01_escape_hz: float = M1_DNP01_LOOM_IPSI_HZ
    escape_duration_steps: int = 8
    escape_pitch: float = 0.45
    escape_forward: float = 0.30
    escape_up: float = 0.20
    escape_yaw_boost: float = 0.35

    # --- Looming dodge (DNp03) ---
    dnp03_yaw_gain: float = 0.90
    dnp03_scale_hz: float = 0.50
    dnp03_deadzone_hz: float = 0.04

    # --- Course steering ---
    dna02_yaw_gain: float = 0.70
    dna02_scale_hz: float = M1_DNA02_LR_HZ
    dnp15_yaw_gain: float = 0.25
    dnp15_scale_hz: float = 1.50

    # --- Thrust / wing amplitude (DNg02 mean L+R only) ---
    dng02_rest_hz: float = 5.0
    dng02_scale_hz: float = 4.0
    dng02_forward_gain: float = 0.28
    dng02_up_gain: float = 0.12
    dng02_stroke_gain: float = 0.20
    forward_bias: float = 0.38
    hover_up: float = 0.32

    # --- HS compensatory damp ---
    hse_yaw_damp: float = 0.35
    hse_scale_hz: float = M1_HSE_LR_HZ
    haltere_omega_hz: float = 5.0

    # --- Engineered looming residual (OFF) ---
    use_engineered_looming_residual: bool = False
    engineered_loom_threshold: float = 0.55
    engineered_loom_gain: float = 3.0

    # --- Closed-loop timing ---
    prewarm_steps: int = 48
    flyvis_stride: int = 2  # hold last flyvis voltages on odd sim steps
    flyvis_torch_threads: int = 2

    # --- Dynamics ---
    action_alpha: float = 0.50
    rate_cap_hz: float = 50.0
    latch_eps_hz: float = 1.0
    action_clip: float = 1.0

    notes: Dict[str, str] = field(default_factory=lambda: {
        "escape": (
            "DNp01 rise over *frozen* post-warmup rest ≥ 0.55 Hz "
            "(above M1 rot 0.33, below loom 2.17); residual is opt-in"
        ),
        "dnp03": "continuous ipsi-loom → contralateral turn; per-side baseline subtracted",
        "dng02": "mean(L,R) amplitude; NO L−R yaw",
        "normalisation": "DN L−R / M1 peak-delta scales before tanh, after baseline",
        "hse": "compensatory yaw -= tanh((L−R)/20)",
        "camera": "right-eye Y flipped so both cameras look +X (r2 bias source)",
        "hover_up": "copied from FlyEnvironment.hover_action at bind time",
    })

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)
