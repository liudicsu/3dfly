"""All engineered closed-loop flight parameters in one place.

These are calibration gains and thresholds on top of the M1 brain stack
(flyvis → handoff → coreTplus RateCore → DN readouts). They are not
connectome weights and they are not hand-coded escape *rules*: escape
fires only when DNp01 rises above its recent rest by a margin taken from
M1 loom vs rotation statistics.

M1 box (PR #17, ``full_reference`` 0.9/4085.83), peak *deltas*:
  loom_ipsi_DNp01 ≈ 2.17 Hz, rot_DNp01_max ≈ 0.15–0.33 Hz,
  DNp03 ipsi/contra ≈ 8.2×, DNa02 L−R ≈ 0.54 Hz, HSE L−R ≈ 20 Hz.
Closed-loop cruise DNp01 sits at ~1.5–3 Hz (same absolute scale as the
loom *delta*), so an absolute 12 Hz gate is unreachable. We threshold the
rise over an EMA rest instead.
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
    # Rise over EMA rest. Midway from rotation (0.33) toward loom (2.17).
    dnp01_loom_hz: float = M1_DNP01_LOOM_IPSI_HZ
    dnp01_rot_hz: float = M1_DNP01_ROT_MAX_HZ
    dnp01_escape_delta_hz: float = 0.85  # (0.33 + 2.17) / 2 − 0.4; above rot, below loom
    dnp01_rest_tau_steps: float = 40.0
    dnp01_warmup_steps: int = 20  # do not fire while rest EMA is still at 0
    # Legacy absolute field kept for latch bookkeeping only (not the trigger).
    dnp01_escape_hz: float = M1_DNP01_LOOM_IPSI_HZ
    escape_duration_steps: int = 8  # 8 × 20 ms = 160 ms burst
    escape_pitch: float = 0.45
    escape_forward: float = 0.30
    escape_up: float = 0.20
    escape_yaw_boost: float = 0.35  # extra DNp03 weight during the burst

    # --- Looming dodge (DNp03), continuous + escape ---
    # In-loop DNp03 ≤ ~1.3 Hz; a 2 Hz gate zeroed the dodge. Scale is the
    # typical ipsi loom (M1 loom_ipsi / T1-ish).
    dnp03_yaw_gain: float = 0.90
    dnp03_scale_hz: float = 0.50
    dnp03_deadzone_hz: float = 0.04

    # --- Course steering (DNa02 + DNp15/DNHS1), M1 L−R ≈ 0.54 Hz ---
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

    # --- HS compensatory damp; M1 |HSE L−R| ≈ 20 Hz ---
    hse_yaw_damp: float = 0.35
    hse_scale_hz: float = M1_HSE_LR_HZ
    haltere_omega_hz: float = 5.0

    # --- Engineered looming residual (OFF by default; ablation only) ---
    use_engineered_looming_residual: bool = False
    engineered_loom_threshold: float = 0.55
    engineered_loom_gain: float = 3.0

    # --- Dynamics / bookkeeping ---
    action_alpha: float = 0.50
    rate_cap_hz: float = 50.0
    latch_eps_hz: float = 1.0
    action_clip: float = 1.0

    notes: Dict[str, str] = field(default_factory=lambda: {
        "escape": (
            "DNp01 rise over EMA rest ≥ 0.85 Hz (between M1 rot 0.33 and loom 2.17); "
            "residual is opt-in"
        ),
        "dnp03": "continuous ipsi-loom → contralateral turn; no 2 Hz gate",
        "dng02": "mean(L,R) amplitude; NO L−R yaw",
        "normalisation": "DN L−R / M1 peak-delta scales before tanh",
        "hse": "compensatory yaw -= tanh((L−R)/20)",
        "hover_up": "copied from FlyEnvironment.hover_action at bind time",
    })

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)
