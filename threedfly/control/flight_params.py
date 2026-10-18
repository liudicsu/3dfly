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
    # U-flip right hex: camera-object DNp01 ~0.70 / 0.86 Hz. Open wander
    # ~0.28 Hz. 0.20 fires earlier (40–60° GF) with slope + integrated
    # laterality so slow open climb cannot pass.
    dnp01_escape_delta_hz: float = 0.20
    # Re-arm only after rise falls below this (rising-edge).
    dnp01_rearm_delta_hz: float = 0.12
    # GF is LC4/LPLC2-driven: allow a lower DNp01 bar when looming VPNs climb
    # so the burst can sit in the 40–60° angular-size window.
    dnp01_vpn_delta_hz: float = 0.12
    # Genuine close is steep (≈0.016–0.025 Hz/step). Open wander is ≈0.009.
    dnp01_slope_min: float = 0.012
    # Threat from the mean L−R over the rise, not the fire instant.
    threat_integrate_steps: int = 10
    # Do not lock side on the first noisy sample (self-motion / early yaw).
    threat_lock_consistent: int = 4
    threat_lock_rise_hz: float = 0.12
    # Standing L−R DC (self-motion / residual LC4_L) is tracked as rest.
    # Only laterality *above* that rest is a loom.
    lat_rest_tau_steps: float = 25.0
    # Ipsilateral LC4/LPLC2 must expand. Residual DC has slope ≈ 0.
    # Closed-loop object LC4 is ~0.1 and rises slowly (hex-mean dilution);
    # 0.005 Hz/step is a real expansion, 0.012 only caught the last 80° jump.
    vpn_amp_min: float = 0.28
    vpn_slope_min: float = 0.015
    # Kept for experiments. Official path does not subtract |HSE| from
    # VPN: HS tracks the loom itself and would hide 40–60° expansion.
    vpn_hse_gate: float = 0.0
    # Slow rest: eats the 16-step settle climb, keeps a 1 s compact loom.
    dnp01_rest_tau_steps: float = 100.0
    dnp01_freeze_rest: bool = True
    dnp01_warmup_steps: int = 40
    # After a settled prewarm this only skips takeoff optic-flow (a few steps).
    dnp01_post_baseline_steps: int = 8
    dnp01_escape_hold_steps: int = 3
    # Closed-loop (baseline set): DNp01 rise alone is not a loom. Require
    # DNp03/LC4/LPLC laterality so the flyvis settle climb cannot fire.
    dnp01_loom_lat_min: float = 0.20
    dnp01_escape_hz: float = M1_DNP01_LOOM_IPSI_HZ
    escape_duration_steps: int = 8
    # One loom = one burst. Short cooldown allowed a second fire that
    # wrecked side-R direction after the object had already passed.
    # One loom = one burst for a 400-step side-loom (object is gone after ~80).
    escape_cooldown_extra: int = 400
    escape_pitch: float = 0.45
    escape_forward: float = 0.30
    escape_up: float = 0.20
    escape_yaw_boost: float = 0.35
    # Motor magnitude of a DNp01 burst. Sign comes from brain laterality
    # (DNp03 / LPLC / LC4 L−R), never from pixel brightness.
    escape_yaw_kick: float = 0.55
    # Map DN yaw onto the existing L−R stroke actuator (not DNg02 L−R).
    # +yaw = left turn = more left stroke. Residual action[4] torque stays.
    yaw_stroke_gain: float = 0.45

    # --- Looming dodge (DNp03 + optic-lobe VPNs) ---
    dnp03_yaw_gain: float = 1.15
    # Frontal / bilateral loom: slow and pitch (no laterality to yaw away).
    loom_brake_gain: float = 0.50
    loom_brake_scale_hz: float = 0.25
    loom_pitch_gain: float = 0.0
    dnp03_scale_hz: float = 0.50
    dnp03_deadzone_hz: float = 0.04
    # Ignore weak binocular L−R so residual LC4_L cannot yaw the fly
    # into a frontal-right object (offset_x+0.8) or spin in open space.
    loom_yaw_deadzone: float = 0.14
    # Residual DNa02/HSE also spin the fly in open space onto a wall.
    steer_deadzone: float = 0.0
    # LPLC2 / LPLC1 / LC4 are ipsilateral looming VPNs (hex-swap / L-only).
    # Use baseline-subtracted L−R, not a contralateral wall-DC hack.
    lplc2_threat_gain: float = 0.45
    lplc1_threat_gain: float = 0.20
    lc4_threat_gain: float = 0.80
    # Opt-in Python eye-mean reflex (OFF). Same class of shortcut as
    # use_engineered_looming_residual — not the official path.
    use_engineered_eye_laterality: bool = False
    eye_yaw_gain: float = 0.85
    eye_scale: float = 0.06
    eye_threat_min: float = 0.015

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
    prewarm_steps: int = 80
    # One prewarm for every scenario (settle DNp01; no short side-loom path).
    side_loom_prewarm_steps: int = 80
    prewarm_max_steps: int = 140
    prewarm_settle_std: float = 0.02
    prewarm_peak_drop_hz: float = 0.12
    # One stride for room and looms. Batch both eyes; box live frame ~41 ms
    # sequential → target mean ≤ 25 with stride 2 after the batch.
    flyvis_stride: int = 2
    flyvis_torch_threads: int = 2

    # --- Dynamics ---
    action_alpha: float = 0.50
    rate_cap_hz: float = 50.0
    latch_eps_hz: float = 1.0
    action_clip: float = 1.0

    notes: Dict[str, str] = field(default_factory=lambda: {
        "escape": (
            "DNp01 rise = max(L−rest_L, R−rest_R) ≥ 0.20 Hz, or ≥ 0.12 Hz "
            "while LC4/LPLC2 expand (biological GF at 40–60°); laterality "
            "and VPN are rises over a slow rest so residual left DC / "
            "self-motion cannot fire; course freeze only on a lateral "
            "DNp01 climb (bilateral wall keeps HS); "
            "HS-orthogonal VPN; p90 LC4/LPLC2; T4/T5 handoff gain; "
            "consistent-sign lock; course frozen; U-flip; stride 2"
        ),
        "dnp03": (
            "DNp03 + LC4/LPLC ipsilateral L−R after per-side baseline → "
            "contralateral turn; compact dark loom object, not wall-drift"
        ),
        "dng02": "mean(L,R) amplitude; NO L−R yaw",
        "yaw_actuator": (
            "DN yaw (DNa02+DNp15+DNp03+LC4/LPLC−HSE) drives residual torque "
            "AND L−R stroke; DNg02 stays bilateral"
        ),
        "normalisation": "DN L−R / M1 peak-delta scales before tanh, after baseline",
        "hse": "compensatory yaw -= tanh((L−R)/20)",
        "camera": (
            "Forward 150° square-FOV cameras; flyvis BoxEye hex-box "
            "average (5.8° lattice, not a 5° Gaussian on top). Lattice "
            "centre = frontal. Drosophila envelope (−15°…+170° ipsilateral, "
            "~15° overlap). Adaptive p80 (τ=2 s) → 0.5, fixed contrast "
            "gain. Matte-black loom. 320 px. 1/f RMS 0.05, gain 1.10. "
            "Laterality is per-side DNp01 rise. LPi (Glu/−) T4/T5→LPi→"
            "LPLC2 is in coreTplus. "
            "Eye input is rendered pixels only — no sim object pose/radius. "
            "No U-flip, no 40° cant."
        ),
        "geometry": (
            "Lateral |az|≥30° at rise onset must turn away; frontal GF "
            "must escape/brake without turning into the object. "
            "Official gate: frontolateral collision-course |az0| 20–70°, "
            "visible+approaching (dCA≤0) and turn away (≥80%). Pure frontal "
            "|az0|<20° must fire (either side / brake ok). Eccentric "
            "|az|≈90–120° (old official L/R) are report-only: LPLC2/GF "
            "frontal bias. Solid dark disc; no hide-as-loom."
        ),
        "hover_up": "copied from FlyEnvironment.hover_action at bind time",
    })

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)
