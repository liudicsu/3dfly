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
    # Alias of gf_threshold (pd form). Live fire is the readout, not 0.20.
    dnp01_escape_delta_hz: float = 0.23
    # Frozen GF readout from scripts/calibrate_gf_readout.py (do not retune
    # on closed-loop gates). Canonical set: flyvis loom() l/v=40 ms, fly
    # still, gray; both-eyes x0=0 and L/R at az ±30°.
    # form=pd, k=0.045 s, threshold=0.23 Hz
    #   both_x0_0  37.7°   L_az30  49.3°   R_az-30  54.8°
    # Binocular hex loom is stronger (rise@50°=0.25 vs 0.18); no (k, thr)
    # puts all three in 40–60°. Fit maximises n_in_window (2/3) then RMS
    # to 50°. von Reyn et al. 2014 Nat Neurosci / 2017 Curr Biol.
    gf_form: str = "pd"
    gf_k: float = 0.045
    gf_threshold: float = 0.23
    gf_vel_window_s: float = 0.04
    gf_target_deg: float = 50.0
    # Re-arm only after rise falls below this (rising-edge).
    dnp01_rearm_delta_hz: float = 0.12
    # Hold / steep-climb helper, not a second fire bar. The calibrated
    # GF score already includes velocity; VPN no longer lowers the bar.
    dnp01_vpn_delta_hz: float = 0.12
    # Genuine close is steep (≈0.016–0.025 Hz/step). Open wander is ≈0.009.
    dnp01_slope_min: float = 0.012
    # Threat from the mean L−R over the rise, not the fire instant.
    threat_integrate_steps: int = 10
    # Do not lock side on the first noisy sample (self-motion / early yaw).
    threat_lock_consistent: int = 4
    threat_lock_rise_hz: float = 0.12
    # Lock also on onset DNp01 L−R (|Δrise| ≥ 0.08 for threat_lock_consistent
    # steps) and let the episode lock win at GF fire over instantaneous
    # binocular VPN/DNp01 L−R. Controller logic, not brain weights.
    threat_lock_onset_dnp01: bool = True
    threat_lock_first: bool = True
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
    # Rest is frozen after a *no-object* prewarm whose photoreceptor
    # rest was τ→∞-initialised to that frame (uniform-field DNp01).
    # Recede keeps the large disc visible and does not initialise, so
    # the standing disc is not an onset. Biological GF has no
    # loom-present / texture-DC subtraction.
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
    # Plant (stroke→N·m and residual torque_max) is frozen at ×12 vs the
    # torque-limited 8e-6 / 5e-6 gains: unit DNa02 L−R → 90° in 50–80 ms,
    # peak 1000–5000 °/s (scripts/calibrate_saccade_plant.py).
    yaw_stroke_gain: float = 0.45
    saccade_plant_scale: float = 12.0

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
    # Total-yaw deadzone stays off; course-only deadzone is the plant
    # threshold (unit DNa02 term ≈ 0.53; post-baseline residual ≪ 0.1).
    # Documented with the saccade plant; not a gate retune.
    steer_deadzone: float = 0.0
    # Cap of the course (DNa02+DNp15) threshold; the live bar is
    # clip(k·σ_quiet, floor, cap) with σ the EMA RMS of the high-passed
    # course residual on quiet steps (no DNp01 climb / VPN expansion).
    # k = 0 restores the old absolute 0.15.
    course_deadzone: float = 0.15
    course_deadzone_sigma_k: float = 2.5
    course_deadzone_floor: float = 0.06
    course_sigma_tau_steps: float = 100.0
    # Slow rest on course L−R (DNa02 / DNp15 only). Scene/connectome DC
    # after baseline is subtracted; a 50–80 ms saccade is a transient
    # and is passed. HSE is not rested — it is the optomotor damp.
    # Not a brain-weight retune. Plant calibration disables this.
    course_rest_tau_steps: float = 40.0
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
    # Visual / photoreceptor tick. Room loop is 20 ms; loom gates use 10 ms.
    visual_dt: float = 0.02
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
            "GF readout calibrated once on flyvis loom() l/v=40 ms "
            "(both-eyes centred + L/R az ±30°, fly still, gray): "
            "pd score = rise + 0.045·d(rise)/dt, threshold 0.23 Hz "
            "(canonical fire 37.7° / 49.3° / 54.8°; binocular hex is "
            "early, no 3/3 in 40–60°). scripts/calibrate_gf_readout.py; "
            "von Reyn et al. 2014 Nat Neurosci / 2017 Curr Biol. "
            "VPN/laterality remain gates, not a second 0.12 Hz fire bar. "
            "Course freeze only on a lateral DNp01 climb; stride 2"
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
        "hse": (
            "compensatory: left turn raises HSE_R (progressive on the "
            "right eye) → yaw += tanh((L−R)/20) is negative. The old "
            "minus sign was positive feedback (open-space ωz/HSE trace). "
            "Torque is divided by saccade_plant_scale so HS stays damp "
            "and does not ride the DN saccade plant."
        ),
        "camera": (
            "Forward 150° + ipsilateral 150° side camera per eye so the "
            "flyvis ±75° lattice (1:1 deg, centre = frontal) is fully "
            "sampled; rear 75–165° has no flyvis cells. Photoreceptor rest "
            "τ=5 s, exp(−dt/τ), init at calib mean 0.5 (not frame 1); "
            "no-object prewarm initialises rest to that frame (τ→∞ "
            "equivalent). Recede skips that initialisation. "
            "BoxEye hex-box at the scaled spacing. Matte-black loom. "
            "320 px. 1/f RMS 0.05 (L/R-even), gain 1.10. Laterality is per-side "
            "DNp01 rise. LPi (Glu/−) T4/T5→LPi→LPLC2 is in coreTplus. "
            "Eye input is rendered pixels only — no sim object pose/radius. "
            "No U-flip, no 40° cant."
        ),
        "geometry": (
            "Frontolat |az0| 20–70° flight: surface clearance "
            "d_centre−0.58 > 0.1 m and far-side/heading-away (≥80%); "
            "GF fire is reported, not required. Frontal GF takeoff "
            "when both sides cross the calibrated score. Official "
            "looms start ≤5°. Receding discs must not fire. Static "
            "30–40° onset is report-only. Eccentric |az|≈90–120° "
            "are report-only."
        ),
        "saccade_plant": (
            "DN→yaw/roll plant frozen ×12 (WING_YAW_GAIN 9.6e-5, "
            "torque_max 6e-5): unit DNa02 L−R step, 90° in 50–80 ms, "
            "peak 1000–5000 °/s. Fry, Sayaman & Dickinson 2003; "
            "Tammero & Dickinson 2002; Muijres et al. 2014. "
            "scripts/calibrate_saccade_plant.py. Izz ~1e-8, "
            "τ=I/kd ~4 ms — not inertia-limited. Course deadzone "
            "= clip(k·σ_rest, floor, 0.15) with σ_rest the quiet-step "
            "RMS of the high-passed DNa02+DNp15 (unit DNa02 ≈ 0.53). "
            "HSE torque ÷12."
        ),
        "hover_up": "copied from FlyEnvironment.hover_action at bind time",
    })

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)
