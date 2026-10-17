"""Map brain activity to flight control commands."""

import numpy as np
import pandas as pd
from typing import Dict, Optional

# 8 proximity sectors from StereoVision.compute_looming_features:
# 0 forward, 1 FL, 2 L, 3 BL, 4 back, 5 BR, 6 R, 7 FR.
LOOMING_N_SECTORS = 8
LOOMING_LEFT_SECTORS = (0, 1, 2, 3)
LOOMING_RIGHT_SECTORS = (0, 7, 6, 5)


def pack_looming_features(left, right=None) -> Dict[str, np.ndarray]:
    """Pack per-eye 8-sector expansion vectors for retinotopic LPLC2/LC4 encoding."""
    def _vec(x):
        s = np.atleast_1d(np.asarray(x, dtype=np.float32)).ravel()
        if s.size < LOOMING_N_SECTORS:
            s = np.pad(s, (0, LOOMING_N_SECTORS - int(s.size)))
        return np.asarray(s[:LOOMING_N_SECTORS], dtype=np.float32)

    left_v = _vec(left)
    right_v = left_v.copy() if right is None else _vec(right)
    return {"left": left_v, "right": right_v, "sectors": np.maximum(left_v, right_v)}


class FlightController:
    """
    Map descending neuron (DN) activity to MuJoCo flight commands.
    
    Uses a simple hand-designed linear mapping from brain activity features
    to thrust and torque commands. No training/learning.
    """
    
    def __init__(
        self,
        n_neurons: int,
        use_rate_model: bool = True,
    ):
        """
        Initialize controller.
        
        Args:
            n_neurons: Number of neurons in brain model
            use_rate_model: Whether using rate model (vs LIF)
        """
        self.n_neurons = n_neurons
        self.use_rate_model = use_rate_model
        
        # Identify "descending neuron" indices (last 20% of neurons as proxy)
        # In reality, would use annotated DN types from connectome
        self.dn_start = int(0.8 * n_neurons)
        self.dn_end = n_neurons
        self.n_dn = self.dn_end - self.dn_start
        
        # Fixed mapping gains (2D horizontal exploration - minimal vertical control)
        self.thrust_forward_gain = 1.5  # Strong forward
        self.thrust_up_gain = 0.0  # Zero - let fixed bias handle altitude
        self.yaw_gain = 1.2  # Strong turning
        self.pitch_gain = 0.0  # Zero - no pitch control
        self.roll_gain = 0.0  # Zero - no roll control
        
        # Fixed upward thrust to maintain altitude (like a drone).
        # ~0.45 * thrust_max exceeds body weight after SI actuator scaling.
        self.thrust_up_bias = 0.45
        
        # Smoothing filter
        self.prev_action = np.zeros(5)
        self.alpha = 0.7  # Exponential smoothing
        self.last_wing_strokes = np.zeros(2, dtype=np.float64)
    
    def compute_action(
        self,
        brain_activity: np.ndarray,
        exploration_signal: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute flight action from brain activity.
        
        Args:
            brain_activity: Neural activity (spikes or rates)
            exploration_signal: Optional exploration bias [forward, turn, up]
        
        Returns:
            Action vector: [thrust_forward, thrust_up, roll, pitch, yaw]
        """
        # Extract DN activity
        dn_activity = brain_activity[self.dn_start:self.dn_end]
        
        # Normalize
        if dn_activity.max() > 0:
            dn_norm = dn_activity / (dn_activity.max() + 1e-6)
        else:
            dn_norm = dn_activity
        
        # Simple linear mapping (engineered, not learned)
        # Forward thrust: balanced strong baseline for stable horizontal flight
        forward_baseline = 0.7  # Strong but stable
        thrust_forward = forward_baseline + self.thrust_forward_gain * dn_norm.mean()
        
        # Upward thrust: bias + activity (counters gravity)
        thrust_up = self.thrust_up_bias + self.thrust_up_gain * dn_norm[:self.n_dn//3].mean()
        
        # Turning: based on left-right asymmetry
        left_activity = dn_norm[:self.n_dn//2].mean()
        right_activity = dn_norm[self.n_dn//2:].mean()
        yaw = self.yaw_gain * (left_activity - right_activity)
        
        # Pitch and roll: based on front-back asymmetry
        front_activity = dn_norm[:self.n_dn//3].mean()
        back_activity = dn_norm[2*self.n_dn//3:].mean()
        pitch = self.pitch_gain * (front_activity - back_activity)
        roll = self.roll_gain * (left_activity - right_activity)
        
        # Combine into action (2D + altitude maintenance)
        action = np.array([thrust_forward, thrust_up, 0.0, 0.0, yaw])  # No roll/pitch
        # exploration_signal is applied as visual input bias in the runner, not here.
        
        # Smooth action
        action = self.alpha * action + (1 - self.alpha) * self.prev_action
        self.prev_action = action.copy()
        
        # Clip to valid range
        action = np.clip(action, -1.0, 1.0)
        mean = float(np.clip(action[1], 0.0, 1.0))
        diff = float(np.clip(action[4], -1.0, 1.0))
        self.last_wing_strokes = np.array(
            [float(np.clip(mean + 0.5 * diff, 0.0, 1.0)),
             float(np.clip(mean - 0.5 * diff, 0.0, 1.0))],
            dtype=np.float64,
        )
        
        return action
    
    def get_status(self) -> Dict[str, float]:
        """Get controller status."""
        return {
            "last_forward": float(self.prev_action[0]),
            "last_up": float(self.prev_action[1]),
            "last_yaw": float(self.prev_action[4]),
        }


class BiologicalFlightController:
    """
    Flight controller using biological neuron mappings.
    
    Maps visual features to specific neuron types based on annotations,
    and reads motor output from descending neuron populations.
    """
    
    def __init__(
        self,
        n_neurons: int,
        annotations: pd.DataFrame,
        body_ids: np.ndarray,
        body_id_to_idx: Dict[int, int],
        nt_data: Optional[pd.DataFrame] = None,
        warmup_steps: int = 800,
        min_warmup_steps: int = 200,
        use_engineered_looming_residual: bool = False,
    ):
        """
        Initialize biological controller.
        
        Args:
            n_neurons: Total number of neurons
            annotations: Annotation DataFrame
            body_ids: Array of body IDs
            body_id_to_idx: Body ID to index mapping
            nt_data: Neurotransmitter data (optional)
            warmup_steps: Maximum grounded brain steps for rest calibration
            min_warmup_steps: Minimum steps before convergence may stop warmup
            use_engineered_looming_residual: Ablation-only engineered looming
                threshold that bypasses DNp01. Default OFF; escape is brain-driven.
        """
        from threedfly.connectome.neuron_selection import (
            select_input_neurons_for_luminance,
            select_input_neurons_for_optic_flow,
            select_input_neurons_for_looming,
            select_output_neurons_for_motor,
            select_haltere_afferent_neurons,
            find_optic_lobe_hex_columns,
            create_fixed_input_assignment,
        )
        
        self.n_neurons = n_neurons
        self.warmup_steps = int(warmup_steps)
        self.min_warmup_steps = int(min(min_warmup_steps, self.warmup_steps)) if self.warmup_steps else 0
        self._warmup_count = 0
        self.create_fixed_input_assignment = create_fixed_input_assignment
        # Rest lift equals weight (env.hover_action ≈ 0.32). Engineered, not a DN.
        self.hover_action = 0.32
        self.input_drive = 0.8
        self.input_drive_clip = 1.5
        # Fixed references (typical full-scale), not the instantaneous population mean.
        self.luminance_reference = 1.0
        self.flow_reference = 1.0
        self.haltere_reference = 2.0
        self.looming_reference = 1.0
        # Phasic LPLC2-like encoding: small tonic scale, strong onset kick.
        self.looming_input_gain = 0.5
        self.looming_onset_gain = 2.0
        self.yaw_baseline_alpha = 0.05  # engineered slow L−R baseline (~1 s)
        self._yaw_baseline = 0.0
        self.silent_side_hz = 1.0
        self.silent_side_rel = 0.05
        self._last_looming = 0.0
        self.brain_escape_count = 0
        self.engineered_escape_count = 0
        self.dnp03_turn_count = 0
        self.use_engineered_looming_residual = bool(use_engineered_looming_residual)
        self.dnp01_target_rest = 2.0
        self.escape_rate_margin = 3.0
        self._last_warmup_activity = None
        self._escape_yaw_sign = 0.0
        self._dnp03_prev_active = False
        
        # Input mapping: visual features → neuron populations
        print("\n[Biological I/O Mapping]")
        print("="*70)
        
        # Luminance → R1-R6
        self.input_mapping = {}
        self.input_mapping["luminance_left"] = select_input_neurons_for_luminance(
            annotations, body_ids, body_id_to_idx, side="L"
        )
        self.input_mapping["luminance_right"] = select_input_neurons_for_luminance(
            annotations, body_ids, body_id_to_idx, side="R"
        )
        print(f"Luminance → R1-R6: L={len(self.input_mapping['luminance_left'])}, "
              f"R={len(self.input_mapping['luminance_right'])}")
        
        # Optic flow → T4a-d and T5a-d separately
        flow_subtypes = []
        for cell_class in ("T4", "T5"):
            for subtype in ["a", "b", "c", "d"]:
                for side, side_name in (("L", "left"), ("R", "right")):
                    key = f"{cell_class.lower()}{subtype}_{side_name}"
                    neurons = select_input_neurons_for_optic_flow(
                        annotations, body_ids, body_id_to_idx,
                        subtype=subtype, side=side, cell_class=cell_class,
                    )
                    self.input_mapping[key] = neurons
                    if len(neurons) > 0:
                        flow_subtypes.append(f"{cell_class}{subtype}_{side}={len(neurons)}")
        
        print(f"Optic flow → T4/T5: {', '.join(flow_subtypes) if flow_subtypes else 'none found'}")
        
        # Looming → LPLC2 + LC4
        self.input_mapping["looming_left"] = select_input_neurons_for_looming(
            annotations, body_ids, body_id_to_idx, side="L"
        )
        self.input_mapping["looming_right"] = select_input_neurons_for_looming(
            annotations, body_ids, body_id_to_idx, side="R"
        )
        print(f"Looming → LPLC2+LC4: L={len(self.input_mapping['looming_left'])}, "
              f"R={len(self.input_mapping['looming_right'])}")
        self._looming_rf_left = self._seeded_rf_indices(
            len(self.input_mapping["looming_left"]), len(LOOMING_LEFT_SECTORS), seed=42
        )
        self._looming_rf_right = self._seeded_rf_indices(
            len(self.input_mapping["looming_right"]), len(LOOMING_RIGHT_SECTORS), seed=43
        )
        self._prev_looming_left = np.zeros(
            len(self.input_mapping["looming_left"]), dtype=np.float32
        )
        self._prev_looming_right = np.zeros(
            len(self.input_mapping["looming_right"]), dtype=np.float32
        )

        # Haltere afferents: body angular velocity. Empty → engineered damping fallback.
        hex_cols = find_optic_lobe_hex_columns(annotations)
        if hex_cols is None:
            print(
                "Optic-lobe hex columns: not present in annotations "
                "(no hexX/hexY/column fields). Using synthetic ~750-ommatidia hex lattice."
            )
        else:
            print(f"Optic-lobe hex columns: {hex_cols[0]}, {hex_cols[1]}")
        self.optic_lobe_hex_columns = hex_cols

        haltere = select_haltere_afferent_neurons(annotations, body_ids, body_id_to_idx)
        self.input_mapping["haltere_left"] = haltere["left"]
        self.input_mapping["haltere_right"] = haltere["right"]
        self.haltere_fallback = haltere["fallback"]
        
        # Output mapping: DN types → motor commands
        self.output_mapping = select_output_neurons_for_motor(
            annotations, body_ids, body_id_to_idx
        )
        
        print(f"\nMotor readout from descending neurons:")
        for dn_type in ("DNp01", "DNp03", "DNg02", "DNa01", "DNa02", "DNb"):
            indices = self.output_mapping.get(dn_type, np.array([], dtype=np.int32))
            if len(indices) > 0:
                print(f"  {dn_type}: {len(indices)} neurons")
            else:
                print(f"  {dn_type}: not found (graceful fallback)")
        
        print("="*70)
        
        # Resting rates (set during warmup) — robust median / MAD
        self.resting_rates = {}
        self.resting_std = {}
        self.resting_pair_mean = {}
        self.resting_pair_scale = {}
        self.warmup_history = {}
        self.warmup_complete = False
        self.escape_timer = 0
        self.escape_duration = 8
        self.dnp01_prev_z = 0.0
        
        # Smoothing
        self.prev_action = np.zeros(5)
        self.alpha = 0.7
        self.last_wing_strokes = np.array([self.hover_action, self.hover_action], dtype=np.float64)
    
    @staticmethod
    def _seeded_rf_indices(n_cells: int, n_sectors: int, seed: int) -> np.ndarray:
        """Seeded RF assignment: shuffle a repeating sector cycle so every sector is used."""
        n_cells = int(n_cells)
        n_sectors = max(int(n_sectors), 1)
        if n_cells <= 0:
            return np.array([], dtype=np.int32)
        rng = np.random.RandomState(seed)
        idx = (np.arange(n_cells, dtype=np.int32) % n_sectors)
        rng.shuffle(idx)
        return idx

    def _encode_fixed_ref(self, indices, raw_values, reference: float, extra_gain: float = 1.0):
        """Map a feature vector onto a population with a fixed full-scale reference."""
        indices = np.asarray(indices, dtype=np.int32)
        if indices.size == 0:
            return indices, np.array([], dtype=np.float32)
        values = np.atleast_1d(np.asarray(raw_values, dtype=np.float32)).ravel()
        if values.size == 1:
            values = np.full(indices.size, float(values[0]), dtype=np.float32)
        elif values.size != indices.size:
            assignment = self.create_fixed_input_assignment(
                values.size, indices.size, seed=42
            )
            values = values[assignment].astype(np.float32)
        ref = max(float(reference), 1e-6)
        values = values * (self.input_drive / ref) * float(extra_gain)
        values = np.clip(values, -self.input_drive_clip, self.input_drive_clip)
        return indices, values.astype(np.float32)

    def _local_looming(self, raw, side: str) -> np.ndarray:
        """Per-cell looming in a side's hemifield (scalar, 4-sector, or 8-sector)."""
        mapping = self.input_mapping.get(f"looming_{side}", np.array([], dtype=np.int32))
        n_cells = int(mapping.size)
        if n_cells == 0:
            return np.array([], dtype=np.float32)
        values = np.atleast_1d(np.asarray(raw, dtype=np.float32)).ravel()
        rf = self._looming_rf_left if side == "left" else self._looming_rf_right
        hemifield = np.asarray(
            LOOMING_LEFT_SECTORS if side == "left" else LOOMING_RIGHT_SECTORS,
            dtype=np.int32,
        )
        if rf.size != n_cells:
            rf = self._seeded_rf_indices(n_cells, max(hemifield.size, 1), seed=42 if side == "left" else 43)
        if values.size == 1:
            return np.full(n_cells, float(values[0]), dtype=np.float32)
        if values.size >= LOOMING_N_SECTORS:
            local_sectors = values[:LOOMING_N_SECTORS][hemifield]
            return np.asarray(local_sectors[rf], dtype=np.float32)
        if values.size == hemifield.size:
            return np.asarray(values[rf], dtype=np.float32)
        if values.size == n_cells:
            return values.astype(np.float32)
        assignment = self.create_fixed_input_assignment(values.size, n_cells, seed=42)
        return values[assignment].astype(np.float32)

    def _encode_looming(self, raw, side: str) -> np.ndarray:
        """Absolute looming rate code plus an onset kick on rising expansion."""
        local = self._local_looming(raw, side)
        if local.size == 0:
            return local
        prev = self._prev_looming_left if side == "left" else self._prev_looming_right
        if prev.size != local.size:
            prev = np.zeros_like(local)
        onset = np.maximum(local - prev, 0.0)
        level_gain = (
            float(self.looming_input_gain)
            * float(self.input_drive)
            / max(float(self.looming_reference), 1e-6)
        )
        drive = level_gain * local + float(self.looming_onset_gain) * onset
        drive = np.clip(drive, 0.0, self.input_drive_clip).astype(np.float32)
        if side == "left":
            self._prev_looming_left = local.copy()
        else:
            self._prev_looming_right = local.copy()
        self._last_looming = max(self._last_looming, float(np.max(np.abs(local))))
        return drive

    def set_visual_input(
        self,
        left_luminance: np.ndarray,
        right_luminance: np.ndarray,
        optic_flow_features: Dict[str, np.ndarray],
        looming_features: Dict[str, np.ndarray],
        brain,
        angular_velocity: Optional[np.ndarray] = None,
    ):
        """
        Set visual input to brain via biological neuron mappings.
        
        Args:
            left_luminance: Left eye luminance (scalar or array)
            right_luminance: Right eye luminance (scalar or array)
            optic_flow_features: Dict with keys like "t4a_left", "t5c_right"
            looming_features: Dict with keys "left", "right" (scalar, hemifield,
                or 8-sector vector). Encoding is absolute (not mean-normalised)
                and retinotopic within each side's hemifield.
            brain: Brain simulator instance
        """
        pending_idx: list = []
        pending_val: list = []

        def _extend(indices, raw_values, reference: float, extra_gain: float = 1.0):
            idx, values = self._encode_fixed_ref(indices, raw_values, reference, extra_gain)
            if idx.size == 0:
                return
            pending_idx.extend(idx.tolist())
            pending_val.extend(values.tolist())

        if self.input_mapping["luminance_left"].size > 0:
            _extend(
                self.input_mapping["luminance_left"],
                left_luminance,
                self.luminance_reference,
            )
        if self.input_mapping["luminance_right"].size > 0:
            _extend(
                self.input_mapping["luminance_right"],
                right_luminance,
                self.luminance_reference,
            )

        for key, feature_val in optic_flow_features.items():
            neurons = self.input_mapping.get(key, np.array([], dtype=np.int32))
            if neurons.size == 0:
                continue
            _extend(neurons, feature_val, self.flow_reference)

        self._last_looming = 0.0
        if "left" in looming_features and self.input_mapping["looming_left"].size > 0:
            drive = self._encode_looming(looming_features["left"], "left")
            pending_idx.extend(self.input_mapping["looming_left"].tolist())
            pending_val.extend(drive.tolist())
        if "right" in looming_features and self.input_mapping["looming_right"].size > 0:
            drive = self._encode_looming(looming_features["right"], "right")
            pending_idx.extend(self.input_mapping["looming_right"].tolist())
            pending_val.extend(drive.tolist())

        if angular_velocity is not None:
            self._extend_haltere(pending_idx, pending_val, angular_velocity)

        if pending_idx:
            brain.set_input(
                np.array(pending_val, dtype=np.float32),
                neuron_indices=np.array(pending_idx, dtype=np.int32),
            )

    def _extend_haltere(self, pending_idx, pending_val, angular_velocity) -> None:
        """Encode body-frame angular velocity onto left/right haltere afferents."""
        ang = np.atleast_1d(np.asarray(angular_velocity, dtype=np.float32)).ravel()
        if ang.size < 3:
            ang = np.pad(ang, (0, 3 - ang.size))
        wx, wy, wz = float(ang[0]), float(ang[1]), float(ang[2])
        left = self.input_mapping.get("haltere_left", np.array([], dtype=np.int32))
        right = self.input_mapping.get("haltere_right", np.array([], dtype=np.int32))
        if left.size > 0:
            idx, values = self._encode_fixed_ref(
                left, np.array([wx, wy, wz], dtype=np.float32), self.haltere_reference
            )
            pending_idx.extend(idx.tolist())
            pending_val.extend(values.tolist())
        if right.size > 0:
            idx, values = self._encode_fixed_ref(
                right, np.array([wx, -wy, wz], dtype=np.float32), self.haltere_reference
            )
            pending_idx.extend(idx.tolist())
            pending_val.extend(values.tolist())

    def set_haltere_input(self, angular_velocity: np.ndarray, brain) -> None:
        """Write angular-velocity drive onto annotated haltere neurons (test helper)."""
        pending_idx: list = []
        pending_val: list = []
        self._extend_haltere(pending_idx, pending_val, angular_velocity)
        if pending_idx:
            brain.set_input(
                np.array(pending_val, dtype=np.float32),
                neuron_indices=np.array(pending_idx, dtype=np.int32),
            )
    
    def record_resting_rates(self, brain_activity: np.ndarray):
        """Record resting rates during warmup."""
        if self.warmup_complete:
            return
        self._warmup_count += 1
        self._last_warmup_activity = np.asarray(brain_activity, dtype=float).copy()
        for dn_type, indices in self.output_mapping.items():
            if len(indices) > 0:
                self.warmup_history.setdefault(dn_type, []).append(
                    float(brain_activity[indices].mean())
                )

    @staticmethod
    def _robust_center_scale(samples) -> tuple:
        arr = np.asarray(samples, dtype=float)
        if arr.size == 0:
            return 0.0, 0.5
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med)))
        scale = max(1.4826 * mad, 0.25 * abs(med), 0.5)
        return med, scale

    def warmup_converged(self, window: int = 50, rel_tol: float = 1.0) -> bool:
        """True when window medians of primary DNs are stable relative to spread."""
        if self._warmup_count < self.min_warmup_steps:
            return False
        primary = ("DNp01", "DNg02", "DNa02", "DNb", "DNa01")
        tracked = [
            self.warmup_history[k]
            for k in primary
            if k in self.warmup_history and len(self.warmup_history[k]) >= 20
        ]
        if not tracked:
            return self._warmup_count >= self.min_warmup_steps
        w = min(window, min(len(s) // 2 for s in tracked))
        if w < 5:
            return False
        for samples in tracked:
            recent = np.asarray(samples[-w:], dtype=float)
            prev = np.asarray(samples[-2 * w:-w], dtype=float)
            m1 = float(np.median(recent))
            m0 = float(np.median(prev))
            spread = max(float(np.median(np.abs(recent - m1))) * 1.4826, 0.05)
            if abs(m1 - m0) > rel_tol * spread:
                return False
        return True
    
    def finalize_warmup(self):
        """Compute robust (median/MAD) resting rates after warmup."""
        if self.warmup_complete:
            return
        
        for dn_type, samples in self.warmup_history.items():
            arr = np.asarray(samples, dtype=float)
            if arr.size == 0:
                continue
            med, scale = self._robust_center_scale(arr)
            self.resting_rates[dn_type] = med
            self.resting_std[dn_type] = scale

        for left_key, right_key in (
            ("DNa02_L", "DNa02_R"),
            ("DNg02_L", "DNg02_R"),
            ("DNa01_L", "DNa01_R"),
            ("DNp03_L", "DNp03_R"),
        ):
            if left_key in self.warmup_history and right_key in self.warmup_history:
                diffs = (
                    np.asarray(self.warmup_history[left_key], dtype=float)
                    - np.asarray(self.warmup_history[right_key], dtype=float)
                )
                rest, scale = self._robust_center_scale(diffs)
                self.resting_pair_mean[(left_key, right_key)] = rest
                self.resting_pair_scale[(left_key, right_key)] = scale
        
        self.warmup_complete = True
        
        print("\n[Warmup complete]")
        print("Resting rates (median ± MAD-scale):")
        for dn_type, rate in self.resting_rates.items():
            print(f"  {dn_type}: {rate:.4f} ± {self.resting_std.get(dn_type, 0):.4f}")

    def apply_giant_fiber_calibration(self, brain, target_rest: Optional[float] = None) -> None:
        """Lower DNp01/DNp03 rest via per-neuron thresholds so looming can spike above rest."""
        target = self.dnp01_target_rest if target_rest is None else float(target_rest)
        if not hasattr(brain, "apply_rest_thresholds"):
            return
        activity = self._last_warmup_activity
        for key in ("DNp01", "DNp03", "DNp03_L", "DNp03_R"):
            idx = self.output_mapping.get(key, np.array([], dtype=np.int32))
            if idx is None or len(idx) == 0:
                continue
            if activity is not None and len(activity) > int(np.max(idx)):
                observed = activity[idx]
            else:
                observed = self.resting_rates.get(key, 0.0)
            brain.apply_rest_thresholds(idx, observed, target_rest=target)
            if key in self.resting_rates:
                self.resting_rates[key] = min(float(self.resting_rates[key]), target)

    def _threat_yaw_sign(self, brain_activity: np.ndarray) -> float:
        """+1 turn left (away from right threat), -1 turn right (away from left threat)."""
        if self._pair_usable("DNp03_L", "DNp03_R", brain_activity):
            diff = self._raw_rate("DNp03_L", brain_activity) - self._raw_rate(
                "DNp03_R", brain_activity
            )
            if abs(diff) > 1e-6:
                return -1.0 if diff > 0 else 1.0
        left_idx = self.input_mapping.get("looming_left", np.array([], dtype=np.int32))
        right_idx = self.input_mapping.get("looming_right", np.array([], dtype=np.int32))
        left_r = float(brain_activity[left_idx].mean()) if len(left_idx) else 0.0
        right_r = float(brain_activity[right_idx].mean()) if len(right_idx) else 0.0
        if left_r == right_r:
            return 0.0
        return -1.0 if left_r > right_r else 1.0

    def _dnp01_escape_threshold(self) -> float:
        rest = float(self.resting_rates.get("DNp01", 0.0))
        scale = max(float(self.resting_std.get("DNp01", 0.0)), 0.5)
        return rest + max(float(self.escape_rate_margin), 1.5 * scale)

    def _raw_rate(self, dn_type: str, brain_activity: np.ndarray) -> float:
        indices = self.output_mapping.get(dn_type, np.array([], dtype=np.int32))
        if len(indices) == 0:
            return 0.0
        return float(brain_activity[indices].mean())

    def _side_z(self, dn_type: str, brain_activity: np.ndarray) -> float:
        """Z-score one DN channel against its own rest/spread."""
        rate = self._raw_rate(dn_type, brain_activity)
        if not self.warmup_complete:
            return float(np.tanh(rate))
        rest = self.resting_rates.get(dn_type, 0.0)
        scale = max(self.resting_std.get(dn_type, 0.0), 0.25 * abs(rest), 0.5)
        return float((rate - rest) / scale)

    def _normalized_readout(self, dn_type: str, brain_activity: np.ndarray) -> float:
        left_key, right_key = f"{dn_type}_L", f"{dn_type}_R"
        left_idx = self.output_mapping.get(left_key, np.array([], dtype=np.int32))
        right_idx = self.output_mapping.get(right_key, np.array([], dtype=np.int32))
        if len(left_idx) > 0 and len(right_idx) > 0:
            return 0.5 * (
                float(np.tanh(self._side_z(left_key, brain_activity)))
                + float(np.tanh(self._side_z(right_key, brain_activity)))
            )
        indices = self.output_mapping.get(dn_type, np.array([], dtype=np.int32))
        if len(indices) == 0:
            return 0.0
        return float(np.tanh(self._side_z(dn_type, brain_activity)))

    def _channel_energy(self, dn_type: str, brain_activity: np.ndarray) -> float:
        current = abs(self._raw_rate(dn_type, brain_activity))
        rest = abs(self.resting_rates.get(dn_type, 0.0))
        return max(current, rest)

    def _side_near_silent(self, key: str, partner_key: str, brain_activity: np.ndarray) -> bool:
        energy = self._channel_energy(key, brain_activity)
        partner = self._channel_energy(partner_key, brain_activity)
        return energy < self.silent_side_hz and energy < self.silent_side_rel * max(partner, 1e-9)

    def _channel_usable(self, dn_type: str, brain_activity: np.ndarray, eps: float = 1e-6) -> bool:
        indices = self.output_mapping.get(dn_type, np.array([], dtype=np.int32))
        if len(indices) == 0:
            return False
        return self._channel_energy(dn_type, brain_activity) > eps

    def _pair_usable(self, left_key: str, right_key: str, brain_activity: np.ndarray) -> bool:
        if not self._channel_usable(left_key, brain_activity) or not self._channel_usable(
            right_key, brain_activity
        ):
            return False
        if self._side_near_silent(left_key, right_key, brain_activity) or self._side_near_silent(
            right_key, left_key, brain_activity
        ):
            return False
        return True

    def _pair_yaw(self, left_key: str, right_key: str, brain_activity: np.ndarray) -> float:
        """Zero-mean L−R of per-side z-scores; 0 if either side is near-silent."""
        if not self._pair_usable(left_key, right_key, brain_activity):
            return 0.0
        return float(
            np.tanh(
                self._side_z(left_key, brain_activity) - self._side_z(right_key, brain_activity)
            )
        )
    
    def compute_action(
        self,
        brain_activity: np.ndarray,
        exploration_signal: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute flight action from DN activity.
        
        Maps specific DN types to motor commands:
        - DNp01: escape → short brake + turn away from the looming side
        - DNp03 L−R: looming-evoked evasive yaw (and escape-turn side)
        - DNg02: thrust/altitude deviations around hover = weight
        - DNa02 L−R and DNg02 L−R → yaw (bilateral, left/right symmetric)
        - DNb: baseline motor activity
        
        Args:
            brain_activity: Neural activity array
            exploration_signal: Ignored here. Exploration biases visual input
                in the runner; it must not override the biological action.
        
        Returns:
            Action: [thrust_forward, thrust_up, roll, pitch, yaw]
        """
        dng = self._normalized_readout("DNg02", brain_activity)
        dng_l = self._normalized_readout("DNg02_L", brain_activity)
        dng_r = self._normalized_readout("DNg02_R", brain_activity)
        dnb = self._normalized_readout("DNb", brain_activity)
        dnp_rate = self._raw_rate("DNp01", brain_activity)
        # DNg02 left/right set wing stroke amplitudes around hover (lift = weight at rest).
        hover = float(self.hover_action)
        self.last_wing_strokes = np.array([
            float(np.clip(hover + 0.25 * dng_l, 0.0, 1.0)),
            float(np.clip(hover + 0.25 * dng_r, 0.0, 1.0)),
        ], dtype=np.float64)
        if not self._pair_usable("DNg02_L", "DNg02_R", brain_activity):
            mean_dev = 0.20 * dng
            self.last_wing_strokes = np.array(
                [float(np.clip(hover + mean_dev, 0.0, 1.0))] * 2,
                dtype=np.float64,
            )
        # Engineered residual: ablation-only (default OFF). Documented as engineered.
        loom_assist = 0.0
        if self.use_engineered_looming_residual:
            loom_assist = float(np.tanh(3.0 * (self._last_looming - 0.35)))
        thresh = self._dnp01_escape_threshold()
        brain_fire = dnp_rate > thresh
        engineered_fire = (not brain_fire) and (loom_assist > 0.55)
        fire_level = 1.0 if (brain_fire or engineered_fire) else 0.0

        # DNp03 looming-evoked saccade count (L/R activity, not the raw looming vector).
        dnp03_diff = self._raw_rate("DNp03_L", brain_activity) - self._raw_rate(
            "DNp03_R", brain_activity
        )
        dnp03_active = abs(dnp03_diff) > max(self.escape_rate_margin, 1.0)
        if dnp03_active and not self._dnp03_prev_active:
            self.dnp03_turn_count += 1
        self._dnp03_prev_active = dnp03_active

        # DNp01: event trigger on threshold crossing over rest, not a continuous pitch command
        if fire_level > 0.5 and self.dnp01_prev_z <= 0.5:
            self.escape_timer = self.escape_duration
            self._escape_yaw_sign = self._threat_yaw_sign(brain_activity)
            if brain_fire:
                self.brain_escape_count += 1
            else:
                self.engineered_escape_count += 1
        self.dnp01_prev_z = fire_level

        escaping = self.escape_timer > 0
        # Rest lift = weight; DNg02 is a bounded deviation around hover.
        thrust_forward = 0.05 if escaping else (0.35 + 0.25 * dng)
        thrust_up = self.hover_action + 0.15 * dng
        roll = 0.12 * dnb
        pitch = 0.35 if escaping else 0.0
        if self.escape_timer > 0:
            self.escape_timer -= 1

        yaw = 0.0
        n_pairs = 0
        dropped = False
        if self._pair_usable("DNa02_L", "DNa02_R", brain_activity):
            yaw += 0.55 * self._pair_yaw("DNa02_L", "DNa02_R", brain_activity)
            n_pairs += 1
        else:
            dropped = True
        if self._pair_usable("DNg02_L", "DNg02_R", brain_activity):
            yaw += 0.45 * self._pair_yaw("DNg02_L", "DNg02_R", brain_activity)
            n_pairs += 1
        else:
            dropped = True
        if n_pairs == 0 and self._pair_usable("DNa01_L", "DNa01_R", brain_activity):
            yaw = 0.5 * self._pair_yaw("DNa01_L", "DNa01_R", brain_activity)
        if escaping:
            yaw = 0.8 * self._escape_yaw_sign + 0.2 * yaw
        elif self._pair_usable("DNp03_L", "DNp03_R", brain_activity):
            yaw += 0.35 * self._pair_yaw("DNp03_L", "DNp03_R", brain_activity)
        else:
            dropped = True
        # Engineered: slow running baseline of the remaining L−R signal so a
        # dropped silent-side pair cannot leave a DC yaw offset at rest.
        a = float(self.yaw_baseline_alpha)
        self._yaw_baseline = (1.0 - a) * self._yaw_baseline + a * float(yaw)
        if dropped and not escaping:
            yaw = float(yaw) - self._yaw_baseline
        
        # Assemble action. exploration_signal is intentionally unused here.
        action = np.array([thrust_forward, thrust_up, roll, pitch, yaw])
        
        # Smooth
        action = self.alpha * action + (1 - self.alpha) * self.prev_action
        self.prev_action = action.copy()
        
        # Clip
        action = np.clip(action, -1.0, 1.0)
        
        return action
    
    def get_status(self) -> Dict[str, float]:
        """Get controller status."""
        return {
            "last_forward": float(self.prev_action[0]),
            "last_up": float(self.prev_action[1]),
            "last_yaw": float(self.prev_action[4]),
            "warmup_complete": float(self.warmup_complete),
            "brain_escapes": float(self.brain_escape_count),
            "engineered_escapes": float(self.engineered_escape_count),
            "dnp03_turns": float(self.dnp03_turn_count),
            "wing_stroke_left": float(self.last_wing_strokes[0]),
            "wing_stroke_right": float(self.last_wing_strokes[1]),
        }
