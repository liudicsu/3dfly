"""Map brain activity to flight control commands."""

import numpy as np
import pandas as pd
from typing import Dict, Optional


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
        """
        from threedfly.connectome.neuron_selection import (
            select_input_neurons_for_luminance,
            select_input_neurons_for_optic_flow,
            select_input_neurons_for_looming,
            select_output_neurons_for_motor,
            create_fixed_input_assignment,
        )
        
        self.n_neurons = n_neurons
        self.warmup_steps = int(warmup_steps)
        self.min_warmup_steps = int(min(min_warmup_steps, self.warmup_steps)) if self.warmup_steps else 0
        self._warmup_count = 0
        self.create_fixed_input_assignment = create_fixed_input_assignment
        # Rest lift equals weight (env.hover_action ≈ 0.32). Engineered, not a DN.
        self.hover_action = 0.32
        self.looming_input_gain = 8.0
        self._last_looming = 0.0
        
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
        
        # Output mapping: DN types → motor commands
        self.output_mapping = select_output_neurons_for_motor(
            annotations, body_ids, body_id_to_idx
        )
        
        print(f"\nMotor readout from descending neurons:")
        for dn_type in ("DNp01", "DNg02", "DNa01", "DNa02", "DNb"):
            indices = self.output_mapping.get(dn_type, np.array([], dtype=np.int32))
            if len(indices) > 0:
                print(f"  {dn_type}: {len(indices)} neurons")
            else:
                print(f"  {dn_type}: not found (graceful fallback)")
        
        print("="*70)
        
        # Resting rates (set during warmup)
        self.resting_rates = {}
        self.resting_std = {}
        self.warmup_history = {}
        self.warmup_complete = False
        self.escape_timer = 0
        self.escape_duration = 8
        self.dnp01_prev_z = 0.0
        
        # Smoothing
        self.prev_action = np.zeros(5)
        self.alpha = 0.7
    
    def set_visual_input(
        self,
        left_luminance: np.ndarray,
        right_luminance: np.ndarray,
        optic_flow_features: Dict[str, np.ndarray],
        looming_features: Dict[str, np.ndarray],
        brain,
    ):
        """
        Set visual input to brain via biological neuron mappings.
        
        Args:
            left_luminance: Left eye luminance (scalar or array)
            right_luminance: Right eye luminance (scalar or array)
            optic_flow_features: Dict with keys like "t4a_left", "t5c_right"
            looming_features: Dict with keys "left", "right"
            brain: Brain simulator instance
        """
        # Build complete input vector
        all_indices = []
        all_values = []
        
        # Luminance → R1-R6
        if self.input_mapping["luminance_left"].size > 0:
            pop_size = len(self.input_mapping["luminance_left"])
            lum_left_val = np.atleast_1d(left_luminance)[0] if np.isscalar(left_luminance) else left_luminance.mean()
            all_indices.extend(self.input_mapping["luminance_left"])
            all_values.extend([lum_left_val] * pop_size)
        
        if self.input_mapping["luminance_right"].size > 0:
            pop_size = len(self.input_mapping["luminance_right"])
            lum_right_val = np.atleast_1d(right_luminance)[0] if np.isscalar(right_luminance) else right_luminance.mean()
            all_indices.extend(self.input_mapping["luminance_right"])
            all_values.extend([lum_right_val] * pop_size)
        
        # Optic flow → T4/T5 (keys like t4a_left)
        for key, feature_val in optic_flow_features.items():
            neurons = self.input_mapping.get(key, np.array([], dtype=np.int32))
            if neurons.size == 0:
                continue
            values = np.atleast_1d(np.asarray(feature_val, dtype=np.float32)).ravel()
            if values.size == 1:
                all_indices.extend(neurons)
                all_values.extend([float(values[0])] * len(neurons))
            else:
                assignment = self.create_fixed_input_assignment(values.size, len(neurons), seed=42)
                all_indices.extend(neurons)
                all_values.extend(values[assignment].tolist())
        
        # Looming → LPLC2 + LC4 (gain so current can reach DNp01 through signed weights)
        self._last_looming = 0.0
        if "left" in looming_features and self.input_mapping["looming_left"].size > 0:
            neurons = self.input_mapping["looming_left"]
            raw = np.atleast_1d(looming_features["left"])[0] if np.isscalar(looming_features["left"]) else looming_features["left"].mean()
            raw = float(raw)
            self._last_looming = max(self._last_looming, abs(raw))
            val = raw * self.looming_input_gain
            all_indices.extend(neurons)
            all_values.extend([val] * len(neurons))
        
        if "right" in looming_features and self.input_mapping["looming_right"].size > 0:
            neurons = self.input_mapping["looming_right"]
            raw = np.atleast_1d(looming_features["right"])[0] if np.isscalar(looming_features["right"]) else looming_features["right"].mean()
            raw = float(raw)
            self._last_looming = max(self._last_looming, abs(raw))
            val = raw * self.looming_input_gain
            all_indices.extend(neurons)
            all_values.extend([val] * len(neurons))
        
        # Set input all at once
        if len(all_indices) > 0:
            input_vec = np.array(all_values, dtype=np.float32)
            neuron_indices = np.array(all_indices, dtype=np.int32)
            brain.set_input(input_vec, neuron_indices=neuron_indices)
    
    def record_resting_rates(self, brain_activity: np.ndarray):
        """Record resting rates during warmup."""
        if self.warmup_complete:
            return
        self._warmup_count += 1
        for dn_type, indices in self.output_mapping.items():
            if len(indices) > 0:
                self.warmup_history.setdefault(dn_type, []).append(
                    float(brain_activity[indices].mean())
                )

    def warmup_converged(self, window: int = 50, rel_tol: float = 0.03) -> bool:
        """True when recent resting-rate windows have settled."""
        if self._warmup_count < self.min_warmup_steps:
            return False
        tracked = [s for s in self.warmup_history.values() if len(s) > 0]
        if not tracked:
            return self._warmup_count >= self.min_warmup_steps
        w = min(window, min(len(s) for s in tracked))
        if w < 5:
            return False
        for samples in tracked:
            recent = np.asarray(samples[-w:], dtype=float)
            if np.std(recent) > rel_tol * (np.abs(np.mean(recent)) + 0.05):
                return False
        return True
    
    def finalize_warmup(self):
        """Compute mean/std resting rates after warmup."""
        if self.warmup_complete:
            return
        
        for dn_type, samples in self.warmup_history.items():
            arr = np.asarray(samples, dtype=float)
            tail = arr[-50:] if arr.size >= 50 else arr
            if tail.size == 0:
                continue
            self.resting_rates[dn_type] = float(tail.mean())
            self.resting_std[dn_type] = float(tail.std(ddof=0))
        
        self.warmup_complete = True
        
        print("\n[Warmup complete]")
        print("Resting rates:")
        for dn_type, rate in self.resting_rates.items():
            print(f"  {dn_type}: {rate:.4f} ± {self.resting_std.get(dn_type, 0):.4f}")

    def _normalized_readout(self, dn_type: str, brain_activity: np.ndarray) -> float:
        indices = self.output_mapping.get(dn_type, np.array([], dtype=np.int32))
        if len(indices) == 0:
            return 0.0
        rate = float(brain_activity[indices].mean())
        if not self.warmup_complete:
            return float(np.tanh(rate))
        rest = self.resting_rates.get(dn_type, 0.0)
        scale = max(self.resting_std.get(dn_type, 0.0), 0.25 * abs(rest), 0.5)
        return float(np.tanh((rate - rest) / scale))

    def _channel_usable(self, dn_type: str, brain_activity: np.ndarray, eps: float = 1e-6) -> bool:
        indices = self.output_mapping.get(dn_type, np.array([], dtype=np.int32))
        if len(indices) == 0:
            return False
        current = float(np.abs(brain_activity[indices]).mean())
        rest = abs(self.resting_rates.get(dn_type, 0.0))
        return max(current, rest) > eps
    
    def compute_action(
        self,
        brain_activity: np.ndarray,
        exploration_signal: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute flight action from DN activity.
        
        Maps specific DN types to motor commands:
        - DNp01: escape → emergency pitch (plus engineered looming residual)
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
        dnb = self._normalized_readout("DNb", brain_activity)
        dnp = self._normalized_readout("DNp01", brain_activity)
        # Engineered residual: strong looming can still trigger escape if DNp01 is silent.
        loom_assist = float(np.tanh(3.0 * (self._last_looming - 0.35)))
        dnp_eff = max(dnp, loom_assist)

        # Rest lift = weight; DNg02 is a bounded deviation around hover.
        thrust_forward = 0.35 + 0.25 * dng
        thrust_up = self.hover_action + 0.15 * dng
        roll = 0.12 * dnb

        # DNp01: event trigger on threshold crossing, not a continuous pitch command
        if dnp_eff > 0.55 and self.dnp01_prev_z <= 0.55:
            self.escape_timer = self.escape_duration
        self.dnp01_prev_z = dnp_eff
        pitch = 0.35 if self.escape_timer > 0 else 0.0
        if self.escape_timer > 0:
            self.escape_timer -= 1

        yaw = 0.45 * (
            self._normalized_readout("DNa02_L", brain_activity)
            - self._normalized_readout("DNa02_R", brain_activity)
        ) + 0.35 * (
            self._normalized_readout("DNg02_L", brain_activity)
            - self._normalized_readout("DNg02_R", brain_activity)
        )
        
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
        }
