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
        
        # Fixed upward thrust to maintain altitude (like a drone)
        self.thrust_up_bias = 0.12  # Strong enough to counter gravity and maintain ~1.5m
        
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
        
        # Add exploration signal if provided (favor horizontal movement)
        if exploration_signal is not None:
            action[0] += 0.8 * exploration_signal[0]  # Strong forward bias for XY coverage
            action[4] += 1.0 * exploration_signal[1]  # Strong yaw for directional changes
            action[1] += 0.0 * exploration_signal[2]  # Ignore vertical exploration bias
        
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
        warmup_steps: int = 0,
    ):
        """
        Initialize biological controller.
        
        Args:
            n_neurons: Total number of neurons
            annotations: Annotation DataFrame
            body_ids: Array of body IDs
            body_id_to_idx: Body ID to index mapping
            nt_data: Neurotransmitter data (optional)
            warmup_steps: Number of warmup steps for resting rate calibration
        """
        from threedfly.connectome.neuron_selection import (
            select_input_neurons_for_luminance,
            select_input_neurons_for_optic_flow,
            select_input_neurons_for_looming,
            select_output_neurons_for_motor,
            create_fixed_input_assignment,
        )
        
        self.n_neurons = n_neurons
        self.warmup_steps = warmup_steps
        
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
        
        # Optic flow → T4a-d, T5a-d
        flow_subtypes = []
        for subtype in ["a", "b", "c", "d"]:
            for side in ["L", "R"]:
                key = f"optic_flow_t4{subtype}_{side.lower()}"
                neurons = select_input_neurons_for_optic_flow(
                    annotations, body_ids, body_id_to_idx, subtype=subtype, side=side
                )
                self.input_mapping[key] = neurons
                if len(neurons) > 0:
                    flow_subtypes.append(f"T4{subtype}_{side}={len(neurons)}")
        
        for subtype in ["a", "b", "c", "d"]:
            for side in ["L", "R"]:
                key = f"optic_flow_t5{subtype}_{side.lower()}"
                neurons = select_input_neurons_for_optic_flow(
                    annotations, body_ids, body_id_to_idx, subtype=subtype, side=side
                )
                self.input_mapping[key] = neurons
                if len(neurons) > 0:
                    flow_subtypes.append(f"T5{subtype}_{side}={len(neurons)}")
        
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
        for dn_type, indices in self.output_mapping.items():
            if len(indices) > 0:
                print(f"  {dn_type}: {len(indices)} neurons")
            else:
                print(f"  {dn_type}: not found (graceful fallback)")
        
        print("="*70)
        
        # Resting rates (set during warmup)
        self.resting_rates = {}
        self.warmup_complete = False
        
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
        
        # Optic flow → T4/T5
        for key, feature_val in optic_flow_features.items():
            mapping_key = f"optic_flow_{key}"
            if mapping_key in self.input_mapping and self.input_mapping[mapping_key].size > 0:
                neurons = self.input_mapping[mapping_key]
                val = np.atleast_1d(feature_val)[0] if np.isscalar(feature_val) else feature_val.mean()
                all_indices.extend(neurons)
                all_values.extend([val] * len(neurons))
        
        # Looming → LPLC2 + LC4
        if "left" in looming_features and self.input_mapping["looming_left"].size > 0:
            neurons = self.input_mapping["looming_left"]
            val = np.atleast_1d(looming_features["left"])[0] if np.isscalar(looming_features["left"]) else looming_features["left"].mean()
            all_indices.extend(neurons)
            all_values.extend([val] * len(neurons))
        
        if "right" in looming_features and self.input_mapping["looming_right"].size > 0:
            neurons = self.input_mapping["looming_right"]
            val = np.atleast_1d(looming_features["right"])[0] if np.isscalar(looming_features["right"]) else looming_features["right"].mean()
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
        
        for dn_type, indices in self.output_mapping.items():
            if len(indices) > 0:
                if dn_type not in self.resting_rates:
                    self.resting_rates[dn_type] = []
                self.resting_rates[dn_type].append(brain_activity[indices].mean())
    
    def finalize_warmup(self):
        """Compute mean resting rates after warmup."""
        if self.warmup_complete:
            return
        
        for dn_type in self.resting_rates:
            self.resting_rates[dn_type] = np.mean(self.resting_rates[dn_type])
        
        self.warmup_complete = True
        
        print("\n[Warmup complete]")
        print("Resting rates:")
        for dn_type, rate in self.resting_rates.items():
            print(f"  {dn_type}: {rate:.4f}")
    
    def compute_action(
        self,
        brain_activity: np.ndarray,
        exploration_signal: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute flight action from DN activity.
        
        Maps specific DN types to motor commands:
        - DNp01: escape → emergency pitch
        - DNg02: thrust/altitude → forward + upward thrust
        - DNa01/DNa02 difference → yaw
        - DNb: baseline motor activity
        
        Args:
            brain_activity: Neural activity array
            exploration_signal: Optional exploration bias
        
        Returns:
            Action: [thrust_forward, thrust_up, roll, pitch, yaw]
        """
        # Read DN activity (deviation from resting if warmup complete)
        dn_activity = {}
        for dn_type, indices in self.output_mapping.items():
            if len(indices) > 0:
                activity = brain_activity[indices].mean()
                if self.warmup_complete and dn_type in self.resting_rates:
                    activity -= self.resting_rates[dn_type]
                dn_activity[dn_type] = activity
            else:
                dn_activity[dn_type] = 0.0
        
        # Map to motor commands
        thrust_forward = 0.7  # baseline
        thrust_up = 0.12      # maintain altitude
        pitch = 0.0
        roll = 0.0
        yaw = 0.0
        
        # DNg02 → thrust/altitude
        if "DNg02" in dn_activity:
            thrust_forward += 1.0 * dn_activity["DNg02"]
            thrust_up += 0.5 * dn_activity["DNg02"]
        
        # DNp01 → escape (sharp pitch)
        if "DNp01" in dn_activity:
            pitch += 2.0 * dn_activity["DNp01"]
        
        # DNa01/DNa02 difference → yaw
        dna_diff = dn_activity.get("DNa01", 0.0) - dn_activity.get("DNa02", 0.0)
        yaw += 1.5 * dna_diff
        
        # DNb → roll
        if "DNb" in dn_activity:
            roll += 0.5 * dn_activity["DNb"]
        
        # Assemble action
        action = np.array([thrust_forward, thrust_up, roll, pitch, yaw])
        
        # Add exploration if provided
        if exploration_signal is not None:
            action[0] += 0.8 * exploration_signal[0]
            action[4] += 1.0 * exploration_signal[1]
        
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
