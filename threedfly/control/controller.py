"""Map brain activity to flight control commands."""

import numpy as np
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
