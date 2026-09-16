"""Neural simulator for connectome dynamics."""

import numpy as np
from scipy import sparse
from typing import Optional, Dict


class BrainSimulator:
    """
    Leaky Integrate-and-Fire (LIF) neural simulator for connectome.
    
    Implements sparse discrete-time spiking dynamics on connectome adjacency.
    """
    
    def __init__(
        self,
        adjacency: sparse.csr_matrix,
        dt: float = 0.001,  # 1ms timestep
        tau: float = 0.010,  # 10ms membrane time constant
        v_thresh: float = 1.0,  # Spike threshold
        v_reset: float = 0.0,  # Reset voltage
        v_rest: float = 0.0,  # Resting potential
        refrac_period: float = 0.002,  # 2ms refractory period
        syn_weight_scale: float = 0.1,  # Scale synaptic weights
    ):
        """
        Initialize simulator.
        
        Args:
            adjacency: Sparse adjacency matrix (pre x post)
            dt: Simulation timestep (seconds)
            tau: Membrane time constant (seconds)
            v_thresh: Spike threshold
            v_reset: Reset voltage after spike
            v_rest: Resting potential
            refrac_period: Refractory period (seconds)
            syn_weight_scale: Scaling factor for synaptic weights
        """
        self.adj = adjacency
        self.n_neurons = adjacency.shape[0]
        self.dt = dt
        self.tau = tau
        self.v_thresh = v_thresh
        self.v_reset = v_reset
        self.v_rest = v_rest
        self.refrac_steps = int(refrac_period / dt)
        self.syn_weight_scale = syn_weight_scale
        
        # Decay factor for membrane potential
        self.decay = np.exp(-dt / tau)
        
        # State variables
        self.v = np.ones(self.n_neurons) * v_rest  # Membrane potentials
        self.spikes = np.zeros(self.n_neurons, dtype=bool)  # Current spikes
        self.refrac_count = np.zeros(self.n_neurons, dtype=int)  # Refractory counter
        
        # Input and activity tracking
        self.external_input = np.zeros(self.n_neurons)
        self.spike_history = []  # For analysis
        
        # Normalize adjacency by max weight for stability
        if self.adj.nnz > 0:
            max_weight = self.adj.data.max()
            if max_weight > 0:
                self.adj = self.adj / max_weight * syn_weight_scale
    
    def set_input(self, input_vector: np.ndarray, neuron_indices: Optional[np.ndarray] = None):
        """
        Set external input to specific neurons (e.g., from photoreceptors).
        
        Args:
            input_vector: Input values
            neuron_indices: Indices of neurons to receive input (default: first N)
        """
        self.external_input.fill(0.0)
        
        if neuron_indices is None:
            n = min(len(input_vector), self.n_neurons)
            self.external_input[:n] = input_vector[:n]
        else:
            self.external_input[neuron_indices] = input_vector
    
    def step(self) -> np.ndarray:
        """
        Simulate one timestep.
        
        Returns:
            Binary spike array (n_neurons,)
        """
        # Neurons in refractory period don't integrate
        active_mask = self.refrac_count == 0
        
        # Leak towards resting potential
        self.v[active_mask] = (
            self.v[active_mask] * self.decay +
            self.v_rest * (1 - self.decay)
        )
        
        # Synaptic input from previous spikes
        if self.spikes.any():
            # Sparse matrix-vector multiply: (pre x post) @ spikes = inputs to each post neuron
            syn_input = self.adj.T @ self.spikes.astype(float)
            self.v[active_mask] += syn_input[active_mask]
        
        # External input
        self.v[active_mask] += self.external_input[active_mask]
        
        # Detect spikes
        self.spikes = (self.v >= self.v_thresh) & active_mask
        
        # Reset spiked neurons
        self.v[self.spikes] = self.v_reset
        self.refrac_count[self.spikes] = self.refrac_steps
        
        # Decrement refractory counters
        self.refrac_count[self.refrac_count > 0] -= 1
        
        # Track history (optional, for analysis)
        self.spike_history.append(self.spikes.copy())
        
        return self.spikes
    
    def get_firing_rates(self, window: int = 100) -> np.ndarray:
        """
        Get recent firing rates (spikes/sec) from history.
        
        Args:
            window: Number of recent timesteps to average
        
        Returns:
            Firing rates (n_neurons,)
        """
        if len(self.spike_history) < window:
            window = len(self.spike_history)
        
        if window == 0:
            return np.zeros(self.n_neurons)
        
        recent_spikes = np.array(self.spike_history[-window:])
        rates = recent_spikes.sum(axis=0) / (window * self.dt)
        
        return rates
    
    def get_population_activity(self) -> Dict[str, float]:
        """Get summary statistics of current activity."""
        return {
            "mean_voltage": float(self.v.mean()),
            "max_voltage": float(self.v.max()),
            "spike_count": int(self.spikes.sum()),
            "active_fraction": float(self.spikes.sum() / self.n_neurons),
        }
    
    def reset(self):
        """Reset simulator state."""
        self.v.fill(self.v_rest)
        self.spikes.fill(False)
        self.refrac_count.fill(0)
        self.external_input.fill(0.0)
        self.spike_history.clear()


class RateSimulator:
    """
    Simplified firing-rate model (faster alternative to LIF).
    
    Uses continuous firing rates instead of discrete spikes.
    """
    
    def __init__(
        self,
        adjacency: sparse.csr_matrix,
        dt: float = 0.010,  # 10ms timestep (coarser than LIF)
        tau: float = 0.050,  # 50ms time constant
        activation: str = "relu",  # relu, sigmoid, tanh
        syn_weight_scale: float = 0.1,  # Reduced from 0.5 to prevent saturation
        max_rate: float = 10.0,  # Maximum firing rate (Hz)
    ):
        """
        Initialize rate-based simulator.
        
        Args:
            adjacency: Sparse adjacency matrix
            dt: Simulation timestep
            tau: Time constant
            activation: Activation function
            syn_weight_scale: Synaptic weight scaling
            max_rate: Maximum firing rate (prevents saturation)
        """
        self.adj = adjacency * syn_weight_scale
        self.n_neurons = adjacency.shape[0]
        self.dt = dt
        self.tau = tau
        self.activation = activation
        self.max_rate = max_rate
        
        self.decay = np.exp(-dt / tau)
        
        # State
        self.rates = np.zeros(self.n_neurons)  # Firing rates
        self.external_input = np.zeros(self.n_neurons)
    
    def _activation_fn(self, x: np.ndarray) -> np.ndarray:
        """Apply activation function."""
        if self.activation == "relu":
            return np.maximum(0, x)
        elif self.activation == "sigmoid":
            return 1 / (1 + np.exp(-x))
        elif self.activation == "tanh":
            return np.tanh(x)
        else:
            return x
    
    def set_input(self, input_vector: np.ndarray, neuron_indices: Optional[np.ndarray] = None):
        """Set external input."""
        self.external_input.fill(0.0)
        if neuron_indices is None:
            n = min(len(input_vector), self.n_neurons)
            self.external_input[:n] = input_vector[:n]
        else:
            self.external_input[neuron_indices] = input_vector
    
    def step(self) -> np.ndarray:
        """
        Simulate one timestep.
        
        Returns:
            Firing rates (n_neurons,)
        """
        # Synaptic input
        syn_input = self.adj.T @ self.rates
        
        # Total input
        total_input = syn_input + self.external_input
        
        # Activation
        activated = self._activation_fn(total_input)
        
        # Temporal dynamics (exponential moving average)
        self.rates = self.rates * self.decay + activated * (1 - self.decay)
        
        # Clip to max rate to prevent saturation
        self.rates = np.clip(self.rates, 0, self.max_rate)
        
        return self.rates
    
    def get_population_activity(self) -> Dict[str, float]:
        """Get summary statistics."""
        return {
            "mean_rate": float(self.rates.mean()),
            "max_rate": float(self.rates.max()),
            "active_fraction": float((self.rates > 0.01).sum() / self.n_neurons),
        }
    
    def reset(self):
        """Reset state."""
        self.rates.fill(0.0)
        self.external_input.fill(0.0)
