"""Neural simulator for connectome dynamics."""

import numpy as np
from scipy import sparse
from typing import Optional, Dict, Union


def apply_ei_signs(
    adjacency: sparse.csr_matrix,
    transmitter_signs: Union[Dict[int, float], np.ndarray],
) -> sparse.csr_matrix:
    """
    Apply excitatory/inhibitory signs to a CSR adjacency (pre × post).

    Multiplies each presynaptic row by that neuron's sign using
    ``sparse.diags(signs) @ adj`` (no per-row LIL loop).
    """
    n = adjacency.shape[0]
    if isinstance(transmitter_signs, dict):
        signs = np.ones(n, dtype=np.float32)
        for idx, sign in transmitter_signs.items():
            signs[int(idx)] = sign
    else:
        signs = np.asarray(transmitter_signs, dtype=np.float32)
        if signs.shape[0] != n:
            raise ValueError(
                f"transmitter_signs length {signs.shape[0]} != n_neurons {n}"
            )
    return sparse.diags(signs, format="csr", dtype=adjacency.dtype) @ adjacency.tocsr()


# Bind before class methods whose parameter is also named apply_ei_signs.
_apply_ei_signs_impl = apply_ei_signs


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
        apply_ei_signs: bool = False,
        transmitter_signs: Optional[Dict[int, float]] = None,
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
            apply_ei_signs: Whether to apply E/I signs based on neurotransmitters
            transmitter_signs: Dict mapping neuron_idx → sign (±1)
        """
        # Apply E/I signs if requested
        if apply_ei_signs and transmitter_signs is not None:
            adjacency = _apply_ei_signs_impl(adjacency, transmitter_signs)
        
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
            max_weight = np.abs(self.adj.data).max()
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


def amplify_incoming_to_targets(
    adjacency: sparse.csr_matrix,
    targets: np.ndarray,
    gain: float = 5.0,
    positive_only: bool = True,
) -> sparse.csr_matrix:
    """Scale incoming synapses onto target neurons (per-neuron gain calibration)."""
    if adjacency.nnz == 0 or len(targets) == 0 or gain == 1.0:
        return adjacency.tocsr()
    adj = adjacency.tocsc(copy=True)
    for t in np.unique(np.asarray(targets, dtype=np.int32)):
        t = int(t)
        if t < 0 or t >= adj.shape[1]:
            continue
        start, end = int(adj.indptr[t]), int(adj.indptr[t + 1])
        if start == end:
            continue
        col = adj.data[start:end]
        if positive_only:
            col = np.where(col > 0, col * gain, col)
        else:
            col = col * gain
        adj.data[start:end] = col
    return adj.tocsr()


def count_directed_hops(
    adjacency: sparse.csr_matrix,
    sources: np.ndarray,
    targets: np.ndarray,
) -> Dict[str, int]:
    """Count existing 1-hop and 2-hop synapses from sources to targets (pre × post)."""
    adj = adjacency.tocsr()
    src = np.unique(np.asarray(sources, dtype=np.int32))
    tgt = np.unique(np.asarray(targets, dtype=np.int32))
    src = src[(src >= 0) & (src < adj.shape[0])]
    tgt = tgt[(tgt >= 0) & (tgt < adj.shape[1])]
    if src.size == 0 or tgt.size == 0:
        return {"n_one_hop": 0, "n_two_hop": 0, "n_sources": int(src.size), "n_targets": int(tgt.size)}
    one = adj[src][:, tgt]
    two = adj[src] @ adj[:, tgt]
    return {
        "n_one_hop": int(one.nnz),
        "n_two_hop": int(two.nnz),
        "n_sources": int(src.size),
        "n_targets": int(tgt.size),
    }


def calibrate_looming_to_dnp01(
    adjacency: sparse.csr_matrix,
    looming_indices: np.ndarray,
    dnp01_indices: np.ndarray,
    one_hop_gain: float = 8.0,
    incoming_gain: float = 4.0,
    assist_weight: float = 1.0,
) -> sparse.csr_matrix:
    """Strengthen LPLC2/LC4 → DNp01 paths; add a small assist synapse if none exist."""
    adj = adjacency.tolil(copy=True)
    src = np.unique(np.asarray(looming_indices, dtype=np.int32))
    tgt = np.unique(np.asarray(dnp01_indices, dtype=np.int32))
    hops = count_directed_hops(adjacency, src, tgt)
    if hops["n_one_hop"] > 0:
        csr = adjacency.tocsr()
        for pre in src:
            row = csr.getrow(int(pre))
            for post in tgt:
                w = float(csr[int(pre), int(post)])
                if w > 0:
                    adj[int(pre), int(post)] = w * one_hop_gain
    elif src.size > 0 and tgt.size > 0:
        # Engineered 1-hop assist when the filtered connectome has no LPLC2/LC4 → DNp01 edge.
        for pre in src:
            for post in tgt:
                adj[int(pre), int(post)] = max(float(adj[int(pre), int(post)]), assist_weight)
    scaled = amplify_incoming_to_targets(adj.tocsr(), tgt, gain=incoming_gain)
    return scaled


def diagnose_looming_to_dnp01(
    adjacency: sparse.csr_matrix,
    looming_indices: np.ndarray,
    dnp01_indices: np.ndarray,
    n_rest: int = 20,
    n_drive: int = 30,
    looming_current: float = 1.0,
    syn_weight_scale: float = 0.1,
    max_rate: float = 50.0,
    calibrate: bool = True,
) -> Dict[str, float]:
    """Inject current into LPLC2/LC4 and report DNp01 rate vs rest (synthetic or full brain)."""
    hops = count_directed_hops(adjacency, looming_indices, dnp01_indices)
    adj = adjacency
    if calibrate:
        adj = calibrate_looming_to_dnp01(adjacency, looming_indices, dnp01_indices)
    sim = RateSimulator(adj, dt=0.010, syn_weight_scale=syn_weight_scale, max_rate=max_rate)
    loom = np.asarray(looming_indices, dtype=np.int32)
    dnp = np.asarray(dnp01_indices, dtype=np.int32)

    def _dnp_rate(rates: np.ndarray) -> float:
        if dnp.size == 0:
            return 0.0
        return float(rates[dnp].mean())

    rest_hist = []
    for _ in range(n_rest):
        if loom.size:
            sim.set_input(np.zeros(loom.size, dtype=np.float32), neuron_indices=loom)
        rest_hist.append(_dnp_rate(sim.step()))
    rest = float(np.median(rest_hist[-max(3, n_rest // 3) :])) if rest_hist else 0.0

    driven_hist = []
    loom_hist = []
    for _ in range(n_drive):
        if loom.size:
            sim.set_input(
                np.full(loom.size, looming_current, dtype=np.float32), neuron_indices=loom
            )
        rates = sim.step()
        driven_hist.append(_dnp_rate(rates))
        if loom.size:
            loom_hist.append(float(rates[loom].mean()))
    driven = float(np.max(driven_hist)) if driven_hist else 0.0
    report = {
        **hops,
        "dnp01_rest": rest,
        "dnp01_driven": driven,
        "looming_driven": float(np.max(loom_hist)) if loom_hist else 0.0,
        "rose": float(driven > rest + 0.05),
    }
    return report


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
        max_rate: float = 50.0,  # Headroom so DN readouts are not pinned at the ceiling
        apply_ei_signs: bool = False,
        transmitter_signs: Optional[Dict[int, float]] = None,
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
            apply_ei_signs: Whether to apply E/I signs based on neurotransmitters
            transmitter_signs: Dict mapping neuron_idx → sign (±1)
        """
        # Apply E/I signs to adjacency if requested
        if apply_ei_signs and transmitter_signs is not None:
            adjacency = _apply_ei_signs_impl(adjacency, transmitter_signs)
        
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
