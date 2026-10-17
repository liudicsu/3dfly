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


def attenuate_nonpath_incoming(
    adjacency: sparse.csr_matrix,
    targets: np.ndarray,
    keep_sources: np.ndarray,
    scale: float = 0.25,
    positive_only: bool = True,
) -> sparse.csr_matrix:
    """Scale non-pathway excitatory incoming onto targets (keeps LPLC2/LC4 and inhibition)."""
    if adjacency.nnz == 0 or len(targets) == 0 or scale == 1.0:
        return adjacency.tocsr()
    keep = set(int(i) for i in np.unique(np.asarray(keep_sources, dtype=np.int32)))
    adj = adjacency.tocsc(copy=True)
    for t in np.unique(np.asarray(targets, dtype=np.int32)):
        t = int(t)
        if t < 0 or t >= adj.shape[1]:
            continue
        start, end = int(adj.indptr[t]), int(adj.indptr[t + 1])
        if start == end:
            continue
        rows = adj.indices[start:end]
        data = adj.data[start:end]
        for i, pre in enumerate(rows):
            if int(pre) in keep:
                continue
            if positive_only and data[i] <= 0:
                continue
            data[i] = data[i] * scale
        adj.data[start:end] = data
    return adj.tocsr()


def _boost_one_hop_path(
    adj_lil,
    csr: sparse.csr_matrix,
    sources: np.ndarray,
    targets: np.ndarray,
    one_hop_gain: float,
    assist_weight: float,
) -> None:
    src = np.unique(np.asarray(sources, dtype=np.int32))
    tgt = np.unique(np.asarray(targets, dtype=np.int32))
    hops = count_directed_hops(csr, src, tgt)
    if hops["n_one_hop"] > 0:
        for pre in src:
            for post in tgt:
                w = float(csr[int(pre), int(post)])
                if w > 0:
                    adj_lil[int(pre), int(post)] = w * one_hop_gain
    elif src.size > 0 and tgt.size > 0:
        # Assist synapse only when the filtered connectome has no 1-hop edge.
        for pre in src:
            for post in tgt:
                adj_lil[int(pre), int(post)] = max(
                    float(adj_lil[int(pre), int(post)]), assist_weight
                )


DEFAULT_LOOMING_PATH_GAIN = 2.0
DEFAULT_LOOMING_REST_INCOMING_SCALE = 1.0


def calibrate_looming_to_dnp01(
    adjacency: sparse.csr_matrix,
    looming_indices: np.ndarray,
    dnp01_indices: np.ndarray,
    one_hop_gain: float = DEFAULT_LOOMING_PATH_GAIN,
    incoming_gain: float = 1.0,
    assist_weight: float = 1.0,
    rest_incoming_scale: float = DEFAULT_LOOMING_REST_INCOMING_SCALE,
    dnp03_indices: Optional[np.ndarray] = None,
) -> sparse.csr_matrix:
    """Engineered LPLC2/LC4 → DNp01/DNp03 path gain; other incoming is unchanged by default.

    ``one_hop_gain`` defaults to 2 (boost the looming path at most 2×).
    ``rest_incoming_scale`` defaults to 1 (do not attenuate other excitatory inputs).
    ``incoming_gain`` defaults to 1 (do not amplify all incoming — that pinned DNp01
    at the rate ceiling). Pass ``incoming_gain>1`` or ``rest_incoming_scale<1`` only
    for ablation.
    """
    csr = adjacency.tocsr()
    adj = adjacency.tolil(copy=True)
    src = np.unique(np.asarray(looming_indices, dtype=np.int32))
    tgt = np.unique(np.asarray(dnp01_indices, dtype=np.int32))
    _boost_one_hop_path(adj, csr, src, tgt, one_hop_gain, assist_weight)
    extra = []
    if dnp03_indices is not None and len(dnp03_indices) > 0:
        dnp03 = np.unique(np.asarray(dnp03_indices, dtype=np.int32))
        _boost_one_hop_path(adj, csr, src, dnp03, one_hop_gain, assist_weight)
        extra.append(dnp03)
    scaled = adj.tocsr()
    all_tgt = tgt if not extra else np.unique(np.concatenate([tgt] + extra))
    scaled = attenuate_nonpath_incoming(
        scaled, all_tgt, keep_sources=src, scale=rest_incoming_scale
    )
    if incoming_gain != 1.0 and tgt.size > 0:
        scaled = amplify_incoming_to_targets(scaled, tgt, gain=incoming_gain)
    return scaled


def _mean_rate(rates: np.ndarray, indices: np.ndarray) -> float:
    idx = np.asarray(indices, dtype=np.int32)
    if idx.size == 0:
        return 0.0
    return float(rates[idx].mean())


def _latency_steps(hist, rest: float, margin: float = 0.05) -> float:
    for i, v in enumerate(hist):
        if float(v) > rest + margin:
            return float(i)
    return float("nan")


def _selectivity(left: float, right: float) -> float:
    denom = abs(left) + abs(right)
    if denom < 1e-9:
        return 0.0
    return float((left - right) / denom)


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
    dnp03_indices: Optional[np.ndarray] = None,
    looming_left: Optional[np.ndarray] = None,
    looming_right: Optional[np.ndarray] = None,
    n_baseline: int = 40,
    baseline_current: float = 0.05,
    closed_loop_margin: float = 0.05,
    n_pulse: int = 8,
) -> Dict[str, float]:
    """Inject current into LPLC2/LC4 and report DNp01/DNp03 rest, drive, latency, L/R selectivity.

    Also runs a closed-loop-like protocol: tonic ``baseline_current`` for
    ``n_baseline`` steps, then a brief pulse of ``looming_current`` (LPLC2 is
    phasic). Reports whether DNp01 crosses rest+margin after that warm-up,
    not just the onset spike from silence.
    """
    hops = count_directed_hops(adjacency, looming_indices, dnp01_indices)
    dnp03 = np.asarray(
        dnp03_indices if dnp03_indices is not None else [], dtype=np.int32
    )
    adj = adjacency
    if calibrate:
        adj = calibrate_looming_to_dnp01(
            adjacency, looming_indices, dnp01_indices, dnp03_indices=dnp03
        )
    sim = RateSimulator(adj, dt=0.010, syn_weight_scale=syn_weight_scale, max_rate=max_rate)
    loom = np.asarray(looming_indices, dtype=np.int32)
    dnp = np.asarray(dnp01_indices, dtype=np.int32)

    rest_dnp, rest_dnp03 = [], []
    for _ in range(n_rest):
        if loom.size:
            sim.set_input(np.zeros(loom.size, dtype=np.float32), neuron_indices=loom)
        rates = sim.step()
        rest_dnp.append(_mean_rate(rates, dnp))
        rest_dnp03.append(_mean_rate(rates, dnp03))
    tail = max(3, n_rest // 3)
    rest = float(np.median(rest_dnp[-tail:])) if rest_dnp else 0.0
    rest03 = float(np.median(rest_dnp03[-tail:])) if rest_dnp03 else 0.0

    def _drive(indices, current) -> tuple:
        sim.reset()
        for _ in range(n_rest):
            if loom.size:
                sim.set_input(np.zeros(loom.size, dtype=np.float32), neuron_indices=loom)
            sim.step()
        dnp_hist, dnp03_hist, loom_hist = [], [], []
        idx = np.asarray(indices, dtype=np.int32)
        for _ in range(n_drive):
            if idx.size:
                sim.set_input(
                    np.full(idx.size, current, dtype=np.float32), neuron_indices=idx
                )
            rates = sim.step()
            dnp_hist.append(_mean_rate(rates, dnp))
            dnp03_hist.append(_mean_rate(rates, dnp03))
            if idx.size:
                loom_hist.append(float(rates[idx].mean()))
        return dnp_hist, dnp03_hist, loom_hist

    driven_hist, driven03_hist, loom_hist = _drive(loom, looming_current)
    driven = float(np.max(driven_hist)) if driven_hist else 0.0
    driven03 = float(np.max(driven03_hist)) if driven03_hist else 0.0

    left_idx = looming_left if looming_left is not None else loom
    right_idx = looming_right if looming_right is not None else loom
    left_dnp, left_dnp03, _ = _drive(left_idx, looming_current)
    right_dnp, right_dnp03, _ = _drive(right_idx, looming_current)
    left_peak = float(np.max(left_dnp)) if left_dnp else 0.0
    right_peak = float(np.max(right_dnp)) if right_dnp else 0.0
    left_peak03 = float(np.max(left_dnp03)) if left_dnp03 else 0.0
    right_peak03 = float(np.max(right_dnp03)) if right_dnp03 else 0.0

    report = {
        **hops,
        "dnp01_rest": rest,
        "dnp01_driven": driven,
        "dnp01_latency_steps": _latency_steps(driven_hist, rest),
        "dnp01_left_driven": left_peak,
        "dnp01_right_driven": right_peak,
        "dnp01_lr_selectivity": _selectivity(left_peak, right_peak),
        "dnp03_rest": rest03,
        "dnp03_driven": driven03,
        "dnp03_latency_steps": _latency_steps(driven03_hist, rest03),
        "dnp03_left_driven": left_peak03,
        "dnp03_right_driven": right_peak03,
        "dnp03_lr_selectivity": _selectivity(left_peak03, right_peak03),
        "looming_driven": float(np.max(loom_hist)) if loom_hist else 0.0,
        "rose": float(driven > rest + 0.05),
    }

    # Closed-loop-like: tonic cruise, then a brief pulse (phasic LPLC2). Score
    # the pulse after warm-up, not a long hold that the network can suppress.
    sim.reset()
    n_base = max(int(n_baseline), 1)
    baseline_hist = []
    for _ in range(n_base):
        if loom.size:
            sim.set_input(
                np.full(loom.size, float(baseline_current), dtype=np.float32),
                neuron_indices=loom,
            )
        rates = sim.step()
        baseline_hist.append(_mean_rate(rates, dnp))
    tail_b = max(3, n_base // 3)
    cl_rest = float(np.median(baseline_hist[-tail_b:])) if baseline_hist else 0.0

    n_ramp = max(int(n_drive), 1)
    n_on = max(1, min(int(n_pulse), n_ramp))
    pulse_hist = []
    for i in range(n_ramp):
        cur = float(looming_current) if i < n_on else float(baseline_current)
        if loom.size:
            sim.set_input(
                np.full(loom.size, cur, dtype=np.float32), neuron_indices=loom
            )
        rates = sim.step()
        if i < n_on:
            pulse_hist.append(_mean_rate(rates, dnp))
    cl_driven = float(np.max(pulse_hist)) if pulse_hist else 0.0
    tail_p = max(1, len(pulse_hist) // 2)
    cl_late = float(np.median(pulse_hist[-tail_p:])) if pulse_hist else 0.0
    report["dnp01_closed_loop_rest"] = cl_rest
    report["dnp01_closed_loop_driven"] = cl_driven
    report["dnp01_closed_loop_late_driven"] = cl_late
    report["dnp01_closed_loop_crossed"] = float(
        cl_driven > cl_rest + float(closed_loop_margin)
    )
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
        # Per-neuron subtractive threshold / bias (warmup calibration for giant fiber).
        self.bias = np.zeros(self.n_neurons, dtype=np.float64)
        self.threshold = np.zeros(self.n_neurons, dtype=np.float64)

    def apply_rest_thresholds(
        self,
        indices: np.ndarray,
        observed_rates,
        target_rest: float = 2.0,
    ) -> None:
        """Raise ReLU threshold so a high-rest cell sits near ``target_rest`` Hz."""
        idx = np.asarray(indices, dtype=np.int32)
        if idx.size == 0:
            return
        if np.isscalar(observed_rates):
            obs = np.full(idx.size, float(observed_rates))
        else:
            obs = np.asarray(observed_rates, dtype=float).ravel()
            if obs.size == 1:
                obs = np.full(idx.size, float(obs[0]))
            elif obs.size == self.n_neurons:
                obs = obs[idx]
            elif obs.size != idx.size:
                obs = np.full(idx.size, float(np.mean(obs)))
        self.threshold[idx] = np.maximum(0.0, obs - float(target_rest))
    
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
        
        # Total input (bias / threshold from warmup keep giant-fiber rest low)
        total_input = syn_input + self.external_input + self.bias
        
        # Activation
        activated = self._activation_fn(total_input - self.threshold)
        
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
