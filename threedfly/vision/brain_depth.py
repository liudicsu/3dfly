"""Brain-based depth estimation using connectome neural pathways.

This module provides depth estimation derived from the fruit fly connectome
neural activity using motion parallax and optic flow - the primary depth cues
used by real fruit flies.

**SCIENTIFIC APPROACH**:
Real Drosophila use motion parallax and optic flow for depth perception, not
stereo disparity like humans. As the fly moves, nearer objects move faster
across the retina than farther objects. This implementation:

1. Computes optic flow between consecutive frames
2. Combines flow with egomotion (flight velocity) to estimate depth
3. Routes flow-based features through the MaleCNS connectome simulator
4. Reads out depth from visual neuron population activity

**ENGINEERING VS BIOLOGY**:
- ✅ Biologically motivated: Uses motion parallax/flow (real fly cues)
- ✅ Brain-routed: Features pass through connectome (set_input → simulate → readout)
- ⚙️ Engineered: Flow extraction, depth calculation, and neural readout are hand-designed
- ⚙️ Not validated: Mapping is inspired by fly vision, not biologically verified

This honors real fly depth perception mechanisms and connectome structure while
acknowledging the I/O mappings are engineering approximations.
"""

import numpy as np
import cv2
from typing import Optional, Dict, Tuple
from scipy import sparse


class BrainDepthEstimator:
    """
    Brain-based depth estimation using motion parallax and optic flow.
    
    Implements real Drosophila-style depth perception:
    1. Compute optic flow between consecutive frames
    2. Use egomotion (flight velocity) to estimate depth from flow
    3. Route flow features through connectome neural pathways
    4. Read out spatial depth map from visual neuron population
    
    This replaces the old arbitrary 8×6 hand-designed readout with a
    biologically-motivated flow/parallax pipeline.
    """
    
    def __init__(
        self,
        n_neurons: int,
        adjacency: sparse.csr_matrix,
        use_rate_model: bool = True,
        flow_grid_size: Tuple[int, int] = (12, 16),  # Height x Width flow grid
    ):
        """
        Initialize brain depth estimator with motion parallax.
        
        Args:
            n_neurons: Total number of neurons in connectome subgraph
            adjacency: Sparse adjacency matrix (connectome wiring)
            use_rate_model: Whether using rate-based (vs LIF) simulator
            flow_grid_size: (height, width) of optical flow sampling grid
        """
        self.n_neurons = n_neurons
        self.adjacency = adjacency
        self.use_rate_model = use_rate_model
        self.flow_grid_h, self.flow_grid_w = flow_grid_size
        self.n_flow_cells = self.flow_grid_h * self.flow_grid_w
        
        # Visual neuron indices for flow/depth processing
        # In biology: T4/T5 (motion), lobula plate (optic flow), lobula (form+motion)
        # Here: use larger middle section for visual processing (40-75%)
        self.visual_start = int(0.40 * n_neurons)
        self.visual_end = int(0.75 * n_neurons)
        self.n_visual = self.visual_end - self.visual_start
        
        # Flow/parallax feature dimensions
        # Both eyes: (flow_x, flow_y) per cell per eye
        self.n_flow_features = self.n_flow_cells * 2 * 2  # 2 components × 2 eyes
        
        # Depth readout: map visual neuron activity → depth per grid cell
        # Initialize with spatial structure: neurons map to retinotopic positions
        np.random.seed(42)
        self.depth_readout_matrix = self._init_depth_readout()
        
        # Previous frame for flow computation
        self.prev_left_gray = None
        self.prev_right_gray = None
        
        # Egomotion state (flight velocity)
        self.velocity = np.zeros(3)  # [vx, vy, vz] in m/s
        
        # Normalization factors
        self.depth_min = 0.2  # Minimum depth (meters)
        self.depth_max = 5.0  # Maximum depth (meters)
        
        # State tracking
        self.last_depth_estimate = 1.0  # For API compatibility
        self.depth_history = []
        
        # Optical flow computation
        self.flow_pyr_scale = 0.5
        self.flow_levels = 3
        self.flow_winsize = 15
        self.flow_iterations = 3
    
    def _init_depth_readout(self) -> np.ndarray:
        """
        Initialize depth readout matrix mapping visual neurons to flow grid cells.
        
        Creates a (n_flow_cells, n_visual) matrix where each row corresponds to
        a spatial location in the flow grid and reads from a subset of visual
        neurons with retinotopic organization.
        
        Returns:
            Readout matrix (n_flow_cells, n_visual)
        """
        readout = np.zeros((self.n_flow_cells, self.n_visual))
        
        # Assign neurons to grid cells with Gaussian spatial tuning
        neurons_per_cell = max(1, self.n_visual // self.n_flow_cells)
        sigma = max(1.0, neurons_per_cell * 0.5)  # Overlap between adjacent cells
        
        for cell_idx in range(self.n_flow_cells):
            # Center neuron index for this cell
            center_neuron = cell_idx * neurons_per_cell + neurons_per_cell // 2
            center_neuron = min(center_neuron, self.n_visual - 1)
            
            # Gaussian weights centered on this cell's neurons
            neuron_indices = np.arange(self.n_visual)
            weights = np.exp(-((neuron_indices - center_neuron) ** 2) / (2 * sigma ** 2 + 1e-8))
            weights /= (weights.sum() + 1e-8)  # Normalize
            
            readout[cell_idx, :] = weights
        
        return readout
        
    def compute_optical_flow(
        self,
        left_img: np.ndarray,
        right_img: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Compute dense optical flow from previous frame to current frame.
        
        Args:
            left_img: Current left eye image (H, W, 3) uint8
            right_img: Current right eye image (H, W, 3) uint8
        
        Returns:
            (flow_left, flow_right): Flow fields (H, W, 2) or (None, None) if first frame
        """
        # Convert to grayscale
        left_gray = cv2.cvtColor(left_img, cv2.COLOR_RGB2GRAY)
        right_gray = cv2.cvtColor(right_img, cv2.COLOR_RGB2GRAY)
        
        # First frame: store and return None
        if self.prev_left_gray is None:
            self.prev_left_gray = left_gray
            self.prev_right_gray = right_gray
            return None, None
        
        # Compute optical flow using Farneback method (dense flow)
        flow_left = cv2.calcOpticalFlowFarneback(
            self.prev_left_gray,
            left_gray,
            None,
            pyr_scale=self.flow_pyr_scale,
            levels=self.flow_levels,
            winsize=self.flow_winsize,
            iterations=self.flow_iterations,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        
        flow_right = cv2.calcOpticalFlowFarneback(
            self.prev_right_gray,
            right_gray,
            None,
            pyr_scale=self.flow_pyr_scale,
            levels=self.flow_levels,
            winsize=self.flow_winsize,
            iterations=self.flow_iterations,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        
        # Update previous frames
        self.prev_left_gray = left_gray
        self.prev_right_gray = right_gray
        
        return flow_left, flow_right
    
    def flow_to_depth(
        self,
        flow: np.ndarray,
        velocity: np.ndarray,
        focal_length_pixels: float,
    ) -> np.ndarray:
        """
        Convert optical flow to depth using motion parallax.
        
        Motion parallax equation:
        depth = (velocity * focal_length) / flow_magnitude
        
        Where:
        - velocity: 3D camera velocity (m/s)
        - flow_magnitude: pixels/frame
        - focal_length: pixels
        
        Args:
            flow: Optical flow field (H, W, 2)
            velocity: Camera velocity [vx, vy, vz] in m/s
            focal_length_pixels: Camera focal length in pixels
        
        Returns:
            Depth map (H, W) in meters
        """
        # Flow magnitude in pixels per frame
        flow_mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        
        # Lateral velocity (perpendicular to viewing direction)
        # For a forward-facing camera, lateral velocity is mainly vx, vy
        v_lateral = np.sqrt(velocity[0]**2 + velocity[1]**2)
        
        # Motion parallax: depth = v * f / flow
        # Add small epsilon to avoid division by zero
        depth = np.zeros_like(flow_mag)
        valid_flow = flow_mag > 0.5  # Require minimum flow (0.5 pixels)
        
        if v_lateral > 0.01:  # Require minimum velocity (1 cm/s)
            depth[valid_flow] = (v_lateral * focal_length_pixels) / (flow_mag[valid_flow] + 1e-6)
        else:
            # If velocity too low, use default depth
            depth[valid_flow] = 1.5
        
        # Clip to reasonable range
        depth = np.clip(depth, self.depth_min, self.depth_max)
        
        return depth
        
    def sample_flow_grid(
        self,
        flow_left: np.ndarray,
        flow_right: np.ndarray,
    ) -> np.ndarray:
        """
        Sample optical flow on a coarse grid (similar to ommatidial sampling).
        
        Args:
            flow_left: Left eye flow (H, W, 2)
            flow_right: Right eye flow (H, W, 2)
        
        Returns:
            Flow features (n_flow_cells * 2,) concatenating left and right flows
        """
        h, w = flow_left.shape[:2]
        
        # Sample at grid centers
        flow_samples_left = []
        for i in range(self.flow_grid_h):
            for j in range(self.flow_grid_w):
                y = int((i + 0.5) * h / self.flow_grid_h)
                x = int((j + 0.5) * w / self.flow_grid_w)
                flow_samples_left.append(flow_left[y, x, :])
        
        flow_samples_right = []
        for i in range(self.flow_grid_h):
            for j in range(self.flow_grid_w):
                y = int((i + 0.5) * h / self.flow_grid_h)
                x = int((j + 0.5) * w / self.flow_grid_w)
                flow_samples_right.append(flow_right[y, x, :])
        
        # Flatten and concatenate
        flow_features_left = np.array(flow_samples_left).flatten()
        flow_features_right = np.array(flow_samples_right).flatten()
        
        # Normalize flow features to [0, 1] range for neural input
        # Typical flow magnitudes: 0-20 pixels
        flow_features = np.concatenate([flow_features_left, flow_features_right])
        flow_features = np.clip(flow_features / 20.0, -1.0, 1.0)  # Normalize to [-1, 1]
        
        return flow_features
        
    def estimate_depth_from_activity(
        self,
        brain_activity: np.ndarray,
        stereo_features: Optional[Dict] = None,
        velocity: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Estimate depth from brain activity using motion parallax/optic flow.
        
        **NEW APPROACH (motion parallax)**:
        This method uses brain-processed optical flow features to estimate depth,
        following real Drosophila depth perception mechanisms:
        
        1. Optical flow has been routed through the connectome
        2. Visual neuron activity encodes flow/parallax information
        3. Read out spatial depth map from population activity
        
        Args:
            brain_activity: Neural activity vector (spikes or rates) after
                           processing flow features through connectome
            stereo_features: Optional dict with visual features including:
                - left_img: left eye image (for flow computation)
                - right_img: right eye image (for flow computation)
                - depth_map: classical StereoBM depth (comparison only)
            velocity: Fly velocity [vx, vy, vz] in m/s (egomotion for parallax)
        
        Returns:
            Dictionary containing:
                - depth_map: Brain-estimated depth map (flow_grid_h, flow_grid_w)
                - confidence: Confidence map (flow_grid_h, flow_grid_w)
                - mean_depth: Scalar mean depth estimate
                - flow_magnitude: Mean flow magnitude (for diagnostics)
        """
        # Extract visual neuron activity (these have processed flow features)
        visual_activity = brain_activity[self.visual_start:self.visual_end]
        
        # Normalize activity to [0, 1]
        activity_max = visual_activity.max()
        if activity_max > 0:
            visual_norm = visual_activity / (activity_max + 1e-8)
        else:
            visual_norm = visual_activity
        
        # Read out depth per grid cell from visual neuron population
        # depth_readout_matrix: (n_flow_cells, n_visual)
        # visual_norm: (n_visual,)
        # Result: (n_flow_cells,) depth value per grid cell
        depth_per_cell = self.depth_readout_matrix @ visual_norm
        
        # Map depth values to reasonable range using sigmoid-like transformation
        # Higher neural activity → closer depth
        # Lower neural activity → farther depth
        # Use inverse relationship: depth = max_depth / (1 + k * activity)
        k = 5.0  # Scaling factor
        depth_values = self.depth_max / (1.0 + k * depth_per_cell + 1e-6)
        depth_values = np.clip(depth_values, self.depth_min, self.depth_max)
        
        # Reshape to spatial grid
        depth_map = depth_values.reshape(self.flow_grid_h, self.flow_grid_w)
        
        # Confidence based on activity level (higher activity = more confident)
        confidence_per_cell = depth_per_cell  # Already normalized
        confidence_map = confidence_per_cell.reshape(self.flow_grid_h, self.flow_grid_w)
        
        # Normalize confidence to [0, 1]
        conf_max = confidence_map.max()
        if conf_max > 0:
            confidence_map = confidence_map / (conf_max + 1e-8)
        
        # Mean depth
        mean_depth = float(np.mean(depth_values))
        
        # Update last_depth_estimate for API compatibility
        self.last_depth_estimate = mean_depth
        
        # Store history
        self.depth_history.append(mean_depth)
        if len(self.depth_history) > 100:
            self.depth_history.pop(0)
        
        # Compute flow magnitude for diagnostics (if available)
        flow_magnitude = 0.0
        if stereo_features is not None and 'flow_magnitude' in stereo_features:
            flow_magnitude = stereo_features['flow_magnitude']
        
        # Depth distribution for API compatibility (population activity → depth votes)
        depth_distribution = {
            "bins": depth_values,  # Depth value per cell
            "weights": depth_per_cell,  # Activity/confidence per cell
        }
        
        return {
            "depth_map": depth_map,
            "confidence": confidence_map,
            "mean_depth": mean_depth,
            "flow_magnitude": flow_magnitude,
            "depth_distribution": depth_distribution,  # API compatibility
        }
    
    def prepare_flow_features_for_brain(
        self,
        left_img: np.ndarray,
        right_img: np.ndarray,
        velocity: np.ndarray,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Prepare optical flow features for brain input.
        
        This is the key method that computes flow and packages it for
        routing through the connectome.
        
        Args:
            left_img: Current left eye image (H, W, 3) uint8
            right_img: Current right eye image (H, W, 3) uint8
            velocity: Fly velocity [vx, vy, vz] in m/s
        
        Returns:
            (flow_features, info_dict) where:
                - flow_features: Array to concatenate with other brain inputs
                - info_dict: Diagnostic information (flow magnitude, etc.)
        """
        # Update velocity state
        self.velocity = velocity
        
        # Compute optical flow
        flow_left, flow_right = self.compute_optical_flow(left_img, right_img)
        
        # First frame: return zero features
        if flow_left is None:
            zero_features = np.zeros(self.n_flow_features)
            return zero_features, {"flow_magnitude": 0.0, "has_flow": False}
        
        # Sample flow on grid
        flow_features = self.sample_flow_grid(flow_left, flow_right)
        
        # Compute mean flow magnitude for diagnostics
        flow_mag_left = np.sqrt(flow_left[..., 0]**2 + flow_left[..., 1]**2)
        mean_flow_magnitude = float(np.mean(flow_mag_left))
        
        info = {
            "flow_magnitude": mean_flow_magnitude,
            "has_flow": True,
            "velocity": velocity,
        }
        
        return flow_features, info
    
    def upsample_depth_map(
        self,
        depth_map: np.ndarray,
        target_height: int,
        target_width: int,
    ) -> np.ndarray:
        """
        Upsample coarse brain depth map to target resolution.
        
        Args:
            depth_map: Coarse depth map from brain (flow_grid_h, flow_grid_w)
            target_height: Target height
            target_width: Target width
        
        Returns:
            Upsampled depth map (target_height, target_width)
        """
        return cv2.resize(depth_map, (target_width, target_height), 
                         interpolation=cv2.INTER_LINEAR)
    
    def get_statistics(self) -> Dict[str, float]:
        """Get depth estimation statistics."""
        stats = {
            "last_depth": self.last_depth_estimate,  # API compatibility
            "n_visual_neurons": self.n_visual,
            "flow_grid_size": f"{self.flow_grid_h}x{self.flow_grid_w}",
            "velocity": tuple(self.velocity.tolist()),
        }
        
        if len(self.depth_history) > 0:
            stats["mean_depth_history"] = float(np.mean(self.depth_history))
            stats["std_depth_history"] = float(np.std(self.depth_history))
        
        return stats
    
    def reset(self):
        """Reset state."""
        self.last_depth_estimate = 1.0  # API compatibility
        self.depth_history = []
        self.prev_left_gray = None
        self.prev_right_gray = None
        self.velocity = np.zeros(3)
