"""Interactive web-based visualization using Dash and Plotly."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, ctx
import dash_bootstrap_components as dbc
from flask import Flask
import threading
import time
import queue
from typing import Optional, Dict
import io
import base64
from PIL import Image


class WebVisualizer:
    """
    Interactive web-based visualization for fly exploration.
    
    Features:
    - Interactive 3D point cloud (orbit/zoom/pan)
    - Live stereo camera views
    - Flight trajectory and pose
    - Brain activity visualization
    - Flight commands display
    - Playback controls (play/pause, speed, reset)
    """
    
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8050,
        simulation_runner=None,
    ):
        """
        Initialize web visualizer.
        
        Args:
            host: Host address
            port: Port number
            simulation_runner: SimulationRunner instance
        """
        self.host = host
        self.port = port
        self.simulation_runner = simulation_runner
        
        # State
        self.state_queue = queue.Queue(maxsize=100)
        self.current_state = {
            "step": 0,
            "left_img": None,
            "right_img": None,
            "fly_position": np.zeros(3),
            "fly_trajectory": [],
            "brain_activity": None,
            "control_action": None,
            "point_cloud": {"points": np.zeros((0, 3)), "colors": np.zeros((0, 3))},
            "brain_stats": {},
            "map_stats": {},
            "control_stats": {},
        }
        self.is_running = False
        self.thread = None
        
        # Create Dash app
        server = Flask(__name__)
        self.app = dash.Dash(
            __name__,
            server=server,
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            suppress_callback_exceptions=True,
        )
        
        self._build_layout()
        self._setup_callbacks()
    
    def _build_layout(self):
        """Build Dash layout."""
        self.app.layout = dbc.Container(
            [
                dbc.Row([
                    dbc.Col([
                        html.H1("🪰 3dfly: Interactive Exploration", className="text-center mb-3"),
                        html.P(
                            "Connectome-driven flight simulation with real-time 3D mapping",
                            className="text-center text-muted mb-4"
                        ),
                    ])
                ]),
                
                # Controls
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H5("Controls", className="mb-3"),
                                dbc.Row([
                                    dbc.Col([
                                        dbc.ButtonGroup([
                                            dbc.Button("▶ Play", id="play-btn", color="success", size="sm"),
                                            dbc.Button("⏸ Pause", id="pause-btn", color="warning", size="sm"),
                                            dbc.Button("⟳ Reset", id="reset-btn", color="secondary", size="sm"),
                                        ]),
                                    ], width=6),
                                    dbc.Col([
                                        html.Div([
                                            html.Label("Speed:", className="me-2"),
                                            dcc.Slider(
                                                id="speed-slider",
                                                min=1, max=10, step=1, value=5,
                                                marks={i: f"{i}x" for i in [1, 5, 10]},
                                                tooltip={"placement": "bottom", "always_visible": False},
                                            ),
                                        ]),
                                    ], width=6),
                                ]),
                                html.Hr(className="my-2"),
                                dbc.Row([
                                    dbc.Col([
                                        html.Div([
                                            html.Label("Random Seed:", className="me-2"),
                                            dcc.Input(
                                                id="seed-input",
                                                type="number",
                                                value=42,
                                                style={"width": "100px"},
                                                className="form-control form-control-sm",
                                            ),
                                        ]),
                                    ], width=6),
                                    dbc.Col([
                                        html.Div(id="status-text", className="text-muted small"),
                                    ], width=6),
                                ]),
                            ])
                        ], className="mb-3"),
                    ], width=12),
                ]),
                
                # Main visualization area
                dbc.Row([
                    # Left column: Stereo views and trajectory
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Left Eye View"),
                            dbc.CardBody([
                                html.Img(id="left-eye-img", style={"width": "100%", "height": "auto"}),
                            ]),
                        ], className="mb-3"),
                        dbc.Card([
                            dbc.CardHeader("Right Eye View"),
                            dbc.CardBody([
                                html.Img(id="right-eye-img", style={"width": "100%", "height": "auto"}),
                            ]),
                        ], className="mb-3"),
                        dbc.Card([
                            dbc.CardHeader("Flight Trajectory"),
                            dbc.CardBody([
                                dcc.Graph(id="trajectory-plot", config={"displayModeBar": False}),
                            ]),
                        ], className="mb-3"),
                    ], width=4),
                    
                    # Right column: 3D point cloud
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("3D Point Cloud Map (Interactive)"),
                            dbc.CardBody([
                                dcc.Graph(id="pointcloud-plot", style={"height": "70vh"}),
                            ]),
                        ], className="mb-3"),
                    ], width=8),
                ]),
                
                # Bottom row: Brain activity and control commands
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Brain Activity"),
                            dbc.CardBody([
                                dcc.Graph(id="brain-plot", config={"displayModeBar": False}),
                            ]),
                        ]),
                    ], width=6),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Flight Commands"),
                            dbc.CardBody([
                                dcc.Graph(id="control-plot", config={"displayModeBar": False}),
                            ]),
                        ]),
                    ], width=3),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("System Status"),
                            dbc.CardBody([
                                html.Div(id="stats-display", style={"font-family": "monospace", "font-size": "0.85em"}),
                            ]),
                        ]),
                    ], width=3),
                ]),
                
                # Auto-update interval
                dcc.Interval(id="interval-component", interval=200, n_intervals=0),
                
                # Hidden state storage
                dcc.Store(id="playback-state", data={"playing": False, "speed": 5}),
            ],
            fluid=True,
            className="p-4",
        )
    
    def _setup_callbacks(self):
        """Setup Dash callbacks."""
        
        @self.app.callback(
            Output("playback-state", "data"),
            [
                Input("play-btn", "n_clicks"),
                Input("pause-btn", "n_clicks"),
                Input("reset-btn", "n_clicks"),
                Input("speed-slider", "value"),
            ],
            State("playback-state", "data"),
            prevent_initial_call=True,
        )
        def control_playback(play_clicks, pause_clicks, reset_clicks, speed, state):
            """Handle playback controls."""
            if ctx.triggered_id == "play-btn":
                state["playing"] = True
                if self.simulation_runner:
                    self.simulation_runner.resume()
            elif ctx.triggered_id == "pause-btn":
                state["playing"] = False
                if self.simulation_runner:
                    self.simulation_runner.pause()
            elif ctx.triggered_id == "reset-btn":
                state["playing"] = False
                if self.simulation_runner:
                    self.simulation_runner.reset()
            elif ctx.triggered_id == "speed-slider":
                state["speed"] = speed
                if self.simulation_runner:
                    self.simulation_runner.set_speed(speed)
            
            return state
        
        @self.app.callback(
            [
                Output("left-eye-img", "src"),
                Output("right-eye-img", "src"),
                Output("trajectory-plot", "figure"),
                Output("pointcloud-plot", "figure"),
                Output("brain-plot", "figure"),
                Output("control-plot", "figure"),
                Output("stats-display", "children"),
                Output("status-text", "children"),
            ],
            Input("interval-component", "n_intervals"),
            State("playback-state", "data"),
        )
        def update_visualizations(n_intervals, playback_state):
            """Update all visualizations."""
            # Get latest state from queue
            while not self.state_queue.empty():
                try:
                    self.current_state = self.state_queue.get_nowait()
                except queue.Empty:
                    break
            
            state = self.current_state
            
            # Convert images to base64
            left_img_src = self._array_to_img_src(state.get("left_img"))
            right_img_src = self._array_to_img_src(state.get("right_img"))
            
            # Trajectory plot
            traj_fig = self._create_trajectory_plot(state.get("fly_trajectory", []))
            
            # Point cloud plot
            pc_fig = self._create_pointcloud_plot(state.get("point_cloud", {}))
            
            # Brain activity plot
            brain_fig = self._create_brain_plot(state.get("brain_activity"))
            
            # Control commands plot
            control_fig = self._create_control_plot(state.get("control_action"))
            
            # Stats display
            stats_html = self._create_stats_html(state)
            
            # Status text
            status = "Playing" if playback_state.get("playing", False) else "Paused"
            status_text = f"Status: {status} | Step: {state.get('step', 0)}"
            
            return (
                left_img_src,
                right_img_src,
                traj_fig,
                pc_fig,
                brain_fig,
                control_fig,
                stats_html,
                status_text,
            )
    
    def _array_to_img_src(self, img_array):
        """Convert numpy array to base64 image source."""
        if img_array is None:
            return ""
        
        try:
            # Ensure uint8
            if img_array.dtype != np.uint8:
                img_array = (img_array * 255).astype(np.uint8)
            
            # Convert to PIL Image
            img = Image.fromarray(img_array)
            
            # Convert to base64
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            buffer.seek(0)
            img_base64 = base64.b64encode(buffer.read()).decode()
            
            return f"data:image/png;base64,{img_base64}"
        except Exception:
            return ""
    
    def _create_trajectory_plot(self, trajectory):
        """Create 2D trajectory plot."""
        fig = go.Figure()
        
        if len(trajectory) > 1:
            traj = np.array(trajectory)
            fig.add_trace(go.Scatter(
                x=traj[:, 0],
                y=traj[:, 1],
                mode="lines+markers",
                line=dict(color="blue", width=2),
                marker=dict(size=3),
                name="Trajectory",
            ))
            fig.add_trace(go.Scatter(
                x=[traj[-1, 0]],
                y=[traj[-1, 1]],
                mode="markers",
                marker=dict(size=12, color="red", symbol="circle"),
                name="Current",
            ))
        
        fig.update_layout(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            xaxis=dict(range=[-5, 5], scaleanchor="y", scaleratio=1),
            yaxis=dict(range=[-5, 5]),
            height=300,
            margin=dict(l=40, r=20, t=20, b=40),
            showlegend=False,
            hovermode="closest",
        )
        
        return fig
    
    def _create_pointcloud_plot(self, point_cloud_data):
        """Create interactive 3D point cloud plot."""
        fig = go.Figure()
        
        points = point_cloud_data.get("points", np.zeros((0, 3)))
        colors = point_cloud_data.get("colors", np.zeros((0, 3)))
        
        if len(points) > 0:
            # Sample points for performance (max 10k points)
            n_points = len(points)
            if n_points > 10000:
                indices = np.random.choice(n_points, 10000, replace=False)
                points = points[indices]
                colors = colors[indices]
            
            # Convert colors to RGB strings
            rgb_colors = [f"rgb({int(r*255)},{int(g*255)},{int(b*255)})" 
                         for r, g, b in colors]
            
            fig.add_trace(go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers",
                marker=dict(
                    size=2,
                    color=rgb_colors,
                    opacity=0.8,
                ),
                name="Point Cloud",
                hovertemplate="X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>",
            ))
        
        fig.update_layout(
            scene=dict(
                xaxis=dict(title="X (m)", range=[-5, 5]),
                yaxis=dict(title="Y (m)", range=[-5, 5]),
                zaxis=dict(title="Z (m)", range=[0, 5]),
                aspectmode="manual",
                aspectratio=dict(x=1, y=1, z=0.5),
            ),
            margin=dict(l=0, r=0, t=30, b=0),
            hovermode="closest",
            showlegend=False,
        )
        
        return fig
    
    def _create_brain_plot(self, brain_activity):
        """Create brain activity bar plot."""
        fig = go.Figure()
        
        if brain_activity is not None and len(brain_activity) > 0:
            # Sample for visualization (max 200 neurons)
            n_sample = min(200, len(brain_activity))
            indices = np.linspace(0, len(brain_activity) - 1, n_sample, dtype=int)
            
            fig.add_trace(go.Bar(
                x=indices,
                y=brain_activity[indices],
                marker=dict(color="steelblue"),
                name="Activity",
            ))
        
        fig.update_layout(
            xaxis_title="Neuron Index (sampled)",
            yaxis_title="Activity",
            height=250,
            margin=dict(l=40, r=20, t=20, b=40),
            showlegend=False,
        )
        
        return fig
    
    def _create_control_plot(self, control_action):
        """Create control commands bar plot."""
        fig = go.Figure()
        
        if control_action is not None and len(control_action) > 0:
            labels = ["Fwd", "Up", "Roll", "Pitch", "Yaw"][:len(control_action)]
            colors = ["blue", "green", "red", "orange", "purple"][:len(control_action)]
            
            fig.add_trace(go.Bar(
                x=labels,
                y=control_action,
                marker=dict(color=colors),
                name="Commands",
            ))
        
        fig.update_layout(
            yaxis=dict(title="Command Value", range=[-1, 1]),
            height=250,
            margin=dict(l=40, r=20, t=20, b=40),
            showlegend=False,
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        
        return fig
    
    def _create_stats_html(self, state):
        """Create HTML for stats display."""
        pos = state.get("fly_position", np.zeros(3))
        brain_stats = state.get("brain_stats", {})
        map_stats = state.get("map_stats", {})
        
        html_parts = [
            f"Position: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) m<br>",
            f"Trajectory points: {len(state.get('fly_trajectory', []))}<br><br>",
        ]
        
        if brain_stats:
            html_parts.append("<b>Brain:</b><br>")
            for key, val in brain_stats.items():
                html_parts.append(f"  {key}: {val:.3f}<br>")
            html_parts.append("<br>")
        
        if map_stats:
            html_parts.append("<b>Mapping:</b><br>")
            html_parts.append(f"  Points: {map_stats.get('total_points', 0)}<br>")
            html_parts.append(f"  Voxels: {map_stats.get('occupied_voxels', 0)}<br>")
            html_parts.append(f"  Explored: {100*map_stats.get('exploration_ratio', 0):.1f}%<br>")
        
        return html.Div([html.Span(dangerously_allow_html=True, children="".join(html_parts))])
    
    def update(
        self,
        step: int = 0,
        left_img: Optional[np.ndarray] = None,
        right_img: Optional[np.ndarray] = None,
        fly_position: Optional[np.ndarray] = None,
        brain_activity: Optional[np.ndarray] = None,
        control_action: Optional[np.ndarray] = None,
        point_cloud_mapper = None,
        brain_stats: Optional[Dict] = None,
        control_stats: Optional[Dict] = None,
    ):
        """
        Update visualization state (called from simulation loop).
        
        Args:
            step: Current simulation step
            left_img: Left eye image
            right_img: Right eye image
            fly_position: Current fly position
            brain_activity: Brain activity vector
            control_action: Control commands
            point_cloud_mapper: PointCloudMapper instance
            brain_stats: Brain statistics
            control_stats: Control statistics
        """
        # Build state update
        state_update = {
            "step": step,
            "left_img": left_img,
            "right_img": right_img,
            "fly_position": fly_position if fly_position is not None else np.zeros(3),
            "brain_activity": brain_activity,
            "control_action": control_action,
            "brain_stats": brain_stats or {},
            "control_stats": control_stats or {},
        }
        
        # Update trajectory
        if fly_position is not None:
            traj = self.current_state.get("fly_trajectory", []).copy()
            traj.append(fly_position.copy())
            state_update["fly_trajectory"] = traj[-500:]  # Keep last 500 points
        else:
            state_update["fly_trajectory"] = self.current_state.get("fly_trajectory", [])
        
        # Extract point cloud
        if point_cloud_mapper is not None:
            pc = point_cloud_mapper.downsample()
            state_update["point_cloud"] = {
                "points": np.asarray(pc.points),
                "colors": np.asarray(pc.colors),
            }
            state_update["map_stats"] = point_cloud_mapper.get_statistics()
        else:
            state_update["point_cloud"] = self.current_state.get("point_cloud", {})
            state_update["map_stats"] = {}
        
        # Add to queue (drop old if full)
        try:
            self.state_queue.put_nowait(state_update)
        except queue.Full:
            try:
                self.state_queue.get_nowait()
                self.state_queue.put_nowait(state_update)
            except (queue.Empty, queue.Full):
                pass
    
    def run(self, debug: bool = False):
        """
        Run the web server (blocking).
        
        Args:
            debug: Enable debug mode
        """
        print("=" * 70)
        print(f"🪰 3dfly Web UI starting at http://{self.host}:{self.port}")
        print("=" * 70)
        print(f"Open your browser to http://{self.host}:{self.port}")
        print("Press Ctrl+C to stop")
        print("=" * 70)
        
        self.app.run_server(host=self.host, port=self.port, debug=debug)
    
    def run_async(self, debug: bool = False):
        """Run the web server in a background thread."""
        self.thread = threading.Thread(
            target=self.run,
            kwargs={"debug": debug},
            daemon=True,
        )
        self.thread.start()
        
        # Wait for server to start
        time.sleep(2)
    
    def close(self):
        """Close the web server."""
        # Dash doesn't have a clean shutdown method, so we just let the daemon thread die
        pass
