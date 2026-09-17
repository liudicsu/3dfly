# 3dfly

**Fruit fly connectome-driven 3D exploration and mapping**

A research simulator where a male fruit fly's brain, structured by the real [MaleCNS v1.0 connectome](https://male-cns.janelia.org/), controls flight in a MuJoCo physics environment while building a 3D point cloud map of its surroundings through stereo vision.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

---

## Overview

**3dfly** demonstrates exploratory behavior driven by a biologically-inspired neural architecture. The system:

- **Brain**: Uses the MaleCNS v1.0 connectome (male *Drosophila* CNS, ~166,700 neurons) as the neural wiring diagram
- **Vision**: Dual cameras provide stereo views, processed through the connectome to estimate depth
- **Motor**: Descending neuron activity maps to flight thrust and torques in a MuJoCo simulation
- **Depth Perception**: **Brain-based depth estimation** from connectome neural pathways (primary) with classical StereoBM as dense comparison baseline
    - **Mapping**: Accumulates brain-estimated depth into a global 3D point cloud, with StereoBM providing informative comparison data
- **Exploration**: Biases flight toward under-mapped regions using occupancy-based curiosity
- **Soft Collision Recovery**: Automatically recovers from wall/obstacle hits to continue exploration

**Important scientific note**: This is a *connectome-structured simulator* with engineered I/O mapping, not a claim of neuron-for-neuron biological accuracy. The brain dynamics use simplified LIF/rate models applied to the synaptic graph. The mapping from visual input to photoreceptor activity, the depth readout from visual neurons, and the mapping from descending neurons to motor commands are hand-designed, not learned or biologically validated. The brain-based depth estimation routes visual features through the connectome and extracts depth from neural activity patterns in visual processing regions—an engineering design that honors the connectome structure.

---

## Features

- **Realistic Fly Model**: Uses anatomically-detailed 3D meshes from the [flybody project](https://github.com/TuragaLab/flybody) (Google DeepMind & HHMI Janelia)
- **Biologically-Inspired Brain**: MaleCNS v1.0 connectome structure with ~166,700 neurons
- **Brain-Based Depth Perception**: Depth estimates derived from connectome neural activity (primary reconstruction)
- **Stereo Vision**: Dual cameras (160×120) with classical StereoBM depth (dense comparison baseline) and ommatidial sampling
- **Interactive 3D Visualization**: Real-time god's-eye view with orbit/pan/zoom controls
- **Live Point Cloud Mapping**: Brain-estimated 3D map accumulates and renders in real-time on the same interactive web page
- **Curiosity-Driven Exploration**: Biases flight toward under-mapped regions
- **Soft Collision Recovery**: Automatically backs off and reorients after hitting obstacles, enabling continuous exploration instead of stopping on first collision

## Architecture

```mermaid
graph TB
    subgraph Environment
        A[MuJoCo Simulation<br/>Room + Obstacles + Realistic Fly]
        B[Left Eye Camera]
        C[Right Eye Camera]
    end
    
    subgraph Vision
        D[Stereo Vision<br/>Classical StereoBM]
        E[Ommatidial Sampling<br/>~100 samples/eye]
        F[Brain Depth Estimator<br/>Neural Pathways → Depth]
        G[Point Cloud Mapper<br/>Voxel Occupancy]
    end
    
    subgraph Brain
        H[Visual Input<br/>Photoreceptor Layer]
        I[Connectome Subgraph<br/>Visual + Flight Neurons]
        J[Neural Simulator<br/>LIF or Rate Model]
        K[Descending Neurons<br/>Motor Output]
    end
    
    subgraph Control
        L[Flight Controller<br/>DN → Thrust/Torque]
        M[Exploration Policy<br/>Curiosity Drive]
    end
    
    subgraph Visualization
        N[Interactive UI<br/>Eye Views + Trajectory<br/>Brain Activity + Map]
    end
    
    B --> D
    C --> D
    D --> E
    E --> H
    H --> I
    I --> J
    I --> F
    F --> G
    D -.-> G
    J --> L
    G --> M
    M --> L
    L --> A
    A --> B
    A --> C
    
    D -.-> N
    F -.-> N
    I -.-> N
    L -.-> N
    G -.-> N
    A -.-> N
```

---

## Installation

### Requirements

- Python 3.9+
- Linux or macOS (Windows may work but not tested)

### Install

```bash
# Clone repository
git clone https://github.com/liudicsu/3dfly.git
cd 3dfly

# Install package with development dependencies
pip install -e ".[dev]"
```

---

## Quick Start

### Option 1: Interactive Unified Web Interface (Recommended)

**🎮 Single-page web interface with 3D god's-eye view + dashboard panels**

```bash
# Create small demo subgraph (500 neurons, synthetic)
python scripts/create_demo_subgraph.py

# Run interactive demo
threedfly run-interactive --subgraph data/demo_subgraph.npz --steps 5000

# Open your browser to http://localhost:8080
# Everything is in one page: 3D view + all dashboard panels
```

**Interactive features:**
- **3D god's-eye view**: See the fly, trajectory, and **live updating brain-depth point cloud** together with orbit/pan/zoom
- **Real-time point cloud rendering**: The brain-estimated 3D map accumulates and updates live in the same view as the fly moves
- **Dashboard panels in same page**: Left/right eye views, trajectory plot, brain activity, flight commands, system status
- **Playback controls**: Play, pause, step frame-by-frame, or reset
- **Adjustable speed**: Control simulation playback speed

**One browser tab, everything together** — no separate matplotlib windows! The brain-based point cloud builds up in real-time as the fly explores. Classical StereoBM depth is also computed as a comparison baseline.

### Option 2: Run with Static Visualization

```bash
# Create small demo subgraph (500 neurons, synthetic)
python scripts/create_demo_subgraph.py

# Run demo with matplotlib plots
threedfly run --subgraph data/demo_subgraph.npz --steps 1000 --viz-mode matplotlib

# Advanced: Configure collision recovery
threedfly run --subgraph data/demo_subgraph.npz --steps 2000 --max-collisions 50
```

### Option 3: Download Full Connectome and Extract Subgraph

**Warning**: Downloads ~1.1 GB of data.

```bash
# Download MaleCNS v1.0 data
threedfly download

# Extract visual-flight subgraph (may take a few minutes)
threedfly extract --output data/subgraph.npz

# Run with extracted subgraph (interactive)
threedfly run-interactive --subgraph data/subgraph.npz --steps 5000

# Or run with static visualization
threedfly run --subgraph data/subgraph.npz --steps 2000
```

### Headless Mode (for CI/servers)

```bash
threedfly run --subgraph data/demo_subgraph.npz --steps 1500 --headless --save-output output/
# Outputs: 
#   output/point_cloud_brain.ply (primary: brain-depth reconstruction)
#   output/point_cloud_stereo.ply (comparison: classical StereoBM, dense sampling)
#   output/final_state.png (visualization snapshot)
# Both point clouds exported at 1cm voxel resolution for detailed comparison
```

---

## Usage

### CLI Commands

#### `threedfly download`
Download MaleCNS v1.0 connectome data from Google Cloud Storage.

```bash
threedfly download [--output-dir DIR] [--files annotations,weights]
```

#### `threedfly extract`
Extract visual-flight subgraph from full connectome.

```bash
threedfly extract [--data-dir DIR] [--output subgraph.npz] [--max-neurons N]

# Create small demo subgraph
threedfly extract --demo
```

#### `threedfly run`
Run the simulation with static visualization.

```bash
threedfly run [OPTIONS]

Options:
  --subgraph PATH              Path to subgraph file [default: data/demo_subgraph.npz]
  --steps N                    Simulation steps [default: 2000]
  --viz-mode MODE              matplotlib|open3d|both|none [default: matplotlib]
  --headless                   Run without visualization
  --save-output DIR            Output directory [default: output]
  --simulator-type TYPE        rate|lif [default: rate]
  --seed N                     Random seed [default: 42]
```

#### `threedfly run-interactive`
Run the simulation with unified interactive web interface.

```bash
threedfly run-interactive [OPTIONS]

Options:
  --subgraph PATH              Path to subgraph file [default: data/demo_subgraph.npz]
  --steps N                    Maximum simulation steps [default: 5000]
  --save-output DIR            Output directory [default: output]
  --simulator-type TYPE        rate|lif [default: rate]
  --seed N                     Random seed [default: 42]
  --host HOST                  Visualization server host [default: 0.0.0.0]
  --port PORT                  Visualization server port [default: 8080]

Features:
  - Single-page web interface combining 3D view and dashboard panels
  - Interactive 3D view with orbit/pan/zoom
  - Dashboard: stereo views, trajectory, brain activity, flight commands, status
  - Play/pause/step/reset controls
  - Adjustable playback speed

After starting, open your browser to http://localhost:8080
All visualizations are in one page — no separate windows!
```

#### `threedfly demo`
Quick demo (automatically handles data setup).

```bash
threedfly demo
```

### Python API

```python
from threedfly import ConnectomeLoader, BrainSimulator, FlyEnvironment
from threedfly.runner import run_simulation

# Run complete simulation
run_simulation(
    subgraph_path="data/demo_subgraph.npz",
    n_steps=2000,
    viz_mode="matplotlib",
    save_output_dir="output",
    simulator_type="rate",
    seed=42
)
```

---

## Data

### MaleCNS v1.0 Connectome

Source: **Male Central Nervous System (MaleCNS) v1.0**  
- Project: https://male-cns.janelia.org/  
- Publication: https://research.google/blog/a-connectomics-milestone-mapping-the-complete-male-fruit-fly-brain/  
- Neurons: ~166,700 (full CNS)  
- Synapses: ~109 million  
- License: **CC-BY 4.0** (must cite)

**Data files**:
- Body annotations: `body-annotations-male-cns-v1.0-minconf-0.5.feather` (~14 MB)
- Connectome weights: `connectome-weights-male-cns-v1.0-minconf-0.5.feather` (~1.1 GB)

**Citation**:
```
Dorkenwald et al. (2024). Male Central Nervous System Connectome v1.0.
https://male-cns.janelia.org/
```

### Subgraph Extraction

The full connectome is too large for realtime simulation. We extract a visual-flight subgraph focusing on:

- **Visual system**: Photoreceptors (R1-R8), lamina (L1-L5), medulla (Mi, Tm, T4, T5), lobula complex (LC, LPLC), lobula plate (LPi)
- **Descending neurons**: Motor control pathways (DN, DNa, DNb, DNp)
- **Central complex**: Navigation circuits (fan-shaped body, ellipsoid body, protocerebral bridge)

Filtering is based on neuron type annotations. If type-based filtering yields insufficient neurons, neuropil region-based selection is used.

---

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=threedfly --cov-report=html

# Run specific test modules
pytest tests/test_connectome.py -v
pytest tests/test_vision.py -v
pytest tests/test_control.py -v
```

---

## Project Structure

```
threedfly/
├── connectome/         # Connectome loading, subgraph extraction, neural simulation
│   ├── downloader.py   # Download MaleCNS data
│   ├── loader.py       # Load and extract subgraph
│   └── simulator.py    # LIF and rate-based simulators
├── vision/             # Vision processing and depth estimation
│   ├── stereo.py       # Classical StereoBM (comparison baseline)
│   ├── brain_depth.py  # Brain-based depth estimation (primary)
│   └── pointcloud.py   # 3D map accumulation, occupancy tracking
├── sim/                # MuJoCo physics simulation
│   ├── mjcf.py         # MJCF (XML) model definitions
│   └── environment.py  # Fly environment with stereo cameras
├── control/            # Flight control and exploration
│   ├── controller.py   # Brain activity → motor commands
│   └── exploration.py  # Curiosity-driven policy
├── viz/                # Interactive visualization
│   └── visualizer.py   # Real-time plots (matplotlib/Open3D)
├── cli.py              # Command-line interface
└── runner.py           # Main simulation loop

data/                   # Connectome data and subgraphs (gitignored except demo)
scripts/                # Utilities (demo subgraph generation)
tests/                  # Unit tests
```

---

## Configuration

### Neural Simulator Parameters

**LIF Model** (`BrainSimulator`):
- `dt`: 1ms timestep
- `tau`: 10ms membrane time constant
- `v_thresh`: 1.0 spike threshold
- `refrac_period`: 2ms refractory period

**Rate Model** (`RateSimulator`):
- `dt`: 10ms timestep
- `tau`: 50ms time constant
- `activation`: ReLU (default) | sigmoid | tanh

**Recommendation**: Use rate model for faster realtime simulation. LIF for more detailed spike dynamics (slower).

### Brain-Based Depth Estimation

The `BrainDepthEstimator` extracts depth information from connectome neural activity:
- **Visual input** → ommatidial samples → photoreceptor layer
- **Connectome processing** → visual neuron activity patterns
- **Depth readout** → population code from visual neurons (hand-designed linear mapping)
- **Spatial resolution** → coarse retinotopic mapping (8×6 regions)

The depth readout uses a population code where each visual neuron votes for its preferred depth weighted by its activity level. This is **engineered, not biologically validated**, but routes visual information through actual connectome pathways rather than using a disconnected neural network.

### StereoBM Comparison Baseline

Classical stereo block matching provides a dense comparison baseline:
- **High resolution**: 160×120 eye cameras for detailed stereo matching
- **Dense sampling**: Stereo depth computed every 2 steps
- **Low confidence threshold**: 0.08 minimum confidence for informative comparison (typical confidences 0.1–0.4)
- **Tuned parameters**: 64 disparities, block size 5, optimized for fly-scale depth range
- **Typical output**: ~200K raw points → thousands of unique points @1-3cm voxel resolution on a 1500-step run

### Flight Control Mapping

The `FlightController` uses a simple hand-designed linear mapping:
- **Thrust forward**: Baseline (0.6) + Mean DN activity (strong forward bias for room traversal)
- **Thrust up**: No bias + Minimal DN activity (reduced to eliminate ceiling bouncing)
- **Yaw**: Left-right DN asymmetry (strong turning for horizontal exploration)
- **Pitch/Roll**: Front-back DN asymmetry (reduced to minimize vertical oscillation)

**Horizontal exploration priority**: Controller gains and baselines are tuned to strongly favor horizontal (XY) room coverage over vertical bobbing. Forward thrust has a high baseline (0.6) and gain (1.0), while upward thrust has zero bias and minimal gain (0.05). Yaw gain is high (0.8) to encourage turning and room traversal.

This mapping is **engineered, not biologically validated**. It provides plausible control but does not claim to replicate actual fly motor control.

### Brain-Mediated Obstacle Avoidance

**How the fly avoids obstacles:**

1. **Visual features → Brain input**:
   - Ommatidial intensity samples (left/right eyes, ~100 samples each)
   - **Looming/proximity signals** (8 directional sectors): Engineered visual features computed from stereo depth map, indicating obstacle proximity in each direction (forward, forward-left, left, back-left, back, back-right, right, forward-right)
   - These 208 features (200 ommatidial + 8 looming) are concatenated and fed as input to the connectome

2. **Brain processing**:
   - Input flows through the MaleCNS connectome subgraph (visual neurons → descending neurons)
   - Neural dynamics (LIF or rate model) propagate activity through actual synaptic connections
   - The brain's response to looming signals emerges from connectome structure + dynamics

3. **Motor output → Steering away**:
   - Descending neuron (DN) activity is read out by the flight controller
   - DN asymmetry generates yaw/pitch/roll to steer away from obstacles
   - The motor response is **brain-mediated**: looming features → connectome activity → DN output → motor commands

**What's brain-controlled vs engineered:**
- ✅ **Brain-controlled**: The transformation from visual input (ommatidia + looming) to motor output (DN activity) routes through the real connectome structure with neural dynamics
- ⚙️ **Engineered**: 
  - Looming feature extraction (inverse depth by sector from StereoBM)
  - Input mapping (which neurons receive looming signals)
  - DN-to-motor readout (linear mapping from DN activity to thrust/torque)
  - Soft collision recovery (physics script as last-resort safety net, NOT primary avoidance)

**Honest assessment**: The fly steers based on brain activity that is influenced by obstacle proximity signals routed through the connectome. This is brain-structured avoidance with engineered feature extraction and motor readout, not a claim that fruit flies use exactly these visual features or that our DN readout matches biological motor control. The primary avoidance mechanism operates through the brain pathway; soft collision recovery serves only as a safety fallback.

### Exploration Policy

- Samples exploration score in 8 directions around current position (horizontal plane only)
- Scores based on voxel occupancy (low occupancy → high score)
- Biases flight toward frontiers with strong forward signal (1.5) and turning
- **No vertical exploration bias**: Altitude control delegated to brain/controller to avoid ceiling bouncing
- Random exploration with probability 0.2

### Collision Recovery

- **Soft recovery** automatically handles collisions instead of stopping
- Backs off from collision point, dampens velocity, randomizes orientation
- Configurable max collisions (default: 50 non-interactive, 100 interactive)
- Enables continuous exploration: 1500+ steps vs ~255 without recovery
- See [COLLISION_RECOVERY.md](COLLISION_RECOVERY.md) for details

---

## Troubleshooting

### ModuleNotFoundError: mujoco

```bash
pip install mujoco>=3.0.0
```

### OpenCV or Open3D installation issues

```bash
# Ubuntu/Debian
sudo apt-get install -y python3-opencv libgl1-mesa-glx

# macOS
brew install open3d
pip install opencv-python open3d
```

### Visualization window not showing

- Ensure you're not in headless mode
- Check `DISPLAY` environment variable (Linux)
- Try `--viz-mode matplotlib` instead of `open3d`

### Out of memory during subgraph extraction

```bash
# Limit number of neurons
threedfly extract --max-neurons 5000
```

---

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Write tests for new functionality
4. Ensure all tests pass: `pytest`
5. Format code: `black threedfly/ tests/`
6. Submit a pull request

---

## License

**Code**: MIT License (see [LICENSE](LICENSE))  
**Connectome Data**: CC-BY 4.0 (cite male-cns.janelia.org)

---

## Citation

If you use this project in research, please cite:

```bibtex
@software{threedfly2024,
  title={3dfly: Connectome-Driven 3D Exploration},
  author={3dfly contributors},
  year={2024},
  url={https://github.com/liudicsu/3dfly}
}
```

And cite the MaleCNS connectome:

```bibtex
@misc{malecns2024,
  title={Male Central Nervous System Connectome v1.0},
  author={Dorkenwald, Sven and others},
  year={2024},
  url={https://male-cns.janelia.org/}
}
```

---

## Acknowledgments

- **Google Research** and **HHMI Janelia Research Campus** for the MaleCNS v1.0 connectome
- **Google DeepMind** and **HHMI Janelia** for the flybody anatomical meshes (Apache-2.0 license)
- **MuJoCo** physics engine (DeepMind)
- **Open3D** library
- The fruit fly research community

---

## 中文说明

**3dfly** 是一个果蝇脑驱动的3D探索模拟器。

### 主要特点

- **写实的果蝇模型**：使用来自 [flybody 项目](https://github.com/TuragaLab/flybody)（Google DeepMind & HHMI Janelia）的解剖学精确3D网格
- 使用真实的雄性果蝇中枢神经系统连接组（MaleCNS v1.0，约16.67万个神经元）
- **大脑驱动的深度感知**：从连接组神经活动中提取深度估计（主要重建方式）
- 双目视觉深度估计（经典StereoBM作为对比基准）和3D点云建图
- MuJoCo物理仿真环境
- 基于好奇心的探索策略
- **交互式3D上帝视角可视化**

### 快速开始

#### 方式一：交互式统一网页界面（推荐）

```bash
# 安装
pip install -e ".[dev]"

# 创建演示子图
python scripts/create_demo_subgraph.py

# 运行交互式模拟
threedfly run-interactive --subgraph data/demo_subgraph.npz --steps 5000

# 在浏览器中打开 http://localhost:8080
# 所有内容在同一页面：3D视图 + 仪表盘面板
```

**交互功能：**
- **3D上帝视角**：同时看到果蝇、飞行轨迹和**实时更新的大脑深度点云地图**，支持旋转、平移和缩放
- **实时点云渲染**：大脑估计的3D地图随着果蝇移动在同一视图中实时累积和更新
- **仪表盘面板**（同页面）：左右眼视图、飞行轨迹图、大脑活动、飞行指令、系统状态
- **播放控制**：播放、暂停、单步前进、重置模拟
- **速度调节**：控制模拟播放速度

**一个浏览器标签页，所有功能齐全** — 不再需要单独的matplotlib窗口！大脑驱动的点云在果蝇探索时实时构建。经典StereoBM深度同时作为对比基准计算。

#### 方式二：静态可视化

```bash
# 运行模拟（静态matplotlib图表）
threedfly run --subgraph data/demo_subgraph.npz --steps 1000
```

### 重要说明

这是一个**基于连接组结构的模拟器**，使用简化的神经元模型（LIF或速率模型）和工程化的输入输出映射。大脑驱动的深度估计通过连接组神经通路处理视觉特征，但深度读取机制是手工设计的工程方案，而非生物学验证的机制。不是对果蝇大脑的逐神经元生物学准确模拟。

---

**Happy exploring! 🪰**
