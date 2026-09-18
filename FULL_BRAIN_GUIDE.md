# Full-Brain MaleCNS Connectome Guide

## Overview

3dfly now supports running simulations using the **complete male fruit fly brain** connectome from MaleCNS v1.0, with ~185,000 neurons and ~10.6 million connections.

## Quick Start

```bash
# 1. Download MaleCNS v1.0 data (~1.1 GB)
threedfly download

# 2. Build full-brain connectome (~1-2 minutes)
threedfly build-full --output data/full_connectome.npz

# 3. Run simulation with full brain
threedfly run-interactive --subgraph data/full_connectome.npz --steps 5000
```

## Technical Specifications

### Default Configuration (min-synapses=3)
- **Neurons**: 184,526 annotated neurons with connections
- **Connections**: 10,653,945 synapses
- **Sparsity**: 0.031% (extremely sparse)
- **File Size**: ~35 MB (compressed NPZ)
- **Memory Usage**: 
  - Build time: 4-8 GB RAM
  - Runtime: ~82 MB for sparse matrix
- **Build Time**: ~1-2 minutes
- **Runtime Performance**: Near real-time with RateSimulator

### Filtering Options

The `--min-synapses` parameter controls connection filtering:

| min-synapses | Neurons | Connections | File Size | RAM (build) |
|--------------|---------|-------------|-----------|-------------|
| 1 (all)      | ~185k   | ~50M+       | ~200 MB   | 10-15 GB    |
| 3 (default)  | ~185k   | ~10.6M      | ~35 MB    | 4-8 GB      |
| 5 (strict)   | ~165k   | ~6M         | ~25 MB    | 3-5 GB      |

**Recommendation**: Use default `min-synapses=3` for best balance of biological realism and memory efficiency. This threshold filters out weak connections while retaining all biologically significant synapses.

## Data Pipeline

### 1. Download Raw Data
```bash
threedfly download
```

Downloads from `gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/`:
- `body-annotations-male-cns-v1.0-minconf-0.5.feather` (14 MB)
- `connectome-weights-male-cns-v1.0-minconf-0.5.feather` (1.1 GB)

### 2. Build Full Connectome

The build process:
1. **Loads annotations** → 211,577 annotated neurons
2. **Filters connections** → Uses PyArrow to efficiently filter by weight threshold before loading into memory
3. **Maps body IDs** → Only includes annotated neurons (avoids spurious IDs)
4. **Builds sparse matrix** → CSR format with float32 weights, int32 indices
5. **Saves compressed** → NPZ format with metadata

```bash
# Default: min-synapses=3 (recommended)
threedfly build-full

# Stricter filtering for lower memory usage
threedfly build-full --min-synapses 5

# Complete unfiltered connectome
threedfly build-full --min-synapses 1
```

### 3. Load and Simulate

```python
from threedfly.connectome import ConnectomeLoader
from threedfly.connectome.simulator import RateSimulator

# Load full brain
subgraph = ConnectomeLoader.load_subgraph("data/full_connectome.npz")

# Initialize simulator (RateSimulator recommended for large graphs)
brain = RateSimulator(
    subgraph["adjacency"],
    dt=0.010,  # 10ms timestep
    activation="relu"
)

# Set input and simulate
brain.set_input(input_vector)
rates = brain.step()
```

## Memory Optimization

### Why Filtering?

The raw MaleCNS dataset contains 151.8M connections. Many are weak connections that may not be biologically significant. Filtering by `min-synapses=3` reduces this to 10.6M connections while retaining strong, reliable connections.

### PyArrow Streaming

The loader uses PyArrow to filter connections **before** loading into pandas, dramatically reducing memory usage:

```python
# Memory-efficient: filter during load
loader.load_weights(min_weight=3)  # Loads only ~23M rows

# vs. naive approach (would run out of memory)
loader.load_weights()  # Would load all 151M rows
```

### Sparse Matrix Storage

- **Format**: Compressed Sparse Row (CSR)
- **Data types**: float32 (weights), int32 (indices)
- **Memory**: ~82 MB for 10.6M connections
- **Operations**: Efficient sparse matrix-vector multiplication (critical for realtime simulation)

## Comparison: Demo vs Full Brain

| Feature | Demo Subgraph | Full Brain |
|---------|---------------|------------|
| Neurons | 500 | 184,526 |
| Connections | ~2,500 | 10,653,945 |
| File Size | 69 KB | 35 MB |
| Load Time | <1 sec | ~2 sec |
| Step Time (Rate) | <1 ms | ~5-10 ms |
| Coverage | Visual-flight only | Entire CNS |
| Use Case | Quick testing | Full simulation |

## Performance Tips

### 1. Use RateSimulator
```python
# Faster for large graphs
brain = RateSimulator(adjacency, dt=0.010)
```

### 2. Reduce Simulation Timestep
```python
# Coarser timesteps = faster simulation
brain = RateSimulator(adjacency, dt=0.020)  # 20ms instead of 10ms
```

### 3. Target Specific Input/Output Neurons

When setting input or reading output, target specific neuron indices rather than random subsets:

```python
# Use annotation-based selection for input
visual_neurons = [i for i, bid in enumerate(subgraph['body_ids']) 
                  if annotations.loc[annotations['bodyId']==bid, 'type'].str.contains('photoreceptor')]

# Set input to visual neurons only
brain.set_input(input_vector, neuron_indices=visual_neurons)
```

## Biological Accuracy

### What's Realistic
- ✅ Real MaleCNS connectivity structure (~185k neurons)
- ✅ Actual body IDs from Janelia dataset
- ✅ Synaptic weight magnitudes (filtered for quality)
- ✅ Sparse connectivity pattern (0.031% dense)

### What's Engineered
- ⚙️ Simplified neuron dynamics (LIF/rate models, not Hodgkin-Huxley)
- ⚙️ Arbitrary input mapping (photoreceptor assignment)
- ⚙️ Hand-designed motor readout (DN → motor commands)
- ⚙️ Weight threshold filtering (min-synapses parameter)

### Honest Assessment
This is a **connectome-structured simulator** using the real brain wiring diagram with simplified neuron models and engineered I/O mappings. It demonstrates that the full brain structure can drive behavior, but does not claim neuron-for-neuron biological accuracy.

## Citation

When using the full-brain connectome, please cite:

```bibtex
@misc{malecns2024,
  title={Male Central Nervous System Connectome v1.0},
  author={Dorkenwald, Sven and others},
  year={2024},
  url={https://male-cns.janelia.org/},
  note={License: CC-BY 4.0}
}
```

## Troubleshooting

### Out of Memory During Build
- Use stricter filtering: `--min-synapses 5`
- Close other applications to free RAM
- Ensure at least 8GB RAM available

### Slow Simulation
- Use `RateSimulator` instead of `BrainSimulator` (LIF)
- Increase timestep: `dt=0.020` or `dt=0.050`
- Reduce input frequency

### File Not Found
- Ensure data downloaded: `threedfly download`
- Check path: `ls -lh data/connectome-raw/`

## Future Work

Potential enhancements for full-brain simulation:

1. **Neuron type-specific input/output**
   - Use annotations to automatically identify photoreceptors
   - Map descending neurons by type (DNa, DNb, DNp)

2. **Multi-region dynamics**
   - Track activity by brain region (medulla, lobula, central complex)
   - Visualize information flow across regions

3. **Optimized sparse operations**
   - GPU-accelerated sparse matrix multiplication
   - Batch processing for multiple timesteps

4. **Biological neuron models**
   - Support Hodgkin-Huxley dynamics for select neuron types
   - Integrate neurotransmitter data when available

---

**Questions?** Open an issue at https://github.com/liudicsu/3dfly/issues
