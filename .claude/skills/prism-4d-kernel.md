---
name: prism-4d-kernel
description: "PRISM-4D GPU kernel architecture and implementation guide. Use when working on: (1) CUDA kernel development for mega_fused_batch.cu or mega_fused_pocket_kernel.cu, (2) Rust-CUDA FFI in prism-gpu crate, (3) FluxNet RL optimizer in ve_optimizer.rs, (4) Feature extraction pipeline (101-dim), (5) VASIL benchmark integration, (6) Stage 7 Fitness or Stage 8 Cycle module implementation. This skill encodes the complete architecture from the Master Blueprint including naming conventions, feature layouts, win conditions, and debug patterns."
---

# PRISM>4D Kernel Development Guide

## Win Conditions (Hard Constraints)

Every code change must satisfy:
- **SPEED**: Full batch pipeline <60 seconds for 14,917 structures
- **ACCURACY**: >92.0% mean accuracy across 12-country VASIL dataset
- **INTEGRITY**: NO hardcoded VASIL weights (γ = 0.65×escape + 0.35×transmit is FORBIDDEN in RL)

## Architecture Overview

```
User Application (prism-ve::predict_viral_evolution)
         ↓
Rust API Layer (MegaFusedBatchGpu::detect_pockets_batch)
         ↓
GPU Kernel Layer (mega_fused_batch_detection_prism4d)
         ↓
8 Computation Stages → 101-dim feature output
         ↓
FluxNet RL (AdaptiveVEOptimizer) → RISE/FALL prediction
```

## Feature Layout (101 dimensions)

| Index | Feature | Stage | Range |
|-------|---------|-------|-------|
| 0-47 | TDA (Betti, persistence, topology) | Stage 3 | varies |
| 48-79 | Dendritic Reservoir (neuromorphic) | Stage 4 | [-1,1] |
| 80-91 | Physics (burial, centrality, etc.) | Stage 5 | varies |
| **92** | **ΔΔG_binding** | **Stage 7** | kcal/mol |
| **93** | **ΔΔG_stability** | **Stage 7** | kcal/mol |
| **94** | **Expression fitness** | **Stage 7** | [0,1] |
| **95** | **Transmissibility (RAW)** | **Stage 7** | [0,1] |
| **96** | **Cycle phase** | **Stage 8** | 0-5 int |
| **97** | **Emergence probability** | **Stage 8** | [0,1] |
| **98** | **Time to peak** | **Stage 8** | 0-24 months |
| **99** | **Frequency (current)** | **Stage 8** | [0,1] |
| **100** | **Velocity** | **Stage 8** | [-0.5,+0.5] |

## Critical File Map

### CUDA Kernels
```
crates/prism-gpu/src/kernels/
├── mega_fused_pocket_kernel.cu   # Single-structure (101-dim working)
├── mega_fused_batch.cu           # Batch (NEEDS Stage 7-8 integration)
├── prism_4d_stages.cuh           # Shared header
└── viral_evolution_fitness.cu    # ΔΔG helpers
```

### Rust API
```
crates/prism-gpu/src/
├── lib.rs                        # Public exports
├── mega_fused_batch.rs           # BatchStructureDesc, PackedBatch, BatchOutput
├── mega_fused.rs                 # MegaFusedGpu, MegaFusedOutput
└── mega_fused_config.rs          # MegaFusedConfig, MegaFusedMode
```

### Benchmark Layer
```
crates/prism-ve-bench/src/
├── main.rs                       # Entry point, build_mega_batch()
├── data_loader.rs                # GisaidFrequencies, DmsEscapeData
├── ve_optimizer.rs               # AdaptiveVEOptimizer (FluxNet RL)
├── gpu_benchmark.rs              # FeatureExtractor, VariantFeatures
├── immunity_model.rs             # Time-varying immunity, CrossReactivityMatrix
└── pdb_parser.rs                 # PdbStructure, apply_mutations()
```

## Naming Conventions

### Variables
| Pattern | Usage | Example |
|---------|-------|---------|
| `d_*` | GPU device buffers | `d_atoms`, `d_combined_features` |
| `h_*` | Host arrays | `h_consensus_scores` |
| `*_packed` | Packed batch data | `atoms_packed`, `burial_packed` |
| `UPPER_CASE` | Constants | `TILE_SIZE`, `ALPHA_ESCAPE` |
| `*_idx` | Indices | `structure_idx`, `residue_idx` |
| `n_*` | Dimensions | `n_structures`, `n_residues` |

### Functions
| Pattern | Context | Example |
|---------|---------|---------|
| `stage*_*` | CUDA device functions | `stage7_fitness_features` |
| `mega_fused_*` | CUDA global kernels | `mega_fused_batch_detection_prism4d` |
| `snake_case` | Rust public/private | `detect_pockets_batch` |

## Key Structs

### BatchStructureDesc (MUST match CUDA)
```rust
#[repr(C, align(16))]
pub struct BatchStructureDesc {
    pub atom_offset: i32,     // Start in atoms_packed
    pub residue_offset: i32,  // Start in residue arrays
    pub n_atoms: i32,
    pub n_residues: i32,
}
```

### VEState (RL Input)
```rust
pub struct VEState {
    pub escape: f32,              // From DMS data
    pub transmit: f32,            // Literature R0
    pub frequency: f32,           // From GISAID
    pub effective_escape: f32,    // Immunity-adjusted
    pub relative_fitness: f32,    // Escape advantage over competition
    pub frequency_velocity: f32,  // df/dt momentum
}
```

## Stage 7: Fitness Module Formulas

```cuda
// Feature 92: ΔΔG_binding
float ddg_binding = (hydro - 0.5f) * centrality * (1.0f - burial);

// Feature 93: ΔΔG_stability
float ddg_stability = burial * (volume - 0.5f) * (1.0f - bfactor);

// Feature 94: Expression fitness
float expression = 0.3f + 0.5f * (1.0f - burial) + 0.2f * bfactor;

// Feature 95: Transmissibility (RAW - NOT gamma!)
float transmit = sigmoid(ddg_binding) * sigmoid(ddg_stability) * expression;
```

## Stage 8: Cycle Phase Classification

| Phase | ID | Condition |
|-------|-----|-----------|
| NAIVE | 0 | freq < 0.01 AND velocity < 0.01 AND escape < 0.5 |
| EXPLORING | 1 | velocity > 0.05 AND freq < 0.50 |
| ESCAPED | 2 | freq > 0.50 AND velocity >= -0.02 |
| COSTLY | 3 | freq > 0.20 AND velocity < -0.02 AND gamma < 0 |
| REVERTING | 4 | velocity < -0.05 |
| FIXED | 5 | freq > 0.80 AND |velocity| < 0.02 AND gamma > -0.1 |

## Build Commands

### Compile PTX
```bash
cd /mnt/c/Users/Predator/Desktop/PRISM/crates/prism-gpu
CUDA_HOME=/usr/local/cuda-12.6 \
LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH \
/usr/local/cuda-12.6/bin/nvcc \
  -ptx -O3 --gpu-architecture=sm_75 --use_fast_math \
  -o target/ptx/mega_fused_batch.ptx \
  src/kernels/mega_fused_batch.cu
```

### Build Rust
```bash
cd /mnt/c/Users/Predator/Desktop/PRISM
PATH="/home/diddy/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin:$PATH" \
CUDA_HOME=/usr/local/cuda-12.6 \
cargo build --release -p prism-ve-bench
```

### Run Benchmark
```bash
RUST_LOG=warn ./target/release/vasil-benchmark
```

---

# CUDA Kernel Patterns

## Kernel Launch Configuration

### RTX 4090/3060 Optimal Settings
```cuda
extern "C" __global__ void __launch_bounds__(256, 2)
mega_fused_batch_detection(...)

// Host-side launch
dim3 grid(n_structures);  // One block per structure
dim3 block(256);          // 256 threads per block
```

## Shared Memory Structure

```cuda
struct __align__(16) BatchSharedMem {
    float3 ca_coords[TILE_SIZE];        // 256×12 = 3KB
    float conservation[TILE_SIZE];       // 256×4 = 1KB
    float bfactor[TILE_SIZE];           // 256×4 = 1KB
    float burial[TILE_SIZE];            // 256×4 = 1KB
    float degree[TILE_SIZE];            // 256×4 = 1KB
    float centrality[TILE_SIZE];        // 256×4 = 1KB
    float consensus_score[TILE_SIZE];   // 256×4 = 1KB
    float fitness_features[TILE_SIZE][4]; // 256×4×4 = 4KB
    float cycle_features[TILE_SIZE][5];   // 256×5×4 = 5KB
};
// Total: ~22KB (fits in 48KB limit)
```

## Tiled Processing Pattern

```cuda
__global__ void mega_fused_batch_detection(...) {
    int structure_idx = blockIdx.x;
    BatchStructureDesc desc = descriptors[structure_idx];

    __shared__ BatchSharedMem smem;

    int n_tiles = (desc.n_residues + TILE_SIZE - 1) / TILE_SIZE;

    for (int tile = 0; tile < n_tiles; tile++) {
        int local_idx = threadIdx.x;
        int global_idx = tile * TILE_SIZE + local_idx;
        bool active = (global_idx < desc.n_residues);

        // Load, process, write per tile
        if (active) {
            // Load from global with __ldg()
            // Process through stages
            // Write to output
        }
        __syncthreads();
    }
}
```

## Constant Memory for Amino Acid Properties

```cuda
__constant__ float c_residue_hydrophobicity[20] = {
    0.616f, 0.000f, 0.236f, 0.028f, 0.680f,  // A,R,N,D,C
    0.251f, 0.043f, 0.501f, 0.165f, 0.943f,  // Q,E,G,H,I
    0.943f, 0.283f, 0.738f, 1.000f, 0.711f,  // L,K,M,F,P
    0.359f, 0.450f, 0.878f, 0.880f, 0.825f   // S,T,W,Y,V
};
```

---

# Debug Playbook

## CUDA Errors

### CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES
**Fix**: Reduce `__launch_bounds__` or unroll pragmas

### CUDA_ERROR_ILLEGAL_ADDRESS
**Debug**: `compute-sanitizer --tool memcheck ./target/release/vasil-benchmark`

### Kernel Hangs
**Debug**: `CUDA_LAUNCH_BLOCKING=1 timeout 30s ./benchmark`

## FluxNet RL Issues

### Training Doesn't Converge (50% accuracy)
- Check feature normalization
- Verify non-zero values in features 92-100
- Try coarser binning (256 states vs 4096)

### Class Imbalance (Always Predicts FALL)
- Apply asymmetric rewards
- Increase RISE weight by 1.5x

### Features 92-100 Are All Zero
- Verify stage7/stage8 called in kernel
- Check stage6_5_combine_features writes all 101 dims

---

# FluxNet RL Reference

## Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `alpha` | 0.15 | Learning rate |
| `gamma` | 0.0 | Discount (single-step) |
| `epsilon_start` | 0.2 | Initial exploration |
| `epsilon_min` | 0.02 | Minimum exploration |
| `epochs` | 300 | Training iterations |
| `num_states` | 256 | State space size |

## Q-Learning Update

```rust
fn train_step(&mut self, exp: &VEExperience) {
    let state_idx = exp.state.discretize();
    let action_idx = exp.action as usize;

    self.q_table[state_idx][action_idx] +=
        self.alpha * (exp.reward - self.q_table[state_idx][action_idx]);
}
```

## Class Imbalance Handling

```rust
let rise_weight = data.len() as f32 / (2.0 * rise_count as f32);
let fall_weight = data.len() as f32 / (2.0 * fall_count as f32);

let reward = if is_correct {
    if is_rise { rise_weight } else { fall_weight }
} else {
    if is_rise { -rise_weight * 1.5 } else { -fall_weight }
};
```

---

# Current Performance Gap Analysis

## The Core Problem

| Feature | RISE mean | FALL mean | Discriminates? |
|---------|-----------|-----------|----------------|
| Raw escape | 0.455 | 0.452 | NO |
| Effective escape | 0.180 | 0.178 | NO |
| Relative fitness | 0.506 | 0.502 | NO |
| **Frequency velocity** | **0.112** | **-0.029** | **YES** |

## Root Causes

1. **Escape is static per-variant** - doesn't change over time
2. **Immunity dampening collapses variance** - 60% reduction applied uniformly
3. **Temporal distribution shift** - Delta patterns don't transfer to Omicron
4. **Missing competitive dynamics** - need escape advantage over dominant variant

## Hypotheses to Test

1. **Wrong Task**: Predict dominance (>X% share), not week-over-week direction
2. **Wrong Features**: Use escape RANK/PERCENTILE, not absolute value
3. **Wrong Competition**: Use `escape / dominant_escape` ratio
4. **Wrong Window**: Use 4-week horizon instead of 1-week
5. **Missing Signal**: Flag recombinants (XBB) with higher priors

---

# Validation Checklist

- [ ] PTX compiles without errors
- [ ] Kernel launches without CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES
- [ ] combined_features has correct dimensions (n_residues × 101)
- [ ] Features 92-100 contain non-zero values
- [ ] RL training converges (train accuracy > 85%)
- [ ] Test accuracy competitive with VASIL (target: >92%)
