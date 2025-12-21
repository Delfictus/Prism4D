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
├── prism_4d_stages.cuh           # Shared header (TO CREATE)
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
    pub escape: f32,        // From DMS data
    pub frequency: f32,     // From GISAID
    pub gamma: f32,         // Computed fitness
    pub growth_potential: f32,
    pub escape_dominance: f32,
}
```

### Discretization (256 states)
```rust
fn discretize(&self) -> usize {
    let escape_bin = ((self.escape * 3.99) as usize).min(3);
    let freq_bin = ((self.frequency * 3.99) as usize).min(3);
    let gp_bin = ((self.growth_potential * 3.99) as usize).min(3);
    let ed_bin = (((self.escape_dominance + 1.0) / 2.0 * 3.99) as usize).min(3);
    escape_bin * 64 + freq_bin * 16 + gp_bin * 4 + ed_bin
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

### Phase Multipliers for Emergence
```cuda
NAIVE:     0.3  // Can emerge but slow
EXPLORING: 1.0  // Actively emerging NOW
ESCAPED:   0.1  // Already happened
COSTLY:    0.4  // Might shift mutation
REVERTING: 0.2  // Declining
FIXED:     0.05 // Stabilized
```

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

## Common Errors & Fixes

See [references/debug_playbook.md](references/debug_playbook.md) for detailed solutions.

## Implementation Tasks

**Task 1**: Create `prism_4d_stages.cuh` - extract stage7/stage8 from mega_fused_pocket_kernel.cu

**Task 2**: Update `mega_fused_batch.cu`:
1. Add `#include "prism_4d_stages.cuh"`
2. Add `combined_features_out` parameter
3. Call stage7_fitness_features() after Stage 6
4. Call stage8_cycle_features() after Stage 7
5. Call stage6_5_combine_features() to write 101-dim

**Task 3**: Update `mega_fused_batch.rs`:
1. Add `d_combined_features` to BatchBufferPool
2. Add `combined_features` to BatchStructureOutput
3. Pass new buffer to kernel launch

## Validation Checklist

- [ ] PTX compiles without errors
- [ ] Kernel launches without CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES
- [ ] combined_features has correct dimensions (n_residues × 101)
- [ ] Features 92-100 contain non-zero values
- [ ] RL training converges (train accuracy > 85%)
- [ ] Test accuracy competitive with VASIL (target: >92%)
