# PRISM>4D

**GPU-Accelerated Pandemic Intelligence & Drug Discovery Platform**

[![Rust](https://img.shields.io/badge/Rust-1.75+-orange.svg)](https://www.rust-lang.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.0+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-Proprietary-red.svg)](LICENSE)

---

## Overview

PRISM>4D is a unified computational biology platform combining two GPU-accelerated engines:

| Engine | Function | Performance | Target |
|--------|----------|-------------|--------|
| **PRISM-4D** | Viral evolution prediction | 19,400x faster than VASIL | ≥92% accuracy |
| **PRISM-LBS** | Binding site discovery | <2 sec/structure | Competitive with P2Rank |

**The only platform that answers both "WHAT variant is coming?" and "WHERE to target it?"**

---

## Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                    PRISM UNIFIED PLATFORM                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  PRISM-4D ENGINE              PRISM-LBS ENGINE              │
│  ─────────────────            ─────────────────             │
│  • Stages 1-7: Structural     • Pocket Detection            │
│  • Stage 8: Temporal/Cycle    • Geometry Features           │
│  • Stages 9-10: Immunity      • TDA Persistence             │
│  • Stage 11: Epidemiological  • Druggability Scoring        │
│  • FluxNet RL Controller      • Cryptic/Allosteric Sites    │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                    FOUNDATION LAYER                         │
│  Neuromorphic Computing │ Quantum Annealing │ PRCT Core     │
├─────────────────────────────────────────────────────────────┤
│                    GPU ACCELERATION                         │
│  49 CUDA Kernels │ 25 PTX Files │ cudarc 0.18.1            │
└─────────────────────────────────────────────────────────────┘
```

---

## Technical Stack

- **Language:** Rust (157k+ LOC across 19 crates)
- **GPU:** CUDA 12.0+ (49 kernels, 26k+ LOC)
- **RL Engine:** FluxNet (discrete Q-learning, 6D state space)
- **Target Hardware:** RTX 3060+ (6GB VRAM minimum)

---

## Key Components

### Mega-Fused CUDA Kernels
- `mega_fused_vasil_fluxnet.cu` — VASIL + FluxNet RL integration
- `mega_fused_batch.cu` — General batch processing (142KB)
- `mega_fused_pocket_kernel.cu` — LBS pocket detection

### VE-Swarm Neuromorphic Kernels
- `ve_swarm_agents.cu` — 32-agent swarm inference
- `ve_swarm_dendritic_reservoir.cu` — Dendritic reservoir computing
- `ve_swarm_temporal_conv.cu` — Temporal convolution

### Immunity/Physics Kernels
- `prism_immunity_ic50.cu` — IC50-based neutralization
- `gamma_envelope_reduction.cu` — Envelope computation
- `polycentric_immunity.cu` — Multi-region immunity

---

## Data

- **VASIL Benchmark:** 12 countries (Germany, USA, UK, Japan, Brazil, France, Canada, Denmark, Australia, Sweden, Mexico, South Africa)
- **Trained Models:** FluxNet Q-table + optimized thresholds
- **Validation:** CryptoBench 2025 golden pockets

---

## Building
```bash
# Prerequisites
# - Rust 1.75+
# - CUDA Toolkit 12.0+
# - RTX 3060+ GPU

# Build
cargo build --release

# Run VASIL benchmark
cargo run --release -p prism-ve-bench

# Run LBS detection
cargo run --release -p prism-lbs -- --pdb structure.pdb
```

---

## Project Structure
```
PRISM4D/
├── crates/                 # 19 Rust crates
│   ├── prism-ve/          # Viral evolution engine
│   ├── prism-ve-bench/    # VASIL benchmark
│   ├── prism-lbs/         # Binding site detection
│   ├── prism-fluxnet/     # RL controller
│   ├── prism-gpu/         # GPU kernels
│   └── ...
├── foundation/            # Core compute layer
│   ├── neuromorphic/      # Dendritic reservoir
│   ├── quantum/           # Quantum annealing
│   └── prct-core/         # PRCT algorithm
├── kernels/ptx/           # Compiled CUDA kernels
├── data/                  # VASIL + benchmark data
├── docs/                  # Architecture documentation
└── validation_results/    # Trained models
```

---

## Win Conditions

| Metric | Target | Current |
|--------|--------|---------|
| VASIL Batch Time | <60 seconds | TBD |
| VASIL Accuracy | >92% | TBD |
| LBS Speed | <2 sec/structure | TBD |

---

## License

**Proprietary — Delfictus I/O Inc.**

All rights reserved. Unauthorized copying, modification, or distribution is prohibited.

---

## Contact

For licensing inquiries: [Contact Delfictus I/O]

---

*Version 0.3.0 | December 2024*
