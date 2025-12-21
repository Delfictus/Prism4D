# PRISM-4D Viral Evolution Prediction Platform: Ultra-Specialized Optimization Agent Prompt

## Mission Statement
You are a world-class computational biology and GPU systems optimization agent. Your mission is to transform the PRISM-4D VASIL benchmark from ~42% accuracy to VASIL's published 92% accuracy target. This requires deep understanding of viral evolution dynamics, GPU-accelerated feature extraction, and machine learning optimization.

---

## Platform Architecture Overview

### Core Technology Stack
- **Language**: Rust (2021 edition) with CUDA PTX kernels
- **GPU Framework**: cudarc (Rust CUDA bindings)
- **Target Hardware**: NVIDIA RTX 4090 / Tesla V100+
- **Data Sources**: GISAID frequencies, Bloom Lab DMS escape scores, PDB structures

### Repository Structure
```
/mnt/c/Users/Predator/Desktop/PRISM/
├── crates/
│   ├── prism-gpu/                    # GPU kernels and CUDA infrastructure
│   │   └── src/
│   │       ├── kernels/
│   │       │   ├── mega_fused_pocket_kernel.cu    # Main 101-dim feature kernel
│   │       │   └── prism_4d_stages.cuh            # Stage definitions and constants
│   │       ├── mega_fused.rs                      # Rust GPU orchestration
│   │       ├── mega_fused_batch.rs                # Batch processing infrastructure
│   │       └── lib.rs                             # Public API exports
│   │
│   └── prism-ve-bench/               # VASIL Benchmark implementation
│       └── src/
│           ├── main.rs                            # Benchmark orchestration
│           ├── ve_optimizer.rs                    # Grid search & prediction
│           ├── immunity_model.rs                  # Time-varying immunity
│           ├── data_loader.rs                     # GISAID/DMS data loading
│           ├── gpu_benchmark.rs                   # GPU feature extraction
│           └── pdb_parser.rs                      # Spike RBD structure parsing
│
├── target/
│   ├── ptx/                          # Compiled CUDA PTX kernels
│   └── release/                      # Release binaries
│
└── data/
    └── spike_rbd_6m0j.pdb            # Reference Spike RBD structure
```

### External Data Paths
```
/mnt/f/VASIL_Data/
├── Germany/
│   ├── gisaid_frequencies.csv        # Daily lineage frequencies
│   ├── lineage_mutations.json        # Per-lineage spike mutations
│   └── dms_escape_scores.csv         # Bloom Lab antibody escape
├── USA/
├── UK/
├── Japan/
├── Brazil/
├── France/
├── Canada/
├── Denmark/
├── Australia/
├── Sweden/
├── Mexico/
└── SouthAfrica/
    └── [same structure as Germany]
```

---

## GPU Kernel Architecture (CRITICAL)

### Mega-Fused Pocket Kernel Pipeline
The kernel processes structures through 8 stages, producing 101-dimensional features per residue:

```
STAGE PIPELINE:
┌─────────────────────────────────────────────────────────────────────┐
│ Stage 1: Distance Matrix (Cα-Cα distances)                          │
│ Stage 2: Contact Graph (8Å cutoff neighbor detection)               │
│ Stage 3: TDA Features (Betti numbers, persistence diagrams) [48-dim]│
│ Stage 4: Pocket Detection (geometric cavity analysis)               │
│ Stage 5: Physics Features (electrostatics, hydrophobicity) [12-dim] │
│ Stage 6: Base Features (burial, conservation, B-factor) [32-dim]    │
│ Stage 7: Fitness Features (ddG binding, stability, expression)[4-dim]│
│ Stage 8: Cycle Features (VE temporal dynamics) [5-dim]              │
│ Stage 6.5: Feature Combination → 101-dim output                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Feature Index Map (101 dimensions)
```
Indices 0-47:   TDA topological features (Betti curves, persistence)
Indices 48-79:  Base structural features (burial, conservation, B-factor)
Indices 80-91:  Physics features (electrostatics, hydrophobicity)
Indices 92-95:  Fitness features (ddG_binding, ddG_stability, expression, transmit)
Indices 96-100: Cycle features (phase, emergence, time_to_peak, frequency, velocity)
```

### Key Constants (`prism_4d_stages.cuh`)
```cuda
#define TILE_SIZE 256
#define MAX_NEIGHBORS 32
#define TOTAL_COMBINED_FEATURES 101
#define TDA_FEATURE_COUNT 48
#define BASE_FEATURE_COUNT 32
#define PHYSICS_FEATURE_COUNT 12
#define FITNESS_FEATURE_COUNT 4
#define CYCLE_FEATURE_COUNT 5
```

---

## Current Performance Analysis

### Benchmark Results (Latest Run)
```
GPU Pipeline: WORKING (7,294 structures/sec)
Training samples: 1,745 (2021-Jun 2022, Delta era)
Testing samples: 9,340 (Jun 2022-2023, Omicron era)
Training accuracy: 66.5% (balanced)
Test accuracy: 42-44%
VASIL target: 92.0%
```

### Feature Discrimination Analysis (THE CORE PROBLEM)
```
┌────────────────────┬────────────┬────────────┬──────────────┐
│ Feature            │ RISE mean  │ FALL mean  │ Discriminates│
├────────────────────┼────────────┼────────────┼──────────────┤
│ Raw escape         │ 0.455      │ 0.452      │ ❌ NO        │
│ Effective escape   │ 0.180      │ 0.178      │ ❌ NO        │
│ Relative fitness   │ 0.506      │ 0.502      │ ❌ NO        │
│ Frequency velocity │ 0.112      │ -0.029     │ ✅ YES       │
│ Transmissibility   │ 0.650      │ 0.650      │ ❌ NO        │
└────────────────────┴────────────┴────────────┴──────────────┘
```

### Root Cause Diagnosis
1. **Escape scores are static per-variant** - they don't change over time, so all samples of "BA.5" have identical escape regardless of competition context
2. **Immunity dampening collapses variance** - 60% reduction applied uniformly removes discriminative signal
3. **Temporal distribution shift** - Delta era patterns (high escape = RISE) don't transfer to Omicron era (everyone has high escape)
4. **Missing competitive dynamics** - Current features don't capture "escape advantage over what's currently dominant"

---

## Implemented Components (Current State)

### 1. Immunity Model (`immunity_model.rs`)
```rust
pub struct PopulationImmunityLandscape {
    pub country: String,
    pub immunity_events: Vec<ImmunityEvent>,  // Vaccinations + infections
}

pub struct CrossReactivityMatrix {
    families: Vec<String>,  // [Wuhan, Alpha, Beta, Gamma, Delta, BA.1, BA.2, BA.45, BQ.1, XBB]
    matrix: [[f32; 10]; 10], // Cross-neutralization values
}

// Key method
pub fn compute_effective_escape(
    &self,
    raw_epitope_escape: &[f32; 10],
    target_lineage: &str,
    date: NaiveDate,
    cross_matrix: &CrossReactivityMatrix,
) -> f32
```

### 2. VE Optimizer (`ve_optimizer.rs`)
```rust
pub struct VEState {
    pub escape: f32,              // Raw DMS escape
    pub transmit: f32,            // Literature R0 [0.3-0.9]
    pub frequency: f32,           // Current GISAID frequency
    pub ddg_binding: f32,         // GPU feature 92
    pub ddg_stability: f32,       // GPU feature 93
    pub expression: f32,          // GPU feature 94
    pub epitope_escape: [f32; 10],// Per-epitope escape scores
    pub effective_escape: f32,    // Immunity-modulated escape
    pub relative_fitness: f32,    // Escape advantage vs competition
    pub frequency_velocity: f32,  // df/dt momentum signal
}

// Grid search formula
// γ = α×relative_fitness + β×velocity + γ×transmit
// Prediction: γ > threshold → RISE, else FALL
```

### 3. Batch Processing (`main.rs`)
```rust
struct BatchMetadata {
    country: String,
    lineage: String,
    date: NaiveDate,
    frequency: f32,
    next_frequency: f32,
    escape_score: f32,
    epitope_escape: [f32; 10],
    effective_escape: f32,
    transmissibility: f32,
    relative_fitness: f32,
    frequency_velocity: f32,
    is_train: bool,
}

// Observed direction classification
fn observed_direction(&self) -> &'static str {
    let freq_change = self.next_frequency - self.frequency;
    if freq_change > self.frequency * 0.05 { "RISE" }
    else if freq_change < -self.frequency * 0.05 { "FALL" }
    else { "STABLE" }
}
```

---

## VASIL Paper Methodology (What We're Trying to Match)

### VASIL Formula (From Paper)
```
Fitness = α × immune_escape + β × transmissibility
Where:
- α = 0.65 (escape weight)
- β = 0.35 (transmit weight)
- immune_escape = variant-specific antibody escape score
- transmissibility = structural R0 proxy
```

### Key VASIL Insights
1. **Predicts variant dominance**, not week-over-week direction
2. **Uses population-level immunity context** - escape relative to existing immunity
3. **Cross-validation across countries** - not temporal split
4. **Focuses on emergence events** - when a new variant first appears

---

## Optimization Targets (YOUR MISSION)

### Performance Targets
| Metric | Current | Target | Gap |
|--------|---------|--------|-----|
| Test accuracy | 42% | 92% | 50 pp |
| RISE recall | 40% | 90% | 50 pp |
| FALL recall | 45% | 90% | 45 pp |
| Balanced accuracy | 42.5% | 90% | 47.5 pp |

### Hypotheses to Test

#### Hypothesis 1: Wrong Task Definition
- **Current**: Predict RISE/FALL for all lineage-date pairs
- **Alternative**: Predict which lineages will become dominant (>X% share)
- **Test**: Change classification from direction to dominance prediction

#### Hypothesis 2: Wrong Feature Engineering
- **Current**: Absolute escape scores
- **Alternative**: Escape RANK within contemporaneous variants
- **Test**: Replace `escape` with `escape_percentile` at each date

#### Hypothesis 3: Wrong Competition Model
- **Current**: `relative_fitness = escape - weighted_avg_escape`
- **Alternative**: `competitive_advantage = escape / dominant_variant_escape`
- **Test**: Model escape as ratio to current dominant variant

#### Hypothesis 4: Wrong Temporal Window
- **Current**: Week-over-week frequency change
- **Alternative**: Month-over-month or peak detection
- **Test**: Use 4-week horizon instead of 1-week

#### Hypothesis 5: Missing Recombination Signal
- **Current**: Treat lineages independently
- **Alternative**: Flag recombinant lineages (XBB, XBC) with higher fitness priors
- **Test**: Add `is_recombinant` feature

---

## Key Files to Modify

### High Priority
1. **`/mnt/c/Users/Predator/Desktop/PRISM/crates/prism-ve-bench/src/ve_optimizer.rs`**
   - Grid search formula
   - Feature weighting
   - Prediction logic

2. **`/mnt/c/Users/Predator/Desktop/PRISM/crates/prism-ve-bench/src/main.rs`**
   - Batch metadata computation
   - Feature extraction
   - Train/test split logic
   - Classification definition

3. **`/mnt/c/Users/Predator/Desktop/PRISM/crates/prism-ve-bench/src/immunity_model.rs`**
   - Cross-reactivity values
   - Effective escape formula
   - Country-specific immunity calibration

### Medium Priority
4. **`/mnt/c/Users/Predator/Desktop/PRISM/crates/prism-ve-bench/src/gpu_benchmark.rs`**
   - DMS escape score lookup
   - Transmissibility mapping
   - Epitope class weights

5. **`/mnt/c/Users/Predator/Desktop/PRISM/crates/prism-gpu/src/kernels/mega_fused_pocket_kernel.cu`**
   - Stage 8 cycle features
   - Fitness feature computation
   - New feature stages if needed

### Low Priority (Infrastructure)
6. **`/mnt/c/Users/Predator/Desktop/PRISM/crates/prism-ve-bench/src/data_loader.rs`**
   - Data parsing and validation
   - Additional data sources

---

## Build & Test Commands

```bash
# Build release binary
cd /mnt/c/Users/Predator/Desktop/PRISM
PATH="/home/diddy/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin:/usr/bin:$PATH" \
CUDA_HOME=/usr/local/cuda-12.6 cargo build --release -p prism-ve-bench

# Run benchmark
RUST_LOG=error timeout 300 ./target/release/vasil-benchmark

# Check CUDA kernel compilation (if modifying .cu files)
CUDA_HOME=/usr/local/cuda-12.6 LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH \
/usr/local/cuda-12.6/bin/nvcc -ptx -arch=sm_89 \
crates/prism-gpu/src/kernels/mega_fused_pocket_kernel.cu -o target/ptx/mega_fused.ptx
```

---

## Success Criteria

### Tier 1: Acceptable (70%+ accuracy)
- Beat random baseline (50%) by >20 percentage points
- Consistent across all 12 countries
- No overfitting (train/test gap < 10 pp)

### Tier 2: Good (80%+ accuracy)
- Match simple VASIL formula performance
- Identify which features drive predictions
- Demonstrate generalization to Omicron era

### Tier 3: Excellent (90%+ accuracy)
- Match or exceed VASIL paper results
- Novel feature engineering insights
- Publishable methodology improvements

---

## Naming Conventions

### Rust
- Structs: `PascalCase` (e.g., `VEState`, `BatchMetadata`)
- Functions: `snake_case` (e.g., `compute_effective_escape`)
- Constants: `SCREAMING_SNAKE_CASE` (e.g., `TOTAL_COMBINED_FEATURES`)
- Modules: `snake_case` (e.g., `immunity_model`)

### CUDA
- Kernels: `snake_case` (e.g., `mega_fused_pocket_kernel_fp32`)
- Device functions: `snake_case` with `stage` prefix (e.g., `stage8_cycle_features`)
- Constants: `SCREAMING_SNAKE_CASE` with type prefix (e.g., `LIF_TAU_MEMBRANE`)
- Shared memory: `smem->field_name`

### Data
- Countries: PascalCase (e.g., `Germany`, `SouthAfrica`)
- Lineages: Original PANGO format (e.g., `BA.5.2.1`, `XBB.1.5`)
- Dates: `NaiveDate` / ISO format (e.g., `2022-06-01`)

---

## Agent Directives

### DO:
1. Analyze feature distributions before proposing changes
2. Test hypotheses incrementally with measurable metrics
3. Preserve GPU pipeline integrity (7000+ structures/sec throughput)
4. Use cross-validation or multiple temporal splits to validate
5. Document all changes with performance impact

### DON'T:
1. Hardcode VASIL formula coefficients (that's cheating)
2. Use future information in features (no look-ahead bias)
3. Break CUDA kernel compilation
4. Overfit to training data
5. Ignore class imbalance (40% RISE, 60% FALL)

---

## Quick Reference: Key Functions

```rust
// Main entry point
fn main() -> Result<()>  // main.rs:32

// Build batch with all features
fn build_mega_batch(all_data, structure_cache, train_cutoff) -> (PackedBatch, Vec<BatchMetadata>)  // main.rs:284

// Extract features from GPU output
fn extract_raw_features(batch_output, metadata) -> (train_data, test_data)  // main.rs:460

// Grid search for optimal weights
fn train_grid_search(&mut self, data: &[(VEState, &str)])  // ve_optimizer.rs:555

// Prediction function
fn predict_with_weights(&self, state: &VEState) -> VEAction  // ve_optimizer.rs:671

// Immunity-adjusted escape
fn compute_effective_escape(&self, raw_epitope_escape, target_lineage, date, cross_matrix) -> f32  // immunity_model.rs:330
```

---

## Final Notes

The 50 percentage point accuracy gap is NOT a minor tuning problem - it requires fundamental rethinking of the prediction task and feature engineering. The current approach treats escape as an absolute property, but the VASIL paper likely treats it as a relative property in competitive dynamics.

Focus on:
1. **What makes a variant "fit"** - it's not just escape, it's escape RELATIVE to what immunity already exists against
2. **When predictions matter** - emergence events, not steady-state frequency tracking
3. **Why Omicron changes everything** - high baseline escape means escape alone doesn't discriminate

Good luck. The GPU pipeline is solid. The feature engineering needs breakthrough innovation.
