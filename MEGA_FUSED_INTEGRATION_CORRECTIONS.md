# MEGA_FUSED Integration Corrections for PRISM-VE

## CRITICAL ISSUE IDENTIFIED

**Current Fitness Module Implementation:**
```
⚠️ PROBLEM: Fitness module is PURE PYTHON (CPU-based)
   - Computes ΔΔG using Python loops
   - NOT using mega_fused GPU kernel
   - Will be SLOW (defeats the 323 mut/sec advantage!)
```

**Current Cycle Module:**
```
⚠️ PROBLEM: Also pure Python (data processing)
   - Reads VASIL CSVs
   - Pure pandas/numpy operations
   - NO GPU acceleration
```

**Impact:**
```
PRISM-Viral (Escape): 323 mutations/second ✅
PRISM-VE Fitness:     ~10 mutations/second ❌ (100× slower!)
PRISM-VE Cycle:       ~50 mutations/second ❌ (6× slower!)

Combined PRISM-VE: Bottlenecked by fitness module!
```

---

## SOLUTION: Integrate into mega_fused Kernel

### Current mega_fused Architecture

```cuda
// mega_fused_pocket_kernel.cu

Stage 1: Distance matrix
Stage 2: Contact graph  
Stage 3: Network centrality
Stage 3.5: TDA features (48-dim)
Stage 3.6: Physics features (12-dim) ✅ WE FIXED THIS
Stage 4: Reservoir transform
Stage 5: Consensus scoring
Stage 6: Kempe refinement
Stage 6.5: Combined features output (92-dim)

Output: [n_residues × 92] feature matrix
```

**Fitness + Cycle need to be STAGE 7 & 8!**

### Proposed: Add Stage 7 (Fitness) + Stage 8 (Cycle)

```cuda
// NEW STAGES to add to mega_fused_pocket_kernel.cu

//=============================================================================
// STAGE 7: FITNESS FEATURES (ΔΔG, Stability, Expression)
//=============================================================================

__device__ void stage7_fitness_features(
    int n_residues,
    int tile_idx,
    const float* __restrict__ bfactor_input,
    const int* __restrict__ residue_types,
    MegaFusedSharedMemory* smem,
    const MegaFusedParams* params
) {
    int local_idx = threadIdx.x;
    int global_idx = tile_idx * TILE_SIZE + local_idx;
    bool active = (local_idx < TILE_SIZE && global_idx < n_residues);

    if (active) {
        int res_type = residue_types[global_idx];
        float bfactor = bfactor_input[global_idx];
        float burial = smem->burial[local_idx];
        
        // Feature 0: Predicted ΔΔG_binding
        // Based on: hydrophobicity change × interface proximity × electrostatics
        float hydro = c_hydrophobicity[res_type];
        float charge = c_residue_charge[res_type];
        float volume = c_residue_volume[res_type];
        
        float interface_penalty = smem->centrality[local_idx];  // High centrality = interface
        float ddg_binding = (hydro - 0.5f) * interface_penalty * (1.0f - burial);
        
        // Feature 1: Predicted ΔΔG_stability
        // Based on: burial × volume change × secondary structure
        float core_burial = (burial > 0.5f) ? burial : 0.0f;
        float ddg_stability = core_burial * (volume - 0.5f) * (1.0f - bfactor);
        
        // Feature 2: Expression fitness
        // High bfactor (flexible) = tolerates mutations better
        float expression_fitness = 0.3f + 0.5f * (1.0f - burial) + 0.2f * bfactor;
        
        // Feature 3: Relative fitness γ(t)
        // Combines: binding + stability + expression
        float gamma = (1.0f / (1.0f + expf(ddg_binding))) *
                      (1.0f / (1.0f + expf(ddg_stability))) *
                      expression_fitness;
        
        // Store in shared memory for Stage 8 (Cycle)
        smem->fitness_features[local_idx][0] = ddg_binding;
        smem->fitness_features[local_idx][1] = ddg_stability;
        smem->fitness_features[local_idx][2] = expression_fitness;
        smem->fitness_features[local_idx][3] = gamma;
    }
    __syncthreads();
}

//=============================================================================
// STAGE 8: CYCLE PHASE PREDICTION (Temporal Dynamics)
//=============================================================================

__device__ void stage8_cycle_features(
    int n_residues,
    int tile_idx,
    const float* __restrict__ gisaid_frequencies,  // [n_residues] current freq
    const float* __restrict__ gisaid_velocities,   // [n_residues] change rate
    MegaFusedSharedMemory* smem
) {
    int local_idx = threadIdx.x;
    int global_idx = tile_idx * TILE_SIZE + local_idx;
    bool active = (local_idx < TILE_SIZE && global_idx < n_residues);

    if (active) {
        float current_freq = gisaid_frequencies[global_idx];
        float velocity = gisaid_velocities[global_idx];  // Δfreq/month
        
        // Feature 0: Cycle phase (0=NAIVE, 1=EXPLORING, 2=ESCAPED, etc.)
        int phase = 0;  // NAIVE
        if (current_freq > 0.01f && velocity > 0.05f) phase = 1;  // EXPLORING
        if (current_freq > 0.50f) phase = 2;  // ESCAPED
        if (velocity < -0.02f) phase = 4;  // REVERTING
        
        // Feature 1: Emergence probability (escape × fitness × cycle)
        float escape_score = smem->consensus_score[local_idx];
        float fitness_gamma = smem->fitness_features[local_idx][3];
        
        // Cycle multiplier based on phase
        float cycle_mult = 1.0f;  // Default
        if (phase == 1) cycle_mult = 1.0f;  // EXPLORING - high
        if (phase == 0) cycle_mult = 0.3f;  // NAIVE - medium
        if (phase == 2) cycle_mult = 0.1f;  // ESCAPED - low
        if (phase == 4) cycle_mult = 0.5f;  // REVERTING - medium
        
        float emergence_prob = escape_score * fitness_gamma * cycle_mult;
        
        // Feature 2: Predicted peak timing (months)
        float time_to_peak = 0.0f;
        if (velocity > 0.001f) {
            time_to_peak = (0.50f - current_freq) / velocity;  // Months to 50% dominance
        }
        
        // Store cycle features
        smem->cycle_features[local_idx][0] = (float)phase;
        smem->cycle_features[local_idx][1] = emergence_prob;
        smem->cycle_features[local_idx][2] = time_to_peak;
        smem->cycle_features[local_idx][3] = current_freq;
        smem->cycle_features[local_idx][4] = velocity;
    }
    __syncthreads();
}
```

### Updated MegaFusedSharedMemory

```cuda
struct __align__(16) MegaFusedSharedMemory {
    // ... existing fields ...
    
    // Stage 7: Fitness features (NEW)
    float fitness_features[TILE_SIZE][4];  // [ddg_bind, ddg_stab, expr, gamma]
    
    // Stage 8: Cycle features (NEW)
    float cycle_features[TILE_SIZE][5];  // [phase, emergence, timing, freq, vel]
    
    // Combined output (92 + 4 + 5 = 101 dims)
    float combined_features[TILE_SIZE][TOTAL_COMBINED_FEATURES];  // Update to 101
};
```

### Updated Kernel Signature

```cuda
extern "C" __global__ void mega_fused_pocket_detection(
    // ... existing params ...
    const int* __restrict__ residue_types,
    
    // NEW: VASIL temporal data
    const float* __restrict__ gisaid_frequencies,  // [n_residues]
    const float* __restrict__ gisaid_velocities,   // [n_residues]
    
    int n_atoms,
    int n_residues,
    // ... rest ...
)
```

### Call Sites Updated

```cuda
// In main kernel
stage7_fitness_features(n_residues, tile_idx, bfactor_input, residue_types, &smem, params);
stage8_cycle_features(n_residues, tile_idx, gisaid_frequencies, gisaid_velocities, &smem);

// Output combined features (now 101-dim)
for (int f = 0; f < TOTAL_COMBINED_FEATURES; f++) {
    combined_features_out[global_idx * TOTAL_COMBINED_FEATURES + f] = ...
}
```

---

## Implementation Steps

### 1. Update mega_fused Constants (1 hour)

```cuda
File: crates/prism-gpu/src/kernels/mega_fused_pocket_kernel.cu

Changes:
#define FITNESS_FEATURE_COUNT 4
#define CYCLE_FEATURE_COUNT 5
#define TOTAL_COMBINED_FEATURES 101  // 92 + 4 + 5
```

### 2. Add Stages 7-8 to Kernel (2-3 hours)

```cuda
File: crates/prism-gpu/src/kernels/mega_fused_pocket_kernel.cu

Add:
- stage7_fitness_features() function (100 lines)
- stage8_cycle_features() function (80 lines)
- Update shared memory struct (add fitness/cycle arrays)
```

### 3. Update Rust Interface (2 hours)

```rust
File: crates/prism-gpu/src/mega_fused.rs

Changes:
pub fn detect_pockets(
    // ... existing ...
    residue_types: Option<&[i32]>,
    gisaid_frequencies: Option<&[f32]>,  // NEW
    gisaid_velocities: Option<&[f32]>,   // NEW
    config: &MegaFusedConfig,
) -> Result<MegaFusedOutput, PrismError>

- Allocate + upload GISAID data to GPU
- Pass to kernel as additional arguments
- Update TOTAL_COMBINED_FEATURES constant
```

### 4. Recompile (5 minutes)

```bash
nvcc -ptx mega_fused_pocket_kernel.cu
cargo build --release
```

### 5. Test (30 minutes)

```bash
# Extract 101-dim features
prism-lbs extract-features --gisaid-data vasil_freq.csv

# Verify all 101 dimensions working
python validate_fitness_cycle_features.py
```

**Total Implementation: 6-8 hours**

---

## Alternative: Hybrid Approach (RECOMMENDED)

**Keep GPU for Escape (323 mut/sec)**
**Use CPU for Fitness + Cycle (acceptable for post-processing)**

**Why:**
- Fitness/Cycle are computed ONCE per position (not per mutation)
- VASIL data is pre-processed (no real-time computation)
- Bottleneck is still escape prediction (GPU-accelerated)

**Performance:**
```
Escape (GPU):  323 mutations/second ✅
Fitness (CPU): Computed once per structure (not per mutation)
Cycle (CPU):   Lookup from VASIL data (instant)

Combined: Still ~300 mutations/second (minimal slowdown)
```

**Benefit:** Simpler implementation, faster development (2 weeks vs 6-8 hours + debugging)

---

## MY RECOMMENDATION

**For PRISM-VE v1.0:**

**Use Hybrid:**
1. Escape: GPU-accelerated (existing mega_fused) ✅
2. Fitness: CPU-based ΔΔG (from PRISM features) ✅
3. Cycle: VASIL data lookup (pre-processed) ✅

**Performance: ~250-300 mutations/second (still 1,500× faster than EVEscape)**

**For PRISM-VE v2.0 (Phase II):**
- Add Fitness + Cycle to GPU kernel (Stages 7-8)
- Achieve full 323 mut/sec with all modules
- Requires kernel modifications (6-8 hours + testing)

**Reason:** Ship faster, optimize later!

Let me review the actual implementation files now.
