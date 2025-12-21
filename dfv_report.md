# Data Flow Validator (DFV) Report

**Date**: 2025-12-19  
**Pipeline**: PRISM-VE VASIL Benchmark (14,917 structures)  
**Status**: ⚠️ ISSUES FOUND - FIXES IN PROGRESS

---

## Executive Summary

The DFV scan identified the root cause of the constant `spike_momentum = 0.500` issue. **It is NOT a nullptr issue** as previously hypothesized. The data flow is correct - the problem is LIF neuron threshold calibration combined with insufficient input signal variance.

### Key Findings

| Issue | Severity | Status |
|-------|----------|--------|
| LIF thresholds too high | HIGH | ✅ FIXED (staggered 0.15-0.90) |
| Spike momentum formula default | MEDIUM | ⚠️ BY DESIGN |
| Metadata propagation gaps | MEDIUM | ⚠️ 11 fields not reaching GPU |
| Feature dimension mismatch | LOW | 📋 DOCUMENTED |

---

## Data Flow Verification

### ✅ Stage 8: Frequency/Velocity Buffer Flow

**Status: VERIFIED WORKING**

```
BatchMetadata.frequency_velocity (prism-ve-bench/main.rs)
    ↓
StructureMetadata { frequency, velocity } (mega_fused_batch.rs:165)
    ↓
PackedBatch.frequencies_packed/velocities_packed (mega_fused_batch.rs:509-510)
    ↓
d_frequencies/d_velocities GPU buffers (mega_fused_batch.rs:920-923)
    ↓
memcpy_htod upload (mega_fused_batch.rs:1078-1081)
    ↓
Kernel arg: frequencies_packed/velocities_packed (mega_fused_batch.rs:1220-1223)
    ↓
batch_stage8_cycle_features_v2() receives struct_freq/struct_vel (mega_fused_batch.cu:2638-2640)
```

**Verification Debug Output** (from logs):
```
[DEBUG] Starting Stage 8 uploads...
[DEBUG] Uploading 14917 frequencies, 14917 velocities
[DEBUG] Stage 8 uploads OK
```

### ✅ Cycle Feature Computation

**File**: `mega_fused_batch.cu:1539-1604`

The cycle features compute correctly:
- F96: `phase` (0-5 cycle stage)
- F97: `emergence_prob` = escape × transmit × cycle_mult
- F98: `time_to_peak` = (0.5 - freq) / velocity
- F99: `freq_normalized` = min(current_freq, 1.0)
- F100: `velocity_normalized` = clamp(velocity, -0.5, 0.5)

### ⚠️ Stage 8.5: LIF Spike Issue Analysis

**Root Cause**: The LIF neuron simulation works correctly, but the **spike momentum formula** has a **design default of 0.5** when no spikes occur.

**Formula** (mega_fused_batch.cu:1742-1746):
```cuda
float early = (spikes[0] + spikes[1] > 0) ? 1.0f : 0.0f;
float late = (spikes[6] + spikes[7] > 0) ? 1.0f : 0.0f;
smem->spike_features[local_idx][5] = (late - early + 1.0f) * 0.5f;
```

**Behavior**:
| early | late | Result |
|-------|------|--------|
| 0 | 0 | (0-0+1)*0.5 = **0.500** |
| 1 | 0 | (0-1+1)*0.5 = 0.000 |
| 0 | 1 | (1-0+1)*0.5 = 1.000 |
| 1 | 1 | (1-1+1)*0.5 = 0.500 |

**Why spikes=0 for most structures**:

The LIF thresholds are now **staggered** (0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.90), but the input current `I` depends on cycle features which are often **low variance** when:

1. Most structures have `velocity ≈ 0` (stable frequencies)
2. Most structures have similar `frequency` values (Omicron-dominated era)
3. `escape_score` is saturated at ~0.97 for Omicron variants

**Input Current Formula** (mega_fused_batch.cu:1691-1695):
```cuda
float I = 0.0f;
for (int f = 0; f < 5; f++) {
    I += cycle_inputs[f] * c_spike_input_weights[n * 5 + f];
}
```

With weights summing to 1.0 per neuron and normalized inputs [0,1], the typical `I ≈ 0.3-0.5` which only exceeds the lowest threshold (0.15) occasionally.

---

## Metadata Propagation Analysis

### Rich Metadata Available (prism-ve-bench/main.rs: BatchMetadata)

| Field | Type | Reaches GPU? |
|-------|------|--------------|
| country | String | ❌ NO |
| lineage | String | ❌ NO |
| date | NaiveDate | ❌ NO |
| date_idx | usize | ❌ NO |
| lineage_idx | usize | ❌ NO |
| frequency | f32 | ✅ YES |
| next_frequency | f32 | ❌ NO |
| frequency_velocity | f32 | ✅ YES |
| escape_score | f32 | ❌ NO (per-structure) |
| epitope_escape | [f32; 10] | ✅ YES (per-residue) |
| effective_escape | f32 | ❌ NO |
| transmissibility | f32 | ❌ NO |
| relative_fitness | f32 | ❌ NO |
| is_train | bool | ❌ NO (used in Rust) |

### What GPU Kernels Actually Receive

1. **Per-structure scalars**: frequency, velocity
2. **Per-residue arrays**: epitope_escape[10], PK immunity levels[75]
3. **Global constants**: current_day, variant_family_idx

---

## Recommendations

### 1. ✅ LIF Threshold Calibration (ALREADY FIXED)

The thresholds were updated to a staggered distribution (0.15-0.90) which provides neuron diversity. The first neuron (N0) with threshold 0.15 should spike for high-velocity variants.

### 2. ⚠️ Alternative Spike Momentum Formula

Consider modifying the formula to avoid the 0.5 default:

```cuda
// Option A: Continuous formula (no binary spike dependency)
float momentum = (cycle_inputs[0] - 0.5f) * 2.0f;  // [-1, 1] from velocity

// Option B: Smooth sigmoid instead of binary
float early_smooth = 1.0f / (1.0f + expf(-10.0f * (spikes[0] + spikes[1] - 0.5f)));
float late_smooth = 1.0f / (1.0f + expf(-10.0f * (spikes[6] + spikes[7] - 0.5f)));
```

### 3. 📋 Extend Metadata Propagation

Add more metadata to GPU kernels:
- `transmissibility` - Key for RISE prediction
- `effective_escape` - Pre-computed immunity-modulated escape
- `relative_fitness` - Competition context

### 4. 📋 Feature Variance Validation

Add runtime check in `execute_batch()`:

```rust
// After downloading features, check for constant values
for feature_idx in 101..109 {  // Spike features
    let values: Vec<f32> = features.iter()
        .skip(feature_idx).step_by(136).collect();
    let variance = compute_variance(&values);
    if variance < 1e-6 {
        log::warn!("DFV-002: Feature {} has zero variance", feature_idx);
    }
}
```

---

## Verified Buffer Allocations

All buffers properly allocated with `alloc_zeros` pattern:

| Buffer | Size | Status |
|--------|------|--------|
| d_atoms | total_atoms × 3 | ✅ |
| d_ca_indices | total_residues | ✅ |
| d_conservation | total_residues | ✅ |
| d_bfactor | total_residues | ✅ |
| d_burial | total_residues | ✅ |
| d_residue_types | total_residues | ✅ |
| d_frequencies | n_structures | ✅ |
| d_velocities | n_structures | ✅ |
| d_epitope_escape | total_residues × 10 | ✅ |
| d_combined_features | total_residues × 136 | ✅ |
| d_p_neut_time_series_75pk | n_countries × 75 × 86 | ✅ (conditional) |
| d_current_immunity_levels_75 | n_structures × 75 | ✅ (conditional) |

---

## Integrity Status

### ✅ No Scientific Fraud Detected

- Train/test split is temporal (date-based)
- No future data leakage in feature computation
- VASIL coefficients from paper are documented as reference
- All GPU features computed from raw inputs

### ⚠️ Engineering Issues (Not Integrity Violations)

- Spike features have low discrimination power due to threshold calibration
- Some metadata fields not propagated to GPU
- Feature dimension documentation inconsistent (125 vs 136)

---

## Action Items

1. **[DONE]** LIF thresholds calibrated to staggered distribution
2. **[PENDING]** Consider spike momentum formula modification
3. **[PENDING]** Add feature variance validation in Rust wrapper
4. **[PENDING]** Extend metadata propagation for transmissibility/fitness
5. **[PENDING]** Update feature documentation to reflect 136-dim layout

---

## Files Modified/Reviewed

| File | Lines | Status |
|------|-------|--------|
| `mega_fused_batch.cu` | 546-548 | ✅ Thresholds updated |
| `mega_fused_batch.rs` | 1067-1082 | ✅ Verified data upload |
| `mega_fused_batch.cu` | 2638-2640 | ✅ Verified kernel receives data |
| `prism_ve_model.rs` | - | 📋 Review feature dimension |

---

**Report Generated**: 2025-12-19  
**DFV Version**: 1.0  
**Pipeline Version**: PRISM-4D VE v0.1.0
