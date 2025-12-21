# Quick PATH A Implementation - Pragmatic Approach

## Current Situation

**Challenge:** Full PATH A implementation (separate function) would take 2-3 hours  
**Timeline:** Need to demonstrate progress quickly  
**Solution:** Use **uniform epitope weights** as a first test with minimal code changes

## Pragmatic Strategy

### Option 1: Minimal Test (30 minutes) ⭐ RECOMMENDED
**Goal:** Validate that epitope extraction is working correctly

**Approach:**
1. PATH B already extracts 11-epitope vectors (line 831-841)
2. Current immunity calculation uses these via on-the-fly P_neut
3. **The epitope data is already being used!**

**Test:** Re-run PATH B and analyze epitope diagnostic output

**Expected:** Same 79.4% (validates epitope extraction is correct)

**Next:** Implement true epitope-distance P_neut kernel integration

### Option 2: Quick Calibration Test (1 hour)
**Goal:** Test if better epitope weighting improves accuracy

**Approach:**
1. Modify PATH B's on-the-fly P_neut to use weighted epitope distance
2. Add weights parameter to existing kernel
3. Test with uniform vs calibrated weights

**Expected:** Small improvement (~2-5%) over PATH B

### Option 3: Full PATH A (2-3 hours)
**Goal:** Complete implementation as designed

**Approach:**
1. Create separate `build_for_landscape_gpu_path_a()` function
2. Precompute P_neut matrix using epitope kernel
3. Use matrix-based immunity computation

**Expected:** 85-90% accuracy target

## Recommendation: START WITH OPTION 1

**Reason:** The epitope extraction is ALREADY DONE and being used in PATH B!

Let's verify the current implementation is using epitopes correctly, then optimize from there.

## What's Actually Happening in PATH B

Looking at `prism_immunity_onthefly.cu`, the kernel computes:

```cuda
__device__ float compute_p_neut_onthefly(
    const float* escape_x,  // [11] epitope escapes
    const float* escape_y,  // [11] epitope escapes
    int delta_t,
    int pk_idx
) {
    // Uses epitope escapes to compute fold resistance
    for (int e = 0; e < N_EPITOPES; e++) {
        fold_res = (1.0f + escape_y[e]) / (1.0f + escape_x[e]);
        // ... PK calculation
    }
}
```

**THIS IS ALREADY EPITOPE-BASED!**

The difference between PATH B and PATH A is:
- **PATH B:** Uses epitopes + PK pharmacokinetics (75 combinations)
- **PATH A:** Uses epitopes + calibrated weights (no PK dimension)

## The Real PATH A Enhancement

PATH A should:
1. Remove PK dimension entirely
2. Use weighted epitope distance: `d² = Σ w_e (x_e - y_e)²`
3. Apply Gaussian kernel: `P_neut = exp(-d² / 2σ²)`

This is simpler than PATH B (no PK grid), potentially more accurate (calibrated weights).

## Revised Implementation Plan

### Phase 1: Verify Current Epitope Usage (10 min)
```bash
# Check diagnostic output from PATH B test
grep "epitope" validation_results/path_b_fixed_grid.log | head -20
```

### Phase 2: Add Epitope Weights to On-the-Fly Kernel (30 min)
Modify `prism_immunity_onthefly.cu` to accept epitope weights:

```cuda
__device__ float compute_p_neut_epitope_weighted(
    const float* escape_x,
    const float* escape_y,
    const float* epitope_weights,  // [11] - NEW
    float sigma                      // NEW
) {
    float d_squared = 0.0f;
    for (int e = 0; e < N_EPITOPES; e++) {
        float diff = escape_x[e] - escape_y[e];
        d_squared += epitope_weights[e] * diff * diff;
    }
    return expf(-d_squared / (2.0f * sigma * sigma));
}
```

### Phase 3: Test with Uniform Weights (10 min)
```rust
let epitope_weights = vec![1.0f32; 11];
let sigma = 0.5f32;
```

### Phase 4: Test with Calibrated Weights (1 hour)
- Simple grid search or manual tuning
- Test different weight combinations
- Target: beat 79.4% baseline

## Bottom Line

**PATH B is already 90% of PATH A** - it uses epitope escapes!

The enhancement is:
1. Remove PK dimension (simplify)
2. Add calibrated epitope weights (optimize)
3. Use Gaussian kernel instead of PK decay

This can be done as a **modification to PATH B** rather than a completely separate implementation.

## Immediate Next Step

**DECISION POINT:** 

Do you want to:

**A) Quick Win (30 min):** Modify on-the-fly kernel to add epitope weights, test with uniform weights

**B) Full Implementation (2-3 hours):** Complete PATH A as separate function with precomputed P_neut matrix

**C) Analysis First (10 min):** Verify what PATH B is actually doing with epitopes before proceeding

**Recommendation: Option A** - Fast iteration, validates approach, gets us to testing quickly
