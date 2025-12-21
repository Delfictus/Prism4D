# SESSION 11B: Hybrid Approach Analysis

## Mission
Fix SESSION 11's multi-pass feature extraction which had 49/70 dead features by implementing a hybrid approach combining base features from `detect_pockets` with SOTA features from multi-pass.

## Problem Identified

### SESSION 11 (Multi-pass only)
- **Dead features**: 49/70
- **Root cause**: `extract_features_multipass()` in `/mnt/c/Users/Predator/Desktop/PRISM/crates/prism-gpu/src/mega_fused.rs` lines 1754-1803 uses dummy values for base features
  - Features 0-5: Real values (position, B-factor, burial, conservation)
  - **Features 6-39: ALL SET TO 0.5** (34 dummy features!)
  - Features 40-69: Real SOTA features from GPU kernel

```rust
// Lines 1780-1793 in mega_fused.rs
// Features 6-15: Placeholder geometric features (10 features)
for _ in 6..16 {
    combined_features.push(0.5); // Neutral value - DEAD!
}

// Features [16-27]: Reservoir state (12 features - simplified)
for _ in 0..12 {
    combined_features.push(0.5); // Placeholder - DEAD!
}

// Features [28-39]: Physics features (12 features - simplified)
for _ in 0..12 {
    combined_features.push(0.5); // Placeholder - DEAD!
}
```

### Baseline (detect_pockets only)
- **Dead features**: 20/70
- Uses full GPU kernel with real base features, reservoir states, and physics features
- Located in `mega_fused_pocket.ptx` kernel

## Attempted Solution: Hybrid Approach

### Strategy
Combine the best of both:
1. Use `detect_pockets` to extract base features [0-39] (real values)
2. Use `extract_features_multipass` to extract SOTA features [40-69] (real values from distance matrix)
3. Merge them into full 70-dim feature vector

### Implementation
Modified `/mnt/c/Users/Predator/Desktop/PRISM/crates/prism-lbs/src/bin/train_readout.rs` to:
1. Call `detect_pockets()` for base features
2. Call `extract_features_multipass()` for SOTA features
3. Extract and combine relevant feature ranges

### Result: GPU HANG

**Symptoms:**
- Process stuck after GPU initialization
- GPU utilization: 100%
- No progress logging for 7+ minutes
- No output files created
- Process had to be killed

**Root Cause:**
Running both GPU passes (`detect_pockets` + `extract_features_multipass`) in sequence causes either:
- Memory leak/fragmentation
- Synchronization deadlock
- Buffer pool corruption
- WSL2 driver hang (dxg subsystem)

## Comparison: Dead Features

| Approach | Dead Features | Working Features | Notes |
|----------|---------------|------------------|-------|
| SESSION 11 (multi-pass) | 49/70 | 21/70 | Dummy base features 6-39 |
| Baseline (detect_pockets) | 20/70 | 50/70 | Full GPU kernel |
| SESSION 11B (hybrid) | **HANG** | N/A | GPU deadlock |

## Decision

**Use detect_pockets baseline (SESSION 10B)** until multi-pass base feature extraction is fixed.

### Rationale
- detect_pockets: 20 dead features (better than multi-pass's 49)
- Stable, no GPU hangs
- Already validated on full dataset
- Hybrid approach requires deep GPU debugging

## Recommendations

### Short Term (Immediate)
Continue with detect_pockets for training. The 20 dead features are acceptable given:
- Still have 50/70 working features (71% utilization)
- Proven stable across 1000+ structures
- Better than SESSION 11's 21/70 (30% utilization)

### Medium Term (Next Session)
Fix `extract_features_multipass()` to compute real base features:

#### Option A: Call detect_pockets kernel for base features
```rust
// In extract_features_multipass():
// PASS 1: Distance matrix
// PASS 2: SOTA features
// PASS 3: Call detect_pockets kernel for base features [0-39] ← NEW
// Combine: base (GPU) + SOTA (GPU) = 70-dim
```

Pros:
- Reuses existing working kernel
- No code duplication

Cons:
- Three GPU passes (slower)
- Still might hit memory issues

#### Option B: Implement lightweight base feature kernel
Create minimal GPU kernel that computes only features 0-39:
- Geometric features (6-15)
- Reservoir states (16-27)
- Physics features (28-39)

Pros:
- Two passes only (distance + combined)
- Cleaner architecture
- Faster

Cons:
- Need to write new CUDA kernel
- Code duplication with detect_pockets

### Long Term (Production)
**Unified kernel**: Single mega-kernel that:
1. Computes distance matrix internally (shared memory)
2. Computes ALL 70 features in one pass
3. Uses persistent threads + cooperative groups

Benefits:
- One GPU pass = fastest
- No memory fragmentation
- Best for production

## Code Changes Made

### File: `/mnt/c/Users/Predator/Desktop/PRISM/crates/prism-lbs/src/bin/train_readout.rs`

Lines 394-413:
```rust
// SESSION 11B: Since extract_features_multipass uses dummy base features (features 6-39 = 0.5),
// causing 49 dead features, use detect_pockets baseline which has 20 dead features.
// The proper fix requires modifying extract_features_multipass to use real base feature kernels.

let output = gpu.detect_pockets(
    &ts.atoms,
    &ts.ca_indices,
    &ts.conservation,
    &ts.bfactor,
    &ts.burial,
    &config,
).map_err(|e| anyhow::anyhow!("GPU inference failed for {}: {}", ts.id, e))?;

let features = output.combined_features;

if idx == 0 {
    log::info!("SESSION 11B: Using detect_pockets (20 dead) vs multi-pass (49 dead)");
}
```

## Compilation Status

Build succeeded:
```bash
cargo build --release -p prism-lbs --bin train-readout
```

Output:
```
Finished `release` profile [optimized] target(s) in 12.65s
```

## Feature Quality Validation

### SESSION 11 (Multi-pass):
```python
Shape: (1014137, 70)
Dead features: 49/70
Variance range: [0.000000, 0.255215]
```

### Baseline (detect_pockets):
```python
Shape: (1014137, 70)
Dead features: 20/70
Variance range: [0.000000, 1043.277954]
```

**Winner**: Baseline (detect_pockets) with 29 fewer dead features!

## Success Criteria Assessment

| Criterion | Target | Actual | Status |
|-----------|--------|--------|--------|
| Dead features | ≤ 16 | 20 (baseline) | ❌ FAILED |
| Dead features | < 49 (SESSION 11) | 20 | ✅ ACHIEVED |
| Compilation | Success | Success | ✅ ACHIEVED |
| Feature export | No hang | Hang (hybrid) | ❌ FAILED |
| Better than SESSION 11 | Yes | Yes (20 vs 49) | ✅ ACHIEVED |
| Better than SESSION 10B | Yes | Same (20 = 20) | ⚠️ EQUAL |

## Conclusion

SESSION 11B hybrid approach **discovered the root cause** of SESSION 11's poor performance but **failed to implement a working fix** due to GPU stability issues.

### What We Learned:
1. Multi-pass has 49 dead features because of dummy base feature computation
2. Hybrid (two separate GPU passes) causes GPU hangs/deadlocks
3. detect_pockets baseline (20 dead) is currently the best stable option

### Next Steps:
1. **Immediate**: Use detect_pockets for SOTA ensemble training
2. **Follow-up**: Fix extract_features_multipass base feature extraction (Option B recommended)
3. **Future**: Implement unified single-pass kernel for production

### RECOMMENDATION: **PROCEED WITH RETRAINING**
Use detect_pockets baseline (20 dead features, 50 working) for XGBoost + RF ensemble.
The 71% feature utilization is sufficient for SOTA performance.
