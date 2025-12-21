# VASIL Gamma Envelope Bug Diagnostic Report

## Date: December 20, 2025

## Executive Summary

**CRITICAL BUG FOUND**: The gamma computation has ZERO correlation with actual frequency direction (48.5% accuracy = random guessing).

This is the root cause of the 97.8% "undecided" rate that blocks VASIL benchmark accuracy.

---

## Diagnostic Evidence

### Gamma Cache Analysis (21,881 samples)

| Metric | Value | Implication |
|--------|-------|-------------|
| gamma_min range | [-0.43, +0.06] | Mostly negative |
| gamma_max range | [+0.06, +0.49] | **ALWAYS positive** |
| gamma_mean range | [-0.15, +0.17] | Centered near zero |
| Envelope width | 0.18 - 0.69 (mean: 0.42) | Too wide |

### Classification Results

| Category | Count | Percentage |
|----------|-------|------------|
| RISE (both γ > 0) | 481 | 2.2% |
| FALL (both γ < 0) | 0 | 0.0% |
| **UNDECIDED** | 21,400 | **97.8%** |

### Ground Truth Distribution

| Direction | Count | Percentage |
|-----------|-------|------------|
| RISE (+1) | 8,834 | 40.4% |
| FALL (-1) | 13,047 | 59.6% |

---

## Critical Finding: Gamma Has No Predictive Power

### Test 1: gamma_mean as classifier
```
If gamma_mean > 0 → predict RISE
If gamma_mean < 0 → predict FALL

Result: 48.5% accuracy (random chance = 50%)
```

### Test 2: Inverted gamma as classifier
```
If -gamma_mean > 0 → predict RISE
If -gamma_mean < 0 → predict FALL

Result: 51.5% accuracy (still random)
```

### Confusion Matrix (gamma_mean classifier)
```
                    Predicted
                 RISE      FALL
Actual  RISE    4,788     4,046    (54.2% correct)
        FALL    7,215     5,832    (44.7% correct)
```

**Conclusion**: The gamma values are completely uncorrelated with actual frequency changes.

---

## Root Cause Analysis

### Suspect 1: Hardcoded 0.5 Fallbacks

Found in `gamma_envelope_reduction.cu`:
- Line 205: `d_weighted_avg_75pk[...] = population * 0.5;` (when date out of range)
- Line 240: `weighted_avg_s = population * 0.5;` (when no active variants)

Found in `vasil_exact_metric.rs`:
- Line 739: `self.population * 0.5` (get_susceptible fallback)
- Line 1363: `landscape.population * 0.5` (weighted_avg fallback)

**Impact**: If these fallbacks are triggered frequently, gamma degenerates to:
```
gamma = S_y / (pop * 0.5) - 1 = 2 * S_y / pop - 1
```
This makes gamma depend only on S_y (target variant susceptibility), not on relative fitness.

### Suspect 2: Frequency Data Missing

The GPU kernel at line 218 skips variants with `freq_x < 1e-9`:
```cuda
if (freq_x < 1e-9) continue;  // Skip inactive variants
```

If most frequency data is zero/missing, the fallback is triggered.

### Suspect 3: Immunity Values Uniform

The gamma range [-0.15, +0.17] suggests:
- Immunity values are varying (otherwise gamma = 0)
- But variation doesn't correlate with actual variant fitness
- Could indicate immunity model is computing wrong values

---

## VASIL Formula Reference

From Extended Data Fig 6a:
```
γy(t) = E[Sy(t)] / ⟨Sx(t)⟩_freq - 1

Where:
- Sy(t) = Population - Immunity_y(t)  (susceptibility)
- ⟨Sx(t)⟩_freq = Σ(freq_x × Sx(t)) / Σ(freq_x)  (weighted average)
```

Interpretation:
- γ > 0 → variant y has MORE susceptible hosts than average → RISE
- γ < 0 → variant y has FEWER susceptible hosts than average → FALL

---

## Files Involved

| File | Role |
|------|------|
| `crates/prism-gpu/src/kernels/gamma_envelope_reduction.cu` | GPU kernel for gamma computation |
| `crates/prism-ve-bench/src/vasil_exact_metric.rs` | VasilMetricComputer, ImmunityLandscape |
| `crates/prism-ve-bench/examples/train_gpu_fluxnet_100ep.rs` | Gamma cache creation |
| `isolated_path_b/gamma_cache/` | Cached gamma data (21,881 samples) |

---

## Recommended Next Steps

### Immediate (Debug)
1. Add logging to count fallback frequency
2. Dump sample immunity values to verify they're biologically meaningful
3. Verify frequency data is actually populated for circulating variants

### Investigation
4. Compare with original VASIL Python implementation
5. Check if epitope escape data is being loaded correctly
6. Verify IC50 values are actually affecting immunity computation

### Fix Candidates
- Replace 0.5 fallback with actual immunity-based computation
- Fix frequency data loading pipeline
- Ensure immunity model matches VASIL paper exactly

---

## Session Context

- Previous session applied per-PK weighted_avg fix
- Per-PK fix was implemented correctly but didn't solve the problem
- The issue is upstream: the immunity/gamma values themselves are wrong
- Undecided rate went from 94.1% to 97.8% (got worse, not better)

---

## Key Insight

The gamma envelope is always crossing zero because:
1. gamma_max is ALWAYS positive (100% of samples)
2. gamma_min is almost always negative (97.8%)

This is **mathematically guaranteed** by the fallback behavior: if weighted_avg = 0.5 * pop, then:
- Low immunity → high S_y → gamma > 0
- High immunity → low S_y → gamma < 0

The 75 PK combinations create variation in S_y, ensuring the envelope always spans both positive and negative values.

**FIX REQUIRED**: Compute weighted_avg correctly using actual frequency data, not fallbacks.
