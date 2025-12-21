# Phase 1 Temporal Integration - Failure Diagnosis

## Results: 50.1% Accuracy (WORSE Than Baseline!)

**Expected**: 84% (mechanistic forward simulation)
**Actual**: 50.1% (random guessing)
**VE-Swarm**: 58.1% (structural features only)

## Possible Root Causes:

### 1. Over-Accumulated Immunity (Most Likely)
**Hypothesis**: Integration sums too much immunity → everyone "immune" → all gamma_y negative → predicts all FALL

**Evidence needed**:
- Distribution of gamma_y values
- Are they all negative?
- Or all clustered around zero?

### 2. Incidence Estimation Wrong
**Current**: Returns constant 5000.0
**Problem**: Should vary by date/country based on actual infection waves

**Fix**: Use phi-corrected total frequency as proxy:
```rust
let total_freq_sum: f32 = frequencies.sum_across_lineages(date);
let incidence = (total_freq_sum / phi) * 50000.0;  // Scale to population
```

### 3. Fold-Resistance Formula Backwards
**Current**: `FR = exp(escape_y - escape_x)`
**Problem**: Might have sign wrong

**Fix**: Check VASIL paper Methods section formula

### 4. Population Size Scaling
**Current**: Population in millions, immunity in absolute counts
**Problem**: Might need normalization

**Fix**: Work in fractions:
```rust
let immune_fraction = accumulated_immunity / population;
let susceptible_fraction = 1.0 - immune_fraction;
```

## Recommended Debug Steps:

1. Add logging to gamma computation:
   - Log first 10 gamma_y values
   - Log min/max/mean gamma_y
   - Log susceptible counts

2. Simplify to single-country test (Germany only)

3. Compare our P_neut to VASIL's Figure 2d

4. Validate fold-resistance matches their Extended Data Fig 1

## Next Actions:

- [ ] Add debug logging to compute_gamma()
- [ ] Run on Germany only (fast iteration)
- [ ] Compare P_neut(Wuhan→Delta, 100 days) to paper Fig 2d
- [ ] Fix formula if mismatch found
- [ ] Re-test

**Expected time to fix**: 1-2 hours
**Expected accuracy after fix**: 75-84%
