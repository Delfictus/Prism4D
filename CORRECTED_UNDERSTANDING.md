# ✅ VASIL Benchmark Protocol - CORRECTED

## Critical Clarification

**What I Got Wrong**:
❌ Testing single mutations (E484K, N501Y...)
❌ Comparing escape probabilities
❌ Structure-level predictions

**What VASIL Actually Tests**:
✅ **Lineage dynamics** (BA.5, BQ.1.1, XBB.1.5 - whole variants)
✅ **Weekly predictions**: Will lineage RISE or FALL?
✅ **Time series**: 52 weeks × 12 countries = ~30,000 predictions
✅ **Accuracy metric**: % correct rise/fall classifications

---

## VASIL's Actual Test

### Input (Each Week):
```
Date: 2022-10-01
Country: Germany
Lineages above 1% frequency:
  - BA.5: 65%
  - BQ.1.1: 8%
  - BA.2.75: 5%
  - BF.7: 3%
  - XBB: 1.5%
  (10-20 lineages per week)
```

### VASIL Predicts:
```
For each lineage, predict γ (growth rate):
  BA.5: γ = -0.05 → FALL
  BQ.1.1: γ = +0.15 → RISE
  BA.2.75: γ = -0.02 → FALL
  BF.7: γ = +0.08 → RISE
  XBB: γ = +0.12 → RISE
```

### Validation (7 days later):
```
Observed (2022-10-08):
  BA.5: 58% (−7%) → FALL ✅ Correct
  BQ.1.1: 12% (+4%) → RISE ✅ Correct
  BA.2.75: 4% (−1%) → FALL ✅ Correct
  BF.7: 4% (+1%) → RISE ✅ Correct
  XBB: 1.8% (+0.3%) → RISE ✅ Correct

Week accuracy: 5/5 = 100%
```

### Aggregated Over Time:
```
52 weeks × ~15 lineages/week = ~780 predictions per country
Germany: 733/780 correct = 0.940 accuracy
Mean across 12 countries: 0.920 accuracy
```

---

## How PRISM-VE Should Test

### Correct Workflow:

```python
# For each week
for week_date in weekly_dates:
    
    # Get lineages >1% frequency
    active_lineages = get_lineages_above_threshold(week_date, threshold=0.01)
    
    for lineage in active_lineages:
        # Get lineage's mutation profile
        mutations = get_lineage_mutations(lineage)  # e.g., ["E484A", "F486V", ...]
        
        # PRISM-VE Prediction:
        # Option A (Full): Run mega_fused on lineage structure → extract gamma (feature 95)
        # Option B (Proxy): Compute gamma from escape + fitness + cycle
        
        # For each mutation in lineage:
        escape_scores = [prism_escape.predict(mut) for mut in mutations]
        fitness_scores = [prism_fitness.predict(mut) for mut in mutations]
        
        # Aggregate
        avg_escape = mean(escape_scores)
        avg_fitness = mean(fitness_scores)
        
        # Compute lineage gamma
        gamma = escape_weight * avg_escape + fitness_weight * avg_fitness
        
        # Predict direction
        predicted_direction = "RISE" if gamma > 0 else "FALL"
        
        # Observe 7 days later
        observed_change = get_frequency_change(lineage, week_date, week_date + 7days)
        observed_direction = "RISE" if observed_change > 0.05 else "FALL"
        
        # Score
        if predicted_direction == observed_direction:
            correct += 1
        total += 1

accuracy = correct / total
```

---

## Why We Have 0% Accuracy

**Current Issue**:
```python
# Using velocity as proxy for gamma
gamma_proxy = velocity

# Predicting based on velocity
if velocity > 0.05: RISE
if velocity < -0.05: FALL
else: STABLE

Problem: 
- Velocity threshold too high (0.05 = 5%/month)
- Most lineages have |velocity| < 0.05
- Everything classified as STABLE
- STABLE cases skipped → 0 predictions counted!
```

**Solution**:
```python
# Use lower threshold OR
# Compute actual gamma from escape + fitness

# Better thresholds:
if velocity > 0.001: RISE   # 0.1%/month
if velocity < -0.001: FALL
else: Use escape+fitness to break tie
```

---

## What We Need

### Option A: Full GPU Pipeline (Ideal)
```rust
// For each lineage:
let gamma = mega_fused.detect_pockets(...).combined_features[95];
let prediction = if gamma > 0.0 { "RISE" } else { "FALL" };
```

**Status**: Kernels ready, need Rust build to work

### Option B: Proxy Using Python (For Now)
```python
# For each lineage:
# 1. Load mutations
# 2. Compute escape (simplified)
# 3. Compute fitness (simplified)
# 4. gamma = 0.5 * escape + 0.5 * fitness
# 5. Predict RISE/FALL from gamma
```

**Status**: Can implement immediately

### Option C: Use Velocity Smartly (Baseline)
```python
# Better proxy:
if velocity > 0.0:  # ANY positive trend
    prediction = "RISE"
else:
    prediction = "FALL"
```

**Status**: Trivial, should get ~50-70% accuracy

---

## CORRECTED Timeline

### Step 1: Fix Proxy Prediction (5 min)
- Use velocity > 0 (not > 0.05)
- Should get 50-70% accuracy
- Proves data pipeline works

### Step 2: Implement Simplified Gamma (1 hour)
- Compute escape scores
- Compute fitness
- gamma = weighted combination
- Should get 70-85% accuracy

### Step 3: Full GPU Pipeline (2 hours)
- Fix Rust build
- Run mega_fused  
- Extract feature 95 (gamma)
- Should get >90% accuracy

---

## Next Action

Let me fix the proxy threshold and run again!
