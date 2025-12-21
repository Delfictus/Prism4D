---
description: Test a specific hypothesis with full statistical validation
---

# Hypothesis Test

Test a single hypothesis with proper statistical validation, cross-validation, and integrity checks.

## Hypothesis Template

```yaml
hypothesis:
  id: "HYP-XX-NNN"
  title: "Descriptive title"
  category: [FE|TD|DP|MA|NN]  # Feature Engineering, Task Definition, Data Processing, Model Architecture, Neural Network
  priority: [P0|P1|P2]
  
  scientific_rationale: |
    Why this hypothesis might improve accuracy.
    Reference to biological or statistical reasoning.
  
  null_hypothesis: "H0: This change has no effect on accuracy"
  alternative_hypothesis: "H1: This change improves accuracy by >2pp"
  
  implementation:
    files_to_modify:
      - path: "relative/path/to/file.rs"
        changes: "Description of changes"
    
  expected_effect: "+X to +Y percentage points"
  
  acceptance_criteria:
    effect_threshold: 2.0  # percentage points
    p_value_threshold: 0.05
    all_countries_improve: true
```

## Priority Hypotheses

### P0: High-Impact (Test First)

**HYP-TD-001: Dominance vs Direction Prediction**
```yaml
rationale: |
  Current task predicts "will frequency increase?" but we have 42% accuracy.
  Alternative: predict "will this become dominant?" (>50% frequency)
  Dominant variants are better characterized by escape scores.
expected_effect: "+30-40pp"
```

**HYP-FE-001: Competitive Escape Ratio**
```yaml
rationale: |
  Current escape scores are static per-variant.
  New feature: variant_escape / dominant_escape
  This measures "escape advantage over current dominant"
expected_effect: "+10-15pp"
```

**HYP-FE-004: Frequency Momentum**
```yaml
rationale: |
  Only frequency_velocity shows discrimination currently.
  Add acceleration: df/dt at t vs t-4 weeks
  Captures "is growth accelerating?"
expected_effect: "+5-10pp"
```

## Arguments

$ARGUMENTS

Usage:
- `/prism-ve-swarm:hypothesis-test HYP-FE-001` - Test specific hypothesis
- `/prism-ve-swarm:hypothesis-test --define` - Create new hypothesis interactively
- `/prism-ve-swarm:hypothesis-test --list` - List all hypotheses with status

## Test Protocol

```
HYPOTHESIS TEST PROTOCOL
========================

1. PRE-TEST INTEGRITY CHECK (IG)
   - Verify no forbidden coefficients
   - Check temporal firewall intact
   - Confirm train/test separation

2. IMPLEMENTATION (FE)
   - Modify code per hypothesis spec
   - Log all changes with git diff

3. DATA FLOW VALIDATION (DFV)
   - Verify new features have variance > 0
   - Check RISE/FALL discrimination
   - Confirm no null buffers

4. EXPERIMENT EXECUTION
   - Train: 2021-01 to 2022-05
   - Test: 2022-06 to 2023-12
   - Random seed: 42 (fixed)

5. CROSS-VALIDATION (CV)
   - Leave-one-country-out across all 12
   - Record per-country accuracy

6. STATISTICAL VALIDATION (SV)
   - Compute 95% Wilson confidence intervals
   - McNemar's test vs baseline
   - Effect size (Cohen's d)
   - Overfitting check: train-test gap < 10pp

7. ABLATION (AS) - if accepted
   - Single-feature ablation
   - Measure marginal contribution

8. POST-TEST INTEGRITY AUDIT (IG)
   - Verify results reproducible
   - Confirm no cherry-picking
   - Sign integrity certificate
```

## Statistical Requirements

| Metric | Threshold | Purpose |
|--------|-----------|---------|
| Effect size | >2 pp | Meaningful improvement |
| p-value | <0.05 | Statistical significance |
| 95% CI | Must exclude 0 | Confidence |
| All countries | Improve | Generalization |
| Train-test gap | <10pp | No overfitting |

## Output

Produces hypothesis outcome record:
```markdown
# Hypothesis Outcome: HYP-FE-001

**Status**: ACCEPTED ✓

## Results
- Baseline accuracy: 42.3%
- New accuracy: 54.8%
- Delta: +12.5pp
- p-value: 0.0023
- 95% CI: [10.2, 14.8]

## Per-Country Results
| Country | Baseline | New | Delta |
|---------|----------|-----|-------|
| Germany | 41.2% | 53.1% | +11.9 |
| USA | 43.5% | 56.2% | +12.7 |
...

## Ablation
- Feature contribution: 8.3pp (primary effect)
- Interaction with existing: +4.2pp

## Integrity Certificate
- No look-ahead bias: ✓
- No coefficient contamination: ✓
- Reproducible from seed 42: ✓
- Signed by: Integrity Guardian
- Date: 2025-12-11
```
