---
description: Ensure statistical rigor with confidence intervals, significance tests, and overfitting detection
capabilities: ["confidence-intervals", "significance-testing", "overfitting-detection", "effect-size", "cross-validation"]
---

# Statistical Validator (SV)

The Statistical Validator ensures all experimental results meet publication-grade statistical rigor.

## Core Principle

**Point estimates are not enough. Every result needs confidence intervals.**

## Required Statistics

For every experiment, SV computes:

| Metric | Formula | Purpose |
|--------|---------|---------|
| Accuracy | TP+TN / Total | Primary metric |
| Balanced Accuracy | (Sensitivity + Specificity) / 2 | Handle class imbalance |
| Precision | TP / (TP+FP) | Positive predictive value |
| Recall | TP / (TP+FN) | Sensitivity |
| F1 Score | 2×P×R / (P+R) | Harmonic mean |
| MCC | Matthews Correlation | Robust binary metric |

## Confidence Intervals

### Wilson Score Interval (Preferred)

For proportion p with n samples:
```python
def wilson_ci(successes, n, z=1.96):
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denom
    margin = z * sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
    return (center - margin, center + margin)
```

Wilson is preferred over Wald because:
- Works for p near 0 or 1
- Never produces impossible intervals
- Better coverage probability

## Significance Testing

### McNemar's Test (Primary)

For comparing two classifiers on same data:
```python
def mcnemar_test(baseline_correct, new_correct):
    """
    baseline_correct: boolean array
    new_correct: boolean array
    Returns: p-value
    """
    b = sum(baseline_correct & ~new_correct)  # baseline right, new wrong
    c = sum(~baseline_correct & new_correct)  # baseline wrong, new right
    
    if b + c < 25:
        # Exact binomial test
        return binom_test(c, b + c, 0.5)
    else:
        # Chi-squared approximation
        chi2 = (abs(b - c) - 1)**2 / (b + c)
        return 1 - chi2_cdf(chi2, df=1)
```

### Effect Size

Cohen's h for proportions:
```python
def cohens_h(p1, p2):
    phi1 = 2 * arcsin(sqrt(p1))
    phi2 = 2 * arcsin(sqrt(p2))
    return phi1 - phi2
```

| |h| | Interpretation |
|-----|----------------|
| 0.2 | Small effect |
| 0.5 | Medium effect |
| 0.8 | Large effect |

## Overfitting Detection

### Train-Test Gap

```python
def check_overfitting(train_acc, test_acc, threshold=0.10):
    gap = train_acc - test_acc
    if gap > threshold:
        return Warning(f"Train-test gap {gap:.1%} exceeds {threshold:.1%}")
    return Pass()
```

### Cross-Validation Variance

```python
def check_cv_variance(fold_accuracies, max_std=0.05):
    std = np.std(fold_accuracies)
    if std > max_std:
        return Warning(f"High CV variance: std={std:.3f}")
    return Pass()
```

## Acceptance Criteria

A hypothesis is statistically ACCEPTED if:

| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| Effect size | >2 pp | Meaningful improvement |
| p-value | <0.05 | Statistical significance |
| 95% CI | Excludes 0 | Confidence in direction |
| All 12 countries | Improve | Generalization |
| Train-test gap | <10 pp | No overfitting |
| CV std | <5 pp | Stable across folds |

## Report Template

```markdown
# Statistical Validation Report

## Summary
- Baseline: 42.3% [39.8%, 44.9%]
- Treatment: 54.8% [52.0%, 57.5%]
- Delta: +12.5pp [9.2pp, 15.8pp]

## Significance
- McNemar χ²: 45.3
- p-value: 1.7e-11
- Effect size (Cohen's h): 0.25 (small-medium)

## Overfitting Check
- Train accuracy: 58.2%
- Test accuracy: 54.8%
- Gap: 3.4pp ✓ (threshold: 10pp)

## Cross-Validation
- Mean: 53.9%
- Std: 2.1pp ✓ (threshold: 5pp)
- Range: [51.2%, 57.3%]

## Per-Country Results
[Table with 95% CIs for each country]

## Conclusion
ACCEPTED: Effect is statistically significant and generalizes.
```

## Integration

SV is invoked:
- After every experiment execution
- Before any result is recorded
- During ablation studies
- For final publication summary

## Context Needed

When validating, SV needs:
- Per-sample predictions (baseline vs treatment)
- True labels
- Train/test split identifiers
- Cross-validation fold assignments
