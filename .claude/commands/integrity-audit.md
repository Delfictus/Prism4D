---
description: Run Integrity Guardian audit on codebase
---

# Integrity Guardian Audit

Run comprehensive scientific integrity checks on the PRISM-4D codebase.

## The Integrity Oath

```
"I will not use future information to predict the past.
 I will not train on test data or test on train data.
 I will not hardcode coefficients from the paper I'm trying to beat.
 I will not cherry-pick results or hide failures.
 I will document every modification and its rationale.
 I will ensure every result is reproducible.
 I will report confidence intervals, not just point estimates.
 I will acknowledge when my method differs from VASIL."
```

## Violation Taxonomy

### CRITICAL (HALT SWARM)

| Code | Description | Detection Pattern |
|------|-------------|-------------------|
| IG-001 | Look-ahead bias | Using future dates in features |
| IG-002 | Train/test leakage | Test data in training set |
| IG-003 | Coefficient contamination | VASIL paper values: 0.65, 0.35, 0.92 |

### HIGH (REJECT HYPOTHESIS)

| Code | Description | Detection Pattern |
|------|-------------|-------------------|
| IG-004 | Future date in feature | Dates beyond prediction horizon |
| IG-005 | Non-reproducible result | Seed not fixed, non-deterministic |

### MEDIUM (WARN)

| Code | Description | Detection Pattern |
|------|-------------|-------------------|
| IG-006 | Undocumented modification | Code change without rationale |
| IG-007 | Missing confidence interval | Point estimate only |

### LOW (NOTE)

| Code | Description | Detection Pattern |
|------|-------------|-------------------|
| IG-008 | Methodology deviation | Differs from VASIL approach |

## Forbidden Coefficients

The following values MUST NOT appear in prediction logic:
```
0.65  - VASIL escape weight
0.35  - VASIL fitness weight  
0.92  - VASIL target accuracy (as a magic number)
```

These can appear in documentation or assertions, but NOT in:
- Feature computation
- Model weights
- Prediction formulas
- Threshold values

## Arguments

$ARGUMENTS

Usage:
- `/prism-ve-swarm:integrity-audit` - Full codebase audit
- `/prism-ve-swarm:integrity-audit --quick` - Only check for CRITICAL violations
- `/prism-ve-swarm:integrity-audit --pre-publish` - Pre-publication checklist
- `/prism-ve-swarm:integrity-audit path/to/file.rs` - Audit specific file

## Checks Performed

### 1. Coefficient Scan
```python
FORBIDDEN = [0.65, 0.35, 0.92]
for file in codebase:
    for coefficient in FORBIDDEN:
        if coefficient in file.computation_logic:
            VIOLATION("IG-003", file, coefficient)
```

### 2. Look-Ahead Detection
```python
for feature in features:
    if feature.uses_date(date) and date > prediction_date:
        VIOLATION("IG-001", feature, date)
```

### 3. Train/Test Separation
```python
train_ids = get_train_sample_ids()
test_ids = get_test_sample_ids()
if train_ids.intersection(test_ids):
    VIOLATION("IG-002", overlap=train_ids & test_ids)
```

### 4. Temporal Firewall
```python
TEMPORAL_BOUNDARY = "2022-06-01"
for sample in test_set:
    if sample.collection_date < TEMPORAL_BOUNDARY:
        VIOLATION("IG-002", sample, "test sample from train period")
```

## Output

Produces `integrity_report.md`:
```markdown
# Integrity Audit Report

**Date**: 2025-12-11
**Codebase**: PRISM-4D v1.0

## Summary
- CRITICAL: 0
- HIGH: 0
- MEDIUM: 2
- LOW: 1

## Findings

### IG-006: Undocumented modification
File: ve_optimizer.rs:234
Change: Q-table epsilon modified from 0.1 to 0.05
Required: Add comment explaining rationale

...
```

## Integration with Swarm

- **Pre-cycle**: IG runs before each hypothesis test
- **Post-cycle**: IG validates results
- **CRITICAL violation**: Swarm halts immediately
- **No workarounds**: IG has absolute veto power

## Red Team Protocol

For highest integrity, run adversarial audit:
```
/prism-ve-swarm:integrity-audit --red-team
```

This tries to find:
- Subtle look-ahead bias
- Indirect coefficient usage
- Information leakage through feature engineering
- Cherry-picked temporal boundaries
