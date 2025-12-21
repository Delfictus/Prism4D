---
description: Run a hypothesis cycle in the PRISM-4D VE Swarm
---

# Run Hypothesis Cycle

Execute a single hypothesis testing cycle with full agent coordination.

## Hypothesis Cycle Protocol

```
┌─────────────────────────────────────────────────────────────────────┐
│                    HYPOTHESIS CYCLE FLOW                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. HG: Generate/select hypothesis                                  │
│         ↓                                                           │
│  2. IG: Pre-execution integrity check                               │
│         ↓ (HALT if CRITICAL violation)                              │
│  3. FE: Implement hypothesis                                        │
│         ↓                                                           │
│  4. DFV: Validate data flow to new features                         │
│         ↓ (FIX if pipeline issues)                                  │
│  5. SV: Run experiment with proper validation                       │
│         ↓                                                           │
│  6. DFV: Check feature variance/discrimination                      │
│         ↓                                                           │
│  7. CV: Cross-validate across 12 countries                          │
│         ↓                                                           │
│  8. AS: Ablation study if improvement detected                      │
│         ↓                                                           │
│  9. IG: Post-execution integrity audit                              │
│         ↓                                                           │
│  10. OA: Record results, update state                               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Priority Hypothesis Queue

| Priority | ID | Hypothesis | Expected Δ |
|----------|-----|------------|------------|
| P0 | HYP-TD-001 | Dominance vs Direction prediction | +30-40pp |
| P0 | HYP-FE-001 | Competitive Escape Ratio | +10-15pp |
| P0 | HYP-FE-004 | Frequency Momentum | +5-10pp |
| P1 | HYP-TD-002 | 4-Week Time Horizon | +5-10pp |
| P1 | HYP-FE-002 | Escape Percentile Ranking | +3-5pp |
| P1 | HYP-DP-001 | Class Balancing | +2-5pp |

## Arguments

$ARGUMENTS

Usage:
- `/prism-ve-swarm:swarm-cycle` - Run next hypothesis from queue
- `/prism-ve-swarm:swarm-cycle HYP-FE-001` - Run specific hypothesis
- `/prism-ve-swarm:swarm-cycle --dry-run` - Preview without executing

## Acceptance Criteria

A hypothesis is ACCEPTED if:
- Effect size > 2 percentage points
- p-value < 0.05 (McNemar's test)
- ALL 12 countries show improvement
- No integrity violations detected
- DFV reports no pipeline issues

## Stopping Conditions

The swarm STOPS when ANY of:
- Target reached: accuracy ≥ 92% mean
- Max cycles: 50 hypothesis cycles
- Integrity violation: IG detects CRITICAL violation
- Diminishing returns: 3 consecutive cycles with Δ < 0.5pp

## Output

Each cycle produces:
- Updated `experiment_log.json`
- Hypothesis result in `hypothesis_outcomes.md`
- Statistical report with 95% CI
- Integrity certificate if passed

## The Integrity Oath

Before each cycle, recall:
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
