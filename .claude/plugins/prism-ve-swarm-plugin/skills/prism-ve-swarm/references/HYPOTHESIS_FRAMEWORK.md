# PRISM-4D Hypothesis Framework

Structured approach to generating, testing, and documenting hypotheses.

## Hypothesis Lifecycle

```
PROPOSED → REVIEWED → APPROVED → IMPLEMENTED → TESTED → VALIDATED → ACCEPTED/REJECTED
    │          │          │           │          │          │
    └──────────┴──────────┴───────────┴──────────┴──────────┘
              Integrity Guardian reviews at each stage
```

---

## Hypothesis Template

```yaml
# PRISM-4D Hypothesis Document
# Template Version: 1.0

metadata:
  id: "HYP-YYYYMMDD-NNN"
  created: "YYYY-MM-DD HH:MM:SS"
  author: "Agent/Human"
  status: "proposed|reviewed|approved|implemented|tested|validated|accepted|rejected"
  priority: "P0|P1|P2|P3"  # P0 = highest

# ============================================================
# SCIENTIFIC FOUNDATION
# ============================================================

observation:
  description: |
    What pattern, anomaly, or gap in current performance prompted this hypothesis?
  evidence:
    - "Specific data point or statistic"
    - "Reference to analysis output"
  
mechanism:
  description: |
    What is the proposed causal mechanism? Why would this change improve accuracy?
  biological_basis: |
    How does this relate to viral evolution biology?
  references:
    - "Paper citation or prior work"

# ============================================================
# HYPOTHESIS STATEMENT
# ============================================================

null_hypothesis: |
  H0: [Default/null expectation - no effect]
  
alternative_hypothesis: |
  H1: [Specific expected effect with direction]

prediction:
  qualitative: |
    What should happen if H1 is true?
  quantitative: |
    Expected effect size: [X pp improvement]
    Expected direction: [increase/decrease]

# ============================================================
# IMPLEMENTATION PLAN
# ============================================================

implementation:
  category: "feature_engineering|task_definition|model_architecture|data_processing"
  
  files_to_modify:
    - path: "crates/prism-ve-bench/src/ve_optimizer.rs"
      changes: "Description of changes"
    - path: "crates/prism-ve-bench/src/main.rs"
      changes: "Description of changes"
  
  complexity: "trivial|moderate|complex"
  estimated_time: "X hours"
  
  dependencies:
    - "HYP-YYYYMMDD-NNN (must be completed first)"
  
  reversibility: "fully_reversible|partially_reversible|irreversible"

# ============================================================
# INTEGRITY ASSESSMENT
# ============================================================

integrity:
  look_ahead_risk:
    level: "none|low|medium|high"
    mitigation: |
      How will look-ahead bias be prevented?
  
  leakage_risk:
    level: "none|low|medium|high"  
    mitigation: |
      How will train/test leakage be prevented?
  
  coefficient_risk:
    level: "none|low|medium|high"
    mitigation: |
      How will VASIL coefficient contamination be prevented?
  
  integrity_guardian_notes: |
    [Filled by IG during review]

# ============================================================
# ACCEPTANCE CRITERIA
# ============================================================

acceptance:
  primary_metric: "test_accuracy"
  
  success_threshold:
    minimum_effect: 0.02  # 2 percentage points improvement
    significance_level: 0.05
    confidence_interval_excludes_zero: true
  
  validation_method: "temporal_split|loco|both"
  
  secondary_criteria:
    - "No increase in train-test gap"
    - "Consistent across all 12 countries"
    - "Effect survives ablation"

# ============================================================
# EXPERIMENT DESIGN
# ============================================================

experiment:
  control_condition: |
    Current baseline configuration
  
  treatment_condition: |
    Configuration with hypothesis implemented
  
  sample_size:
    train: "~1,745 (pre-2022-06)"
    test: "~9,340 (post-2022-06)"
  
  randomization:
    seed: 42
    n_replications: 3
  
  blinding: |
    Test accuracy computed AFTER all hyperparameters fixed

# ============================================================
# RESULTS (Filled after testing)
# ============================================================

results:
  baseline_accuracy: null
  treatment_accuracy: null
  effect_size: null
  confidence_interval: null
  p_value: null
  
  per_country_results: []
  
  ablation_results: []
  
  unexpected_observations: |
    [Any surprising findings]

# ============================================================
# CONCLUSION
# ============================================================

conclusion:
  outcome: "accepted|rejected|inconclusive"
  
  interpretation: |
    What do these results mean?
  
  next_steps: |
    What should be done next?
  
  lessons_learned: |
    What was learned regardless of outcome?
```

---

## Priority Ranking Criteria

### P0 (Critical Path)
- Directly addresses root cause of accuracy gap
- High expected impact (>10 pp improvement)
- Clear biological justification
- Low integrity risk

### P1 (High Priority)
- Addresses significant known issue
- Medium expected impact (5-10 pp improvement)
- Solid mechanistic reasoning
- Manageable integrity risk

### P2 (Medium Priority)
- Exploratory with reasonable basis
- Lower expected impact (2-5 pp improvement)
- Some supporting evidence
- Requires careful integrity management

### P3 (Low Priority)
- Speculative or incremental
- Small expected impact (<2 pp improvement)
- Weak or indirect evidence
- May introduce integrity complexity

---

## Candidate Hypotheses Catalog

### Category: Task Definition

#### HYP-TD-001: Dominance vs Direction Prediction
```yaml
observation: |
  VASIL predicts which variant becomes dominant (>X% share),
  not week-over-week frequency direction.
mechanism: |
  Dominance is a cleaner target - fewer noisy fluctuations.
  Direction prediction conflates noise with signal.
null_hypothesis: "Task reformulation has no effect on accuracy"
alternative_hypothesis: "Dominance prediction achieves >80% accuracy vs 42% for direction"
priority: P0
integrity_risk: low
```

#### HYP-TD-002: Time Horizon Extension
```yaml
observation: |
  1-week horizon captures noise, not meaningful trends.
mechanism: |
  4-week horizon smooths noise, captures real growth.
  Epidemiological doubling times are weeks, not days.
null_hypothesis: "Time horizon has no effect"
alternative_hypothesis: "4-week horizon improves accuracy by >5 pp"
priority: P1
integrity_risk: medium (must not look ahead)
```

### Category: Feature Engineering

#### HYP-FE-001: Competitive Escape Ratio
```yaml
observation: |
  Raw escape doesn't discriminate RISE/FALL.
  All Omicron variants have high escape.
mechanism: |
  What matters is escape RELATIVE to competition.
  Compute: variant_escape / dominant_variant_escape
null_hypothesis: "Competitive ratio has no effect"
alternative_hypothesis: "Competitive ratio discriminates with Cohen's d > 0.3"
priority: P0
integrity_risk: medium (must use contemporaneous dominant)
```

#### HYP-FE-002: Escape Percentile Ranking
```yaml
observation: |
  Absolute escape values are context-dependent.
mechanism: |
  Percentile rank within each time point normalizes context.
  "90th percentile escape" means more than "0.8 escape"
null_hypothesis: "Percentile ranking has no effect"
alternative_hypothesis: "Percentile improves discrimination over absolute"
priority: P1
integrity_risk: low
```

#### HYP-FE-003: Recombinant Lineage Flag
```yaml
observation: |
  Recombinants (XBB, XBC) have unique fitness advantages.
mechanism: |
  Recombination combines beneficial mutations from multiple lineages.
  Should have higher prior fitness probability.
null_hypothesis: "Recombinant flag has no effect"
alternative_hypothesis: "Recombinant flag improves RISE precision for XBB lineages"
priority: P2
integrity_risk: none
```

#### HYP-FE-004: Frequency Momentum (df/dt)
```yaml
observation: |
  Current analysis shows frequency_velocity discriminates.
mechanism: |
  Variants with positive momentum tend to continue growing.
  This is the one feature currently showing signal.
null_hypothesis: "Momentum has no additional value"
alternative_hypothesis: "Momentum + escape outperforms either alone"
priority: P0
integrity_risk: none
```

### Category: Model Architecture

#### HYP-MA-001: Q-Table Discretization Boundaries
```yaml
observation: |
  Current discretization may not capture decision-relevant thresholds.
mechanism: |
  Optimal boundaries should separate high/low fitness regions.
  Data-driven boundaries > arbitrary quantiles.
null_hypothesis: "Boundary choice doesn't matter"
alternative_hypothesis: "Optimized boundaries improve accuracy by >2 pp"
priority: P2
integrity_risk: high (must not optimize on test data)
```

#### HYP-MA-002: Expanded State Space
```yaml
observation: |
  Current 4096 states may be too coarse.
mechanism: |
  Finer discretization captures more nuance.
  But: curse of dimensionality limits gains.
null_hypothesis: "State space size doesn't matter"
alternative_hypothesis: "Optimal state space is between 4096 and 65536"
priority: P2
integrity_risk: medium (more states = more overfitting risk)
```

### Category: Data Processing

#### HYP-DP-001: Class Balancing
```yaml
observation: |
  40% RISE, 60% FALL class imbalance.
mechanism: |
  Imbalance biases model toward majority class.
  Upsampling RISE or downsampling FALL may help.
null_hypothesis: "Class balancing has no effect"
alternative_hypothesis: "Balanced training improves minority class recall"
priority: P1
integrity_risk: low
```

#### HYP-DP-002: Sample Weighting by Frequency
```yaml
observation: |
  Low-frequency variants dominate sample count but are less important.
mechanism: |
  Weight samples by frequency to focus on consequential variants.
null_hypothesis: "Weighting has no effect"
alternative_hypothesis: "Frequency weighting improves real-world relevance"
priority: P2
integrity_risk: low
```

---

## Hypothesis Tracking Board

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         HYPOTHESIS KANBAN BOARD                              │
├─────────────────┬─────────────────┬─────────────────┬───────────────────────┤
│    PROPOSED     │    IN REVIEW    │   TESTING       │      COMPLETED        │
├─────────────────┼─────────────────┼─────────────────┼───────────────────────┤
│ HYP-TD-002 (P1) │ HYP-FE-001 (P0) │ HYP-TD-001 (P0) │ ✓ HYP-FE-004 (+3.2pp)│
│ HYP-MA-001 (P2) │                 │                 │ ✗ HYP-DP-001 (no eff)│
│ HYP-MA-002 (P2) │                 │                 │                       │
│ HYP-DP-002 (P2) │                 │                 │                       │
└─────────────────┴─────────────────┴─────────────────┴───────────────────────┘

LEGEND: (P#) = Priority  ✓ = Accepted  ✗ = Rejected  (+X.Xpp) = Effect size
```

---

## Hypothesis Outcome Documentation

### Template: Accepted Hypothesis

```markdown
## HYP-FE-001: Competitive Escape Ratio — ACCEPTED ✓

### Summary
Replacing raw escape with competitive escape ratio (variant_escape / dominant_escape)
improved test accuracy from 42.3% to 51.7% (+9.4 pp, p < 0.001).

### Effect Size
- Absolute improvement: +9.4 percentage points
- Relative improvement: +22%
- 95% CI: [7.1 pp, 11.8 pp]

### Validation
- Temporal split: 51.7% (n=9,340)
- LOCO mean: 50.2% (σ=3.1%)
- All 12 countries improved
- Ablation confirms: removing competitive ratio drops accuracy by 8.9 pp

### Mechanism Confirmed
Cohen's d for competitive ratio: 0.47 (medium effect)
Cohen's d for raw escape: 0.02 (negligible)
Confirms: relative escape discriminates, absolute does not

### Next Steps
- Build on this with HYP-FE-002 (percentile ranking)
- Investigate interaction with frequency momentum
```

### Template: Rejected Hypothesis

```markdown
## HYP-DP-001: Class Balancing — REJECTED ✗

### Summary
Upsampling RISE class to 50/50 balance did not improve test accuracy.
Test accuracy: 42.1% vs 42.3% baseline (p = 0.84).

### Findings
- No significant effect on overall accuracy
- RISE recall improved (+4 pp) but FALL recall decreased (-5 pp)
- Net effect: noise, not signal

### Interpretation
Class imbalance is not the problem. The features simply don't discriminate
regardless of training balance. This is consistent with the feature
discrimination analysis.

### Lessons Learned
- Don't assume class imbalance is the issue
- Fix features first, then consider training tricks

### Archive Note
Hypothesis retired. Re-test only if feature engineering shows discrimination.
```

---

## Meta-Analysis: Hypothesis Success Patterns

After N hypotheses tested, analyze:

1. **Success rate by category**
   - Feature engineering: X/Y successful
   - Task definition: X/Y successful
   - Model architecture: X/Y successful
   
2. **Effect size distribution**
   - Mean effect of accepted hypotheses: +X.X pp
   - Largest effect: +X.X pp (HYP-XXX)
   
3. **Integrity incident rate**
   - Hypotheses flagged by IG: N
   - Hypotheses rejected for integrity: N
   
4. **Cumulative improvement**
   - Baseline: 42.3%
   - After N cycles: XX.X%
   - Gap remaining: XX.X pp to 92%
