# PRISM-4D Swarm Agent Specifications

Detailed protocols for each agent in the optimization swarm.

## Agent 1: Orchestrator Agent (OA)

### State Management
```yaml
swarm_state:
  current_phase: [init|hypothesis|validation|finalization]
  cycle_number: int
  baseline_accuracy: float
  best_accuracy: float
  best_configuration: dict
  active_hypothesis: str | null
  blocked_hypotheses: list[str]
  integrity_violations: list[str]
  data_flow_issues: list[str]  # NEW: Track DFV findings
```

### Decision Logic
```python
def select_next_action(state):
    if state.integrity_violations:
        return HALT_SWARM
    
    if state.data_flow_issues:
        return FIX_PIPELINE  # NEW: Must fix before continuing
    
    if state.best_accuracy >= 0.92:
        return FINALIZE
    
    if state.cycle_number >= 50:
        return FINALIZE
    
    if consecutive_low_delta(state, threshold=0.005, count=3):
        return FINALIZE
    
    return NEXT_HYPOTHESIS_CYCLE
```

### Communication Protocol
- Broadcasts: `CYCLE_START`, `CYCLE_END`, `SWARM_HALT`, `INTEGRITY_ALERT`, `PIPELINE_ISSUE`
- Receives: Agent completion signals, integrity flags, DFV reports, results

---

## Agent 2: Integrity Guardian (IG)

### Violation Categories

| Code | Severity | Description | Action |
|------|----------|-------------|--------|
| IG-001 | CRITICAL | Look-ahead bias detected | HALT |
| IG-002 | CRITICAL | Train/test leakage | HALT |
| IG-003 | CRITICAL | VASIL coefficients hardcoded | HALT |
| IG-004 | HIGH | Future date in feature | REJECT |
| IG-005 | HIGH | Non-reproducible result | REJECT |
| IG-006 | MEDIUM | Undocumented modification | WARN |
| IG-007 | LOW | Missing confidence interval | WARN |

[... rest of IG specification unchanged ...]

---

## Agent 3: Data Flow Validator (DFV) 🆕

### Purpose
Detect GPU pipeline issues that cause zero-discrimination features:
- Null buffers passed to kernels
- Constant features across all structures
- Metadata not propagating to kernel inputs
- Buffer shape mismatches

### Violation Categories

| Code | Severity | Description | Action |
|------|----------|-------------|--------|
| DFV-001 | CRITICAL | Null buffer passed to kernel | FIX_PIPELINE |
| DFV-002 | HIGH | Feature has zero variance | DIAGNOSE |
| DFV-003 | HIGH | Metadata not reaching kernel | FIX_PIPELINE |
| DFV-004 | MEDIUM | Index mismatch in kernel | FIX_PIPELINE |
| DFV-005 | MEDIUM | Buffer shape mismatch | FIX_PIPELINE |

### Feature Variance Check
```python
def check_feature_variance(features, n_structures):
    """
    Detect constant features that cannot discriminate.
    """
    violations = []
    
    for feat_idx in range(features.shape[1]):
        variance = features[:, feat_idx].var()
        
        if variance < 1e-10:
            constant_val = features[0, feat_idx]
            
            violations.append(DFVViolation(
                code="DFV-002",
                severity="HIGH",
                feature_index=feat_idx,
                constant_value=constant_val,
                diagnosis=diagnose_constant_feature(feat_idx, constant_val)
            ))
    
    return violations
```

### Diagnostic Logic for Constant Features
```python
def diagnose_constant_feature(idx, value):
    """
    Trace back from constant feature to root cause.
    """
    # Feature dependency graph
    DEPENDENCIES = {
        (101, 108): (96, 100),  # Spike depends on Cycle
        (96, 100): "d_frequency_velocity buffer",  # Cycle depends on freq/vel input
        (92, 95): "d_escape_scores buffer",  # Fitness depends on escape input
    }
    
    for (start, end), dependency in DEPENDENCIES.items():
        if start <= idx <= end:
            if isinstance(dependency, tuple):
                return f"Features F{start}-F{end} constant. Check upstream features F{dependency[0]}-F{dependency[1]} first."
            else:
                return f"Features F{start}-F{end} constant. Check {dependency} allocation and population."
    
    return f"Feature F{idx} constant. Check corresponding kernel stage."
```

### Null Buffer Detection
```python
def scan_kernel_params(rust_code):
    """
    Find kernel launches with nullptr parameters.
    """
    violations = []
    
    nullptr_patterns = [
        r'std::ptr::null\(\)',
        r'std::ptr::null_mut\(\)',
        r'ptr:\s*0',
    ]
    
    for pattern in nullptr_patterns:
        matches = re.finditer(pattern, rust_code)
        for match in matches:
            context = extract_context(rust_code, match.start())
            if is_kernel_param_context(context):
                violations.append(DFVViolation(
                    code="DFV-001",
                    severity="CRITICAL",
                    message=f"Null pointer in kernel parameter",
                    context=context
                ))
    
    return violations
```

### Activation Points
1. **After PTX compilation** — Check kernel parameter signatures
2. **Before batch processing** — Verify buffer allocation
3. **After feature extraction** — Check feature variance
4. **When accuracy stagnates** — Full pipeline diagnosis

### Integration with Other Agents

**DFV → Orchestrator (OA)**:
- Reports pipeline issues before hypothesis testing
- Can block optimization until issues fixed

**DFV → Feature Engineering (FE)**:
- Validates new feature implementation
- Confirms data reaches kernel correctly

**DFV → Statistical Validator (SV)**:
- Identifies which features to exclude from analysis
- Flags "pipeline issue" vs "scientific issue"

### Example Diagnostic Output
```
╔══════════════════════════════════════════════════════════════════════╗
║  DFV-002: CONSTANT FEATURE DETECTED                                  ║
╠══════════════════════════════════════════════════════════════════════╣
║  Feature: F101 (velocity_spike_density)                              ║
║  Constant Value: 0.500000 across all 14,917 structures               ║
║                                                                      ║
║  DEPENDENCY TRACE:                                                   ║
║  F101-F108 (Spike) ──depends on──► F96-F100 (Cycle)                  ║
║  F96-F100 (Cycle)  ──depends on──► d_frequency_velocity buffer       ║
║                                                                      ║
║  ROOT CAUSE: d_frequency_velocity buffer is nullptr                  ║
║                                                                      ║
║  FIX:                                                                ║
║  1. Add d_frequency_velocity to BatchBufferPool                      ║
║  2. Pack per-lineage frequency/velocity in build_mega_batch()        ║
║  3. Pass to kernel instead of nullptr                                ║
╚══════════════════════════════════════════════════════════════════════╝
```

### Pre-Execution Checks
```python
def pre_execution_audit(hypothesis, implementation):
    checks = [
        verify_no_future_dates(implementation),
        verify_no_test_data_access(implementation),
        verify_no_hardcoded_coefficients(implementation),
        verify_deterministic_seed(implementation),
        verify_train_test_separation(implementation),
    ]
    
    violations = [c for c in checks if c.failed]
    
    if any(v.severity == 'CRITICAL' for v in violations):
        return HALT_SWARM, violations
    elif any(v.severity == 'HIGH' for v in violations):
        return REJECT_HYPOTHESIS, violations
    else:
        return PROCEED, violations
```

### Post-Execution Audit
```python
def post_execution_audit(experiment_log):
    checks = [
        verify_reproducibility(experiment_log),
        verify_no_overfitting(experiment_log, threshold=0.10),
        verify_cross_country_consistency(experiment_log),
        verify_statistical_validity(experiment_log),
    ]
    
    return AuditReport(checks)
```

### Look-Ahead Bias Detection
```python
def detect_look_ahead(feature_computation, sample):
    """
    A feature has look-ahead bias if its computation
    for date T uses any information from date > T.
    """
    # Trace all data dependencies
    dependencies = trace_data_flow(feature_computation)
    
    for dep in dependencies:
        if dep.date > sample.date:
            return LookAheadViolation(
                feature=feature_computation.name,
                sample_date=sample.date,
                dependency_date=dep.date,
                dependency_source=dep.source
            )
    
    return None
```

---

## Agent 3: Hypothesis Generator (HG)

### Hypothesis Template
```yaml
hypothesis:
  id: "HYP-{YYYYMMDD}-{sequence}"
  title: "Brief descriptive title"
  category: [feature_engineering|task_definition|model_architecture|data_processing]
  
  scientific_rationale:
    observation: "What pattern/problem prompted this hypothesis"
    mechanism: "Proposed causal mechanism"
    prediction: "Specific testable prediction"
  
  null_hypothesis: "H0: No effect / baseline behavior"
  alternative_hypothesis: "H1: Specific expected effect"
  
  implementation:
    files_modified: ["path/to/file.rs"]
    estimated_complexity: [trivial|moderate|complex]
    dependencies: ["other hypothesis IDs"]
  
  acceptance_criteria:
    primary_metric: "test_accuracy"
    minimum_effect_size: 0.02  # 2 percentage points
    significance_level: 0.05
    validation_method: "12-country cross-validation"
  
  integrity_considerations:
    look_ahead_risk: [none|low|medium|high]
    leakage_risk: [none|low|medium|high]
    mitigation: "How risks are addressed"
```

### Priority Ranking Algorithm
```python
def rank_hypotheses(candidates, state):
    scores = []
    for h in candidates:
        score = (
            0.40 * expected_impact(h) +
            0.25 * scientific_plausibility(h) +
            0.20 * implementation_feasibility(h) +
            0.15 * integrity_safety(h)
        )
        scores.append((h, score))
    
    return sorted(scores, key=lambda x: -x[1])
```

### Hypothesis Categories (Prioritized)

1. **Task Definition Hypotheses**
   - Dominance prediction vs direction prediction
   - Emergence detection vs frequency tracking
   - Time horizon selection (1-week vs 4-week)

2. **Feature Engineering Hypotheses**
   - Competitive escape ratios
   - Escape percentile rankings
   - Recombinant lineage flags
   - Temporal momentum features

3. **Model Architecture Hypotheses**
   - Q-table discretization boundaries
   - State space dimensionality
   - Reward function design

4. **Data Processing Hypotheses**
   - Sample weighting schemes
   - Class balancing strategies
   - Temporal sampling density

---

## Agent 4: Feature Engineering Agent (FE)

### Feature Implementation Checklist
```
□ Feature has clear biological interpretation
□ Feature computation is deterministic
□ Feature does not use future information
□ Feature is computed identically for train/test
□ Feature has reasonable value distribution
□ Feature variance is non-zero
□ Feature is not redundant with existing features
□ Feature implementation is documented
□ Feature has unit tests
```

### Discrimination Analysis Protocol
```python
def analyze_discrimination(feature, labels):
    """
    Assess whether a feature discriminates between RISE/FALL.
    """
    rise_values = feature[labels == 'RISE']
    fall_values = feature[labels == 'FALL']
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((rise_values.std()**2 + fall_values.std()**2) / 2)
    cohens_d = (rise_values.mean() - fall_values.mean()) / pooled_std
    
    # Statistical test
    t_stat, p_value = scipy.stats.ttest_ind(rise_values, fall_values)
    
    # Distribution overlap
    overlap = compute_distribution_overlap(rise_values, fall_values)
    
    return DiscriminationReport(
        feature_name=feature.name,
        rise_mean=rise_values.mean(),
        fall_mean=fall_values.mean(),
        cohens_d=cohens_d,
        p_value=p_value,
        overlap_coefficient=overlap,
        discriminates=abs(cohens_d) > 0.2 and p_value < 0.05
    )
```

### Feature Categories

| Category | Features | Index Range |
|----------|----------|-------------|
| TDA Topological | Betti numbers, persistence | 0-47 |
| Base Structural | Burial, conservation, B-factor | 48-79 |
| Physics | Electrostatics, hydrophobicity | 80-91 |
| Fitness | ΔΔG_bind, ΔΔG_stab, expression | 92-95 |
| Cycle | Phase, emergence, velocity | 96-100 |
| **Derived** | Competitive ratios, percentiles | Computed |

---

## Agent 5: Statistical Validator (SV)

### Required Statistics for Publication

| Metric | Requirement | Formula |
|--------|-------------|---------|
| Accuracy | Point estimate + 95% CI | Wilson score interval |
| Precision | Per-class | TP / (TP + FP) |
| Recall | Per-class | TP / (TP + FN) |
| F1 Score | Macro-averaged | 2 × (P × R) / (P + R) |
| AUC-ROC | If probabilistic | sklearn.metrics.roc_auc_score |
| Cohen's κ | Inter-rater agreement | (p_o - p_e) / (1 - p_e) |

### Significance Testing Protocol
```python
def test_improvement(baseline_acc, new_acc, n_samples):
    """
    McNemar's test for paired nominal data.
    """
    # Construct contingency table
    # b = baseline correct, new incorrect
    # c = baseline incorrect, new correct
    
    chi2 = (abs(b - c) - 1)**2 / (b + c)
    p_value = 1 - scipy.stats.chi2.cdf(chi2, df=1)
    
    return SignificanceResult(
        chi2=chi2,
        p_value=p_value,
        significant=p_value < 0.05,
        effect_size=new_acc - baseline_acc
    )
```

### Overfitting Detection
```python
def detect_overfitting(train_acc, test_acc, threshold=0.10):
    """
    Flag if train-test gap exceeds threshold.
    """
    gap = train_acc - test_acc
    
    if gap > threshold:
        return OverfittingWarning(
            train_accuracy=train_acc,
            test_accuracy=test_acc,
            gap=gap,
            severity='HIGH' if gap > 0.20 else 'MEDIUM'
        )
    
    return None
```

---

## Agent 6: Ablation Study Agent (AS)

### Ablation Protocol
```python
def run_ablation_study(full_model, features, test_data):
    """
    Measure each feature's marginal contribution.
    """
    baseline_acc = evaluate(full_model, test_data)
    
    ablation_results = []
    for feature in features:
        # Remove single feature
        ablated_model = train_without_feature(full_model, feature)
        ablated_acc = evaluate(ablated_model, test_data)
        
        contribution = baseline_acc - ablated_acc
        
        ablation_results.append(AblationResult(
            feature=feature,
            baseline_accuracy=baseline_acc,
            ablated_accuracy=ablated_acc,
            contribution=contribution,
            is_critical=contribution > 0.02
        ))
    
    return sorted(ablation_results, key=lambda x: -x.contribution)
```

### Feature Importance Reporting
```
ABLATION REPORT
===============
Baseline Accuracy: 87.3%

Feature                  | Ablated Acc | Δ Accuracy | Critical
-------------------------|-------------|------------|----------
competitive_escape_ratio | 79.1%       | -8.2%      | ✓
frequency_velocity       | 82.4%       | -4.9%      | ✓
transmissibility         | 85.8%       | -1.5%      | 
raw_escape               | 86.9%       | -0.4%      | 
...
```

---

## Agent 7: Cross-Validation Agent (CV)

### Validation Strategies

#### 1. Leave-One-Country-Out (Primary)
```python
def leave_one_country_out(data, model_class):
    """
    Train on 11 countries, test on 1. Repeat for all 12.
    """
    countries = data.countries.unique()
    results = []
    
    for held_out in countries:
        train = data[data.country != held_out]
        test = data[data.country == held_out]
        
        model = model_class.fit(train)
        acc = model.evaluate(test)
        
        results.append(CountryResult(
            country=held_out,
            accuracy=acc,
            n_samples=len(test)
        ))
    
    return CrossValidationReport(
        method='LOCO',
        per_country=results,
        mean_accuracy=np.mean([r.accuracy for r in results]),
        std_accuracy=np.std([r.accuracy for r in results])
    )
```

#### 2. Temporal Split (Required)
```python
def temporal_split_validation(data, cutoff_date='2022-06-01'):
    """
    Train on pre-cutoff, test on post-cutoff.
    This is the PRIMARY validation for publication.
    """
    train = data[data.date < cutoff_date]
    test = data[data.date >= cutoff_date]
    
    # Verify no leakage
    assert train.date.max() < test.date.min()
    
    model = train_model(train)
    return evaluate_with_statistics(model, test)
```

#### 3. K-Fold (Supplementary)
```python
def stratified_kfold(data, k=5):
    """
    Stratified by outcome class, not by time.
    Use only for hyperparameter tuning on TRAIN data.
    """
    # NOTE: This should NEVER touch test data
    pass
```

### Generalization Requirements
```yaml
generalization_criteria:
  minimum_per_country_accuracy: 0.70
  maximum_country_variance: 0.15
  no_country_below_random: true
  consistent_direction_of_effect: true
```

---

## Agent 8: Literature Alignment Agent (LA)

### VASIL Paper Reference Points

| Aspect | VASIL Paper | PRISM Implementation | Status |
|--------|-------------|---------------------|--------|
| Prediction target | Variant dominance | Frequency direction | MISMATCH |
| Time horizon | Emergence to peak | Week-over-week | MISMATCH |
| Feature: Escape | Population-adjusted | Raw DMS scores | MISMATCH |
| Feature: Transmit | ACE2 binding proxy | Literature R0 | PARTIAL |
| Validation | Cross-country | Temporal split | DIFFERENT |
| Coefficients | α=0.65, β=0.35 | Learned via RL | VALID |

### Alignment Checklist
```
□ Prediction task matches VASIL's actual task
□ Features capture same biological signals
□ Validation methodology is comparable
□ Results are reported on same metrics
□ Comparison is fair (no VASIL coefficient cheating)
□ Differences from VASIL are documented and justified
```

### Fair Comparison Protocol
```python
def ensure_fair_comparison(prism_results, vasil_baseline=0.92):
    """
    Verify PRISM results are fairly comparable to VASIL.
    """
    checks = [
        # Must use same countries
        verify_same_countries(prism_results),
        
        # Must use similar time period
        verify_overlapping_period(prism_results),
        
        # Must not use VASIL's exact formula
        verify_no_formula_copying(prism_results),
        
        # Must report comparable metrics
        verify_metric_compatibility(prism_results),
    ]
    
    return FairComparisonReport(checks)
```
