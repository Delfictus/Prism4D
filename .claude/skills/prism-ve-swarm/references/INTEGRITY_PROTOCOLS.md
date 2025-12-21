# PRISM-4D Scientific Integrity Protocols

**FOUNDATIONAL PRINCIPLE**: No accuracy improvement is worth a single integrity violation. A 70% accurate model with perfect methodology is infinitely more valuable than a 95% accurate model with questionable integrity.

## The Integrity Oath

Before any optimization work:

```
I will not use future information to predict the past.
I will not train on test data or test on train data.
I will not hardcode coefficients from the paper I'm trying to beat.
I will not cherry-pick results or hide failures.
I will document every modification and its rationale.
I will ensure every result is reproducible.
I will report confidence intervals, not just point estimates.
I will acknowledge when my method differs from VASIL.
```

---

## Integrity Violation Taxonomy

### CRITICAL Violations (Immediate Halt)

#### IG-001: Look-Ahead Bias
**Definition**: Using information from time T+k to compute features for prediction at time T.

**Examples**:
- Using "next week's frequency" as a feature
- Computing escape relative to variants that haven't emerged yet
- Using end-of-study statistics in feature normalization

**Detection**:
```python
def detect_look_ahead(feature_fn, sample):
    # Instrument feature computation
    accessed_dates = trace_data_access(feature_fn, sample)
    
    for date in accessed_dates:
        if date > sample.prediction_date:
            raise LookAheadViolation(
                f"Feature accessed data from {date} "
                f"for prediction at {sample.prediction_date}"
            )
```

**Consequence**: Entire experiment invalidated, results cannot be published.

---

#### IG-002: Train/Test Leakage
**Definition**: Any information flow from test set to training process.

**Examples**:
- Normalizing features using test set statistics
- Selecting hyperparameters based on test performance
- Using test samples in any training decision

**Prevention**:
```python
class DataSplit:
    def __init__(self, data, cutoff='2022-06-01'):
        self.train = data[data.date < cutoff].copy()
        self.test = data[data.date >= cutoff].copy()
        
        # Compute ALL statistics on train only
        self.train_mean = self.train.mean()
        self.train_std = self.train.std()
        
        # Normalize both using TRAIN statistics
        self.train_normalized = (self.train - self.train_mean) / self.train_std
        self.test_normalized = (self.test - self.train_mean) / self.train_std
```

**Consequence**: Model must be retrained from scratch with proper separation.

---

#### IG-003: Coefficient Hardcoding
**Definition**: Using VASIL paper's specific coefficients (α=0.65, β=0.35) in the model.

**Examples**:
- `fitness = 0.65 * escape + 0.35 * transmit`
- Initializing Q-table with VASIL weights
- Using VASIL thresholds for classification

**Why This Matters**: PRISM's claim is that RL can LEARN optimal weights. Using VASIL's coefficients defeats the entire scientific contribution.

**Detection**:
```python
FORBIDDEN_CONSTANTS = [0.65, 0.35, 0.92]  # VASIL's magic numbers

def scan_for_hardcoded_coefficients(code_path):
    with open(code_path) as f:
        code = f.read()
    
    for const in FORBIDDEN_CONSTANTS:
        if str(const) in code:
            # Check if it's in a suspicious context
            context = extract_context(code, const)
            if 'escape' in context or 'transmit' in context:
                raise CoefficientViolation(
                    f"Suspicious constant {const} found near "
                    f"escape/transmit computation"
                )
```

**Consequence**: Complete invalidation. This is scientific fraud.

---

### HIGH Violations (Reject Hypothesis)

#### IG-004: Future Date Access
**Definition**: Feature computation references dates beyond the sample's date.

**Example**: Computing "mean escape of all variants in dataset" includes future variants.

**Fix**: Filter to only contemporaneous data.

---

#### IG-005: Non-Reproducibility
**Definition**: Results change across runs with same configuration.

**Requirements**:
```python
# ALL randomness must be seeded
MASTER_SEED = 42

np.random.seed(MASTER_SEED)
torch.manual_seed(MASTER_SEED)
random.seed(MASTER_SEED)

# CUDA determinism (if applicable)
torch.backends.cudnn.deterministic = True
```

**Verification**: Run same experiment 3 times, results must be identical.

---

### MEDIUM Violations (Warning)

#### IG-006: Undocumented Modification
**Definition**: Code change without recorded rationale.

**Requirement**: Every modification must have:
- Git commit message explaining WHY
- Entry in experiment log
- Link to hypothesis being tested

---

#### IG-007: Missing Confidence Interval
**Definition**: Reporting point estimate without uncertainty.

**Requirement**: All accuracy reports must include:
- 95% confidence interval
- Sample size
- Validation method used

**Format**:
```
Accuracy: 87.3% (95% CI: 85.1% - 89.2%, n=9,340, temporal split)
```

---

## Temporal Integrity Framework

### The Time Firewall

```
TRAINING ERA                    │  TESTING ERA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━│━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2021-01-01  ───────────────────│──────────────────  2023-12-31
        Delta Emergence         │    Omicron Era
        1,745 samples           │    9,340 samples
                                │
        TRAIN HERE ◄────────────│────────── TEST HERE
                                │
                          2022-06-01
                          (CUTOFF DATE)
                                │
        ════════════════════════╪════════════════════════════
        Information can flow ───┼──► Information CANNOT flow
        from train to model     │    from test to anything
```

### What CAN Cross the Firewall
- Model architecture (decided before seeing test data)
- Feature definitions (same for train and test)
- Trained model weights (learned from train only)

### What CANNOT Cross the Firewall
- Test accuracy (until final evaluation)
- Test data statistics
- Test sample characteristics
- Any information derived from test data

---

## The Integrity Audit Checklist

Before any result is reported:

```
TEMPORAL INTEGRITY
□ Train/test split uses 2022-06-01 cutoff
□ No samples from test period used in training
□ No test statistics used in feature normalization
□ Model was frozen before test evaluation

FEATURE INTEGRITY
□ All features computed identically for train/test
□ No features use future information
□ Feature distributions inspected for anomalies
□ Feature importance makes biological sense

COEFFICIENT INTEGRITY
□ No VASIL coefficients (0.65, 0.35) in code
□ All weights learned from data
□ Q-table initialized uniformly or randomly
□ Thresholds determined on training data only

REPRODUCIBILITY
□ Random seed set and documented
□ Results identical across 3 runs
□ Environment fully specified
□ Code version controlled

STATISTICAL INTEGRITY
□ Confidence intervals computed
□ Significance tests performed
□ Effect sizes reported
□ Multiple comparison corrections applied (if relevant)

DOCUMENTATION INTEGRITY
□ All modifications logged
□ Negative results recorded
□ Failed hypotheses documented
□ Methodology differences from VASIL noted
```

---

## The Red Team Protocol

Every major result undergoes adversarial review:

### Red Team Questions
1. "How could this result be an artifact of data leakage?"
2. "What if you accidentally used future information?"
3. "Could this be overfitting to the test set?"
4. "Is there any way VASIL coefficients snuck in?"
5. "Would this result replicate in a new country?"
6. "Is the comparison to VASIL actually fair?"

### Red Team Response Requirements
Each question must be answered with:
- Specific evidence (code reference, data audit)
- Reproducibility demonstration
- Alternative explanation consideration

---

## Integrity Certificate Template

After passing all checks:

```yaml
PRISM-4D INTEGRITY CERTIFICATE
==============================

Experiment ID: EXP-2024-{sequence}
Date: YYYY-MM-DD
Signed By: Integrity Guardian Agent

VERIFICATION RESULTS:
  Temporal Integrity: PASSED
    - Train/test split verified
    - No look-ahead bias detected
    - Cutoff date: 2022-06-01
  
  Feature Integrity: PASSED
    - N features audited: 101
    - Future-date violations: 0
    - Distribution anomalies: 0
  
  Coefficient Integrity: PASSED
    - Code scanned for forbidden constants
    - Q-table initialization: uniform
    - All weights learned from data
  
  Reproducibility: PASSED
    - Random seed: 42
    - 3/3 runs identical
    - Environment hash: {sha256}
  
  Statistical Integrity: PASSED
    - 95% CI computed: [X%, Y%]
    - Significance test: p < 0.001
    - Effect size: +Z pp

CERTIFICATION:
  This experiment meets all PRISM-4D integrity requirements
  and is suitable for publication.

CAVEATS:
  - [Any known limitations]
  - [Differences from VASIL methodology]
```

---

## Recovery Procedures

### If Integrity Violation Detected

1. **STOP** all optimization immediately
2. **DOCUMENT** the violation in detail
3. **IDENTIFY** all affected experiments
4. **INVALIDATE** results that may be contaminated
5. **ROOT CAUSE** analysis
6. **FIX** the underlying issue
7. **RE-RUN** affected experiments from scratch
8. **VERIFY** fix with integrity audit

### Contamination Spread Assessment
```python
def assess_contamination(violation, experiment_log):
    """
    Determine which experiments are affected by a violation.
    """
    contaminated = []
    
    for exp in experiment_log:
        if exp.date >= violation.date:
            if exp.uses_artifact(violation.artifact):
                contaminated.append(exp)
            if exp.inherits_from(violation.experiment):
                contaminated.append(exp)
    
    return contaminated
```

---

## The Nuclear Option

If a CRITICAL violation is discovered after results are published:

1. **Retract** any claims based on contaminated results
2. **Notify** any collaborators or stakeholders
3. **Document** the error publicly
4. **Re-do** the entire analysis with proper methodology
5. **Publish** corrected results with explanation

Scientific integrity is not negotiable. Period.
