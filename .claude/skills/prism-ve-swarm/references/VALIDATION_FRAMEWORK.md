# PRISM-4D Statistical Validation Framework

Publication-grade statistical rigor for all experimental results.

## Required Statistics

### Primary Metrics

| Metric | Formula | When to Report |
|--------|---------|----------------|
| Accuracy | (TP + TN) / N | Always |
| Balanced Accuracy | (TPR + TNR) / 2 | When class imbalanced |
| Precision | TP / (TP + FP) | Per-class |
| Recall (Sensitivity) | TP / (TP + FN) | Per-class |
| Specificity | TN / (TN + FP) | Per-class |
| F1 Score | 2PR / (P + R) | Summary |
| Matthews Correlation | See formula | Binary classification |

### Matthews Correlation Coefficient (MCC)
```python
def mcc(tp, tn, fp, fn):
    """
    MCC is preferred over accuracy for imbalanced data.
    Range: [-1, +1], where 0 is random performance.
    """
    numerator = (tp * tn) - (fp * fn)
    denominator = np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    
    if denominator == 0:
        return 0.0
    
    return numerator / denominator
```

---

## Confidence Interval Requirements

### For Accuracy (Proportion)

Use Wilson score interval (NOT normal approximation):

```python
from scipy import stats

def wilson_ci(successes, n, confidence=0.95):
    """
    Wilson score interval for binomial proportion.
    More accurate than normal approximation, especially for extreme p.
    """
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    
    p_hat = successes / n
    
    denominator = 1 + z**2 / n
    center = (p_hat + z**2 / (2*n)) / denominator
    margin = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denominator
    
    return (center - margin, center + margin)
```

### For Difference Between Methods

Use paired bootstrap:

```python
def bootstrap_difference_ci(pred_a, pred_b, labels, n_bootstrap=10000):
    """
    Bootstrap CI for accuracy difference between two methods.
    """
    n = len(labels)
    diffs = []
    
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        acc_a = (pred_a[idx] == labels[idx]).mean()
        acc_b = (pred_b[idx] == labels[idx]).mean()
        diffs.append(acc_a - acc_b)
    
    return np.percentile(diffs, [2.5, 97.5])
```

---

## Significance Testing

### For Single Method vs Baseline

Use exact binomial test:

```python
from scipy.stats import binomtest

def test_vs_random(accuracy, n_samples, baseline=0.5):
    """
    Test if accuracy significantly exceeds random baseline.
    """
    successes = int(accuracy * n_samples)
    result = binomtest(successes, n_samples, p=baseline, alternative='greater')
    
    return {
        'p_value': result.pvalue,
        'significant': result.pvalue < 0.05,
        'effect_size': accuracy - baseline
    }
```

### For Comparing Two Methods

Use McNemar's test (paired samples):

```python
from scipy.stats import chi2

def mcnemar_test(pred_a, pred_b, labels):
    """
    McNemar's test for paired nominal data.
    
    Null hypothesis: Both methods have equal error rates.
    """
    # Create contingency table
    a_correct = (pred_a == labels)
    b_correct = (pred_b == labels)
    
    # b = A correct, B incorrect
    b = np.sum(a_correct & ~b_correct)
    # c = A incorrect, B correct
    c = np.sum(~a_correct & b_correct)
    
    if b + c == 0:
        return {'p_value': 1.0, 'significant': False}
    
    # McNemar's chi-squared with continuity correction
    chi2_stat = (abs(b - c) - 1)**2 / (b + c)
    p_value = 1 - chi2.cdf(chi2_stat, df=1)
    
    return {
        'chi2': chi2_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'b': b,
        'c': c
    }
```

---

## Cross-Validation Requirements

### Leave-One-Country-Out (LOCO)

**Primary validation method for PRISM-4D.**

```python
def loco_validation(data, model_class, target_col='direction'):
    """
    Train on 11 countries, test on 1.
    Repeat for all 12 countries.
    """
    countries = sorted(data['country'].unique())
    assert len(countries) == 12, "Must have all 12 VASIL countries"
    
    results = []
    
    for held_out in countries:
        train = data[data['country'] != held_out]
        test = data[data['country'] == held_out]
        
        model = model_class()
        model.fit(train.drop(columns=[target_col]), train[target_col])
        
        predictions = model.predict(test.drop(columns=[target_col]))
        accuracy = (predictions == test[target_col]).mean()
        
        results.append({
            'country': held_out,
            'accuracy': accuracy,
            'n_samples': len(test),
            'n_rise': (test[target_col] == 'RISE').sum(),
            'n_fall': (test[target_col] == 'FALL').sum()
        })
    
    return LOCOResult(
        per_country=results,
        mean_accuracy=np.mean([r['accuracy'] for r in results]),
        std_accuracy=np.std([r['accuracy'] for r in results]),
        min_accuracy=min(r['accuracy'] for r in results),
        max_accuracy=max(r['accuracy'] for r in results)
    )
```

### Temporal Split (Publication Standard)

```python
def temporal_split_validation(data, cutoff='2022-06-01'):
    """
    Strict temporal split - THE validation for publication.
    """
    train = data[data['date'] < cutoff].copy()
    test = data[data['date'] >= cutoff].copy()
    
    # Verify no overlap
    assert train['date'].max() < test['date'].min()
    
    # Report split statistics
    print(f"Train: {len(train)} samples, {train['date'].min()} to {train['date'].max()}")
    print(f"Test:  {len(test)} samples, {test['date'].min()} to {test['date'].max()}")
    
    return train, test
```

---

## Overfitting Detection

### Train-Test Gap Monitoring

```python
def monitor_overfitting(train_acc, test_acc, threshold=0.10):
    """
    Flag potential overfitting if gap exceeds threshold.
    """
    gap = train_acc - test_acc
    
    status = 'OK'
    if gap > threshold:
        status = 'WARNING'
    if gap > 0.20:
        status = 'CRITICAL'
    
    return {
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'gap': gap,
        'status': status,
        'interpretation': interpret_gap(gap)
    }

def interpret_gap(gap):
    if gap < 0:
        return "Unusual: test better than train (check for data issues)"
    elif gap < 0.05:
        return "Excellent generalization"
    elif gap < 0.10:
        return "Acceptable generalization"
    elif gap < 0.20:
        return "Moderate overfitting - consider regularization"
    else:
        return "Severe overfitting - model likely memorizing"
```

### Learning Curve Analysis

```python
def learning_curve(train_data, test_data, model_class, steps=10):
    """
    Plot accuracy vs training set size to detect overfitting.
    """
    n_train = len(train_data)
    fractions = np.linspace(0.1, 1.0, steps)
    
    train_scores = []
    test_scores = []
    
    for frac in fractions:
        n = int(frac * n_train)
        subset = train_data.sample(n, random_state=42)
        
        model = model_class()
        model.fit(subset)
        
        train_scores.append(model.score(subset))
        test_scores.append(model.score(test_data))
    
    return {
        'fractions': fractions,
        'train_scores': train_scores,
        'test_scores': test_scores
    }
```

---

## Effect Size Requirements

### Cohen's d for Feature Discrimination

```python
def cohens_d(group1, group2):
    """
    Standardized effect size for continuous features.
    
    Interpretation:
    - |d| < 0.2: negligible
    - |d| 0.2-0.5: small
    - |d| 0.5-0.8: medium
    - |d| > 0.8: large
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(), group2.var()
    
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    
    return (group1.mean() - group2.mean()) / pooled_std
```

### Minimum Detectable Effect

```python
def minimum_detectable_effect(n, power=0.80, alpha=0.05):
    """
    Smallest effect size detectable with given sample size.
    """
    from statsmodels.stats.power import TTestIndPower
    
    analysis = TTestIndPower()
    effect = analysis.solve_power(nobs1=n/2, power=power, alpha=alpha)
    
    return effect
```

---

## Reporting Templates

### Single Experiment Result

```markdown
## Experiment: [Hypothesis ID]

### Configuration
- Model: FluxNet RL (4096 states)
- Features: [list]
- Training samples: N (2021-01-01 to 2022-05-31)
- Test samples: M (2022-06-01 to 2023-12-31)
- Random seed: 42

### Results

| Metric | Value | 95% CI |
|--------|-------|--------|
| Test Accuracy | X.X% | [A.A%, B.B%] |
| Balanced Accuracy | X.X% | [A.A%, B.B%] |
| MCC | 0.XX | [0.AA, 0.BB] |

### Per-Class Performance

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| RISE | X.X% | X.X% | X.XX | N |
| FALL | X.X% | X.X% | X.XX | M |

### Statistical Significance
- vs Random (50%): p < 0.001
- vs Baseline (42%): p < 0.01, Δ = +X.X pp

### Overfitting Check
- Train accuracy: XX.X%
- Test accuracy: XX.X%
- Gap: X.X pp (ACCEPTABLE)
```

### Cross-Validation Report

```markdown
## Leave-One-Country-Out Validation

| Country | Accuracy | N | RISE | FALL |
|---------|----------|---|------|------|
| Germany | XX.X% | N | n | m |
| USA | XX.X% | N | n | m |
| ... | ... | ... | ... | ... |
| **Mean** | **XX.X%** | **Total** | | |
| Std Dev | X.X% | | | |

### Generalization Assessment
- All countries above random: YES/NO
- Minimum country accuracy: XX.X% (Country)
- Maximum country accuracy: XX.X% (Country)
- Coefficient of variation: X.X%
```

---

## Red Flags to Watch

### Statistical Red Flags

| Red Flag | Indication | Action |
|----------|------------|--------|
| p = 0.049 | Barely significant | Be suspicious |
| Train-test gap > 20% | Overfitting | Add regularization |
| Perfect 100% | Too good | Check for leakage |
| Large CI | Small sample | Get more data |
| Inconsistent replication | Non-determinism | Fix random seeds |

### Reporting Red Flags

| Red Flag | Issue | Fix |
|----------|-------|-----|
| Point estimate only | No uncertainty | Add CI |
| No sample sizes | Can't assess power | Report n |
| Cherry-picked metric | Bias | Report all metrics |
| No baseline comparison | Can't interpret | Compare to random/VASIL |
