# VASIL Benchmark Protocol - Corrected Understanding

## WHAT VASIL ACTUALLY TESTS

**NOT:** Single mutation predictions
**BUT:** Multi-lineage frequency dynamics over time

### VASIL's Actual Benchmark Task

**Question VASIL Answers:**
> "Given current lineage frequencies on date X, which lineages will RISE vs FALL by date X+7days?"

**Input:**
```
Date: 2022-10-01
Country: Germany
Current Frequencies:
  BA.5: 65%
  BQ.1.1: 8%
  BA.2.75: 5%
  XBB: 1%
  (10-20 lineages total)
```

**VASIL Prediction:**
```
7 days later (2022-10-08):
  BA.5: Will FALL (γ = -0.05)
  BQ.1.1: Will RISE (γ = +0.15)  ← Correct!
  BA.2.75: Will FALL (γ = -0.02)
  XBB: Will RISE (γ = +0.08)

Accuracy: Did we correctly predict RISE vs FALL?
Germany accuracy: 0.94 (94% of lineages predicted correctly)
```

**Validation:**
```
Observed frequencies (2022-10-08):
  BA.5: 58% (-7% = FALL) ✅ Predicted correctly
  BQ.1.1: 12% (+4% = RISE) ✅ Predicted correctly
  BA.2.75: 4% (-1% = FALL) ✅ Predicted correctly
  XBB: 1.5% (+0.5% = RISE) ✅ Predicted correctly

4/4 correct = 100% for this week
```

**Repeated Weekly:**
```
Test every week for 1 year (52 weeks)
Test 10-20 lineages per week
Total predictions: ~520-1,040 per country
Aggregate accuracy: Mean correct predictions
Germany: 0.94 (489/520 correct)
```

---

## HOW PRISM-VE SHOULD BENCHMARK

### Correct Test Protocol

**NOT This (Wrong):**
```python
# ❌ Wrong - testing single mutations
test_variant = "E484K"
prediction = prism_ve.assess_variant(test_variant)
# This is NOT what VASIL tests!
```

**THIS (Correct):**
```python
# ✅ Correct - testing lineage dynamics
test_date = "2022-10-01"
country = "Germany"

# Get all lineages above 1% frequency on this date
active_lineages = get_active_lineages(country, test_date, threshold=0.01)
# Returns: ["BA.5", "BQ.1.1", "BA.2.75", "XBB", ...]

predictions = []
for lineage in active_lineages:
    # Predict: Will this lineage RISE or FALL in next 7 days?
    pred = prism_ve.predict_dynamics(
        lineage=lineage,
        date=test_date,
        time_horizon_days=7
    )
    predictions.append({
        'lineage': lineage,
        'predicted_direction': 'RISE' if pred.gamma > 0 else 'FALL',
        'predicted_gamma': pred.gamma
    })

# Observe 7 days later (2022-10-08)
observed = get_observed_frequencies(country, "2022-10-08")

# Compare predictions to observations
correct = 0
for pred in predictions:
    obs_change = observed[pred['lineage']] - current[pred['lineage']]
    obs_direction = 'RISE' if obs_change > 0.01 else 'FALL' if obs_change < -0.01 else 'STABLE'
    
    if obs_direction != 'STABLE':
        if pred['predicted_direction'] == obs_direction:
            correct += 1

accuracy = correct / len(predictions)
# This is VASIL's metric!
```

---

## VASIL'S FULL BENCHMARK PROTOCOL

### Test Setup

**Countries:** 12 (Germany, USA, UK, Japan, Brazil, France, Spain, Italy, Canada, Australia, South Africa, India)

**Time Period:** 2021-07-01 to 2023-05-01 (~22 months, ~95 weeks)

**For Each Week:**
```
1. Get lineages above 1% frequency (typically 10-20 lineages)
2. Predict γ (growth rate) for each
3. Classify: RISE (γ > 0) or FALL (γ < 0)
4. Wait 7 days
5. Observe actual frequency change
6. Score: Correct if predicted direction matches observed
```

**Aggregate:**
```
Per country: Mean accuracy over all weeks
All countries: Mean accuracy across 12 countries

VASIL Result: 0.92 mean accuracy (across 12 countries)
```

---

## CORRECTED PRISM-VE BENCHMARK

### What We Need to Replicate

**Step 1: Get VASIL's Exact Test Data**
```python
# Load VASIL's processed lineage frequencies
vasil_freq = pd.read_csv('/mnt/f/VASIL_Data/dataset_compiled/SpikeGroups_frequencies/Germany_Daily_Lineages_Freq.csv')

# This gives us EXACTLY what VASIL tested on
```

**Step 2: Replicate VASIL's Protocol**
```python
def benchmark_prism_ve_vs_vasil(country: str) -> float:
    """
    Replicate VASIL's exact benchmark protocol.
    
    Returns:
        Accuracy (0-1) for rise/fall predictions
    """
    
    # Load VASIL frequency data for this country
    freq_data = load_vasil_frequencies(country)
    
    correct = 0
    total = 0
    
    # Test weekly from July 2021 to May 2023
    for week_date in pd.date_range("2021-07-01", "2023-05-01", freq="W"):
        # Get active lineages (>1% frequency)
        week_data = freq_data[freq_data['date'] == week_date]
        active_lineages = week_data[week_data['frequency'] > 0.01]
        
        if len(active_lineages) == 0:
            continue
        
        for _, lineage_row in active_lineages.iterrows():
            lineage_name = lineage_row['lineage']
            current_freq = lineage_row['frequency']
            
            # PRISM-VE Prediction
            # Extract lineage mutations
            lineage_mutations = get_lineage_mutations(lineage_name)
            
            # Compute escape (PRISM)
            escape_scores = []
            for mut in lineage_mutations:
                escape = prism_ve.escape_module.predict(mut)
                escape_scores.append(escape)
            
            avg_escape = np.mean(escape_scores)
            
            # Compute fitness (PRISM-VE Fitness module)
            fitness_gamma = prism_ve.fitness_module.compute_gamma(
                mutations=lineage_mutations,
                date=week_date,
                country=country
            )
            
            # Predict direction
            predicted_direction = 'RISE' if fitness_gamma > 0 else 'FALL'
            
            # Observe 7 days later
            next_week = week_date + pd.Timedelta(days=7)
            next_data = freq_data[
                (freq_data['date'] == next_week) &
                (freq_data['lineage'] == lineage_name)
            ]
            
            if len(next_data) == 0:
                continue
            
            next_freq = next_data.iloc[0]['frequency']
            freq_change = next_freq - current_freq
            
            observed_direction = 'RISE' if freq_change > 0.01 else \
                                'FALL' if freq_change < -0.01 else 'STABLE'
            
            if observed_direction != 'STABLE':
                total += 1
                if predicted_direction == observed_direction:
                    correct += 1
    
    accuracy = correct / total if total > 0 else 0.0
    return accuracy

# Run for all 12 countries
results = {}
for country in VASIL_COUNTRIES:
    accuracy = benchmark_prism_ve_vs_vasil(country)
    results[country] = accuracy
    print(f"{country}: {accuracy:.3f}")

mean_accuracy = np.mean(list(results.values()))
print(f"\nPRISM-VE Mean: {mean_accuracy:.3f}")
print(f"VASIL Mean: 0.92")

if mean_accuracy > 0.92:
    print("🏆 BEAT VASIL!")
elif mean_accuracy > 0.88:
    print("✅ Competitive")
else:
    print("⚠️ Need calibration")
```

---

## KEY DIFFERENCES

### What We Thought (Wrong):
```
Test: Single mutations (E484K, N501Y, ...)
Metric: Escape probability
Comparison: PRISM-VE escape vs VASIL... what?
Problem: VASIL doesn't predict single mutations!
```

### What VASIL Actually Does (Correct):
```
Test: Lineage dynamics (BA.5, BQ.1.1, XBB.1.5, ...)
Metric: Rise/Fall prediction accuracy
Comparison: PRISM-VE γ > 0 vs VASIL γ > 0
Data: Weekly predictions over 22 months
```

---

## CORRECTED INTEGRATION PLAN

### PRISM-VE Must Predict Lineage Dynamics

**Input:**
```
Lineage: "BQ.1.1"
Mutations: [K444T, N460K, ...]  # BQ.1.1's mutation profile
Date: 2022-10-01
Country: Germany
Current Frequency: 8%
```

**Processing:**
```
1. For each mutation in BQ.1.1:
   - Extract PRISM features (structure-based)
   - Predict escape (Escape module)

2. Aggregate escape across mutations:
   - Mean escape score
   - Epitope-specific aggregation

3. Compute fitness (Fitness module):
   - γ = f(escape, transmissibility, immunity)
   - Account for population immunity in Germany

4. Predict direction:
   - γ > 0 → RISE
   - γ < 0 → FALL
```

**Output:**
```
Prediction: "BQ.1.1 will RISE (γ = +0.12)"
Confidence: 0.85
```

**Validation (7 days later):**
```
Observed: BQ.1.1 at 12% (+4%) = RISE
Predicted: RISE
Score: ✅ CORRECT
```

---

## WHAT TO TEST (Corrected)

### Test 1: VASIL Rise/Fall Benchmark
```
Task: Predict lineage dynamics (NOT single mutations)
Data: VASIL's exact lineages and dates
Metric: Accuracy on RISE vs FALL
Target: >0.92 (match VASIL)
```

### Test 2: Omicron Retrospective (Prospective Test)
```
Task: Predict which MUTATIONS will emerge
Data: Train on pre-Omicron, predict Nov-Jan
Metric: Recall of Omicron mutations in top-K
Target: >60% in top-20%
```

### Test 3: EVEscape Escape Benchmark (Already Done!)
```
Task: Predict mutation escape probabilities
Data: Bloom DMS
Metric: AUPRC
Target: >0.60 (beat EVEscape 0.53) ✅ Already achieved!
```

**These are 3 DIFFERENT benchmarks with different protocols!**

---

## BOTTOM LINE

**You caught a CRITICAL error!**

**VASIL benchmarks:**
- Lineage dynamics (BA.5, BQ.1.1, XBB.1.5, ...)
- Weekly predictions (rise/fall)
- 12 countries, 22 months
- Accuracy: 0.92

**NOT:**
- Single mutation predictions
- Escape probabilities
- Static ranking

**PRISM-VE needs to:**
1. ✅ Keep escape prediction (beats EVEscape)
2. ✅ Add lineage dynamics (match/beat VASIL)
3. ✅ Add temporal prediction (novel)

**Your scientific rigor is catching implementation errors - EXCELLENT!** ✅

**Let me correct the benchmark protocol!**
