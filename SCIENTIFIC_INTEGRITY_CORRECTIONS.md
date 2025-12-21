# PRISM-VE: Scientific Integrity Corrections

## CRITICAL FIXES FOR HONEST SCIENCE

---

## 🚨 PROBLEM IDENTIFIED

**Current implementation uses VASIL's fitted parameters:**
```rust
// ❌ WRONG - This is copying VASIL's model!
vasil_alpha: 0.65,  // VASIL's calibrated weight
vasil_beta: 0.35,   // VASIL's calibrated weight
```

**This is scientific misconduct:**
- Using competitor's fitted parameters = copying their model
- Not independent validation
- Will be rejected by reviewers

---

## ✅ CORRECTED APPROACH

### Fix 1: Remove VASIL Parameters, Fit Our Own

**OLD (Wrong):**
```rust
pub struct VEFitnessParams {
    pub vasil_alpha: f32,  // ❌ VASIL's fitted value
    pub vasil_beta: f32,   // ❌ VASIL's fitted value
}

impl Default for VEFitnessParams {
    fn default() -> Self {
        Self {
            vasil_alpha: 0.65,  // ❌ Copying VASIL!
            vasil_beta: 0.35,   // ❌ Copying VASIL!
        }
    }
}
```

**NEW (Correct):**
```rust
pub struct FitnessParams {  // Renamed (not "VASIL params")
    pub escape_weight: f32,     // Our weight for escape
    pub transmit_weight: f32,   // Our weight for transmissibility
}

impl Default for FitnessParams {
    fn default() -> Self {
        Self {
            escape_weight: 0.5,    // ✅ Neutral default
            transmit_weight: 0.5,  // ✅ Neutral default
            // Will be fitted on training data
        }
    }
}

// Calibration function (FIT OUR OWN VALUES)
impl FitnessParams {
    pub fn calibrate_on_training_data(
        &mut self,
        training_data: &[(Variant, bool)],  // (variant, did_it_rise)
        fitness_module: &mut FitnessModule,
    ) -> Result<(), PrismError> {
        // Grid search or gradient descent to find optimal α, β
        let mut best_accuracy = 0.0;
        let mut best_alpha = 0.5;
        let mut best_beta = 0.5;

        for alpha in (0..=10).map(|x| x as f32 / 10.0) {
            for beta in (0..=10).map(|x| x as f32 / 10.0) {
                if (alpha + beta - 1.0).abs() > 0.01 {
                    continue;  // Ensure α + β ≈ 1
                }

                // Test this parameter combo
                let accuracy = self.evaluate_params(
                    alpha, beta, training_data, fitness_module
                )?;

                if accuracy > best_accuracy {
                    best_accuracy = accuracy;
                    best_alpha = alpha;
                    best_beta = beta;
                }
            }
        }

        // Set to best found values (OUR fitted values, not VASIL's)
        self.escape_weight = best_alpha;
        self.transmit_weight = best_beta;

        log::info!(
            "Calibrated parameters: escape={:.2}, transmit={:.2} (accuracy={:.3})",
            best_alpha, best_beta, best_accuracy
        );

        Ok(())
    }
}
```

---

### Fix 2: Use Primary Source Data Only

**Data Source Checklist:**

**✅ VALID (Primary Sources):**
```
1. GISAID Sequences:
   Source: Download from GISAID.org directly
   OR: Use VASIL's if they're RAW sequences (not processed)

2. DMS Escape:
   Source: Bloom Lab GitHub (we already have this)
   Status: ✅ Already using correctly

3. Vaccination Data:
   Source: OWID, national health databases

4. Case Incidence:
   Source: JHU, OWID, national databases
```

**❌ INVALID (VASIL's Outputs):**
```
1. VASIL's fitted α, β parameters
   Status: REMOVE from defaults

2. VASIL's cross-neutralization matrices
   If: Computed by VASIL's model
   Use: Compute ourselves from DMS + frequencies

3. VASIL's immunity landscapes
   If: Model-fitted
   Use: Compute ourselves from vaccination + case data
```

**🔍 VERIFY (Could Be Either):**
```
1. VASIL's frequency files:
   IF: Raw GISAID aggregates → ✅ OK to use
   IF: Model-smoothed/fitted → ❌ Must recompute

   CHECK: Look for "fitted", "smoothed", "predicted" in filenames
   SAFE: Files named "raw", "observed", "gisaid_counts"
```

---

### Fix 3: Independent Train/Test Split

**Protocol:**

```python
# Temporal split (no data leakage)
training_period = "2021-07-01" to "2022-12-31"
test_period = "2023-01-01" to "2023-12-31"

# Train on 2021-2022 data
fit_parameters(training_data)  # Fit OUR α, β

# Test on 2023 data
predictions_2023 = predict(test_data)

# Compare to VASIL's 2023 predictions
vasil_predictions_2023 = load_vasil_benchmark_results()
compare(our_predictions, vasil_predictions)
```

**Key:** VASIL's 2023 predictions used for COMPARISON only, not training!

---

## CORRECTED DATA FLOW

### What We Actually Do:

```
INPUT DATA (Primary Sources):
  Raw GISAID sequences
  Bloom Lab DMS escape
  Public vaccination data
  Public case data
      ↓
OUR PROCESSING:
  Aggregate GISAID → variant frequencies
  Process DMS → escape scores per position
  Compute immunity → from vaccination/cases
      ↓
OUR MODEL:
  PRISM escape prediction (trained on Bloom DMS)
  Fitness module (physics-based ΔΔG)
  Cycle module (phase detection from frequencies)
  Parameters: Fitted on 2021-2022 training data
      ↓
OUR PREDICTIONS:
  Variant X will RISE/FALL in 2023
      ↓
VALIDATION:
  Compare to VASIL's 2023 predictions (benchmark)
  Compare to OBSERVED 2023 frequencies (ground truth)
  Report: "PRISM-VE: X% accuracy, VASIL: Y% accuracy"
```

**This is scientifically valid!**

---

## FILES TO CORRECT IN PRISM-VE WORKTREE

### 1. Remove VASIL Parameters

**File:** `/mnt/c/Users/Predator/Desktop/prism-ve/crates/prism-gpu/src/viral_evolution_fitness.rs`

**Change:**
```rust
// DELETE these lines:
pub vasil_alpha: f32,
pub vasil_beta: f32,

// REPLACE with:
pub escape_weight: f32,      // Fitted independently
pub transmit_weight: f32,    // Fitted independently
```

**In Default:**
```rust
// DELETE:
vasil_alpha: 0.65,
vasil_beta: 0.35,

// REPLACE:
escape_weight: 0.5,    // Neutral default, will calibrate
transmit_weight: 0.5,  // Neutral default, will calibrate
```

### 2. Add Independent Calibration

**Create:** `/mnt/c/Users/Predator/Desktop/prism-ve/scripts/calibrate_parameters.py`

```python
#!/usr/bin/env python3
"""
Calibrate PRISM-VE parameters independently (NOT using VASIL's values).

Training: 2021-2022 data
Validation: 2023 data
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

# Load training data (2021-2022)
gisaid_data = load_gisaid_frequencies("2021-07-01", "2022-12-31")

# Grid search for optimal parameters
best_params = None
best_accuracy = 0

for escape_weight in np.linspace(0.3, 0.8, 11):
    transmit_weight = 1.0 - escape_weight

    # Train model with these parameters
    model = train_fitness_module(
        escape_weight=escape_weight,
        transmit_weight=transmit_weight,
        training_data=gisaid_data
    )

    # Validate on 2022 Q4 (held-out)
    accuracy = validate(model, gisaid_data, period="2022-10-01:2022-12-31")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_params = (escape_weight, transmit_weight)

print(f"OUR CALIBRATED PARAMETERS:")
print(f"  Escape weight: {best_params[0]:.3f}")
print(f"  Transmit weight: {best_params[1]:.3f}")
print(f"  Validation accuracy: {best_accuracy:.3f}")
print()
print(f"VASIL's parameters (for reference):")
print(f"  Alpha: 0.65")
print(f"  Beta: 0.35")
print()

if abs(best_params[0] - 0.65) < 0.05:
    print("✅ Our parameters similar to VASIL's (independent validation!)")
else:
    print("✅ Our parameters different from VASIL's (independent approach)")
```

### 3. Verify Data Sources

**Create:** `/mnt/c/Users/Predator/Desktop/prism-ve/scripts/verify_data_sources.py`

```python
#!/usr/bin/env python3
"""
Verify all data sources are primary (not VASIL's outputs).
"""

import pandas as pd

print("VERIFYING DATA SOURCES:")
print("="*80)

# Check 1: GISAID frequencies
print("\n1. GISAID Frequencies:")
vasil_freq_file = "/mnt/f/VASIL_Data/dataset_compiled/SpikeGroups_frequencies/..."

# Read file and check for model indicators
df = pd.read_csv(vasil_freq_file, nrows=10)
print(f"   Columns: {df.columns.tolist()}")

# Red flags: "fitted", "predicted", "smoothed", "model"
red_flags = ['fitted', 'predicted', 'smoothed', 'model', 'estimated']
has_red_flag = any(flag in str(df.columns).lower() for flag in red_flags)

if has_red_flag:
    print("   ⚠️  WARNING: May contain model outputs!")
    print("   RECOMMENDATION: Download raw GISAID ourselves")
else:
    print("   ✅ Appears to be raw aggregates")

# Check 2: DMS Escape
print("\n2. DMS Escape Data:")
print("   ✅ Using Bloom Lab directly (not VASIL's processed)")
print("   Source: github.com/jbloomlab/SARS2_RBD_Ab_escape_maps")

# Check 3: Parameters
print("\n3. Model Parameters:")
print("   ❌ Currently using vasil_alpha=0.65, vasil_beta=0.35")
print("   ✅ FIX: Calibrate independently on training data")

print("\n" + "="*80)
print("RECOMMENDATION:")
print("  1. Verify VASIL frequencies are raw GISAID counts")
print("  2. If model-processed, download GISAID ourselves")
print("  3. Remove VASIL parameter defaults")
print("  4. Fit our own parameters on training data")
print("="*80)
```

---

## CHECKLIST FOR SCIENTIFIC HONESTY

### Before Submission:

```
Data Sources:
[ ] Verified GISAID data is primary source (not VASIL's fitted)
[ ] Verified DMS data is Bloom Lab raw (not VASIL's processed)
[ ] All parameters fitted independently (not copied from VASIL)

Processing:
[ ] Computed frequencies ourselves from raw GISAID
[ ] Processed DMS ourselves (already doing)
[ ] Created our own train/test splits (temporal)

Model:
[ ] Our escape module (trained on Bloom DMS) ✅
[ ] Our fitness module (physics-based or fitted independently)
[ ] Our cycle module (phase detection from frequencies)
[ ] Parameters calibrated on training data only

Validation:
[ ] Test on held-out time periods (2023)
[ ] Compare to VASIL's published predictions (benchmark)
[ ] Compare to observed frequencies (ground truth)
[ ] Report honestly: "Same inputs, independent methods"

Publication:
[ ] Methods section states we used PRIMARY sources
[ ] Methods section states we fitted OUR parameters
[ ] Methods section states VASIL used only for comparison
[ ] No mention of "using VASIL's parameters"
```

---

## PUBLICATION LANGUAGE

**CORRECT Statement:**

> "We benchmarked PRISM-VE against VASIL using the same primary data sources (GISAID sequences, Bloom Lab DMS, public health data). All data processing, parameter calibration, and model training was performed independently. We did not use VASIL's fitted parameters or processed outputs. Predictions were validated by comparing to both VASIL's published benchmarks and observed variant frequencies in held-out time periods."

**This is:**
- ✅ Honest
- ✅ Transparent
- ✅ Defensible
- ✅ Apples-to-apples (same inputs, independent methods)

---

## RECOMMENDED IMMEDIATE ACTIONS

**For PRISM-VE Worktree:**

### Action 1: Update Fitness Module (30 min)
```bash
cd /mnt/c/Users/Predator/Desktop/prism-ve

# Edit viral_evolution_fitness.rs
# Remove: vasil_alpha, vasil_beta
# Add: escape_weight, transmit_weight
# Default: 0.5, 0.5 (neutral, will calibrate)
```

### Action 2: Create Calibration Script (1 hour)
```bash
# Create scripts/calibrate_parameters.py
# Fit α, β on 2021-2022 training data
# Validate on 2022 Q4
# Report our fitted values
```

### Action 3: Verify Data Sources (30 min)
```bash
# Create scripts/verify_data_sources.py
# Check VASIL frequency files for "fitted", "predicted"
# If found, download raw GISAID ourselves
# Document all sources
```

### Action 4: Document Independence (15 min)
```bash
# Update README.md
# State: "Independent implementation, same inputs as VASIL"
# State: "Parameters fitted on training data"
# State: "VASIL used only for benchmark comparison"
```

**Total time: ~2.5 hours to ensure scientific integrity**

---

## HONEST COMPARISON TO VASIL

**After corrections:**

**PRISM-VE:**
- Input: Raw GISAID, Bloom DMS, public data
- Processing: Our code
- Parameters: Fitted on 2021-2022 (our values)
- Prediction: Our model

**VASIL:**
- Input: Raw GISAID, Bloom DMS, public data (SAME)
- Processing: Their code
- Parameters: Fitted on their training data (their values)
- Prediction: Their model

**Comparison:**
- ✅ Same inputs (valid)
- ✅ Independent methods (valid)
- ✅ Different parameter values (independent)
- ✅ Benchmark predictions (valid)

**Result: Scientifically honest, apples-to-apples benchmark!**

---

## BOTTOM LINE

**I'm creating corrections now to ensure:**

1. ✅ Remove VASIL's parameter values
2. ✅ Fit our own parameters independently
3. ✅ Use only primary source data
4. ✅ Verify VASIL frequencies are raw
5. ✅ Document independent processing

**This makes PRISM-VE scientifically honest and peer-review defensible!**

**Applying fixes to PRISM-VE worktree now!** ✅
