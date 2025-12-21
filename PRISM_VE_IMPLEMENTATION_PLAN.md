# PRISM-VE HYBRID MODEL - Complete Implementation Blueprint

## Executive Summary
Building **PRISM-VE**: A novel hybrid model combining epidemiological immunity dynamics with GPU-accelerated structural protein analysis for SARS-CoV-2 variant fitness prediction.

**Goal**: Match or beat VASIL's 92% accuracy using our own architecture and independently fitted weights.

---

## 1. Architecture Overview

```
                    ┌─────────────────────────────┐
                    │   PRISM-VE Hybrid Model     │
                    └──────────┬──────────────────┘
                               │
                ┌──────────────┴──────────────┐
                │                             │
        ┌───────▼────────┐           ┌───────▼──────────┐
        │  Epidemiological│           │   Structural     │
        │    Features     │           │    Features      │
        │   (VASIL Data)  │           │  (GPU Pipeline)  │
        └───────┬────────┘           └───────┬──────────┘
                │                             │
    ┌───────────┼─────────────┐      ┌───────┼──────────┐
    │           │             │      │       │          │
    ▼           ▼             ▼      ▼       ▼          ▼
 Immunity   Cross-React   P_neut  ddG    Burial   Expression
 Dynamics     Matrix                Binding
    │           │             │      │       │          │
    └───────────┴─────────────┴──────┴───────┴──────────┘
                               │
                        ┌──────▼──────┐
                        │ VE-Swarm    │
                        │ Ensemble    │
                        │ (32 agents) │
                        └──────┬──────┘
                               │
                        ┌──────▼──────┐
                        │  Weighted   │
                        │ Consensus   │
                        │  γ_PRISM_VE │
                        └─────────────┘
```

---

## 2. Hybrid Fitness Formula

```rust
γ_PRISM_VE = w1 × log_fold_reduction +
              w2 × transmissibility +
              w3 × velocity_inversion +
              w4 × structural_ddG +
              w5 × frequency_saturation +
              w6 × swarm_consensus
```

### Component Details:

#### 2.1 Log Fold Reduction (Epidemiological)
```
fold_reduction = exp(Σ_{i=1}^{10} escape[epitope_i] × immunity[epitope_i, t])

where:
  epitope_i ∈ {A, B, C, D1, D2, E12, E3, F1, F2, F3}
  immunity[epitope_i, t] = I_max × exp(-(t - t_max) × ln(2) / t_half)
  t = days since last major infection wave
  t_half ∈ [25.0, 87.857] days (75 parameter combinations)
  t_max ∈ [14.0, 49.0] days
```

#### 2.2 Velocity Inversion (Our Discovery)
```rust
velocity_inversion = {
    -velocity × 1.5           if freq > 0.5 (dominant => at peak)
    velocity × 0.2 - 0.15     if freq > 0.2 && velocity > 0.05
    velocity × 1.5            if freq < 0.1 && velocity > 0 (true growth)
    velocity × 0.5            otherwise
}
```

**Data Evidence**: RISE velocity = 0.016, FALL velocity = 0.106 (6x higher)

#### 2.3 Structural ddG (GPU Features)
From `MegaFusedBatchGpu` feature 92:
```
ddG_binding = Δ(binding_energy_mutant - binding_energy_wildtype)
Favorable mutations: ddG < 0 (stronger binding)
```

#### 2.4 Frequency Saturation
```
saturation = 1 - freq  // Room to grow
```

#### 2.5 Swarm Consensus
Average of 32 VE-Swarm agents with genetic evolution

---

## 3. Data Sources

### 3.1 VASIL Data (Epidemiological)

**Location**: `/mnt/f/VASIL_Data/ByCountry/{country}/`

| File | Content | Usage |
|------|---------|-------|
| `results/Immunological_Landscape_groups/Immunized_SpikeGroup_{variant}_all_PK.csv` | 75 PK parameter combinations in headers | Waning immunity I(t) |
| `results/Cross_react_dic_spikegroups_ALL.pck` | 10x 136x136 cross-immunity matrices | Inter-variant immunity |
| `results/Immunological_Landscape_groups/P_neut_{variant}.csv` | Neutralization vs days since infection | Immune escape dynamics |
| `results/PK_for_all_Epitopes.csv` | Epitope-specific immunity decay | Fine-grained immunity |
| `results/epitope_data/dms_per_ab_per_site.csv` | Per-antibody, per-site escape | Detailed escape mapping |
| `smoothed_phi_estimates_{country}.csv` | Case ascertainment rate | Testing bias correction |

**Extracted Data**:
- PK parameters: `/mnt/c/Users/Predator/Desktop/prism-ve/data/pk_parameters.json`
- Cross-reactivity: `/mnt/c/Users/Predator/Desktop/prism-ve/data/cross_reactivity_summary.json`

### 3.2 GPU Structural Features

**Source**: `prism-gpu/MegaFusedBatchGpu` kernel

| Feature Index | Description | Value Range |
|---------------|-------------|-------------|
| 92 | ddG binding | [-2.0, +2.0] |
| 93 | ddG stability | [-2.0, +2.0] |
| 94 | Expression | [0, 1] |
| 95 | Transmissibility | [0, 1] |
| 0-91 | Pocket features | Various |

---

## 4. File Structure

```
crates/prism-ve-bench/src/
├── main.rs                      # Main benchmark (UPDATE)
├── prism_ve_model.rs            # NEW: PRISM-VE hybrid model
├── immunity_dynamics.rs         # NEW: Time-varying immunity
├── vasil_data.rs                # UPDATE: Add PK parsing
├── ve_swarm_integration.rs      # KEEP: Swarm ensemble
├── gpu_benchmark.rs             # KEEP: Structural features
└── ve_optimizer.rs              # UPDATE: Add hybrid weight fitting

data/
├── pk_parameters.json           # CREATED: 75 PK combinations
├── cross_reactivity_summary.json # CREATED: Cross-immunity matrix
└── spike_rbd_6m0j.pdb           # EXISTING: Reference structure
```

---

## 5. Core Implementation

### 5.1 `prism_ve_model.rs` - Main Hybrid Model

```rust
//! PRISM-VE: Hybrid Epidemiological + Structural Model
//!
//! Combines VASIL immunity dynamics with GPU-accelerated structural analysis

use anyhow::Result;
use std::collections::HashMap;
use chrono::NaiveDate;

/// PRISM-VE Hybrid Predictor
pub struct PRISMVEPredictor {
    /// Epidemiological features
    immunity_dynamics: ImmunityDynamics,
    cross_reactivity: CrossReactivityMatrix,

    /// VE-Swarm ensemble (32 agents)
    ve_swarm: VeSwarmPredictor,

    /// Learned weights (fitted on training data)
    weights: HybridWeights,

    /// Statistics
    prediction_count: usize,
    correct_count: usize,
}

#[derive(Clone, Debug)]
pub struct HybridWeights {
    pub w_fold_reduction: f32,    // Epidemiological escape
    pub w_transmissibility: f32,  // R0 advantage
    pub w_velocity_inv: f32,      // Momentum inversion
    pub w_structural_ddg: f32,    // Binding energy
    pub w_freq_saturation: f32,   // Room to grow
    pub w_swarm_consensus: f32,   // Ensemble vote
    pub threshold: f32,           // Classification threshold
}

impl Default for HybridWeights {
    fn default() -> Self {
        Self {
            // Initial guesses (to be fitted)
            w_fold_reduction: 0.4,
            w_transmissibility: 0.2,
            w_velocity_inv: -0.3,    // Negative: high velocity = FALL
            w_structural_ddg: 0.15,
            w_freq_saturation: -0.2, // Negative: high freq = saturated
            w_swarm_consensus: 0.25,
            threshold: 0.45,         // Adjusted for 40% RISE base rate
        }
    }
}

impl PRISMVEPredictor {
    /// Create new PRISM-VE model with VASIL data
    pub fn new(
        vasil_data_dir: &Path,
        ve_swarm: VeSwarmPredictor,
    ) -> Result<Self> {
        let immunity_dynamics = ImmunityDynamics::load_from_vasil(vasil_data_dir)?;
        let cross_reactivity = CrossReactivityMatrix::load_from_vasil(vasil_data_dir)?;

        Ok(Self {
            immunity_dynamics,
            cross_reactivity,
            ve_swarm,
            weights: HybridWeights::default(),
            prediction_count: 0,
            correct_count: 0,
        })
    }

    /// Predict RISE/FALL using hybrid model
    pub fn predict(
        &mut self,
        country: &str,
        lineage: &str,
        date: NaiveDate,
        frequency: f32,
        velocity: f32,
        epitope_escape: &[f32; 10],
        structural_features: &StructuralFeatures,
    ) -> (bool, f32) {
        self.prediction_count += 1;

        // 1. Compute time-varying immunity
        let days_since_outbreak = estimate_days_since_outbreak(country, &date);
        let immunity = self.immunity_dynamics.compute_immunity(
            country,
            lineage,
            days_since_outbreak,
        );

        // 2. Compute fold reduction with cross-reactivity
        let fold_reduction = self.compute_fold_reduction(
            epitope_escape,
            &immunity,
            lineage,
        );
        let log_fold_reduction = fold_reduction.ln();

        // 3. Velocity inversion (KEY feature!)
        let velocity_inv = self.correct_velocity(velocity, frequency);

        // 4. Structural features from GPU
        let ddg_binding = structural_features.ddg_binding;
        let transmissibility = structural_features.transmissibility;

        // 5. Frequency saturation
        let freq_saturation = 1.0 - frequency;

        // 6. VE-Swarm consensus
        let swarm_pred = self.ve_swarm.predict_from_structure(
            structural_features.structure,
            structural_features.combined_features,
            &[],  // freq_history
            frequency,
            velocity,
        ).ok();
        let swarm_consensus = swarm_pred
            .map(|p| if p.predicted_rise { 1.0 } else { -1.0 })
            .unwrap_or(0.0);

        // 7. Weighted combination (PRISM-VE formula)
        let gamma = self.weights.w_fold_reduction * log_fold_reduction +
                    self.weights.w_transmissibility * transmissibility +
                    self.weights.w_velocity_inv * velocity_inv +
                    self.weights.w_structural_ddg * ddg_binding +
                    self.weights.w_freq_saturation * freq_saturation +
                    self.weights.w_swarm_consensus * swarm_consensus;

        // 8. Sigmoid + threshold
        let rise_prob = 1.0 / (1.0 + (-gamma * 2.0).exp());
        let predicted_rise = rise_prob > self.weights.threshold;
        let confidence = (rise_prob - 0.5).abs() * 2.0;

        (predicted_rise, confidence)
    }

    /// Compute fold reduction with epitope-specific immunity and cross-reactivity
    fn compute_fold_reduction(
        &self,
        epitope_escape: &[f32; 10],
        immunity: &EpitopeImmunity,
        lineage: &str,
    ) -> f32 {
        let mut sum = 0.0f32;

        for i in 0..10 {
            // Get cross-immunity factor for this epitope
            let cross_immunity = self.cross_reactivity.get_cross_immunity(
                lineage,
                i,
            );

            // Effective immunity = base immunity × cross-reactivity factor
            let effective_immunity = immunity.levels[i] * cross_immunity;

            // Accumulate: escape × immunity
            sum += epitope_escape[i] * effective_immunity;
        }

        // fold_reduction = exp(sum)
        sum.exp()
    }

    /// Velocity inversion correction (from VE-Swarm)
    fn correct_velocity(&self, velocity: f32, frequency: f32) -> f32 {
        if frequency > 0.5 {
            -velocity.abs() * 1.5
        } else if frequency > 0.2 && velocity > 0.05 {
            velocity * 0.2 - 0.15
        } else if frequency < 0.1 && velocity > 0.0 {
            velocity * 1.5
        } else {
            velocity * 0.5
        }
    }

    /// Fit weights on training data (grid search or gradient descent)
    pub fn fit_weights(&mut self, training_data: &[(HybridInput, bool)]) {
        // TODO: Implement grid search or LBFGS weight fitting
        // For now, use default weights
    }
}
```

### 5.2 `immunity_dynamics.rs` - Time-Varying Immunity

```rust
//! Time-varying immunity computation with PK parameters

use anyhow::Result;
use std::collections::HashMap;
use serde::{Deserialize, Serialize};

/// PK parameters for immunity waning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PKParameters {
    pub t_half: f32,  // Half-life in days
    pub t_max: f32,   // Time to peak immunity in days
}

/// Epitope-specific immunity at a given time
#[derive(Debug, Clone)]
pub struct EpitopeImmunity {
    pub levels: [f32; 10],  // A, B, C, D1, D2, E12, E3, F1, F2, F3
}

/// Immunity dynamics with waning curves
pub struct ImmunityDynamics {
    /// PK parameter combinations (75 total)
    pk_params: Vec<PKParameters>,

    /// Per-country outbreak dates
    outbreak_dates: HashMap<String, Vec<NaiveDate>>,
}

impl ImmunityDynamics {
    /// Load from VASIL data
    pub fn load_from_vasil(vasil_data_dir: &Path) -> Result<Self> {
        // Load PK parameters from JSON
        let pk_file = vasil_data_dir.join("../../data/pk_parameters.json");
        let pk_data: serde_json::Value = serde_json::from_reader(
            std::fs::File::open(pk_file)?
        )?;

        let pk_params: Vec<PKParameters> = serde_json::from_value(
            pk_data["pk_combinations"].clone()
        )?;

        // TODO: Load outbreak dates from frequency data
        let outbreak_dates = HashMap::new();

        Ok(Self {
            pk_params,
            outbreak_dates,
        })
    }

    /// Compute immunity at time t using PK waning curve
    /// I(t) = I_max × exp(-(t - t_max) × ln(2) / t_half) for t >= t_max
    pub fn compute_immunity(
        &self,
        country: &str,
        lineage: &str,
        days_since_outbreak: i32,
    ) -> EpitopeImmunity {
        // Use median PK parameters (or fit best combination)
        let pk = &self.pk_params[self.pk_params.len() / 2];

        let mut immunity = EpitopeImmunity {
            levels: [0.0; 10],
        };

        for i in 0..10 {
            immunity.levels[i] = self.compute_epitope_immunity(
                days_since_outbreak as f32,
                pk,
            );
        }

        immunity
    }

    fn compute_epitope_immunity(&self, t: f32, pk: &PKParameters) -> f32 {
        if t < pk.t_max {
            // Rising phase (simplified: linear rise)
            t / pk.t_max
        } else {
            // Waning phase (exponential decay)
            let t_rel = t - pk.t_max;
            (-t_rel * 2.0_f32.ln() / pk.t_half).exp()
        }
    }
}
```

---

## 6. Integration Points

### 6.1 Update `main.rs`

```rust
// Add PRISM-VE predictor
use prism_ve_model::PRISMVEPredictor;

// After VE-Swarm initialization
let mut prism_ve = PRISMVEPredictor::new(
    vasil_data_dir,
    ve_swarm,
)?;

// Fit weights on training data
prism_ve.fit_weights(&train_data);

// Evaluate on test data
for (input, actual_rise) in test_data {
    let (predicted_rise, confidence) = prism_ve.predict(
        &input.country,
        &input.lineage,
        input.date,
        input.frequency,
        input.velocity,
        &input.epitope_escape,
        &input.structural_features,
    );

    if predicted_rise == actual_rise {
        correct += 1;
    }
}

println!("PRISM-VE Test Accuracy: {:.1}%", accuracy * 100.0);
```

---

## 7. Expected Performance

| Model | Method | Expected Accuracy |
|-------|--------|-------------------|
| VASIL Paper | Fitted statistical model | 92% |
| VE-Swarm (current) | Structural + velocity | 59% |
| **PRISM-VE (target)** | **Hybrid epidem + structural** | **75-85%** |

**Improvement drivers**:
1. ✅ Time-varying immunity (not static)
2. ✅ Cross-reactivity between variants
3. ✅ PK waning curves (75 parameter combinations)
4. ✅ Velocity inversion (our discovery)
5. ✅ Structural ddG from GPU
6. ✅ VE-Swarm ensemble consensus

---

## 8. Implementation Steps

1. ✅ Parse PK parameters from VASIL data → `pk_parameters.json`
2. ✅ Parse cross-reactivity matrix → `cross_reactivity_summary.json`
3. ⏳ Create `immunity_dynamics.rs`
4. ⏳ Create `prism_ve_model.rs`
5. ⏳ Update `vasil_data.rs` to load PK data
6. ⏳ Implement weight fitting in `ve_optimizer.rs`
7. ⏳ Integrate into `main.rs`
8. ⏳ Run benchmark and compare to VASIL
9. ⏳ Tune hyperparameters
10. ⏳ Write up results

---

## 9. Scientific Contribution

**Title**: "PRISM-VE: GPU-Accelerated Hybrid Model Combining Epidemiological Immunity Dynamics with Structural Protein Analysis for Variant Fitness Prediction"

**Key Claims**:
- Novel hybrid architecture combining two complementary approaches
- First GPU-accelerated structural analysis for variant fitness
- Discovery of velocity inversion signal (high velocity at peak = about to fall)
- VE-Swarm ensemble with 32 agents
- Achieves X% accuracy on VASIL benchmark (target: 75-85%)

**Fair Use of VASIL Data**:
- Using publicly available benchmark dataset (standard practice)
- Building OUR model, fitting OUR weights
- NOT copying their model or parameters
- Apples-to-apples comparison on same test set

---

## 10. Next Steps

Execute implementation in order:
1. Create immunity_dynamics.rs
2. Create prism_ve_model.rs
3. Update vasil_data.rs
4. Integrate into main.rs
5. Run benchmark
6. Analyze results
7. Iterate on weights

**End Goal**: Demonstrate that hybrid structural + epidemiological modeling outperforms either approach alone, validating the PRISM-VE architecture.
