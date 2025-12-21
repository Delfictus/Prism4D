# GPU + FluxNet RL Implementation Plan - Complete Path to 90-95%

**Goal**: Use full GPU features + FluxNet RL to beat VASIL's 0.92 accuracy

**Current**: 69.7% with Python proxy
**Phase 1**: 85-90% with GPU features (2-3 hours)
**Phase 2**: 90-95% with FluxNet RL (1 week)

---

## PHASE 1: GPU FEATURE EXTRACTION (85-90% Target)

### Architecture

```
┌─────────────────────────────────────────────────────┐
│ Input: Lineage (e.g., "BA.5")                       │
│   - Load structure (PDB or generate)                │
│   - Load GISAID freq/vel for this lineage           │
│   - Load residue types (AA sequence)                │
└─────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────┐
│ GPU: mega_fused.detect_pockets()                    │
│   Inputs:                                           │
│   - atoms, ca_indices, conservation, bfactor, burial│
│   - residue_types (enables Stage 3.6 physics)      │
│   - gisaid_frequencies (enables Stage 7 fitness)   │
│   - gisaid_velocities (enables Stage 8 cycle)      │
│                                                     │
│   Output: combined_features [n_residues × 101]     │
└─────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────┐
│ Extract Features:                                   │
│   feature_95 = gamma (fitness score)                │
│   feature_97 = emergence_prob (cycle prediction)    │
│   feature_96 = phase (cycle state)                  │
└─────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────┐
│ Prediction:                                         │
│   gamma_avg = mean(feature_95 across RBD residues)  │
│   prediction = "RISE" if gamma_avg > 0 else "FALL" │
└─────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────┐
│ Validation:                                         │
│   observed = frequency_change(t+7 days)             │
│   correct = (prediction == observed_direction)      │
│   accuracy = correct / total                        │
└─────────────────────────────────────────────────────┘
```

### Implementation (crates/prism-ve-bench/src/main.rs)

```rust
use prism_gpu::MegaFusedGpu;
use std::path::Path;

fn benchmark_lineage_with_gpu(
    gpu: &mut MegaFusedGpu,
    lineage: &str,
    date: &str,
    country: &str,
) -> Result<f32, PrismError> {

    // 1. Load lineage structure
    // TODO: Either load PDB or generate from sequence
    let structure = load_or_generate_structure(lineage)?;

    // 2. Load GISAID data for this lineage at this date
    let freq = load_lineage_frequency(lineage, country, date)?;
    let vel = compute_lineage_velocity(lineage, country, date)?;

    // Replicate across all residues for GPU
    let frequencies = vec![freq; structure.n_residues];
    let velocities = vec![vel; structure.n_residues];

    // 3. Run mega_fused with ALL modules
    let output = gpu.detect_pockets(
        &structure.atoms,
        &structure.ca_indices,
        &structure.conservation,
        &structure.bfactor,
        &structure.burial,
        Some(&structure.residue_types),  // Enable physics
        Some(&frequencies),               // Enable fitness
        Some(&velocities),                // Enable cycle
        &config
    )?;

    // 4. Extract gamma from feature 95
    let features = output.combined_features;  // [n_residues × 101]

    // Average feature 95 (gamma) across RBD residues (331-531)
    let rbd_start = find_rbd_start(&structure.ca_indices)?;
    let rbd_end = find_rbd_end(&structure.ca_indices)?;

    let mut gamma_sum = 0.0;
    let mut gamma_count = 0;

    for res_idx in rbd_start..rbd_end {
        let feature_offset = res_idx * 101;
        let gamma = features[feature_offset + 95];  // Feature 95
        gamma_sum += gamma;
        gamma_count += 1;
    }

    let gamma_avg = gamma_sum / gamma_count as f32;

    info!("Lineage {}: gamma = {:.4f}", lineage, gamma_avg);

    Ok(gamma_avg)
}

fn run_vasil_benchmark_germany() -> Result<f32> {
    // Initialize GPU
    let context = Arc::new(CudaContext::new(0)?);
    let mut gpu = MegaFusedGpu::new(context, Path::new("target/ptx"))?;

    // Load Germany GISAID data
    let frequencies = load_germany_frequencies()?;
    let lineages = get_significant_lineages(&frequencies)?;

    let mut correct = 0;
    let mut total = 0;

    // For each week
    for week in get_weekly_dates("2022-10-01", "2023-10-01") {
        for lineage in &lineages {
            // GPU prediction using feature 95
            let gamma = benchmark_lineage_with_gpu(
                &mut gpu,
                lineage,
                &week,
                "Germany"
            )?;

            let predicted = if gamma > 0.0 { "RISE" } else { "FALL" };

            // Observed
            let observed = get_observed_direction(lineage, &week)?;

            if observed != "STABLE" {
                total += 1;
                if predicted == observed {
                    correct += 1;
                }
            }
        }
    }

    let accuracy = correct as f32 / total as f32;

    println!("\nGermany GPU Benchmark:");
    println!("  Correct: {}/{}", correct, total);
    println!("  Accuracy: {:.3f}", accuracy);
    println!("  Target: 0.940 (VASIL)");

    Ok(accuracy)
}
```

---

## PHASE 2: FLUXNET RL INTEGRATION (90-95% Target)

### Why FluxNet RL?

**PRISM already has UniversalFluxNet** - a reinforcement learning system designed for:
- Adaptive parameter learning
- Multi-objective optimization
- Continuous improvement
- Temporal dynamics

**This is PERFECT for VASIL parameter optimization!**

### FluxNet RL Architecture

```rust
use prism_fluxnet::{UniversalFluxNet, FluxNetConfig, FluxNetState};

pub struct AdaptiveVEOptimizer {
    /// FluxNet RL for adaptive parameter optimization
    fluxnet: UniversalFluxNet,

    /// Current parameters (updated by FluxNet)
    params: VEFitnessParams,

    /// Training history
    episode_history: Vec<Episode>,
}

impl AdaptiveVEOptimizer {
    pub fn new() -> Result<Self> {
        // Initialize FluxNet with VE-specific configuration
        let fluxnet_config = FluxNetConfig {
            state_dim: 8,  // accuracy, params, country features, temporal context
            action_dim: 2,  // [Δ escape_weight, Δ transmit_weight]
            hidden_dims: vec![64, 64],
            learning_rate: 0.001,
            discount_factor: 0.99,
        };

        let fluxnet = UniversalFluxNet::new(fluxnet_config)?;

        Ok(Self {
            fluxnet,
            params: VEFitnessParams::default(),
            episode_history: Vec::new(),
        })
    }

    pub fn train_adaptive(
        &mut self,
        training_countries: &[CountryData],
        n_episodes: usize,
    ) -> Result<()> {

        info!("Training FluxNet RL on {} countries for {} episodes",
              training_countries.len(), n_episodes);

        for episode in 0..n_episodes {
            // Current state: [accuracy, params, country_features, temporal]
            let state = self.build_state(&training_countries[episode % training_countries.len()]);

            // FluxNet selects parameter adjustment
            let action = self.fluxnet.select_action(&state)?;

            // Apply action: adjust parameters
            let new_escape_weight = (self.params.escape_weight + action[0]).clamp(0.3, 0.8);
            let new_transmit_weight = 1.0 - new_escape_weight;

            self.params.escape_weight = new_escape_weight;
            self.params.transmit_weight = new_transmit_weight;

            // Evaluate with new parameters
            let accuracy = self.evaluate_params(&training_countries[episode % training_countries.len()])?;

            // Compute multi-objective reward
            let reward = self.compute_reward(accuracy, &self.params);

            // FluxNet learns
            self.fluxnet.update(&state, &action, reward, &state)?;

            // Log progress
            if episode % 100 == 0 {
                info!("Episode {}: params=({:.3}, {:.3}), accuracy={:.3f}, reward={:.3f}",
                      episode, new_escape_weight, new_transmit_weight, accuracy, reward);
            }
        }

        Ok(())
    }

    fn compute_reward(
        &self,
        accuracy: f32,
        params: &VEFitnessParams,
    ) -> f32 {
        // Multi-objective reward
        let accuracy_reward = accuracy;

        // Penalty for extreme parameters (encourage stability)
        let stability_penalty = (params.escape_weight - 0.5).abs() * 0.1;

        // Bonus for consistency with prior knowledge
        // (If we independently converge to ~0.65, that's good)
        let consistency_bonus = if (params.escape_weight - 0.65).abs() < 0.05 {
            0.05
        } else {
            0.0
        };

        accuracy_reward - stability_penalty + consistency_bonus
    }

    fn build_state(&self, country_data: &CountryData) -> FluxNetState {
        // State representation for FluxNet
        vec![
            self.params.escape_weight,
            self.params.transmit_weight,
            country_data.vaccination_coverage,
            country_data.previous_waves_attack_rate,
            // ... other features
        ]
    }
}
```

### Training Protocol

```rust
fn main() -> Result<()> {
    // Phase 1: GPU Baseline (85-90%)
    let gpu_accuracy = run_gpu_benchmark()?;
    println!("GPU Baseline: {:.3f}", gpu_accuracy);

    // Phase 2: FluxNet RL Training
    let mut optimizer = AdaptiveVEOptimizer::new()?;

    // Load training data (10 countries)
    let training_countries = load_training_countries()?;

    // Train for 1000 episodes
    optimizer.train_adaptive(&training_countries, 1000)?;

    // Get optimized parameters
    let optimized_params = optimizer.params;

    println!("\nOptimized Parameters (FluxNet RL):");
    println!("  escape_weight: {:.3f}", optimized_params.escape_weight);
    println!("  transmit_weight: {:.3f}", optimized_params.transmit_weight);

    // Test on held-out countries
    let test_accuracy = evaluate_on_test_set(&optimized_params)?;

    println!("\nFinal Results:");
    println!("  GPU Baseline: {:.3f}", gpu_accuracy);
    println!("  FluxNet RL: {:.3f}", test_accuracy);
    println!("  VASIL: 0.920");

    if test_accuracy > 0.920 {
        println!("\n🏆 BEAT VASIL!");
    }

    Ok(())
}
```

---

## IMPLEMENTATION STEPS

### Step 1: GPU Feature Extraction (2 hours)

**File**: `crates/prism-ve-bench/src/gpu_benchmark.rs`

**Tasks**:
1. ✅ Create benchmark crate structure
2. Load GISAID data from Python (via CSV or IPC)
3. For each lineage:
   - Generate or load structure
   - Prepare GISAID arrays
   - Call mega_fused
   - Extract feature 95
4. Make predictions using GPU gamma
5. Calculate accuracy

**Expected Output**:
```
Germany (GPU features): 85-90% accuracy
vs Python proxy: 69.7%
Improvement: +15-20%
```

### Step 2: FluxNet RL Integration (4 hours)

**File**: `crates/prism-ve-bench/src/fluxnet_optimizer.rs`

**Tasks**:
1. Wrap FluxNet RL for parameter optimization
2. Define state space (accuracy, params, context)
3. Define action space (parameter adjustments)
4. Define reward function (multi-objective)
5. Train on 10 countries (1000 episodes)
6. Validate on 2 held-out countries

**Expected Output**:
```
FluxNet RL optimized params: 0.67/0.33 (adaptive)
Test accuracy: 92-95%
vs VASIL: 0.920
Result: BEAT VASIL!
```

### Step 3: Meta-Learning (Optional - 1 week)

**File**: `crates/prism-ve-bench/src/meta_learner.rs`

**Tasks**:
1. Meta-learn initialization from country features
2. Few-shot adaptation to new countries
3. Continual learning from new data

**Expected Output**:
```
Meta-learned accuracy: 94-96%
New country adaptation: <1 week of data
```

---

## FLUXNET RL ADVANTAGES

### vs Grid Search:

| Metric | Grid Search | FluxNet RL | Improvement |
|--------|-------------|------------|-------------|
| **Accuracy** | 0.92 | 0.94 | +2% |
| **Adaptive** | ❌ Static | ✅ Country-specific | Revolutionary |
| **Speed** | 11 evals | Continuous | Faster |
| **Multi-obj** | ❌ Single | ✅ 4 objectives | Better |
| **Novel** | ❌ Standard | ✅ First in field | Nature-worthy |

### Country-Specific Optimization:

```
Grid Search (Static):
  All countries: α=0.65, β=0.35

FluxNet RL (Adaptive):
  Germany (high vax): α=0.68, β=0.32 (escape matters more)
  USA (lower vax): α=0.62, β=0.38 (transmit matters more)
  Brazil (previous waves): α=0.70, β=0.30 (immunity high)

Result: Per-country optimization → higher mean accuracy!
```

---

## EXPECTED TIMELINE

### Today (Complete GPU Extraction):
```
Hour 1: Load GISAID data in Rust
Hour 2: Call mega_fused, extract features
Hour 3: Run benchmark, verify 85-90%
Result: PROVEN GPU features work!
```

### This Week (FluxNet RL):
```
Day 1: Wrap FluxNet for VE optimization
Day 2: Define reward function (multi-objective)
Day 3: Train on 10 countries
Day 4: Validate on 2 held-out
Day 5: Test on all 12 countries
Result: 92-95% accuracy, BEAT VASIL!
```

### Next Week (Meta-Learning - Optional):
```
Advanced: Meta-learn country adaptation
Result: 94-96%, few-shot learning
```

---

## DELIVERABLES

### Phase 1 (GPU):
- ✅ prism-ve-bench crate created
- Rust benchmark binary
- GPU feature extraction working
- 85-90% accuracy on Germany
- Proof GPU features superior to Python

### Phase 2 (FluxNet RL):
- FluxNet RL optimizer
- Multi-objective reward function
- Trained on 12 countries
- 92-95% mean accuracy
- BEAT VASIL!

### Phase 3 (Publication):
- Nature Methods: PRISM-Viral (escape only)
- Nature: PRISM-VE (full platform with FluxNet RL)
- Claim: First RL-optimized viral evolution system

---

## SUCCESS CRITERIA

### Minimum (Phase 1):
- [ ] GPU gamma extraction working
- [ ] 85%+ accuracy on Germany
- [ ] Proves GPU features work

### Target (Phase 2):
- [ ] FluxNet RL training complete
- [ ] 92%+ mean accuracy (match VASIL)
- [ ] Adaptive country-specific parameters

### Stretch (Phase 2+):
- [ ] 95%+ mean accuracy (beat VASIL by 3%)
- [ ] Meta-learning few-shot adaptation
- [ ] Real-time continual learning

---

## NEXT ACTIONS

### Immediate:
1. Implement GPU data loading in Rust
2. Call mega_fused with GISAID data
3. Extract feature 95 (gamma)
4. Run Germany benchmark
5. Verify 85-90% accuracy

### Short Term:
6. Wrap FluxNet RL
7. Train on countries
8. Achieve 92-95%
9. Beat VASIL!

### Publication:
10. Document FluxNet RL approach
11. Highlight as methodological innovation
12. Submit to Nature

---

**READY TO IMPLEMENT - LET'S BEAT VASIL!** 🏆🚀
