# WORLD-CLASS VASIL BENCHMARK IMPLEMENTATION PROMPT
## Scientific Integrity + GPU Acceleration + 92% Accuracy Target

**Version:** 1.0  
**Date:** 2025-12-17  
**Target:** Transform PRISM-VE from 77.4% → 92% accuracy with publication-grade rigor  
**Status:** Ready for Implementation  

---

## 🎯 EXECUTIVE SUMMARY

You are Claude Code operating inside the PRISM-VE repository. Your mission is to implement a world-class VASIL benchmark that combines:

1. **Scientific Integrity** - Zero data leakage, deterministic reproducibility, full auditability
2. **VASIL-Exact Methodology** - 75-PK envelope, per-day predictions, proper exclusion rules
3. **GPU Acceleration** - Maintain 4,772 structures/sec throughput (75× speedup)
4. **Real Temporal Features** - Extract phi peaks, frequency trajectories for polycentric wave model
5. **Production-Ready Engineering** - Strict mode, CI tests, manifest generation, single source of truth

**Current State:** 77.4% accuracy (verified correct, but implementation incomplete)  
**Target State:** 85-92% accuracy with full reproducibility and auditability

---

## 📊 CURRENT STATE ASSESSMENT (from forensic audit)

### What Works ✅

1. **VASIL-Exact Metric Framework** (`vasil_exact_metric.rs`, 1,500+ lines)
   - ✅ Per-day γ predictions
   - ✅ Rising/falling classification
   - ✅ Negligible change exclusion (<5%)
   - ✅ Per-country accuracy → mean across 12 countries
   - ✅ Zero data leakage verified (train < 2022-06-01, test >= 2022-06-01)

2. **Real VASIL Data** (12 countries complete)
   - ✅ Phi estimates: 8,054 days (9/12 countries, 3 use fallback)
   - ✅ DMS escape: 835 antibodies, 10 epitope classes
   - ✅ PK immunity: 75 scenarios per country
   - ✅ Frequency trajectories: all 12 countries
   - ✅ Mutation profiles: 2,065 lineages

3. **GPU Pipeline** (47 CUDA kernels)
   - ✅ Immunity cache: 20.9s build time (75× speedup)
   - ✅ Feature extraction: 4,772 structures/sec
   - ✅ 136-dim features per residue
   - ✅ Real DMS data reaching GPU (23.8M values)

### Critical Gaps ❌

1. **75-PK ENVELOPE NOT IMPLEMENTED** (PRIMARY BLOCKER)
   - GPU computes all 75 PK values
   - **BUT AVERAGES THEM** before returning to CPU
   - CPU receives mean immunity, cannot compute envelope (min/max)
   - Result: Using mean instead of envelope decision rule
   - **Impact:** This is likely WHY we're at 77.4% instead of 92%
   - **Status:** Attempted 3×, all failed (36.6%, 50.5% regressions)
   - **Cause:** `immunity_matrix: Vec<Vec<f64>>` needs to be `Vec<Vec<[f64; 75]>>`

2. **PLACEHOLDERS BLOCK POLYCENTRIC FEATURES**
   ```rust
   // Current code (mega_fused_batch.rs:1910-1916):
   let time_since_infection = vec![30.0f32; n_structures];      // FAKE
   let freq_history_flat = vec![0.10f32; n_structures * 7];    // FAKE
   let current_freq = vec![0.15f32; n_structures];             // FAKE
   ```
   - Wave features (F146-F157) contain no real signal
   - Expected gain: +2-5% accuracy if fixed

3. **NO REPRODUCIBILITY MANIFEST**
   - No run_manifest.json with git hash, config hash, data SHA256s
   - No predictions.csv output
   - No exclusions.json tracking

4. **CONSTANTS DUPLICATED IN 4 PLACES**
   - `ALPHA_ESCAPE = 0.65`, `BETA_TRANSMIT = 0.35` defined in:
     - `gpu_benchmark.rs:205-206`
     - `mega_fused_batch.cu:222-223`
     - `mega_fused_pocket_kernel.cu:1315-1316`
     - `main.rs` (build_pk_params)
   - Risk: Divergence → non-reproducible results

5. **NO STRICT MODE / FAIL-FAST**
   - Env var `PRISM_ENABLE_VASIL_METRIC=1` gates metric (good)
   - But no `--strict` mode that fails on placeholders/approximations

6. **NO CI TESTS FOR LEAKAGE**
   - Manual verification ("5 independent tests")
   - But no automated tests to prevent future regressions

---

## 🏗️ NON-NEGOTIABLE CONSTRAINTS

### Scientific Integrity (must follow)

1. **Zero Data Leakage**
   - No information from test dates (≥ 2022-06-01) may influence training artifacts
   - Any violation = bug that must be fixed
   - All caches/calibrations/normalizations must be train-only

2. **Apples-to-Apples with VASIL**
   - Implement 75-PK envelope as specified (no averaging)
   - "Undecided → excluded" rule (envelope crosses zero)
   - If cannot be implemented: FAIL in strict mode, label "NOT VASIL-exact" otherwise

3. **Deterministic Reproducibility**
   - Same code + data + config = identical outputs (predictions + metrics + manifest)
   - All randomness must be seeded
   - GPU operations must be deterministic

4. **Single Source of Truth**
   - One evaluation entrypoint, one constants module
   - Remove or clearly gate alternative code paths

5. **Full Auditability**
   - Every run must emit manifest proving what happened
   - Code hash, data hashes, config, environment, exclusion counts

### GPU-Centric Platform (from CLAUDE.md)

6. **No CPU Fallbacks**
   - All compute-intensive operations MUST use GPU
   - PTX kernels are REQUIRED (not optional)
   - GPU functions must be implemented, integrated, wired, compiled, and operational

---

## 📋 IMPLEMENTATION PHASES

### PHASE 0: Repository State Assessment (DO THIS FIRST)

**Before writing any code**, read and analyze:

1. `/mnt/c/Users/Predator/Desktop/SESSION_FINAL_SUMMARY_COMPLETE.md`
2. `/mnt/c/Users/Predator/Desktop/PRISM_VE_FORENSIC_AUDIT_REPORT.md`
3. `crates/prism-ve-bench/src/vasil_exact_metric.rs`
4. `crates/prism-ve-bench/src/main.rs` (lines 1000-1110 - VASIL metric section)
5. `VASIL_DATA_INVENTORY.md` (data completeness)

**Produce a "Phase 0 Report" (markdown) with:**

```markdown
## Phase 0: Repository State Assessment

### Evaluation Entrypoint
- File: `crates/prism-ve-bench/src/main.rs`
- Lines: 1006-1109
- Trigger: `PRISM_ENABLE_VASIL_METRIC=1` env var
- Current: Single entrypoint ✅ / Multiple paths ❌

### 75-PK Envelope Status
- Implementation location: `vasil_exact_metric.rs:XXX-YYY`
- GPU output format: [f64] (mean) / [f64; 75] (full envelope)
- Envelope computation: Working ✅ / Broken ❌
- Evidence: [describe current behavior]

### Constants Locations
- Found in: [list all files with ALPHA_ESCAPE, BETA_TRANSMIT, PK params]
- Centralized: Yes ✅ / No ❌
- Duplicates: [count]

### Placeholders Inventory
- time_since_infection: Real ✅ / Placeholder ❌ (location: XXX)
- freq_history_7d: Real ✅ / Placeholder ❌ (location: XXX)
- current_freq: Real ✅ / Placeholder ❌ (location: XXX)
- immunity fallbacks: [list any compute_immunity functions returning 0.5]

### Data Leakage Protection
- Train/test split: File: XXX, Line: YYY, Cutoff: YYYY-MM-DD
- Runtime assertions: Present ✅ / Missing ❌
- Cache provenance: Tracked ✅ / Not tracked ❌

### CI Tests
- Leakage tests: Present ✅ / Missing ❌
- Determinism tests: Present ✅ / Missing ❌
- Golden file tests: Present ✅ / Missing ❌
```

**DO NOT PROCEED** until this report is complete and accurate.

---

## PHASE 1: FIX 75-PK ENVELOPE (HIGHEST PRIORITY - ACCURACY BLOCKER)

**Objective:** Implement true 75-PK envelope computation to match VASIL methodology exactly.

**Current Bug:** GPU computes all 75 PK immunity values, but averages them before returning to CPU. CPU cannot compute envelope decision rule.

**Expected Impact:** +5-10% accuracy (77.4% → 82-87%)

### Step 1.1: Modify GPU Output to Return All 75 PK Values

**File:** `crates/prism-gpu/src/mega_fused_batch.rs`

**Current (WRONG):**
```rust
// GPU returns averaged immunity (single f64 per variant-date)
let immunity_mean: Vec<f64> = download_and_average_75pk(...);
```

**Required (CORRECT):**
```rust
// GPU returns ALL 75 PK values (no averaging)
let immunity_75pk: Vec<[f64; 75]> = download_all_75pk_no_averaging(...);
```

**Implementation:**

1. **Locate GPU immunity download** (search for: `download.*immunity`, `memcpy_dtoh.*immunity`)

2. **Change download logic:**
   ```rust
   // OLD: Download and average
   let mean_immunity = download_mean_only();
   
   // NEW: Download full 75-dimensional array
   let immunity_75pk: Vec<[f64; 75]> = {
       let n_variants = /* count */;
       let mut buffer = vec![0.0f64; n_variants * 75];
       context.dtoh_sync_copy(&d_immunity_75pk, &mut buffer)?;
       
       // Reshape into Vec<[f64; 75]>
       buffer.chunks_exact(75)
           .map(|chunk| {
               let mut arr = [0.0f64; 75];
               arr.copy_from_slice(chunk);
               arr
           })
           .collect()
   };
   ```

3. **Update ImmunityCache struct:**
   ```rust
   // In vasil_exact_metric.rs
   pub struct ImmunityCache {
       // OLD: immunity_matrix: HashMap<(String, NaiveDate), f64>,
       
       // NEW: Store all 75 PK values
       immunity_matrix_75: HashMap<(String, NaiveDate), [f64; 75]>,
       
       // Metadata for provenance
       min_date: NaiveDate,
       max_date: NaiveDate,
       cutoff_used: NaiveDate,
       git_commit: String,
   }
   ```

### Step 1.2: Implement Envelope Decision Rule

**File:** `crates/prism-ve-bench/src/vasil_exact_metric.rs`

**Current (WRONG):**
```rust
// Uses mean immunity
let gamma_mean = compute_gamma_from_mean_immunity(...);
let predicted = if gamma_mean > 0.0 { Rising } else { Falling };
```

**Required (CORRECT):**
```rust
/// Compute gamma envelope across all 75 PK combinations
fn compute_gamma_envelope_75pk(
    immunity_75pk: &[f64; 75],
    weighted_avg_susceptibility: f64,
) -> (f64, f64, f64) {  // (min, max, mean)
    let mut gammas = [0.0f64; 75];
    
    for (i, &immunity) in immunity_75pk.iter().enumerate() {
        gammas[i] = (immunity / weighted_avg_susceptibility) - 1.0;
    }
    
    let min = gammas.iter().copied().fold(f64::INFINITY, f64::min);
    let max = gammas.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mean = gammas.iter().sum::<f64>() / 75.0;
    
    (min, max, mean)
}

/// Apply VASIL decision rule
fn classify_gamma_envelope(min: f64, max: f64) -> EnvelopeDecision {
    if max < 0.0 {
        EnvelopeDecision::Falling  // Entire envelope negative
    } else if min > 0.0 {
        EnvelopeDecision::Rising   // Entire envelope positive
    } else {
        EnvelopeDecision::Undecided  // Envelope crosses zero → EXCLUDE
    }
}

enum EnvelopeDecision {
    Rising,
    Falling,
    Undecided,  // CRITICAL: Must exclude from accuracy calculation
}
```

### Step 1.3: Update Accuracy Computation

**File:** `crates/prism-ve-bench/src/vasil_exact_metric.rs`

**Current (WRONG):**
```rust
// Compare mean prediction with actual
if predicted == actual { correct += 1; }
total += 1;
```

**Required (CORRECT):**
```rust
// Exclude "undecided" predictions
match classify_gamma_envelope(min, max) {
    EnvelopeDecision::Undecided => {
        excluded_undecided += 1;
        continue;  // DO NOT count in accuracy
    }
    EnvelopeDecision::Rising => {
        if actual == DayDirection::Rising {
            correct += 1;
        }
        total += 1;
    }
    EnvelopeDecision::Falling => {
        if actual == DayDirection::Falling {
            correct += 1;
        }
        total += 1;
    }
}
```

### Step 1.4: Add Strict Mode Guard

**File:** `crates/prism-ve-bench/src/main.rs`

```rust
// Check that 75-PK envelope is actually computed
if strict_mode && !has_75pk_envelope {
    eprintln!("ERROR: Strict mode requires 75-PK envelope computation");
    eprintln!("       Found: averaged immunity (single value)");
    eprintln!("       Expected: full 75-dimensional array");
    std::process::exit(1);
}
```

### Step 1.5: Add Golden File Test

**File:** `crates/prism-ve-bench/tests/envelope_decision_test.rs` (NEW)

```rust
#[test]
fn test_envelope_decision_rule() {
    // Case 1: Entirely positive → Rising
    let immunity_rising = [1.2f64; 75];  // All > weighted_avg (1.0)
    let (min, max, _) = compute_gamma_envelope_75pk(&immunity_rising, 1.0);
    assert_eq!(classify_gamma_envelope(min, max), EnvelopeDecision::Rising);
    
    // Case 2: Entirely negative → Falling
    let immunity_falling = [0.8f64; 75];  // All < weighted_avg (1.0)
    let (min, max, _) = compute_gamma_envelope_75pk(&immunity_falling, 1.0);
    assert_eq!(classify_gamma_envelope(min, max), EnvelopeDecision::Falling);
    
    // Case 3: Crosses zero → Undecided (MUST EXCLUDE)
    let mut immunity_mixed = [1.0f64; 75];
    immunity_mixed[0] = 0.8;   // One below
    immunity_mixed[74] = 1.2;  // One above
    let (min, max, _) = compute_gamma_envelope_75pk(&immunity_mixed, 1.0);
    assert_eq!(classify_gamma_envelope(min, max), EnvelopeDecision::Undecided);
}
```

**Success Criteria for Phase 1:**
- ✅ GPU returns `Vec<[f64; 75]>` instead of `Vec<f64>`
- ✅ Envelope (min, max) computed across 75 PK combinations
- ✅ Undecided predictions excluded from accuracy
- ✅ Golden file test passes
- ✅ Strict mode fails if envelope cannot be computed
- ✅ Accuracy improves by ≥5 percentage points

---

## PHASE 2: EXTRACT REAL TEMPORAL DATA (POLYCENTRIC ENABLER)

**Objective:** Replace placeholder constants with real VASIL data to enable polycentric wave features.

**Current Bug:** Polycentric features use fake temporal data (constant values), contributing noise instead of signal.

**Expected Impact:** +2-5% accuracy (via wave features F146-F157)

### Step 2.1: Add Metadata to PackedBatch

**File:** `crates/prism-gpu/src/mega_fused_batch.rs`

```rust
#[derive(Debug, Clone)]
pub struct BatchMetadata {
    pub country: String,
    pub lineage: String,
    pub date: NaiveDate,
    pub frequency: f32,
}

pub struct PackedBatch {
    // ... existing fields ...
    
    /// Metadata for temporal feature extraction
    pub metadata: Vec<BatchMetadata>,
}
```

### Step 2.2: Populate Metadata During Batch Building

**File:** `crates/prism-ve-bench/src/main.rs`

```rust
// In build_mega_batch() function
let mut all_metadata = Vec::new();

for country_data in countries {
    for (lineage, freq_map) in &country_data.frequencies {
        for (date, freq) in freq_map {
            // ... existing structure building code ...
            
            all_metadata.push(BatchMetadata {
                country: country_data.name.clone(),
                lineage: lineage.clone(),
                date: *date,
                frequency: *freq,
            });
        }
    }
}

// Return metadata with batch
Ok((PackedBatch {
    // ... existing fields ...
    metadata: all_metadata,
}, all_metadata))
```

### Step 2.3: Extract time_since_infection from Phi Peaks

**File:** `crates/prism-gpu/src/mega_fused_batch.rs`

```rust
/// Extract time since last major infection wave from phi estimates
fn compute_time_since_infection_from_phi(
    batch: &PackedBatch,
    phi_data: &HashMap<String, Vec<(NaiveDate, f32)>>,
    strict_mode: bool,
) -> Vec<f32> {
    batch.metadata.iter().map(|meta| {
        let country_phi = match phi_data.get(&meta.country) {
            Some(data) => data,
            None => {
                if strict_mode {
                    panic!("Strict mode: missing phi data for {}", meta.country);
                }
                return 90.0;  // Default: 3 months
            }
        };
        
        // Find last major wave (phi > 5000 for major countries, > 1000 for smaller)
        let threshold = if ["Germany", "USA", "UK", "France"].contains(&meta.country.as_str()) {
            5000.0
        } else {
            1000.0
        };
        
        let last_peak = country_phi.iter()
            .filter(|(date, phi)| *date < meta.date && *phi > threshold)
            .max_by_key(|(date, _)| *date);
        
        match last_peak {
            Some((peak_date, _)) => {
                let days = (meta.date - *peak_date).num_days();
                days.max(0) as f32
            }
            None => 90.0,
        }
    }).collect()
}
```

### Step 2.4: Extract freq_history_7d from Frequency Trajectories

**File:** `crates/prism-gpu/src/mega_fused_batch.rs`

```rust
/// Extract 7-day frequency history from VASIL data
fn extract_freq_history_7d_from_data(
    batch: &PackedBatch,
    freq_data: &HashMap<(String, String, NaiveDate), f32>,
    strict_mode: bool,
) -> Vec<f32> {
    let mut result = Vec::with_capacity(batch.metadata.len() * 7);
    
    for meta in &batch.metadata {
        for days_ago in (0..7).rev() {
            let date = meta.date - chrono::Duration::days(days_ago);
            let freq = freq_data
                .get(&(meta.country.clone(), meta.lineage.clone(), date))
                .copied()
                .unwrap_or_else(|| {
                    if strict_mode && days_ago < 3 {  // Missing recent data is error
                        panic!("Strict mode: missing freq for {} {} on {}",
                               meta.country, meta.lineage, date);
                    }
                    // Fallback: exponential decay from current
                    meta.frequency * 0.9_f32.powi(days_ago as i32)
                });
            result.push(freq);
        }
    }
    
    result
}
```

### Step 2.5: Extract current_freq from Metadata

**File:** `crates/prism-gpu/src/mega_fused_batch.rs`

```rust
/// Extract current frequency (simple - already in metadata)
fn extract_current_freq(batch: &PackedBatch) -> Vec<f32> {
    batch.metadata.iter().map(|meta| meta.frequency).collect()
}
```

### Step 2.6: Update enhance_with_polycentric Signature

**File:** `crates/prism-gpu/src/mega_fused_batch.rs`

```rust
pub fn enhance_with_polycentric(
    &self,
    output: BatchOutput,
    batch: &PackedBatch,
    polycentric: &PolycentricImmunityGpu,
    phi_data: &HashMap<String, Vec<(NaiveDate, f32)>>,     // ADD
    freq_data: &HashMap<(String, String, NaiveDate), f32>, // ADD
    strict_mode: bool,                                      // ADD
) -> Result<BatchOutput, PrismError> {
    // REPLACE placeholders with real extraction
    let time_since_infection = compute_time_since_infection_from_phi(
        batch, phi_data, strict_mode
    );
    let freq_history_flat = extract_freq_history_7d_from_data(
        batch, freq_data, strict_mode
    );
    let current_freq = extract_current_freq(batch);
    
    // ... rest of polycentric computation ...
}
```

### Step 2.7: Build phi_data and freq_data in main.rs

**File:** `crates/prism-ve-bench/src/main.rs`

```rust
// After loading VASIL data (around line 280)
println!("Building phi and frequency lookup maps...");

let mut phi_data: HashMap<String, Vec<(NaiveDate, f32)>> = HashMap::new();
let mut freq_data: HashMap<(String, String, NaiveDate), f32> = HashMap::new();

for country_data in &all_data.countries {
    // Extract phi from VasilEnhancedData
    if let Some(enhanced) = vasil_enhanced.get(&country_data.name) {
        phi_data.insert(
            country_data.name.clone(),
            enhanced.phi_estimates.clone()
        );
    }
    
    // Extract frequencies
    for (lineage, date_map) in &country_data.frequencies {
        for (date, freq) in date_map {
            freq_data.insert(
                (country_data.name.clone(), lineage.clone(), *date),
                *freq
            );
        }
    }
}

println!("  ✅ Phi data: {} countries", phi_data.len());
println!("  ✅ Freq data: {} (country, lineage, date) tuples", freq_data.len());
```

### Step 2.8: Update Call Site

**File:** `crates/prism-ve-bench/src/main.rs`

```rust
// Around line 400 (polycentric enhancement call)
let batch_output = gpu.enhance_with_polycentric(
    batch_output,
    &packed_batch,
    &polycentric,
    &phi_data,      // Pass real phi data
    &freq_data,     // Pass real frequency data
    strict_mode,    // From config
)?;
```

### Step 2.9: Add Validation Test

**File:** `crates/prism-ve-bench/tests/temporal_data_test.rs` (NEW)

```rust
#[test]
fn test_temporal_data_not_constant() {
    // Build real batch with metadata
    let batch = build_test_batch();
    let phi_data = load_test_phi_data();
    let freq_data = load_test_freq_data();
    
    // Extract temporal features
    let time_since = compute_time_since_infection_from_phi(&batch, &phi_data, false);
    let freq_history = extract_freq_history_7d_from_data(&batch, &freq_data, false);
    
    // Assert NOT constant (variance > 0)
    let mean_time = time_since.iter().sum::<f32>() / time_since.len() as f32;
    let variance_time = time_since.iter()
        .map(|&x| (x - mean_time).powi(2))
        .sum::<f32>() / time_since.len() as f32;
    
    assert!(variance_time > 1.0, "time_since_infection should vary, got constant");
    
    // Assert values in reasonable range
    assert!(time_since.iter().all(|&x| x >= 0.0 && x <= 365.0));
}
```

**Success Criteria for Phase 2:**
- ✅ time_since_infection shows non-constant values (variance > 0)
- ✅ freq_history_7d extracted from real VASIL data
- ✅ current_freq matches batch metadata
- ✅ Strict mode fails on missing phi/freq data
- ✅ Wave features (F146-F157) show non-constant values in output
- ✅ Accuracy improves by ≥2 percentage points

---

## PHASE 3: STRICT MODE + CLI + MANIFEST (REPRODUCIBILITY)

**Objective:** Add production-grade rigor: fail-fast strict mode, unified CLI, full audit manifest.

### Step 3.1: Create Benchmark Configuration File

**File:** `bench.toml` (NEW, repository root)

```toml
# PRISM-VE VASIL Benchmark Configuration

[dataset]
root_path = "/mnt/f/VASIL_Data/ByCountry"
pdb_reference = "data/spike_rbd_6m0j.pdb"

[evaluation]
train_cutoff = "2022-06-01"  # Train: dates < this, Test: dates >= this
eval_start = "2022-10-01"    # Evaluation window start
eval_end = "2023-10-31"      # Evaluation window end

[behavior]
strict_mode = true           # Fail on placeholders, missing data, approximations
seed = 42                    # Deterministic randomness
num_countries = 12           # Limit for testing (0 = all)

[output]
manifest_dir = "benchmark_results"
save_predictions = true
save_exclusions = true

[features]
enable_75pk_envelope = true     # Required in strict mode
enable_polycentric = true       # Wave features
enable_real_temporal_data = true  # No placeholders

[gpu]
device_id = 0
deterministic = true  # Use deterministic CUDA algorithms
```

### Step 3.2: Implement Unified CLI

**File:** `crates/prism-ve-bench/src/main.rs`

```rust
use clap::{Parser, ValueEnum};

#[derive(Parser)]
#[command(name = "vasil-benchmark")]
#[command(about = "PRISM-VE VASIL Benchmark - Publication-Grade Evaluation")]
struct Args {
    /// Path to configuration TOML file
    #[arg(short, long, default_value = "bench.toml")]
    config: String,
    
    /// Enable strict mode (fail on any approximation)
    #[arg(short, long)]
    strict: bool,
    
    /// Override number of countries (0 = all)
    #[arg(short, long)]
    num_countries: Option<usize>,
    
    /// Output directory for results
    #[arg(short, long)]
    output: Option<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    
    // Load config
    let config = load_config(&args.config)?;
    let strict_mode = args.strict || config.behavior.strict_mode;
    
    println!("🎯 PRISM-VE VASIL Benchmark");
    println!("   Mode: {}", if strict_mode { "STRICT" } else { "Standard" });
    println!("   Config: {}", args.config);
    println!();
    
    // Create manifest
    let mut manifest = BenchmarkManifest::new(strict_mode);
    manifest.record_git_state()?;
    manifest.record_config(&config)?;
    
    // ... rest of benchmark ...
    
    // Save manifest at end
    manifest.save(&config.output.manifest_dir)?;
    
    Ok(())
}
```

### Step 3.3: Implement Benchmark Manifest

**File:** `crates/prism-ve-bench/src/manifest.rs` (NEW)

```rust
use serde::{Serialize, Deserialize};
use sha2::{Sha256, Digest};
use std::collections::HashMap;

#[derive(Serialize, Deserialize)]
pub struct BenchmarkManifest {
    // Execution metadata
    pub timestamp: String,
    pub strict_mode: bool,
    pub git_commit: String,
    pub git_dirty: bool,
    
    // Configuration
    pub config_hash: String,
    pub config_content: String,
    
    // Environment
    pub rustc_version: String,
    pub cuda_version: Option<String>,
    pub gpu_name: Option<String>,
    
    // Input data
    pub data_hashes: HashMap<String, String>,  // file → SHA256
    
    // Results
    pub accuracy_mean: f64,
    pub per_country_accuracy: HashMap<String, f64>,
    
    // Exclusions
    pub total_evaluated: usize,
    pub excluded_negligible: usize,
    pub excluded_undecided: usize,
    pub excluded_low_freq: usize,
    
    // Output artifacts
    pub output_hashes: HashMap<String, String>,  // predictions.csv → SHA256
}

impl BenchmarkManifest {
    pub fn new(strict_mode: bool) -> Self {
        Self {
            timestamp: chrono::Utc::now().to_rfc3339(),
            strict_mode,
            git_commit: String::new(),
            git_dirty: false,
            config_hash: String::new(),
            config_content: String::new(),
            rustc_version: rustc_version::version().to_string(),
            cuda_version: None,
            gpu_name: None,
            data_hashes: HashMap::new(),
            accuracy_mean: 0.0,
            per_country_accuracy: HashMap::new(),
            total_evaluated: 0,
            excluded_negligible: 0,
            excluded_undecided: 0,
            excluded_low_freq: 0,
            output_hashes: HashMap::new(),
        }
    }
    
    pub fn record_git_state(&mut self) -> Result<()> {
        // Get git commit hash
        let output = std::process::Command::new("git")
            .args(&["rev-parse", "HEAD"])
            .output()?;
        self.git_commit = String::from_utf8(output.stdout)?.trim().to_string();
        
        // Check if dirty
        let output = std::process::Command::new("git")
            .args(&["diff-index", "--quiet", "HEAD", "--"])
            .status()?;
        self.git_dirty = !output.success();
        
        Ok(())
    }
    
    pub fn record_config(&mut self, config: &BenchmarkConfig) -> Result<()> {
        let config_str = toml::to_string(config)?;
        self.config_content = config_str.clone();
        
        let mut hasher = Sha256::new();
        hasher.update(config_str.as_bytes());
        self.config_hash = format!("{:x}", hasher.finalize());
        
        Ok(())
    }
    
    pub fn hash_file(&mut self, path: &str) -> Result<()> {
        let data = std::fs::read(path)?;
        let mut hasher = Sha256::new();
        hasher.update(&data);
        let hash = format!("{:x}", hasher.finalize());
        self.data_hashes.insert(path.to_string(), hash);
        Ok(())
    }
    
    pub fn save(&self, dir: &str) -> Result<()> {
        std::fs::create_dir_all(dir)?;
        
        let manifest_path = format!("{}/run_manifest.json", dir);
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(&manifest_path, json)?;
        
        println!("\n✅ Manifest saved: {}", manifest_path);
        
        Ok(())
    }
}
```

### Step 3.4: Add Strict Mode Guards Throughout

**File:** `crates/prism-ve-bench/src/vasil_exact_metric.rs`

```rust
// At start of compute_vasil_metric_exact()
if strict_mode {
    // Check 75-PK envelope is available
    if !has_75pk_envelope(immunity_cache) {
        bail!("STRICT MODE FAILURE: 75-PK envelope not implemented\n\
               Found: averaged immunity (single value per variant-date)\n\
               Required: full [f64; 75] array per variant-date\n\
               \n\
               This is a critical accuracy blocker. See Phase 1 of implementation guide.");
    }
    
    // Check no placeholders in temporal data
    if has_placeholder_temporal_data() {
        bail!("STRICT MODE FAILURE: Placeholder temporal data detected\n\
               time_since_infection, freq_history_7d, or current_freq using constants\n\
               Required: extract from real VASIL phi/frequency CSVs\n\
               \n\
               See Phase 2 of implementation guide.");
    }
}
```

### Step 3.5: Save Predictions and Exclusions

**File:** `crates/prism-ve-bench/src/vasil_exact_metric.rs`

```rust
// After computing accuracy
if config.output.save_predictions {
    let predictions_path = format!("{}/predictions.csv", config.output.manifest_dir);
    save_predictions_csv(&predictions, &predictions_path)?;
    manifest.hash_file(&predictions_path)?;
}

if config.output.save_exclusions {
    let exclusions = ExclusionReport {
        total_evaluated,
        excluded_negligible,
        excluded_undecided,
        excluded_low_freq,
        by_country: per_country_exclusions,
    };
    
    let exclusions_path = format!("{}/exclusions.json", config.output.manifest_dir);
    std::fs::write(&exclusions_path, serde_json::to_string_pretty(&exclusions)?)?;
    manifest.hash_file(&exclusions_path)?;
}

fn save_predictions_csv(predictions: &[Prediction], path: &str) -> Result<()> {
    let mut wtr = csv::Writer::from_path(path)?;
    
    wtr.write_record(&["country", "lineage", "date", "gamma_min", "gamma_max", 
                       "gamma_mean", "prediction", "actual", "correct"])?;
    
    for pred in predictions {
        wtr.write_record(&[
            &pred.country,
            &pred.lineage,
            &pred.date.to_string(),
            &pred.gamma_min.to_string(),
            &pred.gamma_max.to_string(),
            &pred.gamma_mean.to_string(),
            &format!("{:?}", pred.prediction),
            &format!("{:?}", pred.actual),
            &pred.correct.to_string(),
        ])?;
    }
    
    wtr.flush()?;
    Ok(())
}
```

**Success Criteria for Phase 3:**
- ✅ Single CLI: `vasil-benchmark --config bench.toml --strict`
- ✅ Strict mode fails on: missing 75-PK envelope, placeholders, missing config
- ✅ Manifest includes: git hash, config hash, data SHA256s, exclusion counts
- ✅ Outputs: run_manifest.json, predictions.csv, exclusions.json
- ✅ Deterministic: same config → identical manifest hashes

---

## PHASE 4: CENTRALIZE CONSTANTS + CI TESTS (PREVENT REGRESSIONS)

**Objective:** Single source of truth for all constants, automated tests to prevent leakage/non-determinism.

### Step 4.1: Create Constants Module

**File:** `crates/prism-ve-bench/src/constants.rs` (NEW)

```rust
//! PRISM-VE Constants - SINGLE SOURCE OF TRUTH
//!
//! ⚠️ WARNING: These constants MUST match their CUDA counterparts.
//! Any changes here must be synchronized with:
//! - crates/prism-gpu/src/kernels/mega_fused_batch.cu
//! - crates/prism-gpu/src/kernels/mega_fused_pocket_kernel.cu
//!
//! A CI test verifies consistency (see tests/constants_consistency_test.rs)

/// VASIL gamma weighting: γ = α × escape + β × transmit
pub const ALPHA_ESCAPE: f64 = 0.65;
pub const BETA_TRANSMIT: f64 = 0.35;

/// 75 PK parameter combinations (5 tmax × 15 thalf)
pub const TMAX_VALUES: [f32; 5] = [14.0, 17.5, 21.0, 24.5, 28.0];
pub const THALF_VALUES: [f32; 15] = [
    25.0, 28.14, 31.29, 34.43, 37.57,
    40.71, 43.86, 47.0, 50.14, 53.29,
    56.43, 59.57, 62.71, 65.86, 69.0
];
pub const N_PK_COMBINATIONS: usize = 75;

/// 10 Epitope classes (Bloom Lab DMS)
pub const N_EPITOPE_CLASSES: usize = 10;

/// Feature dimensions
pub const BASE_FEATURE_DIM: usize = 136;
pub const POLYCENTRIC_FEATURE_DIM: usize = 22;
pub const TOTAL_FEATURE_DIM: usize = BASE_FEATURE_DIM + POLYCENTRIC_FEATURE_DIM;  // 158

/// VASIL thresholds
pub const NEGLIGIBLE_CHANGE_THRESHOLD: f32 = 0.05;  // 5% relative
pub const MIN_FREQUENCY_THRESHOLD: f32 = 0.03;      // 3% absolute
pub const MIN_PEAK_FREQUENCY: f32 = 0.01;           // 1% peak (VASIL spec)

/// Fold resistance bounds
pub const FR_MIN: f64 = 1.0;   // VASIL: FR ≥ 1 always
pub const FR_MAX: f64 = 100.0;

/// IC50 calibrated values (per VASIL Delta VE calibration)
pub const CALIBRATED_IC50: [f64; 10] = [
    0.85, 1.12, 0.93, 1.05, 0.98,
    1.21, 0.89, 1.08, 0.95, 1.03
];
```

**File:** `crates/prism-ve-bench/src/lib.rs`

```rust
pub mod constants;  // Make constants accessible throughout crate
```

### Step 4.2: Replace All Usages

**Files to update:**
1. `gpu_benchmark.rs` - Remove duplicates, import from constants module
2. `main.rs` - Use `constants::ALPHA_ESCAPE`, etc.
3. `vasil_exact_metric.rs` - Import from constants module

**Example:**
```rust
// OLD
const ALPHA_ESCAPE: f64 = 0.65;

// NEW
use crate::constants::ALPHA_ESCAPE;
```

### Step 4.3: Add Constants Consistency CI Test

**File:** `.github/workflows/constants_check.yml` (NEW)

```yaml
name: Constants Consistency Check

on: [push, pull_request]

jobs:
  check-duplicates:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Check for duplicate constant definitions
        run: |
          # Check ALPHA_ESCAPE (should only be in constants.rs)
          count=$(grep -r "const ALPHA_ESCAPE" --include="*.rs" | wc -l)
          if [ $count -ne 1 ]; then
            echo "ERROR: ALPHA_ESCAPE defined in $count places (expected 1)"
            grep -r "const ALPHA_ESCAPE" --include="*.rs"
            exit 1
          fi
          
          # Check BETA_TRANSMIT
          count=$(grep -r "const BETA_TRANSMIT" --include="*.rs" | wc -l)
          if [ $count -ne 1 ]; then
            echo "ERROR: BETA_TRANSMIT defined in $count places (expected 1)"
            exit 1
          fi
          
          echo "✅ Constants centralized correctly"
```

### Step 4.4: Add Leakage Canary Test

**File:** `crates/prism-ve-bench/tests/leakage_canary_test.rs` (NEW)

```rust
#[test]
fn test_no_test_data_leakage_canary() {
    use chrono::NaiveDate;
    
    // Setup: Create train and test data with unique sentinels
    let train_cutoff = NaiveDate::from_ymd_opt(2022, 6, 1).unwrap();
    
    let mut train_data = load_train_data();
    let mut test_data = load_test_data();
    
    // Inject unique sentinel into TEST data only
    let sentinel_lineage = "CANARY_TEST_ONLY_XBB.1.99";
    let sentinel_date = NaiveDate::from_ymd_opt(2023, 1, 1).unwrap();
    assert!(sentinel_date >= train_cutoff, "Sentinel must be in test period");
    
    test_data.insert(sentinel_lineage.to_string(), sentinel_date, 0.05);
    
    // Build caches and calibrations using ONLY train data
    let cache = build_immunity_cache(&train_data, train_cutoff);
    let calibration = calibrate_parameters(&train_data, train_cutoff);
    
    // ASSERT: Sentinel lineage never appears in train artifacts
    assert!(!cache.contains_lineage(sentinel_lineage),
            "LEAKAGE DETECTED: Test sentinel found in immunity cache");
    
    assert!(!calibration.contains_lineage(sentinel_lineage),
            "LEAKAGE DETECTED: Test sentinel found in calibration");
}
```

### Step 4.5: Add Determinism Test

**File:** `crates/prism-ve-bench/tests/determinism_test.rs` (NEW)

```rust
#[test]
fn test_deterministic_reproducibility() {
    let config = load_test_config();
    
    // Run benchmark twice with same config
    let (manifest1, predictions1) = run_benchmark(&config).unwrap();
    let (manifest2, predictions2) = run_benchmark(&config).unwrap();
    
    // ASSERT: Identical results
    assert_eq!(manifest1.accuracy_mean, manifest2.accuracy_mean,
               "Accuracy not deterministic");
    
    assert_eq!(manifest1.config_hash, manifest2.config_hash,
               "Config hash not deterministic");
    
    assert_eq!(predictions1.len(), predictions2.len(),
               "Number of predictions not deterministic");
    
    for (pred1, pred2) in predictions1.iter().zip(predictions2.iter()) {
        assert_eq!(pred1.gamma_min, pred2.gamma_min, "Gamma not deterministic");
        assert_eq!(pred1.prediction, pred2.prediction, "Prediction not deterministic");
    }
}
```

### Step 4.6: Add No-Placeholder Strict Mode Test

**File:** `crates/prism-ve-bench/tests/strict_mode_test.rs` (NEW)

```rust
#[test]
#[should_panic(expected = "STRICT MODE FAILURE: Placeholder temporal data")]
fn test_strict_mode_fails_on_placeholders() {
    // Create config with strict mode enabled
    let mut config = load_test_config();
    config.behavior.strict_mode = true;
    
    // Simulate placeholder data (constant values)
    let placeholder_temporal_data = TemporalData {
        time_since_infection: vec![30.0; 100],  // Constant = placeholder
        freq_history: vec![0.1; 700],            // Constant = placeholder
    };
    
    // Should panic in strict mode
    validate_temporal_data(&placeholder_temporal_data, &config).unwrap();
}

#[test]
fn test_strict_mode_passes_with_real_data() {
    let mut config = load_test_config();
    config.behavior.strict_mode = true;
    
    // Real data (non-constant)
    let real_temporal_data = TemporalData {
        time_since_infection: vec![30.0, 45.0, 60.0, 15.0],  // Varying
        freq_history: extract_real_freq_history(),           // Real data
    };
    
    // Should NOT panic
    validate_temporal_data(&real_temporal_data, &config).unwrap();
}
```

**Success Criteria for Phase 4:**
- ✅ All constants in `crates/prism-ve-bench/src/constants.rs` (single source)
- ✅ CI test fails if constants duplicated
- ✅ Leakage canary test passes
- ✅ Determinism test passes (two runs → identical hashes)
- ✅ Strict mode test fails on placeholders, passes on real data

---

## PHASE 5: README + DOCUMENTATION (PUBLICATION-READY)

**Objective:** Document exact reproduction steps, scientific guarantees, known limitations.

### Step 5.1: Create Benchmark README

**File:** `README_BENCHMARK.md` (NEW, repository root)

```markdown
# PRISM-VE VASIL Benchmark - Scientific Evaluation

**Version:** 1.0  
**Status:** Production-Ready  
**Target Accuracy:** 92% (VASIL Nature publication standard)  
**Current Accuracy:** [UPDATE AFTER PHASE 1-4]  

---

## Quick Start

### One-Command Reproduction

```bash
# Build
cargo build --release -p prism-ve-bench

# Run benchmark (strict mode)
./target/release/vasil-benchmark --config bench.toml --strict

# Results in: benchmark_results/
#   - run_manifest.json (full provenance)
#   - predictions.csv (all predictions)
#   - exclusions.json (exclusion counts)
```

### Configuration

Edit `bench.toml`:

```toml
[evaluation]
train_cutoff = "2022-06-01"  # Train/test split date
eval_start = "2022-10-01"    # Evaluation window
eval_end = "2023-10-31"

[behavior]
strict_mode = true           # Fail on approximations
```

---

## Scientific Guarantees

### 1. Zero Data Leakage ✅

**Guarantee:** No information from test dates (≥ 2022-06-01) influences training artifacts.

**Enforcement:**
- Hard cutoff in `vasil_exact_metric.rs:1042`
- Cache provenance tracking (min_date, max_date, cutoff_used)
- CI canary test (`tests/leakage_canary_test.rs`)

**Verification:**
```bash
# Run leakage test
cargo test leakage_canary
```

### 2. Deterministic Reproducibility ✅

**Guarantee:** Same config + code + data = identical results.

**Enforcement:**
- Seeded randomness (config.behavior.seed)
- Deterministic CUDA algorithms
- Full provenance in manifest (git hash, data hashes)

**Verification:**
```bash
# Two runs should produce identical manifests
./target/release/vasil-benchmark --config bench.toml
cp benchmark_results/run_manifest.json run1.json

./target/release/vasil-benchmark --config bench.toml
cp benchmark_results/run_manifest.json run2.json

diff run1.json run2.json  # Should be empty
```

### 3. VASIL-Exact Methodology ✅

**Guarantee:** Implements Extended Data Fig 6a from VASIL Nature paper exactly.

**Implementation:**
- 75-PK envelope (all combinations, no averaging)
- Per-day rising/falling predictions
- Exclusions: negligible (<5%), undecided (envelope crosses 0), low freq (<3%)
- Per-country accuracy → mean

**Code:** `crates/prism-ve-bench/src/vasil_exact_metric.rs`

---

## Strict vs. Non-Strict Mode

### Strict Mode (Production, CI)

**Enabled:** `--strict` flag OR `bench.toml`: `strict_mode = true`

**Behavior:**
- ❌ FAILS if 75-PK envelope not available
- ❌ FAILS if placeholder temporal data used
- ❌ FAILS if missing phi/frequency data
- ❌ FAILS if cache provenance invalid
- ✅ PASSES only if 100% VASIL-exact

**Use for:** Publication results, CI validation

### Non-Strict Mode (Development)

**Enabled:** Default if `strict_mode = false`

**Behavior:**
- ⚠️ WARNS on placeholders (continues execution)
- ⚠️ Uses fallbacks for missing data
- 🏷️ Labels results "NOT VASIL-exact" if approximations used

**Use for:** Development, debugging, ablation studies

---

## Output Artifacts

### run_manifest.json

Full provenance record:

```json
{
  "timestamp": "2025-12-17T12:34:56Z",
  "strict_mode": true,
  "git_commit": "abc123...",
  "git_dirty": false,
  "config_hash": "def456...",
  "data_hashes": {
    "/path/to/data/file.csv": "sha256:789..."
  },
  "accuracy_mean": 0.874,
  "per_country_accuracy": {
    "Germany": 0.901,
    "USA": 0.856
  },
  "excluded_negligible": 145,
  "excluded_undecided": 89,
  "total_evaluated": 1193
}
```

### predictions.csv

All per-day predictions:

```csv
country,lineage,date,gamma_min,gamma_max,gamma_mean,prediction,actual,correct
Germany,BA.1,2023-01-15,-0.12,0.08,-0.02,Undecided,Rising,false
USA,XBB.1.5,2023-05-20,0.15,0.34,0.24,Rising,Rising,true
```

### exclusions.json

Detailed exclusion counts:

```json
{
  "total_evaluated": 1193,
  "excluded_negligible": 145,
  "excluded_undecided": 89,
  "excluded_low_freq": 67,
  "by_country": {
    "Germany": {
      "evaluated": 98,
      "excluded_negligible": 12
    }
  }
}
```

---

## Known Limitations

### Implemented ✅

1. 75-PK envelope computation (Phase 1)
2. Real temporal data extraction (Phase 2)
3. Strict mode + manifest (Phase 3)
4. Constants centralization + CI tests (Phase 4)

### Not Yet Implemented ⚠️

1. **Immunological landscape timeseries** (Germany only in VASIL data)
   - Impact: Minor (used for visualization, not prediction)
   - Workaround: Generate from immunity cache

2. **Phi data for 3 countries** (UK, Denmark, South Africa use fallback)
   - Impact: -5-8pp accuracy for those countries
   - Workaround: Fallback = population × 0.001

### Will Not Implement ❌

1. **GPU kernel modifications** (per CLAUDE.md constraint)
   - NTD epitope class (11th) requires GPU update
   - Impact: +3-5pp if implemented
   - Status: Code ready in CPU path, GPU-blocked

---

## Accuracy Breakdown

### Current State

```
MEAN: XX.X% (UPDATE AFTER IMPLEMENTATION)

Per-Country:
  Canada:       XX.X%
  UK:           XX.X%
  Germany:      XX.X%
  ...
```

### VASIL Target

```
MEAN: 92.0%

Per-Country Range: 87-94%
```

### Gap Analysis

[UPDATE AFTER IMPLEMENTATION]

---

## CI Pipeline

### Tests Run on Every Commit

```bash
# Unit tests
cargo test

# Constants consistency
.github/workflows/constants_check.yml

# Leakage detection
cargo test leakage_canary

# Determinism
cargo test determinism

# Strict mode validation
cargo test strict_mode
```

### Benchmark Smoke Test

```bash
# Small dataset slice (1 country, 30 days)
PRISM_COUNTRIES=1 ./target/release/vasil-benchmark --config bench_smoke.toml
```

---

## Citation

If you use this benchmark, please cite:

```bibtex
@software{prism_ve_vasil_benchmark,
  title = {PRISM-VE VASIL Benchmark},
  author = {[Your Name]},
  year = {2025},
  url = {https://github.com/yourusername/prism-ve}
}
```

And the original VASIL methodology:

```bibtex
@article{vasil2024,
  title = {Predicting SARS-CoV-2 variant spread with immune escape and transmissibility},
  author = {Obermeyer et al.},
  journal = {Nature},
  year = {2024}
}
```

---

## Contact

Questions? Issues? Open a GitHub issue or contact [your email].
```

### Step 5.2: Update Main README.md

**File:** `README.md` (UPDATE existing)

Add section:

```markdown
## VASIL Benchmark Evaluation

PRISM-VE includes a production-grade VASIL benchmark with:
- ✅ 77.4% → 92% accuracy target
- ✅ Zero data leakage (CI-tested)
- ✅ Deterministic reproducibility
- ✅ Full audit manifest (git hash, data SHA256s)

**Quick Start:**
```bash
cargo build --release -p prism-ve-bench
./target/release/vasil-benchmark --config bench.toml --strict
```

**Documentation:** See [README_BENCHMARK.md](README_BENCHMARK.md)
```

**Success Criteria for Phase 5:**
- ✅ README_BENCHMARK.md complete with reproduction steps
- ✅ Strict vs non-strict behavior documented
- ✅ Scientific guarantees stated clearly
- ✅ Known limitations listed
- ✅ Citation information included

---

## 🎯 FINAL CHECKLIST

Before declaring implementation complete, verify:

### Code Quality
- [ ] All phases (0-5) implemented
- [ ] 75-PK envelope working (Phase 1)
- [ ] Real temporal data extracted (Phase 2)
- [ ] Strict mode + manifest (Phase 3)
- [ ] Constants centralized (Phase 4)
- [ ] Documentation complete (Phase 5)

### Tests Pass
- [ ] `cargo test` (all unit tests)
- [ ] `cargo test leakage_canary` (no leakage)
- [ ] `cargo test determinism` (reproducible)
- [ ] `cargo test strict_mode` (fail-fast works)
- [ ] CI pipeline green

### Benchmarks Run
- [ ] `vasil-benchmark --strict` completes without errors
- [ ] Manifest generated with all required fields
- [ ] predictions.csv and exclusions.json saved
- [ ] Accuracy ≥ 82% (conservative target)
- [ ] Accuracy ≥ 90% (stretch goal)

### Scientific Integrity
- [ ] Zero data leakage verified
- [ ] Deterministic (two runs → identical hashes)
- [ ] 75-PK envelope used (not averaged)
- [ ] Undecided predictions excluded
- [ ] No placeholders in strict mode

### Documentation
- [ ] README_BENCHMARK.md accurate
- [ ] bench.toml has clear comments
- [ ] All functions have docstrings
- [ ] Known limitations listed

---

## 🚀 EXECUTION TIMELINE

**Estimated Total Effort:** 16-24 hours over 4-5 days

### Day 1: Phase 0-1 (Critical Path)
- Hours 1-2: Phase 0 assessment + report
- Hours 3-8: Phase 1 (75-PK envelope fix)
- **Milestone:** Accuracy improves to 82-87%

### Day 2: Phase 2 (Polycentric Enabler)
- Hours 9-14: Extract real temporal data
- **Milestone:** Wave features contribute signal (+2-5%)

### Day 3: Phase 3 (Rigor)
- Hours 15-19: Strict mode + CLI + manifest
- **Milestone:** Full reproducibility

### Day 4: Phase 4 (Prevent Regressions)
- Hours 20-23: Constants + CI tests
- **Milestone:** CI pipeline green

### Day 5: Phase 5 + Validation
- Hours 24-26: Documentation
- Hours 27-28: End-to-end validation
- **Milestone:** Publication-ready

---

## ⚠️ CRITICAL SUCCESS FACTORS

### 1. DO NOT SKIP PHASE 0
You MUST understand current state before coding. Changing wrong code wastes hours.

### 2. PHASE 1 IS THE ACCURACY BOTTLENECK
75-PK envelope fix is why you're at 77.4% instead of 92%. This is your highest priority.

### 3. TEST EACH PHASE BEFORE PROCEEDING
Don't stack unverified changes. Test Phase 1 → verify accuracy gain → proceed to Phase 2.

### 4. STRICT MODE IS NON-NEGOTIABLE FOR PUBLICATION
If strict mode fails, results cannot be published. Fix the issue, don't disable strict mode.

### 5. MANIFEST MUST BE COMPLETE
Reviewers will demand reproducibility proof. Manifest is your evidence.

---

## 📞 HELP / TROUBLESHOOTING

### Phase 1: GPU returns averaged immunity instead of full array

**Symptom:** Accuracy still 77.4% after Phase 1

**Debug:**
```rust
// Add debug print after GPU download
eprintln!("Immunity shape: {} × {}", immunity_data.len(), immunity_data[0].len());
// Should print: "Immunity shape: N × 75"
// If prints "Immunity shape: N × 1", GPU is still averaging
```

**Fix:** Check GPU kernel launch - ensure `z` dimension = 75, not reduction to mean

### Phase 2: Temporal data still constant

**Symptom:** Wave features show no variance

**Debug:**
```rust
let variance = compute_variance(&time_since_infection);
assert!(variance > 1.0, "time_since_infection is constant!");
```

**Fix:** Check `phi_data` HashMap is populated - if empty, data loading failed

### Phase 3: Manifest missing fields

**Symptom:** `run_manifest.json` has empty/null fields

**Debug:**
```rust
dbg!(&manifest);  // Before saving
```

**Fix:** Ensure `record_git_state()` and `record_config()` called before `save()`

### Phase 4: CI test false positive

**Symptom:** Determinism test fails despite identical code

**Cause:** GPU non-determinism (random order of operations)

**Fix:** Set `config.gpu.deterministic = true` and use `CUBLAS_WORKSPACE_CONFIG=:4096:8`

---

## 📊 SUCCESS METRICS

### Minimum Viable (Publication-Ready)

- ✅ Accuracy ≥ 85% (within 7pp of VASIL)
- ✅ Strict mode passes (100% VASIL-exact)
- ✅ Zero data leakage (CI-verified)
- ✅ Deterministic (manifest hash identical across runs)
- ✅ Documentation complete

### Stretch Goal (Match VASIL)

- ✅ Accuracy ≥ 90% (within 2pp of VASIL)
- ✅ All 12 countries functional (no fallback data)
- ✅ Polycentric features contribute (+3-5% gain)
- ✅ Paper-ready figures + tables generated

---

## 🎓 OPERATING PRINCIPLES

### For Claude Code (You)

1. **Read Before Writing**
   - Understand current state from forensic audit
   - Never guess what VASIL methodology requires
   - Derive from provided documentation

2. **Fail Loudly**
   - Prefer crashing over silently producing wrong results
   - In strict mode: any approximation = immediate failure
   - Clear error messages with fix instructions

3. **Single Source of Truth**
   - One constants module, one config, one entrypoint
   - Delete or gate alternative paths
   - No "temporary" duplicates that become permanent

4. **Test Incrementally**
   - Implement Phase 1 → test → verify accuracy gain
   - Don't stack 5 untested phases
   - Each phase has pass/fail criteria

5. **Document Decisions**
   - If VASIL paper ambiguous, document your interpretation
   - If placeholder needed temporarily, add TODO with issue number
   - Explain WHY in code comments, not just WHAT

### For Reviewers (Future You)

1. **Manifest = Proof**
   - If manifest says "strict_mode: true", results are trustworthy
   - If "git_dirty: true", results are unreproducible
   - SHA256 mismatches = data corrupted

2. **Accuracy < 85% = Incomplete**
   - Either 75-PK envelope not working (Phase 1)
   - Or temporal data still placeholder (Phase 2)
   - Or fundamental methodology difference

3. **No Manifest = Don't Trust**
   - Without git hash + data hashes, can't reproduce
   - Without exclusion counts, can't audit
   - Without config hash, can't verify settings

---

## 🏆 DELIVERABLES SUMMARY

Upon completion, repository will contain:

### Code
- ✅ `vasil-benchmark` binary (single CLI entrypoint)
- ✅ 75-PK envelope implementation (Phase 1)
- ✅ Real temporal data extraction (Phase 2)
- ✅ Strict mode enforcement (Phase 3)
- ✅ `constants.rs` (single source of truth, Phase 4)

### Configuration
- ✅ `bench.toml` (production config)
- ✅ `bench_smoke.toml` (CI smoke test)

### Tests
- ✅ `tests/envelope_decision_test.rs` (golden file)
- ✅ `tests/temporal_data_test.rs` (non-constant check)
- ✅ `tests/leakage_canary_test.rs` (CI leakage detection)
- ✅ `tests/determinism_test.rs` (reproducibility)
- ✅ `tests/strict_mode_test.rs` (fail-fast validation)

### CI
- ✅ `.github/workflows/constants_check.yml`
- ✅ `.github/workflows/benchmark_smoke.yml`

### Documentation
- ✅ `README_BENCHMARK.md` (reproduction guide)
- ✅ Updated `README.md` (quick start)
- ✅ Code comments (methodology explanations)

### Outputs (after run)
- ✅ `benchmark_results/run_manifest.json`
- ✅ `benchmark_results/predictions.csv`
- ✅ `benchmark_results/exclusions.json`

---

**END OF IMPLEMENTATION PROMPT**

---

## READY TO START?

**Begin with Phase 0.** Read the repository documents, analyze current state, produce Phase 0 Report.

**Do not write code until Phase 0 Report is complete and accurate.**

Good luck! 🚀
