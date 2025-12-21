# PRISM-VE Fitness Module Implementation Plan

## Executive Summary

This document outlines the implementation of the **Viral Evolution Fitness Module** for PRISM-VE, designed to predict SARS-CoV-2 variant dynamics and benchmark against VASIL's 0.92 accuracy.

**Goal**: GPU-accelerated immune fitness calculation that integrates with mega_fused.rs architecture to predict which variants will rise or fall in frequency.

---

## 1. Architecture Overview

### 1.1 Fitness Module Position in PRISM-VE Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│ PRISM-VE 7-Phase Pipeline (Existing)                           │
│                                                                 │
│ Phase 1-2: Structure Analysis → Pocket Detection               │
│ Phase 3-4: Network Dynamics → Topological Features             │
│ Phase 5-6: Dendritic Processing → Quantum/Thermodynamic        │
│ Phase 7: Integration & Output                                  │
└─────────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│ NEW: Viral Evolution Fitness Module (Phase 8 - VE)            │
│                                                                 │
│ Stage 1: DMS Escape Score Calculation (GPU)                    │
│ Stage 2: Cross-Neutralization Computation (GPU)                │
│ Stage 3: Population Immunity Integration (CPU/GPU Hybrid)      │
│ Stage 4: Variant Fitness Scoring (GPU)                         │
│ Stage 5: Dynamics Prediction (γ > 0 → RISE, γ < 0 → FALL)     │
└─────────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│ VASIL Benchmark Validation                                     │
│ Target: 0.92 accuracy on variant rise/fall prediction          │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Integration with mega_fused.rs Pattern

Following the established multi-pass kernel architecture:

```rust
// In mega_fused.rs or new viral_evolution_fused.rs

pub struct ViralEvolutionFusedGpu {
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,

    // GPU Kernels
    dms_escape_kernel: Option<CudaFunction>,           // Stage 1
    cross_neutralization_kernel: Option<CudaFunction>, // Stage 2
    fitness_scoring_kernel: Option<CudaFunction>,      // Stage 4

    // Buffer Pool (zero-allocation hot path)
    buffer_pool: VEBufferPool,

    // Runtime Parameters
    params: VEFusedParams,
}

pub struct VEBufferPool {
    // Variant data
    d_spike_mutations: Option<CudaSlice<i32>>,       // Mutation array
    d_epitope_assignments: Option<CudaSlice<i32>>,   // Mutation → epitope class

    // DMS escape data
    d_escape_matrix: Option<CudaSlice<f32>>,         // [836 abs × 201 sites]
    d_antibody_classes: Option<CudaSlice<i32>>,      // [836] → 10 epitope classes

    // Population immunity
    d_immunity_landscape: Option<CudaSlice<f32>>,    // [10 classes × time points]

    // Outputs
    d_escape_scores: Option<CudaSlice<f32>>,         // Per-variant escape scores
    d_fitness_values: Option<CudaSlice<f32>>,        // Final fitness (γ)

    capacity: usize,
}
```

---

## 2. GPU Kernel Design

### 2.1 Kernel Architecture: viral_evolution_fitness.cu

Following PRISM-VE's multi-stage pattern:

```cuda
// ============================================================================
// STAGE 1: DMS Escape Score Calculation
// ============================================================================

// Constant memory for DMS data (faster than global memory)
__constant__ float c_escape_matrix[836 * 201];      // 836 abs × 201 RBD sites
__constant__ int c_antibody_epitopes[836];          // Antibody → epitope class
__constant__ float c_epitope_weights[10];           // Weight per class

/**
 * Compute antibody escape score for a variant based on DMS data.
 *
 * Input:
 *   - spike_mutations: [n_mutations] array of mutation positions
 *   - mutation_aa: [n_mutations] array of amino acid changes
 *   - n_mutations: Number of mutations in variant
 *   - escape_matrix: [836 × 201] DMS escape scores
 *   - antibody_epitopes: [836] antibody → epitope class mapping
 *
 * Output:
 *   - escape_scores: [10] escape score per epitope class
 */
extern "C" __global__ void __launch_bounds__(256, 4)
stage1_dms_escape_scores(
    const int* __restrict__ spike_mutations,     // [n_mutations]
    const char* __restrict__ mutation_aa,        // [n_mutations]
    const int n_mutations,
    const int n_variants,
    float* __restrict__ escape_scores_out        // [n_variants × 10]
) {
    int variant_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (variant_idx >= n_variants) return;

    // Shared memory for per-epitope aggregation
    __shared__ float smem_escape[10];

    // Initialize
    if (threadIdx.x < 10) {
        smem_escape[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    // For each mutation in this variant
    int mutation_offset = variant_idx * MAX_MUTATIONS_PER_VARIANT;
    for (int m = 0; m < n_mutations; m++) {
        int site = spike_mutations[mutation_offset + m];
        char aa = mutation_aa[mutation_offset + m];

        if (site < 331 || site > 531) continue;  // Only RBD (331-531)

        int rbd_site = site - 331;  // 0-200 index

        // Aggregate escape across all 836 antibodies
        for (int ab = 0; ab < 836; ab++) {
            float escape_value = c_escape_matrix[ab * 201 + rbd_site];
            int epitope_class = c_antibody_epitopes[ab];

            // Atomic add to epitope class (10 classes)
            atomicAdd(&smem_escape[epitope_class], escape_value);
        }
    }
    __syncthreads();

    // Write aggregated escape scores per epitope class
    if (threadIdx.x < 10) {
        escape_scores_out[variant_idx * 10 + threadIdx.x] = smem_escape[threadIdx.x];
    }
}


// ============================================================================
// STAGE 2: Cross-Neutralization Computation
// ============================================================================

/**
 * Compute fold-reduction in neutralization between two variants.
 *
 * Based on VASIL formula:
 *   fold_reduction = exp(sum(escape_scores[epitope_class] * weight[class]))
 *
 * Where weights account for population immunity landscape.
 */
extern "C" __global__ void __launch_bounds__(256, 4)
stage2_cross_neutralization(
    const float* __restrict__ escape_scores,      // [n_variants × 10]
    const float* __restrict__ immunity_weights,   // [10] current immunity per class
    const int n_variants,
    float* __restrict__ fold_reduction_out        // [n_variants]
) {
    int variant_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (variant_idx >= n_variants) return;

    float total_escape = 0.0f;

    // Sum weighted escape across 10 epitope classes
    for (int epitope = 0; epitope < 10; epitope++) {
        float escape = escape_scores[variant_idx * 10 + epitope];
        float weight = immunity_weights[epitope];
        total_escape += escape * weight;
    }

    // Fold-reduction in neutralization (exponential)
    fold_reduction_out[variant_idx] = expf(total_escape);
}


// ============================================================================
// STAGE 3: Population Immunity Landscape (CPU-driven, GPU-accelerated)
// ============================================================================

// This stage handled on CPU with data updates to GPU
// - Tracks vaccination campaigns over time
// - Tracks infection waves (from GInPipe incidence reconstruction)
// - Updates immunity_weights[10] based on historical exposure


// ============================================================================
// STAGE 4: Variant Fitness Scoring
// ============================================================================

/**
 * Compute relative fitness advantage/disadvantage (γ) for each variant.
 *
 * VASIL model:
 *   γ_variant = -log(fold_reduction) + intrinsic_transmissibility
 *
 * Where:
 *   - Higher fold_reduction → more escape → lower γ (more fit)
 *   - Intrinsic transmissibility = base R0 adjustment
 *
 * Returns:
 *   γ > 0 → variant RISING
 *   γ < 0 → variant FALLING
 */
extern "C" __global__ void __launch_bounds__(256, 4)
stage4_variant_fitness(
    const float* __restrict__ fold_reduction,     // [n_variants]
    const float* __restrict__ transmissibility,   // [n_variants] intrinsic R0
    const int n_variants,
    float* __restrict__ gamma_out                 // [n_variants] growth rate
) {
    int variant_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (variant_idx >= n_variants) return;

    float fold_red = fold_reduction[variant_idx];
    float r0_boost = transmissibility[variant_idx];

    // Fitness = immune escape advantage + transmissibility
    float gamma = -logf(fold_red) + r0_boost;

    gamma_out[variant_idx] = gamma;
}


// ============================================================================
// STAGE 5: Dynamics Prediction (Integrated with FluxNet RL)
// ============================================================================

/**
 * Predict variant frequency change over time window.
 *
 * Simple logistic growth model:
 *   dF/dt = γ * F * (1 - F)
 *
 * Where F = variant frequency, γ = fitness advantage
 */
extern "C" __global__ void __launch_bounds__(256, 4)
stage5_predict_dynamics(
    const float* __restrict__ gamma,              // [n_variants]
    const float* __restrict__ current_freq,       // [n_variants]
    const float dt,                               // Time step (days)
    const int n_variants,
    float* __restrict__ predicted_freq            // [n_variants]
) {
    int variant_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (variant_idx >= n_variants) return;

    float g = gamma[variant_idx];
    float f = current_freq[variant_idx];

    // Logistic growth: dF = γ * F * (1 - F) * dt
    float df = g * f * (1.0f - f) * dt;
    float f_new = fmaxf(0.0f, fminf(1.0f, f + df));

    predicted_freq[variant_idx] = f_new;
}
```

---

## 3. Rust Integration Layer

### 3.1 Rust Wrapper: viral_evolution_fitness.rs

```rust
// crates/prism-gpu/src/viral_evolution_fitness.rs

use cudarc::driver::{CudaContext, CudaFunction, CudaStream, LaunchConfig, CudaSlice};
use cudarc::nvrtc::Ptx;
use prism_core::PrismError;
use std::path::Path;
use std::sync::Arc;

/// GPU-accelerated viral evolution fitness module
pub struct ViralEvolutionFitnessGpu {
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,

    // Kernels (loaded from PTX)
    dms_escape_kernel: Option<CudaFunction>,
    cross_neutralization_kernel: Option<CudaFunction>,
    fitness_kernel: Option<CudaFunction>,
    dynamics_kernel: Option<CudaFunction>,

    // Pre-loaded DMS data on GPU (constant memory)
    dms_data_loaded: bool,
}

/// Runtime parameters for fitness calculation
#[repr(C, align(16))]
pub struct VEFitnessParams {
    // DMS escape parameters
    pub escape_scale: f32,              // Scaling factor for escape scores
    pub epitope_weights: [f32; 10],     // Weight per epitope class

    // Transmissibility parameters
    pub base_r0: f32,                   // Base reproduction number
    pub r0_variance: f32,               // Variance in R0 estimates

    // Population immunity
    pub immunity_decay_rate: f32,       // Antibody decay half-life (days)
    pub booster_efficacy: f32,          // Vaccine booster effectiveness

    // Prediction parameters
    pub time_horizon_days: f32,         // Forecast window
    pub frequency_threshold: f32,       // Min frequency to track

    // VASIL calibration
    pub vasil_alpha: f32,               // Immune escape weight (from VASIL)
    pub vasil_beta: f32,                // Transmissibility weight
}

impl Default for VEFitnessParams {
    fn default() -> Self {
        Self {
            escape_scale: 1.0,
            epitope_weights: [1.0; 10],  // Equal weight initially
            base_r0: 3.0,                // Omicron-like
            r0_variance: 0.2,
            immunity_decay_rate: 0.0077, // ~90 day half-life
            booster_efficacy: 0.85,
            time_horizon_days: 7.0,      // Weekly prediction
            frequency_threshold: 0.01,   // 1% minimum
            vasil_alpha: 0.65,           // Calibrated from VASIL paper
            vasil_beta: 0.35,
        }
    }
}

/// Variant mutation data
pub struct VariantData {
    pub lineage_name: String,
    pub spike_mutations: Vec<i32>,      // Mutation positions (331-531 RBD)
    pub mutation_aa: Vec<u8>,           // Amino acid changes
    pub current_frequency: f32,         // Current prevalence
    pub collection_date: String,        // YYYY-MM-DD
}

impl ViralEvolutionFitnessGpu {
    /// Create new fitness GPU executor
    pub fn new(
        context: Arc<CudaContext>,
        ptx_dir: &Path,
    ) -> Result<Self, PrismError> {
        let stream = context.default_stream();

        // Load PTX kernels
        let ptx_path = ptx_dir.join("viral_evolution_fitness.ptx");
        let ptx_src = std::fs::read_to_string(&ptx_path)
            .map_err(|e| PrismError::gpu("ve_fitness", format!("Read PTX: {}", e)))?;

        let module = context.load_module(Ptx::from_src(ptx_src))
            .map_err(|e| PrismError::gpu("ve_fitness", format!("Load module: {}", e)))?;

        let dms_escape_kernel = module.load_function("stage1_dms_escape_scores").ok();
        let cross_neutralization_kernel = module.load_function("stage2_cross_neutralization").ok();
        let fitness_kernel = module.load_function("stage4_variant_fitness").ok();
        let dynamics_kernel = module.load_function("stage5_predict_dynamics").ok();

        Ok(Self {
            context,
            stream,
            dms_escape_kernel,
            cross_neutralization_kernel,
            fitness_kernel,
            dynamics_kernel,
            dms_data_loaded: false,
        })
    }

    /// Load DMS escape data into GPU constant memory
    pub fn load_dms_data(
        &mut self,
        escape_matrix: &[f32; 836 * 201],  // 836 abs × 201 sites
        antibody_epitopes: &[i32; 836],    // Antibody → epitope class
    ) -> Result<(), PrismError> {
        // TODO: Upload to constant memory via cuMemcpyToSymbol
        // For now, will use global memory
        self.dms_data_loaded = true;
        Ok(())
    }

    /// Compute fitness for a set of variants
    pub fn compute_fitness(
        &mut self,
        variants: &[VariantData],
        params: &VEFitnessParams,
    ) -> Result<Vec<f32>, PrismError> {
        let n_variants = variants.len();

        if !self.dms_data_loaded {
            return Err(PrismError::gpu("ve_fitness", "DMS data not loaded"));
        }

        // Stage 1: Compute DMS escape scores
        let escape_scores = self.compute_dms_escape(variants)?;

        // Stage 2: Cross-neutralization
        let fold_reductions = self.compute_cross_neutralization(
            &escape_scores,
            &params.epitope_weights,
        )?;

        // Stage 3: Population immunity (CPU-side for now)
        // TODO: Integrate with cycle module for immunity landscape

        // Stage 4: Variant fitness
        let transmissibility = vec![params.base_r0; n_variants];
        let gamma_values = self.compute_variant_fitness(
            &fold_reductions,
            &transmissibility,
        )?;

        Ok(gamma_values)
    }

    /// Predict variant dynamics (rise or fall)
    pub fn predict_dynamics(
        &mut self,
        variants: &[VariantData],
        gamma: &[f32],
        time_horizon_days: f32,
    ) -> Result<Vec<f32>, PrismError> {
        let n_variants = variants.len();
        let func = self.dynamics_kernel.as_ref()
            .ok_or_else(|| PrismError::gpu("ve_fitness", "Dynamics kernel not loaded"))?;

        // Allocate GPU memory
        let mut d_gamma = self.stream.alloc_zeros::<f32>(n_variants)?;
        let current_freq: Vec<f32> = variants.iter()
            .map(|v| v.current_frequency)
            .collect();
        let mut d_current_freq = self.stream.alloc_zeros::<f32>(n_variants)?;
        let mut d_predicted_freq = self.stream.alloc_zeros::<f32>(n_variants)?;

        // Copy to GPU
        self.stream.memcpy_htod(gamma, &mut d_gamma)?;
        self.stream.memcpy_htod(&current_freq, &mut d_current_freq)?;

        // Launch kernel
        let block_size = 256u32;
        let grid_size = ((n_variants + 255) / 256) as u32;
        let launch_config = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        let n_variants_i32 = n_variants as i32;
        unsafe {
            let mut builder = self.stream.launch_builder(func);
            builder.arg(&*d_gamma);
            builder.arg(&*d_current_freq);
            builder.arg(&time_horizon_days);
            builder.arg(&n_variants_i32);
            builder.arg(&mut d_predicted_freq);
            builder.launch(launch_config)?;
        }

        // Synchronize and copy back
        self.stream.synchronize()?;
        let predicted_freq = self.stream.clone_dtoh(&d_predicted_freq)?;

        Ok(predicted_freq)
    }

    // Private helper methods for each stage
    fn compute_dms_escape(&mut self, variants: &[VariantData]) -> Result<Vec<f32>, PrismError> {
        // TODO: Implement DMS escape kernel launch
        Ok(vec![0.0; variants.len() * 10])
    }

    fn compute_cross_neutralization(
        &mut self,
        escape_scores: &[f32],
        epitope_weights: &[f32; 10],
    ) -> Result<Vec<f32>, PrismError> {
        // TODO: Implement cross-neutralization kernel launch
        Ok(vec![1.0; escape_scores.len() / 10])
    }

    fn compute_variant_fitness(
        &mut self,
        fold_reductions: &[f32],
        transmissibility: &[f32],
    ) -> Result<Vec<f32>, PrismError> {
        // TODO: Implement fitness kernel launch
        Ok(fold_reductions.iter().zip(transmissibility)
            .map(|(fr, t)| -fr.ln() + t)
            .collect())
    }
}
```

---

## 4. Data Integration with VASIL Benchmark

### 4.1 DMS Data Loader

```rust
// crates/prism-ve/src/data/dms_loader.rs

use std::path::Path;
use csv::ReaderBuilder;
use prism_core::PrismError;

/// DMS escape data structure (836 antibodies × 201 RBD sites)
pub struct DmsEscapeData {
    pub escape_matrix: Vec<f32>,        // [836 × 201]
    pub antibody_names: Vec<String>,    // [836]
    pub antibody_epitopes: Vec<i32>,    // [836] → 10 classes
    pub site_positions: Vec<i32>,       // [201] RBD positions (331-531)
}

impl DmsEscapeData {
    /// Load from VASIL processed DMS data
    pub fn load_from_vasil(
        vasil_data_dir: &Path,
    ) -> Result<Self, PrismError> {
        let dms_file = vasil_data_dir
            .join("dms/vasil_processed/dms_per_ab_per_site.csv");

        let mut reader = ReaderBuilder::new()
            .has_headers(true)
            .from_path(&dms_file)
            .map_err(|e| PrismError::io("dms_loader", format!("Open DMS file: {}", e)))?;

        let mut escape_matrix = vec![0.0f32; 836 * 201];
        let mut antibody_names = Vec::new();
        let mut antibody_epitopes = Vec::new();

        for result in reader.records() {
            let record = result
                .map_err(|e| PrismError::io("dms_loader", format!("Read record: {}", e)))?;

            let antibody = record.get(0)
                .ok_or_else(|| PrismError::data("dms_loader", "Missing antibody"))?;
            let epitope_class: i32 = record.get(1)
                .and_then(|s| s.parse().ok())
                .ok_or_else(|| PrismError::data("dms_loader", "Invalid epitope class"))?;
            let site: i32 = record.get(2)
                .and_then(|s| s.parse().ok())
                .ok_or_else(|| PrismError::data("dms_loader", "Invalid site"))?;
            let escape_score: f32 = record.get(3)
                .and_then(|s| s.parse().ok())
                .ok_or_else(|| PrismError::data("dms_loader", "Invalid escape score"))?;

            // Populate matrix
            // TODO: Map antibody name to index, site to 0-200 range
        }

        Ok(Self {
            escape_matrix,
            antibody_names,
            antibody_epitopes,
            site_positions: (331..=531).collect(),
        })
    }
}
```

### 4.2 VASIL Frequency Data Integration

```rust
// crates/prism-ve/src/data/vasil_frequencies.rs

use std::path::Path;
use chrono::NaiveDate;
use csv::ReaderBuilder;
use prism_core::PrismError;

/// Lineage frequency time series from VASIL
pub struct VasilFrequencies {
    pub country: String,
    pub lineages: Vec<String>,
    pub dates: Vec<NaiveDate>,
    pub frequencies: Vec<Vec<f32>>,     // [n_dates × n_lineages]
}

impl VasilFrequencies {
    /// Load from VASIL pre-computed frequencies
    pub fn load_from_vasil(
        vasil_data_dir: &Path,
        country: &str,
    ) -> Result<Self, PrismError> {
        let freq_file = vasil_data_dir
            .join(format!("vasil_code/ByCountry/{}/results/Daily_Lineages_Freq_1_percent.csv", country));

        let mut reader = ReaderBuilder::new()
            .has_headers(true)
            .from_path(&freq_file)
            .map_err(|e| PrismError::io("vasil_freq", format!("Open frequency file: {}", e)))?;

        // Parse CSV (wide format: date as rows, lineages as columns)
        let headers = reader.headers()
            .map_err(|e| PrismError::io("vasil_freq", format!("Read headers: {}", e)))?
            .clone();

        let lineages: Vec<String> = headers.iter()
            .skip(1)  // Skip "date" column
            .map(|s| s.to_string())
            .collect();

        let mut dates = Vec::new();
        let mut frequencies = Vec::new();

        for result in reader.records() {
            let record = result
                .map_err(|e| PrismError::io("vasil_freq", format!("Read record: {}", e)))?;

            let date_str = record.get(0)
                .ok_or_else(|| PrismError::data("vasil_freq", "Missing date"))?;
            let date = NaiveDate::parse_from_str(date_str, "%Y-%m-%d")
                .map_err(|e| PrismError::data("vasil_freq", format!("Invalid date: {}", e)))?;

            let freq_row: Vec<f32> = record.iter()
                .skip(1)
                .map(|s| s.parse::<f32>().unwrap_or(0.0))
                .collect();

            dates.push(date);
            frequencies.push(freq_row);
        }

        Ok(Self {
            country: country.to_string(),
            lineages,
            dates,
            frequencies,
        })
    }

    /// Get frequency for a specific lineage at a date
    pub fn get_frequency(&self, lineage: &str, date: &NaiveDate) -> Option<f32> {
        let lineage_idx = self.lineages.iter().position(|l| l == lineage)?;
        let date_idx = self.dates.iter().position(|d| d == date)?;
        Some(self.frequencies[date_idx][lineage_idx])
    }
}
```

---

## 5. Integration into Pipeline

### 5.1 Updated Pipeline Structure

```rust
// crates/prism-pipeline/src/ve_pipeline.rs

use prism_gpu::ViralEvolutionFitnessGpu;
use prism_core::PrismError;

/// PRISM-VE pipeline with viral evolution module
pub struct PRISMVEPipeline {
    // Existing phases
    // ...

    // NEW: Viral evolution fitness
    fitness_gpu: Option<ViralEvolutionFitnessGpu>,
}

impl PRISMVEPipeline {
    pub fn predict_variant_dynamics(
        &mut self,
        variants: &[VariantData],
        date: &str,
        country: &str,
    ) -> Result<Vec<PredictionResult>, PrismError> {
        // Load population immunity landscape for this country/date
        let immunity_landscape = self.load_immunity_landscape(country, date)?;

        // Compute fitness using GPU
        let fitness_gpu = self.fitness_gpu.as_mut()
            .ok_or_else(|| PrismError::config("Fitness GPU not initialized"))?;

        let gamma_values = fitness_gpu.compute_fitness(variants, &self.fitness_params)?;

        // Predict dynamics over time horizon
        let predicted_freqs = fitness_gpu.predict_dynamics(
            variants,
            &gamma_values,
            7.0,  // 7-day horizon
        )?;

        // Convert to rise/fall predictions
        let predictions: Vec<PredictionResult> = variants.iter()
            .zip(gamma_values.iter())
            .zip(predicted_freqs.iter())
            .map(|((variant, gamma), pred_freq)| {
                PredictionResult {
                    lineage: variant.lineage_name.clone(),
                    gamma: *gamma,
                    direction: if *gamma > 0.0 { "RISE" } else { "FALL" }.to_string(),
                    current_frequency: variant.current_frequency,
                    predicted_frequency: *pred_freq,
                    confidence: self.compute_confidence(*gamma),
                }
            })
            .collect();

        Ok(predictions)
    }
}

pub struct PredictionResult {
    pub lineage: String,
    pub gamma: f32,                  // Growth rate (γ)
    pub direction: String,           // "RISE" or "FALL"
    pub current_frequency: f32,
    pub predicted_frequency: f32,
    pub confidence: f32,
}
```

---

## 6. VASIL Benchmark Integration

### 6.1 Benchmark Execution

```rust
// crates/prism-ve/src/benchmark/vasil_benchmark.rs

use crate::data::{DmsEscapeData, VasilFrequencies};
use crate::PRISMVEPipeline;
use prism_core::PrismError;
use std::path::Path;

pub struct VasilBenchmarkRunner {
    pipeline: PRISMVEPipeline,
    dms_data: DmsEscapeData,
    countries: Vec<String>,
}

impl VasilBenchmarkRunner {
    pub fn new(vasil_data_dir: &Path) -> Result<Self, PrismError> {
        // Load DMS data
        let dms_data = DmsEscapeData::load_from_vasil(vasil_data_dir)?;

        // Initialize pipeline with GPU
        let mut pipeline = PRISMVEPipeline::new()?;
        pipeline.fitness_gpu.as_mut().unwrap().load_dms_data(
            &dms_data.escape_matrix.try_into().unwrap(),
            &dms_data.antibody_epitopes.try_into().unwrap(),
        )?;

        Ok(Self {
            pipeline,
            dms_data,
            countries: vec![
                "Germany".to_string(),
                "USA".to_string(),
                "UK".to_string(),
                // ... all 12 countries
            ],
        })
    }

    /// Run benchmark against VASIL for a single country
    pub fn benchmark_country(&mut self, country: &str) -> Result<f32, PrismError> {
        // Load VASIL frequencies
        let frequencies = VasilFrequencies::load_from_vasil(
            Path::new("data/vasil_benchmark"),
            country,
        )?;

        let mut correct = 0;
        let mut total = 0;

        // For each date in dataset (weekly sampling)
        for (date_idx, date) in frequencies.dates.iter().step_by(7).enumerate() {
            // Skip if not enough future data
            if date_idx + 2 >= frequencies.dates.len() {
                continue;
            }

            // Get variants above 3% frequency
            let variants: Vec<VariantData> = frequencies.lineages.iter()
                .enumerate()
                .filter(|(i, _)| frequencies.frequencies[date_idx][*i] > 0.03)
                .map(|(i, lineage)| VariantData {
                    lineage_name: lineage.clone(),
                    spike_mutations: vec![],  // TODO: Load from VASIL
                    mutation_aa: vec![],
                    current_frequency: frequencies.frequencies[date_idx][i],
                    collection_date: date.format("%Y-%m-%d").to_string(),
                })
                .collect();

            if variants.is_empty() {
                continue;
            }

            // Predict using PRISM-VE
            let predictions = self.pipeline.predict_variant_dynamics(
                &variants,
                &date.format("%Y-%m-%d").to_string(),
                country,
            )?;

            // Evaluate predictions
            for (variant_idx, pred) in predictions.iter().enumerate() {
                // Observed direction (compare frequency at date vs date+7days)
                let future_freq = frequencies.frequencies[date_idx + 1][variant_idx];
                let current_freq = pred.current_frequency;

                let observed_direction = if future_freq > current_freq * 1.05 {
                    "RISE"
                } else if future_freq < current_freq * 0.95 {
                    "FALL"
                } else {
                    "STABLE"
                };

                if observed_direction != "STABLE" {
                    total += 1;
                    if pred.direction == observed_direction {
                        correct += 1;
                    }
                }
            }
        }

        let accuracy = if total > 0 {
            correct as f32 / total as f32
        } else {
            0.0
        };

        Ok(accuracy)
    }

    /// Run full benchmark across all countries
    pub fn run_full_benchmark(&mut self) -> Result<BenchmarkResults, PrismError> {
        let mut country_accuracies = Vec::new();

        for country in &self.countries.clone() {
            let accuracy = self.benchmark_country(country)?;
            country_accuracies.push((country.clone(), accuracy));
            println!("  {} accuracy: {:.3}", country, accuracy);
        }

        let mean_accuracy = country_accuracies.iter()
            .map(|(_, acc)| acc)
            .sum::<f32>() / country_accuracies.len() as f32;

        Ok(BenchmarkResults {
            mean_accuracy,
            country_accuracies,
            vasil_baseline: 0.92,
        })
    }
}

pub struct BenchmarkResults {
    pub mean_accuracy: f32,
    pub country_accuracies: Vec<(String, f32)>,
    pub vasil_baseline: f32,
}

impl BenchmarkResults {
    pub fn print_comparison(&self) {
        println!("\n{}", "=".repeat(80));
        println!("PRISM-VE vs VASIL Benchmark Results");
        println!("{}", "=".repeat(80));
        println!("\n{:<15} {:<10} {:<10} {:<10}", "Country", "PRISM-VE", "VASIL", "Delta");
        println!("{}", "-".repeat(50));

        for (country, accuracy) in &self.country_accuracies {
            let vasil_acc = 0.92;  // TODO: Load from VASIL baseline per country
            let delta = accuracy - vasil_acc;
            println!("{:<15} {:.3}      {:.3}      {:+.3}", country, accuracy, vasil_acc, delta);
        }

        println!("{}", "-".repeat(50));
        println!("{:<15} {:.3}      {:.3}      {:+.3}",
                 "MEAN", self.mean_accuracy, self.vasil_baseline,
                 self.mean_accuracy - self.vasil_baseline);

        if self.mean_accuracy > self.vasil_baseline {
            println!("\n🏆 PRISM-VE BEATS VASIL!");
        } else {
            println!("\n⚠️  PRISM-VE behind VASIL by {:.3}",
                     self.vasil_baseline - self.mean_accuracy);
        }
    }
}
```

---

## 7. Build Integration

### 7.1 Update build.rs

```rust
// crates/prism-gpu/build.rs

// Add to compile_kernels()

compile_kernel(
    &nvcc,
    "src/kernels/viral_evolution_fitness.cu",
    &ptx_dir.join("viral_evolution_fitness.ptx"),
    &target_ptx_dir.join("viral_evolution_fitness.ptx"),
);
```

### 7.2 Update Cargo.toml

```toml
# crates/prism-ve/Cargo.toml

[dependencies]
prism-gpu = { path = "../prism-gpu" }
prism-core = { path = "../prism-core" }
csv = "1.3"
chrono = "0.4"
serde = { version = "1.0", features = ["derive"] }
```

---

## 8. Implementation Roadmap

### Phase 1: Data Infrastructure (Week 1)
- [x] Download VASIL benchmark data
- [ ] Implement DMS data loader
- [ ] Implement VASIL frequency loader
- [ ] Create variant mutation database
- [ ] Test data loading pipeline

### Phase 2: GPU Kernel Development (Week 2-3)
- [ ] Implement `stage1_dms_escape_scores` kernel
- [ ] Implement `stage2_cross_neutralization` kernel
- [ ] Implement `stage4_variant_fitness` kernel
- [ ] Implement `stage5_predict_dynamics` kernel
- [ ] Test individual kernels with synthetic data
- [ ] Optimize kernel performance (shared memory, coalescing)

### Phase 3: Rust Integration (Week 3-4)
- [ ] Create `ViralEvolutionFitnessGpu` wrapper
- [ ] Implement buffer pooling for efficiency
- [ ] Integrate with existing PRISM pipeline
- [ ] Create CLI interface for predictions
- [ ] Add telemetry and provenance tracking

### Phase 4: VASIL Benchmark (Week 4-5)
- [ ] Implement `VasilBenchmarkRunner`
- [ ] Run initial benchmarks (expect ~0.5-0.6 accuracy initially)
- [ ] Calibrate parameters using gradient descent
- [ ] Integrate population immunity module (cycle module)
- [ ] Re-benchmark after calibration (target >0.92)

### Phase 5: Optimization & Publication (Week 5-6)
- [ ] Profile GPU kernels, optimize bottlenecks
- [ ] Multi-GPU support for batch country processing
- [ ] FluxNet RL integration for adaptive learning
- [ ] Write Nature Methods manuscript
- [ ] Prepare code release and documentation

---

## 9. Success Criteria

### Minimum Viable Product (MVP)
- ✅ Load VASIL DMS data into GPU memory
- ✅ Compute DMS escape scores for variants
- ✅ Predict variant rise/fall (γ > 0 or γ < 0)
- ✅ Achieve >0.80 accuracy on Germany dataset

### Target Performance
- 🎯 **>0.92 mean accuracy** across 12 countries (match VASIL)
- 🎯 **<100ms per prediction** (GPU acceleration)
- 🎯 Correct inflection point predictions (BA.2, BQ.1.1, XBB.1.5, etc.)
- 🎯 Geographic specificity (BA.2.12.1 in USA vs Germany)

### Stretch Goals
- 🚀 **>0.95 accuracy** (beat VASIL)
- 🚀 Prospective prediction (BA.2.86 emergence)
- 🚀 Real-time variant tracking dashboard
- 🚀 Multi-pathogen generalization (influenza, RSV)

---

## 10. Technical Challenges & Mitigations

| Challenge | Risk | Mitigation Strategy |
|-----------|------|---------------------|
| DMS data quality | Medium | Use VASIL's processed data, validate against EVEscape |
| Population immunity tracking | High | Start with simplified model, integrate GInPipe later |
| GPU memory limits | Medium | Use constant memory for DMS (680KB), buffer pooling |
| Variant mutation mapping | High | Create comprehensive mutation → site index database |
| Calibration complexity | High | Use VASIL parameters as starting point, grid search |
| Real-time data | Low | Use GISAID metadata.tsv, update weekly |

---

## 11. Next Steps

**Immediate Actions:**
1. Create `crates/prism-gpu/src/kernels/viral_evolution_fitness.cu`
2. Implement DMS data loader from VASIL
3. Test DMS escape kernel with BA.2 vs BA.5
4. Validate against VASIL's published escape scores

**This Week:**
- Complete GPU kernel implementation
- Create Rust wrapper with buffer pooling
- Run first benchmark on Germany dataset

**This Month:**
- Achieve >0.85 accuracy on Germany
- Extend to all 12 countries
- Calibrate parameters to match VASIL baseline

---

## Conclusion

This implementation plan provides a **GPU-accelerated, VASIL-compatible fitness module** that integrates seamlessly with PRISM-VE's existing mega_fused.rs architecture.

**Key Design Principles:**
- ✅ Multi-pass kernel architecture (modular, maintainable)
- ✅ Buffer pooling for zero-allocation hot path
- ✅ Runtime-configurable parameters (no PTX recompilation)
- ✅ Direct integration with VASIL benchmark data
- ✅ GPU-centric computation (CPU only for data loading)

**Expected Outcome:**
A production-ready viral evolution prediction system that matches or exceeds VASIL's 0.92 accuracy while running 10-100× faster on GPU.

Ready to proceed with implementation? 🚀
