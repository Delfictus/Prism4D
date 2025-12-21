# Rust API Reference for PRISM>4D

## Core Types (prism-gpu crate)

### MegaFusedBatchGpu
Main GPU executor for batch processing.

```rust
pub struct MegaFusedBatchGpu {
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    batch_func: Option<CudaFunction>,        // mega_fused_batch_detection
    batch_metrics_func: Option<CudaFunction>,
    training_func: Option<CudaFunction>,
    buffer_pool: BatchBufferPool,
    telemetry: GpuTelemetry,
}

impl MegaFusedBatchGpu {
    /// Load PTX and initialize
    pub fn new(context: Arc<CudaContext>, ptx_dir: &Path) -> Result<Self, PrismError>;
    
    /// Process batch of structures
    pub fn detect_pockets_batch(
        &mut self,
        batch: &PackedBatch,
        config: &MegaFusedConfig,
    ) -> Result<BatchOutput, PrismError>;
}
```

### BatchStructureDesc
**MUST match CUDA struct exactly** (16-byte aligned)

```rust
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Default)]
pub struct BatchStructureDesc {
    pub atom_offset: i32,     // Index into atoms_packed
    pub residue_offset: i32,  // Index into residue arrays
    pub n_atoms: i32,
    pub n_residues: i32,
}
```

### StructureInput
Input data for a single structure before packing.

```rust
#[derive(Debug, Clone)]
pub struct StructureInput {
    pub id: String,
    pub atoms: Vec<f32>,        // [x0,y0,z0, x1,y1,z1, ...]
    pub ca_indices: Vec<i32>,   // CA atom index per residue
    pub conservation: Vec<f32>, // [0,1] per residue
    pub bfactor: Vec<f32>,      // Normalized B-factor
    pub burial: Vec<f32>,       // [0,1] burial fraction
}

impl StructureInput {
    pub fn new(id: impl Into<String>) -> Self;
    pub fn n_atoms(&self) -> usize { self.atoms.len() / 3 }
    pub fn n_residues(&self) -> usize { self.ca_indices.len() }
    pub fn validate(&self) -> Result<(), String>;
}
```

### PackedBatch
GPU-ready packed data for all structures.

```rust
#[derive(Debug)]
pub struct PackedBatch {
    pub structure_ids: Vec<String>,
    pub atoms_packed: Vec<f32>,           // All atoms concatenated
    pub ca_indices_packed: Vec<i32>,      // All CA indices
    pub conservation_packed: Vec<f32>,
    pub bfactor_packed: Vec<f32>,
    pub burial_packed: Vec<f32>,
    pub descriptors: Vec<BatchStructureDesc>,
    pub total_atoms: usize,
    pub total_residues: usize,
}

impl PackedBatch {
    pub fn from_structures(structures: &[StructureInput]) -> Result<Self, String>;
    pub fn n_structures(&self) -> usize { self.descriptors.len() }
}
```

### BatchOutput
Results from batch kernel execution.

```rust
#[derive(Debug, Clone)]
pub struct BatchOutput {
    pub structures: Vec<BatchStructureOutput>,
    pub kernel_time_ms: f64,
    pub total_time_ms: f64,
}

#[derive(Debug, Clone)]
pub struct BatchStructureOutput {
    pub id: String,
    pub consensus_scores: Vec<f32>,
    pub confidence: Vec<i32>,
    pub signal_mask: Vec<i32>,
    pub pocket_assignment: Vec<i32>,
    pub centrality: Vec<f32>,
    pub combined_features: Vec<f32>,  // 101 dims per residue
}
```

### MegaFusedConfig
Runtime configuration for kernel.

```rust
#[derive(Debug, Clone)]
pub struct MegaFusedConfig {
    pub use_fp16: bool,
    pub contact_sigma: f32,         // Default: 6.0 Å
    pub consensus_threshold: f32,   // Default: 0.35
    pub mode: MegaFusedMode,
    pub kempe_iterations: i32,
    pub power_iterations: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MegaFusedMode {
    UltraPrecise,  // kempe=15, power=25
    #[default]
    Balanced,      // kempe=10, power=15
    Screening,     // kempe=3, power=5
}
```

## Benchmark Layer (prism-ve-bench crate)

### VEState
RL state representation.

```rust
#[derive(Debug, Clone)]
pub struct VEState {
    pub escape: f32,           // DMS escape score [0,1]
    pub frequency: f32,        // GISAID frequency [0,1]
    pub gamma: f32,            // Fitness score
    pub growth_potential: f32, // gamma × (1-freq)²
    pub escape_dominance: f32, // Relative escape [-1,1]
}

impl VEState {
    pub fn new(escape: f32, frequency: f32, gamma: f32) -> Self;
    
    /// Map to Q-table index (256 states)
    pub fn discretize(&self) -> usize;
    
    /// Coarser discretization (16 states)
    pub fn discretize_coarse(&self) -> usize;
}
```

### VEAction
RL action space.

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VEAction {
    Rise = 0,
    Fall = 1,
}

impl VEAction {
    pub fn from_index(idx: usize) -> Self;
    pub fn to_str(&self) -> &'static str;  // "RISE" or "FALL"
}
```

### AdaptiveVEOptimizer
Q-learning optimizer.

```rust
pub struct AdaptiveVEOptimizer {
    q_table: Vec<[f32; 2]>,        // [state][action] -> Q-value
    visit_counts: Vec<[usize; 2]>,
    replay_buffer: Vec<VEExperience>,
    alpha: f32,                    // Learning rate (0.15)
    gamma: f32,                    // Discount (0.0 for single-step)
    epsilon: f32,                  // Exploration (0.2 → 0.02)
    num_states: usize,             // 256
}

impl AdaptiveVEOptimizer {
    pub fn new() -> Self;
    
    /// Select action (epsilon-greedy)
    pub fn select_action(&self, state: &VEState, explore: bool) -> VEAction;
    
    /// Greedy prediction (no exploration)
    pub fn predict(&self, state: &VEState) -> VEAction;
    
    /// Train on single experience
    pub fn train_step(&mut self, exp: &VEExperience);
    
    /// Train on dataset for N epochs
    pub fn train_on_dataset(&mut self, data: &[(VEState, &str)], epochs: usize);
    
    /// Evaluate accuracy
    pub fn evaluate(&self, data: &[(VEState, &str)]) -> f32;
    
    /// Get Q-values for debugging
    pub fn get_q_values(&self, state: &VEState) -> [f32; 2];
}
```

### Data Loaders

```rust
// GISAID frequency data
pub struct GisaidFrequencies {
    pub dates: Vec<NaiveDate>,
    pub lineages: Vec<String>,
    pub frequencies: Vec<Vec<f32>>,  // [date][lineage]
}

impl GisaidFrequencies {
    pub fn load_from_vasil(data_dir: &Path, country: &str) -> Result<Self>;
}

// DMS escape matrix
pub struct DmsEscapeData {
    pub antibodies: Vec<String>,
    pub sites: Vec<i32>,           // 331-531 (RBD)
    pub escape_scores: Vec<Vec<f32>>,  // [antibody][site]
}

impl DmsEscapeData {
    pub fn load_from_vasil(data_dir: &Path, country: &str) -> Result<Self>;
}

// Lineage mutations
pub struct LineageMutations {
    pub lineage_to_mutations: HashMap<String, Vec<String>>,
}

impl LineageMutations {
    pub fn load_from_vasil(data_dir: &Path, country: &str) -> Result<Self>;
    pub fn get_mutations(&self, lineage: &str) -> Option<&Vec<String>>;
}
```

## GPU Feature Extraction

### FeatureExtractor
Wraps MegaFusedGpu for VASIL benchmark.

```rust
pub struct FeatureExtractor {
    gpu: MegaFusedGpu,
    config: MegaFusedConfig,
}

impl FeatureExtractor {
    pub fn new() -> Result<Self>;
    
    /// Extract 101-dim features for a variant structure
    pub fn extract_features(
        &mut self,
        structure: &VariantStructure,
        gisaid_freq: f32,
        gisaid_vel: f32,
    ) -> Result<VariantFeatures>;
}

pub struct VariantFeatures {
    pub gamma: f32,           // Feature 95 average
    pub emergence_prob: f32,  // Feature 97 average
    pub phase: i32,           // Feature 96 mode
    pub all_features_101: Vec<f32>,
}
```

## Buffer Pool Pattern

### Efficient Memory Reuse
```rust
struct BatchBufferPool {
    // Atom buffers
    atoms_capacity: usize,
    d_atoms: Option<CudaSlice<f32>>,
    
    // Residue buffers
    residue_capacity: usize,
    d_ca_indices: Option<CudaSlice<i32>>,
    d_conservation: Option<CudaSlice<f32>>,
    d_bfactor: Option<CudaSlice<f32>>,
    d_burial: Option<CudaSlice<f32>>,
    
    // Output buffers
    d_consensus_scores: Option<CudaSlice<f32>>,
    d_combined_features: Option<CudaSlice<f32>>,  // NEW
    
    // Statistics
    allocations: usize,
    reuses: usize,
}

impl BatchBufferPool {
    fn ensure_capacity(&mut self, 
                       stream: &CudaStream,
                       total_atoms: usize, 
                       total_residues: usize,
                       n_structures: usize) -> Result<()> {
        // Only reallocate if capacity exceeded
        if total_atoms * 3 > self.atoms_capacity {
            let new_cap = total_atoms * 3 * 6 / 5;  // 20% headroom
            self.d_atoms = Some(stream.alloc_zeros::<f32>(new_cap)?);
            self.atoms_capacity = new_cap;
            self.allocations += 1;
        } else {
            self.reuses += 1;
        }
        // ... similar for other buffers
    }
}
```

## Error Handling

```rust
// Custom error type
#[derive(Debug)]
pub enum PrismError {
    Gpu { context: String, message: String },
    Io(std::io::Error),
    Parse(String),
}

impl PrismError {
    pub fn gpu(context: &str, message: impl Into<String>) -> Self {
        PrismError::Gpu {
            context: context.to_string(),
            message: message.into(),
        }
    }
}

// Usage pattern
let func = module.load_function("mega_fused_batch_detection")
    .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Load kernel: {}", e)))?;
```

## Kernel Launch Pattern

```rust
// In detect_pockets_batch()
pub fn detect_pockets_batch(
    &mut self,
    batch: &PackedBatch,
    config: &MegaFusedConfig,
) -> Result<BatchOutput, PrismError> {
    let n_structures = batch.n_structures();
    
    // 1. Ensure buffer capacity
    self.ensure_buffers(batch.total_atoms, batch.total_residues, n_structures)?;
    
    // 2. Copy data to GPU
    let d_atoms = self.buffer_pool.d_atoms.as_mut().unwrap();
    self.stream.memcpy_htod(&batch.atoms_packed, d_atoms)?;
    // ... copy other buffers
    
    // 3. Setup launch config
    let launch_config = LaunchConfig {
        grid_dim: (n_structures as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    
    // 4. Launch kernel
    let func = self.batch_func.as_ref().unwrap();
    unsafe {
        func.launch(launch_config, (
            d_atoms.as_ref(),
            d_ca_indices.as_ref(),
            // ... other args
        ))?;
    }
    
    // 5. Synchronize and copy results
    self.stream.synchronize()?;
    
    let mut h_consensus = vec![0.0f32; batch.total_residues];
    self.stream.memcpy_dtoh(d_consensus_scores.as_ref(), &mut h_consensus)?;
    
    // 6. Unpack per-structure results
    let mut structures = Vec::with_capacity(n_structures);
    for (i, desc) in batch.descriptors.iter().enumerate() {
        let start = desc.residue_offset as usize;
        let end = start + desc.n_residues as usize;
        
        structures.push(BatchStructureOutput {
            id: batch.structure_ids[i].clone(),
            consensus_scores: h_consensus[start..end].to_vec(),
            // ... other fields
        });
    }
    
    Ok(BatchOutput { structures, kernel_time_ms, total_time_ms })
}
```
