# Buffer Pool Design for GPU Memory Management

## Overview

A buffer pool pre-allocates GPU memory to avoid repeated allocation/deallocation overhead during batch processing. This is critical for PRISM-4D's target of processing 14,917 structures in under 60 seconds.

## Design Goals

1. **Zero allocation during processing** - All memory allocated upfront
2. **Safe Rust lifetimes** - No borrow checker conflicts
3. **Capacity tracking** - Prevent buffer overflows
4. **Reuse statistics** - Monitor memory efficiency

## Production Buffer Pool Implementation

```rust
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, CudaView, CudaViewMut};
use std::sync::Arc;
use anyhow::{Result, anyhow};

/// Configuration for buffer pool sizing
#[derive(Clone, Debug)]
pub struct BufferPoolConfig {
    /// Maximum atoms across all structures in a batch
    pub max_total_atoms: usize,
    /// Maximum residues across all structures
    pub max_total_residues: usize,
    /// Maximum structures per batch
    pub max_structures: usize,
    /// Features per residue (101 for PRISM-4D)
    pub features_per_residue: usize,
}

impl Default for BufferPoolConfig {
    fn default() -> Self {
        Self {
            max_total_atoms: 500_000,      // ~33 atoms/residue * 15K residues
            max_total_residues: 50_000,     // ~200 residues * 250 structures
            max_structures: 500,
            features_per_residue: 101,
        }
    }
}

/// Pre-allocated GPU buffers for batch processing
pub struct GpuBufferPool {
    ctx: Arc<CudaContext>,
    stream: CudaStream,
    config: BufferPoolConfig,
    
    // Input buffers
    atoms_buffer: CudaSlice<f32>,         // [x,y,z] packed
    ca_indices_buffer: CudaSlice<u32>,    // CA atom indices
    descriptors_buffer: CudaSlice<BatchStructureDesc>,
    residue_types_buffer: CudaSlice<u8>,
    conservation_buffer: CudaSlice<f32>,
    bfactor_buffer: CudaSlice<f32>,
    
    // GISAID data buffers
    gisaid_freq_buffer: CudaSlice<f32>,
    gisaid_velocity_buffer: CudaSlice<f32>,
    
    // Output buffer
    features_buffer: CudaSlice<f32>,      // 101-dim per residue
    
    // Current usage tracking
    current_atoms: usize,
    current_residues: usize,
    current_structures: usize,
    
    // Statistics
    pub stats: BufferPoolStats,
}

#[derive(Default, Debug, Clone)]
pub struct BufferPoolStats {
    pub total_batches: u64,
    pub total_structures: u64,
    pub peak_atoms: usize,
    pub peak_residues: usize,
    pub peak_structures: usize,
    pub reallocs_avoided: u64,
}

impl GpuBufferPool {
    /// Create new buffer pool with specified capacity
    pub fn new(ctx: Arc<CudaContext>, config: BufferPoolConfig) -> Result<Self> {
        let stream = ctx.default_stream();
        
        // Pre-allocate all buffers
        let atoms_buffer = stream.alloc_zeros(config.max_total_atoms * 3)?;
        let ca_indices_buffer = stream.alloc_zeros(config.max_total_residues)?;
        let descriptors_buffer = stream.alloc_zeros(config.max_structures)?;
        let residue_types_buffer = stream.alloc_zeros(config.max_total_residues)?;
        let conservation_buffer = stream.alloc_zeros(config.max_total_residues)?;
        let bfactor_buffer = stream.alloc_zeros(config.max_total_residues)?;
        let gisaid_freq_buffer = stream.alloc_zeros(config.max_total_residues)?;
        let gisaid_velocity_buffer = stream.alloc_zeros(config.max_total_residues)?;
        
        let features_buffer = stream.alloc_zeros(
            config.max_total_residues * config.features_per_residue
        )?;
        
        Ok(Self {
            ctx,
            stream,
            config,
            atoms_buffer,
            ca_indices_buffer,
            descriptors_buffer,
            residue_types_buffer,
            conservation_buffer,
            bfactor_buffer,
            gisaid_freq_buffer,
            gisaid_velocity_buffer,
            features_buffer,
            current_atoms: 0,
            current_residues: 0,
            current_structures: 0,
            stats: BufferPoolStats::default(),
        })
    }
    
    /// Check if batch fits in current capacity
    pub fn can_fit(&self, n_atoms: usize, n_residues: usize, n_structures: usize) -> bool {
        n_atoms <= self.config.max_total_atoms &&
        n_residues <= self.config.max_total_residues &&
        n_structures <= self.config.max_structures
    }
    
    /// Upload batch data to GPU, returns views for kernel launch
    pub fn upload_batch<'a>(
        &'a mut self,
        atoms: &[f32],
        ca_indices: &[u32],
        descriptors: &[BatchStructureDesc],
        residue_types: &[u8],
        conservation: &[f32],
        bfactor: &[f32],
        gisaid_freq: &[f32],
        gisaid_velocity: &[f32],
    ) -> Result<BatchViews<'a>> {
        let n_atoms = atoms.len() / 3;
        let n_residues = ca_indices.len();
        let n_structures = descriptors.len();
        
        // Validate capacity
        if !self.can_fit(n_atoms, n_residues, n_structures) {
            return Err(anyhow!(
                "Batch exceeds capacity: atoms={}/{}, residues={}/{}, structures={}/{}",
                n_atoms, self.config.max_total_atoms,
                n_residues, self.config.max_total_residues,
                n_structures, self.config.max_structures
            ));
        }
        
        // Upload to pre-allocated buffers
        self.stream.htod_copy_into(atoms, &mut self.atoms_buffer.try_slice_mut(..atoms.len())?)?;
        self.stream.htod_copy_into(ca_indices, &mut self.ca_indices_buffer.try_slice_mut(..n_residues)?)?;
        self.stream.htod_copy_into(descriptors, &mut self.descriptors_buffer.try_slice_mut(..n_structures)?)?;
        self.stream.htod_copy_into(residue_types, &mut self.residue_types_buffer.try_slice_mut(..n_residues)?)?;
        self.stream.htod_copy_into(conservation, &mut self.conservation_buffer.try_slice_mut(..n_residues)?)?;
        self.stream.htod_copy_into(bfactor, &mut self.bfactor_buffer.try_slice_mut(..n_residues)?)?;
        self.stream.htod_copy_into(gisaid_freq, &mut self.gisaid_freq_buffer.try_slice_mut(..n_residues)?)?;
        self.stream.htod_copy_into(gisaid_velocity, &mut self.gisaid_velocity_buffer.try_slice_mut(..n_residues)?)?;
        
        // Update tracking
        self.current_atoms = n_atoms;
        self.current_residues = n_residues;
        self.current_structures = n_structures;
        
        // Update stats
        self.stats.total_batches += 1;
        self.stats.total_structures += n_structures as u64;
        self.stats.peak_atoms = self.stats.peak_atoms.max(n_atoms);
        self.stats.peak_residues = self.stats.peak_residues.max(n_residues);
        self.stats.peak_structures = self.stats.peak_structures.max(n_structures);
        self.stats.reallocs_avoided += 1;
        
        // Return views for kernel launch
        Ok(BatchViews {
            atoms: self.atoms_buffer.try_slice(..atoms.len())?,
            ca_indices: self.ca_indices_buffer.try_slice(..n_residues)?,
            descriptors: self.descriptors_buffer.try_slice(..n_structures)?,
            residue_types: self.residue_types_buffer.try_slice(..n_residues)?,
            conservation: self.conservation_buffer.try_slice(..n_residues)?,
            bfactor: self.bfactor_buffer.try_slice(..n_residues)?,
            gisaid_freq: self.gisaid_freq_buffer.try_slice(..n_residues)?,
            gisaid_velocity: self.gisaid_velocity_buffer.try_slice(..n_residues)?,
            n_atoms,
            n_residues,
            n_structures,
        })
    }
    
    /// Get mutable view to output buffer for kernel
    pub fn output_view(&mut self) -> Result<CudaViewMut<f32>> {
        let n = self.current_residues * self.config.features_per_residue;
        Ok(self.features_buffer.try_slice_mut(..n)?)
    }
    
    /// Download output features from GPU
    pub fn download_features(&self) -> Result<Vec<f32>> {
        let n = self.current_residues * self.config.features_per_residue;
        let view = self.features_buffer.try_slice(..n)?;
        Ok(self.stream.memcpy_dtov(&view)?)
    }
    
    /// Get reference to stream for kernel launches
    pub fn stream(&self) -> &CudaStream {
        &self.stream
    }
    
    /// Synchronize stream
    pub fn sync(&self) -> Result<()> {
        self.stream.synchronize()?;
        Ok(())
    }
}

/// Immutable views to uploaded batch data
pub struct BatchViews<'a> {
    pub atoms: CudaView<'a, f32>,
    pub ca_indices: CudaView<'a, u32>,
    pub descriptors: CudaView<'a, BatchStructureDesc>,
    pub residue_types: CudaView<'a, u8>,
    pub conservation: CudaView<'a, f32>,
    pub bfactor: CudaView<'a, f32>,
    pub gisaid_freq: CudaView<'a, f32>,
    pub gisaid_velocity: CudaView<'a, f32>,
    pub n_atoms: usize,
    pub n_residues: usize,
    pub n_structures: usize,
}
```

## Usage Pattern

```rust
// Initialize once at startup
let ctx = Arc::new(CudaContext::new(0)?);
let config = BufferPoolConfig::default();
let mut pool = GpuBufferPool::new(ctx.clone(), config)?;

// Load kernel once
let ptx = std::fs::read_to_string("mega_fused_batch.ptx")?;
let module = ctx.load_module(ptx)?;
let kernel = module.load_function("mega_fused_batch_detection_prism4d")?;

// Process multiple batches without reallocation
for batch_data in batches {
    // Upload returns views (borrows from pool)
    let views = pool.upload_batch(
        &batch_data.atoms,
        &batch_data.ca_indices,
        &batch_data.descriptors,
        &batch_data.residue_types,
        &batch_data.conservation,
        &batch_data.bfactor,
        &batch_data.gisaid_freq,
        &batch_data.gisaid_velocity,
    )?;
    
    // Get output view (separate from input views)
    let mut output = pool.output_view()?;
    
    // Launch kernel
    let cfg = LaunchConfig {
        grid_dim: (views.n_structures as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 48 * 1024,
    };
    
    unsafe {
        pool.stream().launch_builder(&kernel)
            .arg(&mut output)
            .arg(&views.atoms)
            .arg(&views.ca_indices)
            .arg(&views.descriptors)
            .arg(&views.residue_types)
            .arg(&views.conservation)
            .arg(&views.bfactor)
            .arg(&views.gisaid_freq)
            .arg(&views.gisaid_velocity)
            .arg(&(views.n_structures as u32))
            .launch(cfg)?;
    }
    
    // Sync and download
    pool.sync()?;
    let features = pool.download_features()?;
    
    // Process features...
}

// Check efficiency
println!("Buffer pool stats: {:?}", pool.stats);
```

## Key Design Decisions

### Why Pre-allocate?
- `cudaMalloc` is expensive (~1ms per call)
- 14,917 structures / 60 seconds = 249 structures/second required
- Allocating each batch would add ~50ms overhead per batch

### Why Separate Input/Output Buffers?
- Allows kernel to read and write simultaneously
- Avoids WAR (write-after-read) hazards
- Enables proper Rust borrow semantics

### Why Views Instead of Cloning?
- Views are zero-cost (just pointer + length)
- Cloning CudaSlice would duplicate GPU memory
- Rust borrow checker ensures safety

### Capacity Sizing
```rust
// For PRISM-4D VASIL benchmark:
// - 14,917 structures total
// - ~200 residues per structure average
// - ~33 atoms per residue (with hydrogens)
// - Process in batches of 200-500 structures

let config = BufferPoolConfig {
    max_total_atoms: 3_000_000,    // 500 structures * 200 res * 30 atoms
    max_total_residues: 100_000,   // 500 structures * 200 residues
    max_structures: 500,
    features_per_residue: 101,
};
```

## Thread Safety

The buffer pool is NOT thread-safe by design. For multi-threaded processing:

```rust
// Option 1: One pool per thread
thread_local! {
    static POOL: RefCell<Option<GpuBufferPool>> = RefCell::new(None);
}

// Option 2: Pool behind mutex (serializes GPU access)
lazy_static! {
    static ref POOL: Mutex<GpuBufferPool> = Mutex::new(
        GpuBufferPool::new(Arc::new(CudaContext::new(0).unwrap()), 
                          BufferPoolConfig::default()).unwrap()
    );
}

// Option 3: Multiple pools with round-robin (best for multi-GPU)
struct PoolManager {
    pools: Vec<Mutex<GpuBufferPool>>,
    next: AtomicUsize,
}
```

## Memory Overhead

```
Buffer                    Size Formula                        Example (default)
------------------------ ----------------------------------- -----------------
atoms_buffer             max_atoms * 3 * 4 bytes            6 MB
ca_indices_buffer        max_residues * 4 bytes             200 KB
descriptors_buffer       max_structures * 24 bytes          12 KB
residue_types_buffer     max_residues * 1 byte              50 KB
conservation_buffer      max_residues * 4 bytes             200 KB
bfactor_buffer           max_residues * 4 bytes             200 KB
gisaid_freq_buffer       max_residues * 4 bytes             200 KB
gisaid_velocity_buffer   max_residues * 4 bytes             200 KB
features_buffer          max_residues * 101 * 4 bytes       20 MB
------------------------ ----------------------------------- -----------------
TOTAL                                                       ~27 MB

RTX 3060 has 12 GB VRAM - buffer pool uses ~0.2% of available memory
```
