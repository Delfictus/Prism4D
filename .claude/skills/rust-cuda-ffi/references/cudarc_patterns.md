# cudarc Advanced Patterns Reference

## Version: cudarc 0.15.x (December 2024)

## Type System Deep Dive

### CudaSlice Internals
```rust
// CudaSlice owns:
// 1. Arc<CudaContext> - keeps device context alive
// 2. Raw device pointer (CUdeviceptr)
// 3. Length in elements

pub struct CudaSlice<T> {
    ctx: Arc<CudaContext>,
    ptr: CUdeviceptr,
    len: usize,
    _marker: PhantomData<T>,
}

// Key trait implementations:
// - Send + Sync (safe to share across threads)
// - Drop (calls cuMemFree automatically)
// - DevicePtr<T> (for kernel args)
```

### View Lifetime Rules
```rust
// Views are borrowed references - cannot outlive source
fn view_lifetimes() {
    let slice: CudaSlice<f32> = stream.alloc_zeros(100)?;
    
    // View borrows from slice
    let view: CudaView<f32> = slice.as_view();
    
    // This would fail - moving slice while view exists
    // drop(slice);  // ERROR: slice borrowed
    
    // Correct: drop view first
    drop(view);
    drop(slice);  // OK
}
```

## Memory Transfer Patterns

### Host to Device
```rust
// Copy with allocation
let host_data: Vec<f32> = vec![1.0, 2.0, 3.0];
let d_data: CudaSlice<f32> = stream.htod_copy(&host_data)?;

// Copy into existing buffer (no allocation)
let mut d_buffer: CudaSlice<f32> = stream.alloc_zeros(1000)?;
stream.htod_copy_into(&host_data, &mut d_buffer.try_slice_mut(..3)?)?;

// Async copy (requires pinned memory)
let pinned: PinnedBuffer<f32> = stream.pin_memory(&host_data)?;
stream.htod_copy_async(&pinned, &mut d_buffer)?;
stream.synchronize()?;
```

### Device to Host
```rust
// Copy to new Vec
let result: Vec<f32> = stream.memcpy_dtov(&d_data)?;

// Copy into existing slice
let mut host_buffer = vec![0.0f32; 100];
stream.memcpy_dtoh(&d_data, &mut host_buffer)?;
```

### Device to Device
```rust
// Copy between slices
let src: CudaSlice<f32> = stream.alloc_zeros(100)?;
let mut dst: CudaSlice<f32> = stream.alloc_zeros(100)?;
stream.memcpy_dtod(&src, &mut dst)?;
```

## Multi-Stream Patterns

### Independent Streams for Overlap
```rust
let ctx = CudaContext::new(0)?;

// Create multiple streams
let stream1 = ctx.fork_default_stream()?;
let stream2 = ctx.fork_default_stream()?;

// Parallel operations
let mut buf1 = stream1.alloc_zeros::<f32>(1000)?;
let mut buf2 = stream2.alloc_zeros::<f32>(1000)?;

// Launch on different streams (truly parallel on GPU)
unsafe {
    stream1.launch_builder(&kernel).arg(&mut buf1).launch(cfg)?;
    stream2.launch_builder(&kernel).arg(&mut buf2).launch(cfg)?;
}

// Sync both
stream1.synchronize()?;
stream2.synchronize()?;
```

### Event-Based Synchronization
```rust
let event = ctx.create_event()?;

// Stream 1 does work, records event
unsafe { stream1.launch_builder(&kernel1).arg(&mut buf).launch(cfg)?; }
stream1.record_event(&event)?;

// Stream 2 waits for event before proceeding
stream2.wait_event(&event)?;
unsafe { stream2.launch_builder(&kernel2).arg(&buf).launch(cfg)?; }
```

## Launch Builder Patterns

### Argument Types
```rust
unsafe {
    let mut builder = stream.launch_builder(&kernel);
    
    // Mutable device buffer (output)
    builder.arg(&mut output_slice);
    
    // Immutable device buffer (input)
    builder.arg(&input_slice);
    
    // View types work too
    builder.arg(&mut output_slice.try_slice_mut(..n)?);
    builder.arg(&input_slice.try_slice(..n)?);
    
    // Scalars by reference
    builder.arg(&(n as u32));
    builder.arg(&threshold);
    
    // Struct by reference (must be #[repr(C)])
    builder.arg(&config);
    
    builder.launch(cfg)?;
}
```

### Dynamic Shared Memory
```rust
let cfg = LaunchConfig {
    grid_dim: (num_blocks, 1, 1),
    block_dim: (256, 1, 1),
    shared_mem_bytes: 48 * 1024,  // Request 48KB
};

// In CUDA kernel:
// extern __shared__ float shared_mem[];
```

## Error Handling

### Comprehensive Error Matching
```rust
use cudarc::driver::DriverError;

match result {
    Ok(slice) => { /* success */ }
    Err(DriverError::OutOfMemory) => {
        // GPU memory exhausted - reduce batch size
        eprintln!("GPU OOM, reducing batch size");
    }
    Err(DriverError::InvalidValue) => {
        // Bad parameter - check kernel args
        eprintln!("Invalid kernel argument");
    }
    Err(DriverError::LaunchFailed) => {
        // Kernel crash - check for out-of-bounds
        eprintln!("Kernel execution failed");
    }
    Err(e) => return Err(e.into()),
}
```

### Deferred Error Detection
```rust
// Kernel errors are asynchronous - must sync to detect
unsafe {
    stream.launch_builder(&kernel).arg(&mut buf).launch(cfg)?;
}

// This will surface kernel errors
if let Err(e) = stream.synchronize() {
    eprintln!("Kernel failed: {}", e);
}
```

## PRISM-4D Specific Patterns

### Packed Batch Data Structure
```rust
#[repr(C)]
pub struct PackedBatch {
    // Concatenated atom coordinates [x,y,z,x,y,z,...]
    pub atoms_packed: CudaSlice<f32>,
    // CA indices per structure
    pub ca_indices_packed: CudaSlice<u32>,
    // Structure descriptors
    pub descriptors: CudaSlice<BatchStructureDesc>,
    // Per-residue data
    pub residue_types: CudaSlice<u8>,
    pub conservation: CudaSlice<f32>,
    pub bfactor: CudaSlice<f32>,
    // GISAID frequency data
    pub gisaid_freq: CudaSlice<f32>,
    pub gisaid_velocity: CudaSlice<f32>,
}

impl PackedBatch {
    pub fn from_inputs(
        stream: &CudaStream,
        inputs: &[StructureInput],
    ) -> Result<Self> {
        // Pack all data contiguously
        let mut all_atoms = Vec::new();
        let mut all_ca = Vec::new();
        let mut descs = Vec::new();
        
        let mut atom_offset = 0u32;
        let mut ca_offset = 0u32;
        
        for input in inputs {
            descs.push(BatchStructureDesc {
                atom_offset,
                ca_offset,
                n_atoms: input.atoms.len() as u32 / 3,
                n_residues: input.ca_indices.len() as u32,
                lineage_hash: input.lineage_hash,
                country_id: input.country_id,
                _padding: [0; 3],
            });
            
            all_atoms.extend_from_slice(&input.atoms);
            all_ca.extend_from_slice(&input.ca_indices);
            
            atom_offset += input.atoms.len() as u32 / 3;
            ca_offset += input.ca_indices.len() as u32;
        }
        
        Ok(Self {
            atoms_packed: stream.htod_copy(&all_atoms)?,
            ca_indices_packed: stream.htod_copy(&all_ca)?,
            descriptors: stream.htod_copy(&descs)?,
            // ... other fields
        })
    }
}
```

### Kernel Launch for Batch Processing
```rust
pub fn launch_mega_fused_batch(
    stream: &CudaStream,
    kernel: &CudaFunction,
    batch: &PackedBatch,
    output: &mut CudaSlice<f32>,
    config: &MegaFusedConfig,
) -> Result<()> {
    let n_structures = batch.descriptors.len();
    
    let cfg = LaunchConfig {
        grid_dim: (n_structures as u32, 1, 1),  // One block per structure
        block_dim: (256, 1, 1),
        shared_mem_bytes: 48 * 1024,
    };
    
    unsafe {
        stream.launch_builder(kernel)
            // Output buffer
            .arg(output)
            // Packed inputs
            .arg(&batch.atoms_packed)
            .arg(&batch.ca_indices_packed)
            .arg(&batch.descriptors)
            .arg(&batch.residue_types)
            .arg(&batch.conservation)
            .arg(&batch.bfactor)
            .arg(&batch.gisaid_freq)
            .arg(&batch.gisaid_velocity)
            // Config
            .arg(&(n_structures as u32))
            .arg(&config.contact_threshold)
            .arg(&config.min_feature_value)
            .launch(cfg)?;
    }
    
    Ok(())
}
```

## Debugging Tips

### Check Device Properties
```rust
let ctx = CudaContext::new(0)?;
let props = ctx.device_properties()?;
println!("Device: {}", props.name);
println!("Total memory: {} MB", props.total_global_mem / 1024 / 1024);
println!("Max shared per block: {} KB", props.shared_mem_per_block / 1024);
println!("Max threads per block: {}", props.max_threads_per_block);
```

### Memory Usage Tracking
```rust
fn check_memory() -> Result<()> {
    let (free, total) = cudarc::driver::result::mem_get_info()?;
    println!("GPU Memory: {}/{} MB free", free / 1024 / 1024, total / 1024 / 1024);
    Ok(())
}
```

### Kernel Timing
```rust
let start = ctx.create_event()?;
let end = ctx.create_event()?;

stream.record_event(&start)?;
unsafe { stream.launch_builder(&kernel).arg(&mut buf).launch(cfg)?; }
stream.record_event(&end)?;
stream.synchronize()?;

let elapsed_ms = end.elapsed_ms(&start)?;
println!("Kernel time: {:.2} ms", elapsed_ms);
```
