---
name: rust-cuda-ffi
description: "Rust-CUDA FFI patterns using cudarc crate. Use when: (1) Working with CudaSlice, CudaView, DevicePointer in Rust GPU code, (2) Implementing buffer pools for GPU memory management, (3) Debugging borrow checker conflicts with raw GPU pointers, (4) Loading PTX modules and launching kernels from Rust, (5) Managing GPU memory lifetimes with Arc references. Encodes cudarc 0.15+ patterns, buffer pool designs, and safe FFI abstractions."
---

# Rust-CUDA FFI Skill

## Purpose
Resolve borrow checker conflicts and provide safe FFI patterns for Rust code interfacing with CUDA kernels via the `cudarc` crate.

## When to Use This Skill
- Implementing `mega_fused_batch.rs` or similar GPU executor modules
- Managing GPU buffer pools with proper lifetime semantics
- Loading PTX files and launching kernels from Rust
- Debugging "borrowed value does not live long enough" errors with CudaSlice
- Implementing batch processing with GPU memory reuse

## Core cudarc Types (v0.15+)

### Memory Types Hierarchy
```
CudaSlice<T>     - Owned GPU memory (like Vec<T> for device)
    |-- .as_view()      -> CudaView<T>     (immutable borrow)
    |-- .as_view_mut()  -> CudaViewMut<T>  (mutable borrow)
    |-- .try_slice(range)     -> CudaView<T>
    +-- .try_slice_mut(range) -> CudaViewMut<T>

CudaView<T>      - Immutable reference to device memory (&[T] equivalent)
CudaViewMut<T>   - Mutable reference to device memory (&mut [T] equivalent)
```

### Critical Ownership Rules
1. **CudaSlice owns an Arc of CudaContext** - Device memory stays valid as long as slice exists
2. **Views borrow from CudaSlice** - Cannot outlive the parent slice
3. **Kernel launches require &mut for output buffers** - Use CudaViewMut
4. **Kernel launches accept & for input buffers** - Use CudaView

## Common Patterns

### Pattern 1: Basic Kernel Launch
```rust
use cudarc::driver::{CudaContext, CudaSlice, LaunchConfig};

let ctx = CudaContext::new(0)?;
let stream = ctx.default_stream();

// Load PTX module
let ptx = std::fs::read_to_string("kernel.ptx")?;
let module = ctx.load_module(ptx)?;
let kernel = module.load_function("my_kernel")?;

// Allocate device memory
let mut output: CudaSlice<f32> = stream.alloc_zeros(1024)?;
let input: CudaSlice<f32> = stream.htod_copy(host_data)?;

// Launch kernel
let cfg = LaunchConfig::for_num_elems(1024);
unsafe {
    stream.launch_builder(&kernel)
        .arg(&mut output)  // mutable output
        .arg(&input)       // immutable input
        .arg(&1024usize)   // scalar by reference
        .launch(cfg)?;
}

// Copy back to host
let result: Vec<f32> = stream.memcpy_dtov(&output)?;
```

### Pattern 2: Avoiding Borrow Checker Conflicts
```rust
// WRONG: Multiple mutable borrows
fn bad_pattern(pool: &mut GpuBufferPool) {
    let output = pool.output_view(1024);  // borrows pool mutably
    let input = pool.upload_atoms(&data);  // ERROR: pool already borrowed
}

// CORRECT: Sequential operations with explicit scope
fn good_pattern(pool: &mut GpuBufferPool, stream: &CudaStream) {
    // Get views from raw slices, not through pool methods
    let input_slice = pool.atoms_buffer.try_slice(..n)?;
    let mut output_view = pool.output_buffer.try_slice_mut(..m)?;
    
    unsafe {
        stream.launch_builder(&kernel)
            .arg(&mut output_view)
            .arg(&input_slice)
            .launch(cfg)?;
    }
}
```

### Pattern 3: Batch Processing with Sub-views
```rust
let mut output: CudaSlice<f32> = stream.alloc_zeros(batch_size * 10)?;

for i_batch in 0..num_batches {
    let offset = i_batch * 10;
    let mut batch_view = output.try_slice_mut(offset..offset + 10)?;
    
    unsafe {
        stream.launch_builder(&kernel)
            .arg(&mut batch_view)
            .launch(cfg)?;
    }
}
```

## LaunchConfig Options

```rust
// For simple element-wise operations
let cfg = LaunchConfig::for_num_elems(n as u32);

// For explicit control (RTX 3060 optimal for PRISM-4D)
let cfg = LaunchConfig {
    grid_dim: ((n + 255) / 256, 1, 1),
    block_dim: (256, 1, 1),
    shared_mem_bytes: 48 * 1024,  // 48KB shared memory
};
```

## PTX Loading Patterns

```rust
// From file
let ptx = std::fs::read_to_string("target/ptx/mega_fused_batch.ptx")?;
let module = ctx.load_module(ptx)?;

// From embedded bytes
const PTX_BYTES: &str = include_str!("../target/ptx/kernel.ptx");
let module = ctx.load_module(PTX_BYTES)?;

// Get function handle
let kernel = module.load_function("mega_fused_batch_detection_prism4d")?;
```

## PRISM-4D FFI Struct Layout

```rust
/// Must match CUDA struct layout exactly
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug)]
pub struct BatchStructureDesc {
    pub atom_offset: u32,
    pub ca_offset: u32,
    pub n_atoms: u32,
    pub n_residues: u32,
    pub lineage_hash: u32,
    pub country_id: u8,
    pub _padding: [u8; 3],
}

// Compile-time size verification
const _: () = assert!(std::mem::size_of::<BatchStructureDesc>() == 24);
```

## Common Errors and Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `borrowed value does not live long enough` | View outlives CudaSlice | Store CudaSlice in struct, not view |
| `cannot borrow as mutable more than once` | Multiple &mut to same buffer | Use separate buffers or sequential access |
| `CUDA_ERROR_INVALID_VALUE` on launch | Mismatched types or null pointer | Verify #[repr(C)] and alignment |
| `CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES` | Too many registers or shared mem | Reduce block size or shared memory |

## References
- See `references/cudarc_patterns.md` for advanced patterns
- See `references/buffer_pool_design.md` for production buffer pool implementation
