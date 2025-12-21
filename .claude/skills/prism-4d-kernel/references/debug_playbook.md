# PRISM>4D Debug Playbook

## CUDA Errors

### CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES
**Cause**: Too many registers per thread or too much shared memory

**Fix**:
```bash
# Check register usage
nvcc --ptxas-options=-v -ptx src/kernels/mega_fused_batch.cu 2>&1 | grep -E "registers|smem"
```

**Solutions**:
1. Reduce `__launch_bounds__(256, 2)` to `__launch_bounds__(256, 1)` (fewer blocks/SM)
2. Add `#pragma unroll 1` to large loops
3. Split shared memory arrays into smaller tiles
4. Move constant arrays from shared to constant memory (`__constant__`)

**RTX 3060 Limits**:
- Max registers/thread: 255
- Max shared memory/block: 48KB
- Target: <64 registers/thread for 2 blocks/SM

### CUDA_ERROR_ILLEGAL_ADDRESS
**Cause**: Out-of-bounds memory access in kernel

**Debug**:
```bash
compute-sanitizer --tool memcheck ./target/release/vasil-benchmark
```

**Common causes**:
1. `structure_idx >= n_structures` without early return
2. `global_idx >= n_residues` accessing residue arrays
3. Misaligned `BatchStructureDesc` (must be 16-byte aligned)

### Kernel Hangs (No Error, No Output)
**Cause**: Infinite loop or synchronization deadlock

**Debug**:
```bash
# Add timeout to kernel launch
export CUDA_LAUNCH_BLOCKING=1
timeout 30s ./target/release/vasil-benchmark
```

**Common causes**:
1. Missing `__syncthreads()` after shared memory writes
2. Conditional `__syncthreads()` (NEVER do this)
3. Infinite `while` loop in convergence check

## Rust-CUDA FFI Errors

### Borrow Checker: "borrow of moved value"
**Pattern**: Buffer pool references after move

**Bad**:
```rust
let d_atoms = self.buffer_pool.d_atoms.as_mut().unwrap();
// ... use d_atoms ...
let d_atoms_ref = self.buffer_pool.d_atoms.as_ref().unwrap(); // ERROR
```

**Good**:
```rust
// Get all refs in one block, then use
let (d_atoms, d_ca) = {
    let pool = &mut self.buffer_pool;
    (pool.d_atoms.as_ref().unwrap(), pool.d_ca_indices.as_ref().unwrap())
};
```

### CudaSlice Lifetime Issues
**Error**: "borrowed value does not live long enough"

**Fix**: Ensure CudaSlice outlives kernel launch:
```rust
// Allocate BEFORE building launch args
let d_output = stream.alloc_zeros::<f32>(size)?;

// Build args with references
unsafe {
    func.launch(config, (d_input.as_ref(), d_output.as_mut(), ...))
}?;

// Copy results AFTER synchronize
stream.synchronize()?;
let mut results = vec![0.0f32; size];
stream.memcpy_dtoh(&d_output, &mut results)?;
```

### PTX Load Failure
**Error**: "Failed to load PTX: invalid PTX input"

**Causes**:
1. PTX compiled for wrong architecture (need sm_75 for RTX 3060)
2. Missing extern "C" on kernel function
3. Syntax error in CUDA code

**Fix**:
```bash
# Verify PTX is valid
cuobjdump -ptx target/ptx/mega_fused_batch.ptx | head -50
# Should show ".target sm_75" and ".entry mega_fused_batch_detection"
```

## FluxNet RL Issues

### Training Doesn't Converge (Accuracy Stuck ~50%)
**Causes**:
1. Features not properly normalized
2. Discretization bins too coarse/fine
3. Learning rate too high/low

**Debug**:
```rust
// Print feature distributions
for (state, _) in &train_data {
    println!("escape={:.3} freq={:.3} gp={:.3}", 
             state.escape, state.frequency, state.growth_potential);
}
```

**Fixes**:
1. Z-score normalize features before discretization
2. Try coarser bins: 4 bins instead of 8 (256 states vs 4096)
3. Adjust α (learning rate) from 0.1 to 0.05

### Class Imbalance (Always Predicts FALL)
**Cause**: ~64% of samples are FALL, RL converges to majority

**Fix** (in ve_optimizer.rs):
```rust
// Asymmetric rewards
let rise_weight = data.len() as f32 / (2.0 * rise_count as f32);
let fall_weight = data.len() as f32 / (2.0 * fall_count as f32);

let reward = if is_correct {
    if is_rise { rise_weight } else { fall_weight }
} else {
    if is_rise { -rise_weight * 1.5 } else { -fall_weight }
};
```

### Q-Values Don't Make Sense
**Debug**:
```rust
// Dump Q-table for inspection
for state_idx in 0..256 {
    let q = optimizer.q_table[state_idx];
    if q[0].abs() > 0.1 || q[1].abs() > 0.1 {
        println!("State {}: Q(RISE)={:.3}, Q(FALL)={:.3}", 
                 state_idx, q[0], q[1]);
    }
}
```

**Expected pattern**:
- High escape + low freq → Q(RISE) > Q(FALL)
- Low escape + high freq → Q(FALL) > Q(RISE)

## Feature Extraction Issues

### Features 92-100 Are All Zero
**Cause**: Stage 7/8 not being called in batch kernel

**Check**: Verify in mega_fused_batch.cu:
```cuda
// After Stage 6, MUST have:
stage7_fitness_features(n_residues, tile_idx, bfactor, residue_types, &smem, params);
stage8_cycle_features(n_residues, tile_idx, gisaid_frequencies, gisaid_velocities, &smem);
stage6_5_combine_features(n_residues, tile_idx, combined_features_out, &smem);
```

### Feature Dimensions Mismatch
**Error**: "Expected 14917*101 features, got X"

**Check**:
1. `TOTAL_COMBINED_FEATURES` constant = 101
2. Output buffer allocated as `n_residues * 101`
3. stage6_5_combine_features writes all 101 dims

### NaN/Inf in Features
**Debug**:
```rust
for i in 0..features.len() {
    if features[i].is_nan() || features[i].is_infinite() {
        println!("Feature {} at index {} is invalid", i % 101, i);
    }
}
```

**Common causes**:
1. Division by zero in ΔΔG calculation
2. exp() overflow in sigmoid
3. log(0) in escape score

**Fix**: Add guards:
```cuda
float sigmoid_safe(float x) {
    x = fminf(fmaxf(x, -20.0f), 20.0f); // Clamp to prevent overflow
    return 1.0f / (1.0f + expf(-x));
}
```

## Performance Issues

### Batch Kernel Too Slow (>60s)
**Profile**:
```bash
nsys profile --stats=true ./target/release/vasil-benchmark
```

**Common bottlenecks**:
1. Too many global memory accesses → Use `__ldg()` for read-only
2. Bank conflicts in shared memory → Pad arrays
3. Warp divergence → Restructure conditionals

**Quick wins**:
```cuda
// Before: Multiple loads
float x = atoms[idx*3];
float y = atoms[idx*3+1];
float z = atoms[idx*3+2];

// After: Single float3 load
float3 coord = *((float3*)&atoms[idx*3]);
```

### Memory Transfer Overhead
**Problem**: D2H copies dominating runtime

**Fix**: Use pinned memory for large transfers:
```rust
// In Rust, use page-locked allocation
let mut h_features = cuda_malloc_host::<f32>(n_residues * 101)?;
stream.memcpy_dtoh_async(&d_features, &mut h_features)?;
stream.synchronize()?;
```

## VASIL Data Issues

### Country Data Not Loading
**Check paths**:
```
/mnt/f/VASIL_Data/ByCountry/{Country}/results/
├── Daily_Lineages_Freq_1_percent.csv
├── Lineage_Spike_Mutations.csv
└── DMS_Escape_Matrix.csv
```

**Country names** (case-sensitive):
Germany, USA, UK, Japan, Brazil, France, Canada, Denmark, Australia, Sweden, Mexico, SouthAfrica

### Date Parsing Errors
**Format**: YYYY-MM-DD (e.g., 2023-07-27)

**Fix in data_loader.rs**:
```rust
let date = NaiveDate::parse_from_str(&date_str, "%Y-%m-%d")
    .or_else(|_| NaiveDate::parse_from_str(&date_str, "%m/%d/%Y"))?;
```

### Missing Lineage Mutations
**Cause**: Lineage name mismatch between files

**Debug**:
```rust
let gisaid_lineages: HashSet<_> = frequencies.lineages.iter().collect();
let mutation_lineages: HashSet<_> = mutations.lineage_to_mutations.keys().collect();
let missing: Vec<_> = gisaid_lineages.difference(&mutation_lineages).collect();
println!("Missing mutations for: {:?}", missing);
```
