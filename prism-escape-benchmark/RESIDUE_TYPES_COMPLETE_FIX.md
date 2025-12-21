# COMPLETE FIX FOR DEAD PHYSICS FEATURES

## ROOT CAUSE IDENTIFIED

**The issue has 3 levels:**

### Level 1: Rust Parsing ✅ FIXED
```rust
// detector.rs lines 212-233
let residue_types: Vec<i32> = structure.residues.iter()
    .map(|res| parse_residue_name_to_index(&res.name))
    .collect();

// Passes to detect_pockets()
Some(&residue_types)
```

### Level 2: GPU Upload ❌ NOT IMPLEMENTED
```rust
// mega_fused.rs - MISSING CODE
// Need to add (around line 1380):

if let Some(res_types) = residue_types {
    let mut d_residue_types = self.stream.alloc_zeros::<i32>(n_residues)?;
    self.stream.memcpy_htod(res_types, &mut d_residue_types)?;
    builder.arg(&*d_residue_types);  // Pass to kernel
} else {
    builder.arg(std::ptr::null::<i32>());  // nullptr
}
```

### Level 3: CUDA Kernel Signature ❌ MISSING PARAMETER
```cuda
// mega_fused_pocket_kernel.cu line 1481
// CURRENT (wrong):
extern "C" __global__ void mega_fused_pocket_detection(
    const float* atoms,
    const int* ca_indices,
    const float* conservation_input,
    const float* bfactor_input,
    const float* burial_input,
    int n_atoms,
    int n_residues,
    // TDA args...  ← NO residue_types parameter!
    ...
)

// SHOULD BE:
extern "C" __global__ void mega_fused_pocket_detection(
    const float* atoms,
    const int* ca_indices,
    const float* conservation_input,
    const float* bfactor_input,
    const float* burial_input,
    const int* residue_types,  ← ADD THIS!
    int n_atoms,
    int n_residues,
    // TDA args...
    ...
)
```

## WHY IT'S STILL BROKEN

Even though we parse and pass residue_types in Rust:
1. ❌ We don't allocate GPU memory for it
2. ❌ We don't upload it to GPU  
3. ❌ We don't pass it to kernel
4. ❌ CUDA kernel signature doesn't expect it

**The kernel still calls stage3_6_physics_features() with nullptr!**

## COMPLETE FIX REQUIRED

**Files to modify:**

1. **crates/prism-gpu/src/kernels/mega_fused_pocket_kernel.cu**
   - Line 1481: Add `const int* residue_types,` parameter
   - Line 1554: Pass actual residue_types instead of nullptr
   - Recompile PTX

2. **crates/prism-gpu/src/mega_fused.rs**
   - Add buffer allocation for residue_types
   - Upload to GPU
   - Pass as kernel argument (line ~1505)

## ESTIMATED TIME

- Modify CUDA kernel: 5 minutes
- Recompile PTX: 1 minute
- Modify Rust upload code: 10 minutes
- Test: 5 minutes
**Total: ~20 minutes**

## EXPECTED RESULT

After fix:
- Features 81-83 (hydrophobicity): ✅ WORKING
- Features 89, 91 (use hydrophobicity): ✅ WORKING
- Total working: 4 → 7 features
- Expected correlation: ρ=0.36 → 0.45-0.50

Ready to implement?
