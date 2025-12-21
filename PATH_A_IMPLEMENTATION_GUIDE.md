# PATH A Implementation Guide

**Status:** PATH B Complete (79.4%), PATH A Ready for Implementation  
**Target:** 85-90% accuracy via epitope-based P_neut  
**Estimated Time:** 2-3 hours

---

## What's Already Done ✅

### 1. GPU Kernel Created
**File:** `crates/prism-gpu/src/kernels/epitope_p_neut.cu` (267 lines)

**Functions:**
- `compute_epitope_p_neut()` - Computes P_neut matrix from 11-D epitope vectors
- `compute_immunity_from_epitope_p_neut()` - Uses precomputed P_neut for immunity
- `compute_p_neut_correlation()` - For calibration (Pearson correlation)

**Compiled:** `target/ptx/epitope_p_neut.ptx` (19 KB)

### 2. DMS Epitope Extraction
**File:** `crates/prism-ve-bench/src/data_loader.rs:476`

**Function:** `get_epitope_escape(lineage, epitope_idx) -> f32`

**Already extracts 11 epitopes:**
- Epitopes 0-9: RBD (A, B, C, D1, D2, E12, E3, F1, F2, F3)
- Epitope 10: NTD (currently hardcoded 0.4)

**Used in:** `vasil_exact_metric.rs:831` - already populates `epitope_escape` vectors

### 3. Test Infrastructure
**Files:**
- `crates/prism-ve-bench/examples/vasil_exact_gpu_test.rs` (PATH B test)
- `crates/prism-ve-bench/examples/vasil_exact_path_a_test.rs` (PATH A skeleton - created)

---

## Implementation Steps

### STEP 1: Add GPU P_neut Matrix Computation

**Location:** `crates/prism-ve-bench/src/vasil_exact_metric.rs`

**Add after line 1084** (end of `build_for_landscape_gpu`):

```rust
/// Build immunity cache using PATH A (epitope-based P_neut)
pub fn build_for_landscape_gpu_path_a(
    landscape: &ImmunityLandscape,
    dms_data: &DmsEscapeData,
    eval_start: NaiveDate,
    eval_end: NaiveDate,
    epitope_weights: &[f32; 11],
    sigma: f32,
) -> Result<ImmunityCache> {
    use cudarc::driver::{CudaContext, CudaStream, LaunchConfig};
    use anyhow::anyhow;
    
    const N_EPITOPES: usize = 11;
    
    eprintln!("[ImmunityCache GPU PATH A] Epitope-based P_neut approach");
    eprintln!("  Weights: [{:.2}, {:.2}, {:.2}, {:.2}, {:.2}, {:.2}, {:.2}, {:.2}, {:.2}, {:.2}, {:.2}]",
              epitope_weights[0], epitope_weights[1], epitope_weights[2], epitope_weights[3],
              epitope_weights[4], epitope_weights[5], epitope_weights[6], epitope_weights[7],
              epitope_weights[8], epitope_weights[9], epitope_weights[10]);
    eprintln!("  Sigma: {:.3}", sigma);
    
    // Initialize CUDA
    let ctx = CudaContext::new(0).map_err(|e| anyhow!("CUDA init: {}", e))?;
    let stream = ctx.fork_default_stream().map_err(|e| anyhow!("Stream: {}", e))?;
    
    // Load PTX module
    let ptx_path = "target/ptx/epitope_p_neut.ptx";
    let ptx = std::fs::read_to_string(ptx_path)
        .map_err(|e| anyhow!("Failed to read PTX: {}", e))?;
    
    ctx.load_ptx(ptx.into(), "epitope_p_neut", &["compute_epitope_p_neut"])
        .map_err(|e| anyhow!("Load PTX: {}", e))?;
    
    let compute_p_neut_func = ctx.get_func("epitope_p_neut", "compute_epitope_p_neut")
        .ok_or_else(|| anyhow!("Function not found"))?;
    
    // Filter significant variants
    let eval_start_idx = landscape.date_to_idx(eval_start).ok_or_else(|| anyhow!("Start date OOB"))?;
    let significant_indices: Vec<usize> = (0..landscape.lineages.len())
        .filter(|&i| landscape.variant_frequencies.iter()
            .any(|day_freqs| day_freqs.get(i).map_or(false, |&f| f >= 0.01)))
        .collect();
    
    let n_variants = significant_indices.len();
    eprintln!("  {} significant variants (of {} total)", n_variants, landscape.lineages.len());
    
    // Extract epitope escape vectors [n_variants × 11]
    let mut epitope_escape = vec![0.0f32; n_variants * N_EPITOPES];
    for (new_idx, &orig_idx) in significant_indices.iter().enumerate() {
        let lineage = &landscape.lineages[orig_idx];
        for e in 0..10 {
            epitope_escape[new_idx * N_EPITOPES + e] =
                dms_data.get_epitope_escape(lineage, e).unwrap_or(0.0);
        }
        // Epitope 10 = NTD
        let ntd = dms_data.get_ntd_escape(lineage).unwrap_or(0.4) as f32;
        epitope_escape[new_idx * N_EPITOPES + 10] = ntd;
    }
    
    // Upload to GPU
    let mut d_epitope_escape = stream.alloc_zeros(n_variants * N_EPITOPES)
        .map_err(|e| anyhow!("Alloc epitope_escape: {}", e))?;
    stream.memcpy_htod(&epitope_escape, &mut d_epitope_escape)
        .map_err(|e| anyhow!("Upload epitope_escape: {}", e))?;
    
    let mut d_epitope_weights = stream.alloc_zeros(N_EPITOPES)
        .map_err(|e| anyhow!("Alloc weights: {}", e))?;
    stream.memcpy_htod(epitope_weights, &mut d_epitope_weights)
        .map_err(|e| anyhow!("Upload weights: {}", e))?;
    
    // Allocate P_neut matrix [n_variants × n_variants]
    let mut d_p_neut_matrix = stream.alloc_zeros(n_variants * n_variants)
        .map_err(|e| anyhow!("Alloc P_neut: {}", e))?;
    
    // Launch P_neut computation kernel
    eprintln!("[PATH A] Computing epitope-based P_neut matrix ({} × {})...", n_variants, n_variants);
    let cfg_p_neut = LaunchConfig {
        grid_dim: (n_variants as u32, n_variants as u32, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    
    let n_variants_i32 = n_variants as i32;
    unsafe {
        let mut builder = stream.launch_builder(&compute_p_neut_func);
        builder.arg(&d_epitope_escape);
        builder.arg(&d_p_neut_matrix);
        builder.arg(&d_epitope_weights);
        builder.arg(&sigma);
        builder.arg(&n_variants_i32);
        builder.launch(cfg_p_neut)
            .map_err(|e| anyhow!("Launch compute_epitope_p_neut: {}", e))?;
    }
    
    stream.synchronize().map_err(|e| anyhow!("Sync: {}", e))?;
    eprintln!("[PATH A] ✓ P_neut matrix computed");
    
    // Download P_neut matrix
    let p_neut_matrix: Vec<f32> = stream.clone_dtoh(&d_p_neut_matrix)
        .map_err(|e| anyhow!("Download P_neut: {}", e))?;
    
    // TODO: Use P_neut matrix to compute immunity via compute_immunity_from_epitope_p_neut kernel
    // TODO: Compute gamma envelopes (reuse PATH B code)
    // TODO: Return ImmunityCache
    
    unimplemented!("PATH A: Complete immunity computation using P_neut matrix")
}
```

**Key differences from PATH B:**
- Computes **P_neut matrix once** (n_variants × n_variants) instead of on-the-fly
- Uses **epitope distance** instead of PK pharmacokinetics
- Requires **calibrated weights** (12 parameters: 11 weights + 1 sigma)

---

### STEP 2: Create Calibration Script

**File:** `scripts/calibrate_epitope_weights.py`

```python
#!/usr/bin/env python3
"""
Calibrate 11 epitope weights + sigma for PATH A

Uses Nelder-Mead to maximize Pearson correlation with VASIL reference P_neut
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import pearsonr
import subprocess
import json

def load_vasil_p_neut(country="Germany"):
    """Load reference P_neut from VASIL output"""
    # TODO: Parse from VASIL results
    # File: data/VASIL/ByCountry/{country}/results/p_neut_reference.csv
    pass

def compute_p_neut_with_params(epitope_weights, sigma, epitope_escape):
    """
    Compute P_neut using epitope distance formula:
    P_neut(x,y) = exp(-d²(x,y) / (2σ²))
    where d²(x,y) = Σ w_e × (escape_x[e] - escape_y[e])²
    """
    n_variants = len(epitope_escape)
    p_neut = np.zeros((n_variants, n_variants), dtype=np.float32)
    
    for i in range(n_variants):
        for j in range(n_variants):
            # Weighted Euclidean distance
            d_squared = 0.0
            for e in range(11):
                diff = epitope_escape[i][e] - epitope_escape[j][e]
                d_squared += epitope_weights[e] * diff * diff
            
            # Gaussian kernel
            p_neut[i, j] = np.exp(-d_squared / (2 * sigma**2))
    
    return p_neut

def objective_function(params, epitope_escape, p_neut_reference):
    """
    Objective: Maximize correlation with VASIL P_neut
    
    params[0:11] = epitope weights
    params[11] = sigma
    """
    weights = params[0:11]
    sigma = params[11]
    
    # Compute predicted P_neut
    p_neut_pred = compute_p_neut_with_params(weights, sigma, epitope_escape)
    
    # Flatten matrices for correlation
    pred_flat = p_neut_pred.flatten()
    ref_flat = p_neut_reference.flatten()
    
    # Pearson correlation
    corr, _ = pearsonr(pred_flat, ref_flat)
    
    # Return negative (minimize)
    return -corr

def calibrate():
    """Run Nelder-Mead calibration"""
    
    # Load data
    print("Loading Germany validation data...")
    epitope_escape = load_epitope_escape("Germany")  # [n_variants, 11]
    p_neut_reference = load_vasil_p_neut("Germany")  # [n_variants, n_variants]
    
    # Initial parameters
    x0 = np.ones(12, dtype=np.float32)  # [1.0; 11] weights + 1.0 sigma
    x0[11] = 0.5  # Initial sigma estimate
    
    # Bounds
    bounds = [(0.0, 10.0)] * 11 + [(0.01, 2.0)]  # Weights [0-10], sigma [0.01-2]
    
    print("Running Nelder-Mead optimization...")
    print(f"  Initial params: {x0}")
    print(f"  Search space: weights [0-10], sigma [0.01-2]")
    
    result = minimize(
        objective_function,
        x0,
        args=(epitope_escape, p_neut_reference),
        method='Nelder-Mead',
        bounds=bounds,
        options={'maxiter': 500, 'disp': True}
    )
    
    # Extract calibrated parameters
    weights_calibrated = result.x[0:11]
    sigma_calibrated = result.x[11]
    correlation = -result.fun  # Negative of objective
    
    print("\n" + "="*60)
    print("CALIBRATION RESULTS")
    print("="*60)
    print(f"Correlation: {correlation:.4f}")
    print(f"\nEpitope Weights:")
    epitope_names = ["A", "B", "C", "D1", "D2", "E12", "E3", "F1", "F2", "F3", "NTD"]
    for i, (name, weight) in enumerate(zip(epitope_names, weights_calibrated)):
        print(f"  {name:4s}: {weight:.4f}")
    print(f"\nSigma: {sigma_calibrated:.4f}")
    
    # Save to JSON
    output = {
        "weights": weights_calibrated.tolist(),
        "sigma": float(sigma_calibrated),
        "correlation": float(correlation),
        "country": "Germany"
    }
    
    with open("validation_results/epitope_weights_calibrated.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print("\n✅ Saved to validation_results/epitope_weights_calibrated.json")
    
    return weights_calibrated, sigma_calibrated, correlation

if __name__ == "__main__":
    calibrate()
```

---

### STEP 3: Integrate Calibrated Weights into Test

**Modify:** `crates/prism-ve-bench/examples/vasil_exact_path_a_test.rs`

```rust
// Load calibrated weights (if available)
let (epitope_weights, sigma) = if let Ok(json_str) = std::fs::read_to_string(
    "validation_results/epitope_weights_calibrated.json"
) {
    let json: serde_json::Value = serde_json::from_str(&json_str)?;
    let weights: Vec<f32> = json["weights"].as_array().unwrap()
        .iter().map(|v| v.as_f64().unwrap() as f32).collect();
    let sigma = json["sigma"].as_f64().unwrap() as f32;
    let mut weights_array = [0.0f32; 11];
    weights_array.copy_from_slice(&weights);
    
    println!("[PATH A] Using CALIBRATED epitope weights:");
    println!("  Correlation: {:.4}", json["correlation"].as_f64().unwrap());
    (weights_array, sigma)
} else {
    println!("[PATH A] Using UNIFORM epitope weights (baseline):");
    ([1.0f32; 11], 0.5f32)
};
```

---

### STEP 4: Test Progression

#### 4A. Baseline Test (Uniform Weights)
```bash
cargo run --release -p prism-ve-bench --example vasil_exact_path_a_test
```

**Expected:** ~75-80% accuracy (uniform weights are suboptimal)

#### 4B. Calibration (Germany Only)
```bash
python scripts/calibrate_epitope_weights.py
```

**Expected:** Correlation > 0.90, optimized weights, ~10 minutes runtime

#### 4C. Calibrated Test (All Countries)
```bash
cargo run --release -p prism-ve-bench --example vasil_exact_path_a_test
```

**Expected:** 85-90% accuracy (calibrated weights)

---

## Expected Performance

| Configuration | Mean Accuracy | Notes |
|---------------|---------------|-------|
| **PATH B** (baseline) | 79.4% | On-the-fly P_neut, PK-based |
| **PATH A** (uniform) | ~77% | Epitope-based, no calibration |
| **PATH A** (calibrated) | **85-90%** | Optimized weights + sigma |
| **VASIL** (reference) | 90.8% | Ground truth |

**Gap analysis:**
- PATH B → PATH A (uniform): -2% (slight drop expected)
- PATH A (uniform) → PATH A (calibrated): +8-13% (calibration impact)
- PATH A (calibrated) → VASIL: -0.8 to -5.8% (remaining gap)

---

## Files to Modify

### Core Implementation
1. `crates/prism-ve-bench/src/vasil_exact_metric.rs`
   - Add `build_for_landscape_gpu_path_a()` function (after line 1084)
   - Add epitope P_neut matrix computation
   - Reuse gamma envelope code from PATH B

### Calibration
2. `scripts/calibrate_epitope_weights.py` (NEW)
   - Nelder-Mead optimization
   - Load VASIL reference P_neut
   - Save calibrated parameters to JSON

### Testing
3. `crates/prism-ve-bench/examples/vasil_exact_path_a_test.rs` (CREATED)
   - Load calibrated weights from JSON
   - Call PATH A GPU function
   - Report accuracy vs PATH B

---

## Implementation Checklist

- [ ] **STEP 1:** Add `build_for_landscape_gpu_path_a()` to `vasil_exact_metric.rs`
- [ ] **STEP 2:** Implement P_neut matrix computation (GPU kernel call)
- [ ] **STEP 3:** Implement immunity computation using P_neut matrix
- [ ] **STEP 4:** Test PATH A with uniform weights (baseline)
- [ ] **STEP 5:** Create `calibrate_epitope_weights.py` script
- [ ] **STEP 6:** Extract VASIL reference P_neut for calibration
- [ ] **STEP 7:** Run calibration on Germany (optimize 12 params)
- [ ] **STEP 8:** Test PATH A with calibrated weights (all 12 countries)
- [ ] **STEP 9:** Compare PATH A vs PATH B performance
- [ ] **STEP 10:** Document results and next steps

---

## Known Challenges

### 1. P_neut Matrix Memory
For France (481 variants): 481² × 4 bytes = 925 KB (trivial)  
✅ **No issue** - fits easily in GPU memory

### 2. Calibration Runtime
Nelder-Mead with 12 parameters, 500 iterations:
- ~10-15 minutes on CPU
- Can parallelize with GPU correlation kernel (optional speedup)

### 3. Parameter Sensitivity
Some epitope weights may be more important than others:
- Class I (A, B): High impact (receptor binding)
- Class IV (F1, F2, F3): Lower impact (peripheral)
- NTD: Moderate impact (non-RBD)

**Solution:** Nelder-Mead will automatically find optimal weights

---

## Success Criteria

### Minimum Viable (MVP)
- [ ] PATH A baseline test runs (uniform weights)
- [ ] Achieves ≥75% accuracy (validates kernel correctness)

### Target Achievement
- [ ] Calibration script completes successfully
- [ ] Calibrated PATH A achieves ≥85% mean accuracy
- [ ] All 12 countries ≥80% (no major outliers)

### Stretch Goal
- [ ] PATH A achieves ≥88% mean accuracy
- [ ] Gap to VASIL <3% (within expected margin)
- [ ] Runtime <10 minutes for full 12-country test

---

## Next Steps After PATH A

1. **Multi-scale fusion** (PATH A+): Combine epitope + PK approaches
2. **Temporal P_neut**: Add time-decay to epitope distance
3. **Vaccination data**: Incorporate booster effects
4. **Production deployment**: Optimize for real-time prediction

---

## Time Estimates

| Task | Time | Priority |
|------|------|----------|
| STEP 1: Rust P_neut function | 30 min | HIGH |
| STEP 2: Immunity computation | 30 min | HIGH |
| STEP 3: Test baseline | 10 min | HIGH |
| STEP 4: Calibration script | 45 min | MEDIUM |
| STEP 5: Run calibration | 15 min | MEDIUM |
| STEP 6: Test calibrated | 10 min | HIGH |
| **Total** | **~2.5 hours** | - |

---

## Files Created This Session

✅ `crates/prism-gpu/src/kernels/epitope_p_neut.cu` (267 lines)  
✅ `target/ptx/epitope_p_neut.ptx` (19 KB, compiled)  
✅ `crates/prism-ve-bench/examples/vasil_exact_path_a_test.rs` (skeleton)  
✅ `PATH_A_IMPLEMENTATION_GUIDE.md` (this file)  
✅ `PATH_B_SUCCESS_SUMMARY.md` (complete session summary)  
✅ `GPU_GRID_FIX.md` (CUDA grid dimension fix)

---

**Status:** Ready for implementation  
**Next:** Start with STEP 1 (add Rust function for P_neut matrix)  
**Blocker:** None - all dependencies ready
