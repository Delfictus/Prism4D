# SESSION 3 HANDOFF - INTEGRATION & BENCHMARK

## COMPLETED SESSIONS

### Session 1: Phase 1 - Kernel (COMPLETE ✅)
- **Commit**: `e15cac5`
- **Tag**: `phase1-92dim-kernel`
- **Deliverable**: 92-dim CUDA kernel with 12 physics features

### Session 2: Phase 2 - Training Scaffolding (COMPLETE ✅)
- **Commit**: `8c52241`
- **Tag**: `phase2-training-modules`
- **Deliverable**: 3 new training modules (simplified placeholders)

---

## CURRENT STATE

**Branch**: `lbs-unified-50-results`
**Latest Commit**: `8c52241`
**PTX File**: `target/ptx/mega_fused_pocket.ptx` (527K, 92-dim)
**Dimension**: 92 features (48 TDA + 32 base + 12 physics)

### Baseline Performance (80-dim):
- AUC-ROC: 0.7050
- F1: 0.0593

---

## SESSION 3 TASKS

### Priority Order:
1. **Test 92-dim kernel first** (most important!)
2. Enhance training modules (if physics helps)
3. Full two-pass system (if time permits)

---

## TASK 1: TEST 92-DIM PHYSICS KERNEL (CRITICAL)

Before building complex training, verify physics features help:

### Step 1.1: Rebuild train-readout binary
```bash
PATH="/home/diddy/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin:/usr/bin:$PATH" \
CUDA_HOME=/usr/local/cuda-12.6 \
cargo build --release -p prism-lbs --bin train-readout
```

### Step 1.2: Run benchmark with 92-dim
```bash
PRISM_PTX_DIR=/mnt/c/Users/Predator/Desktop/PRISM/target/ptx \
RUST_LOG=info \
timeout 600 ./target/release/train-readout \
    --pdb-dir ./benchmarks/datasets/cryptobench/pdb-files \
    --dataset ./benchmarks/datasets/cryptobench/dataset.json \
    --folds ./benchmarks/datasets/cryptobench/folds.json \
    --output /tmp/physics_92dim_test.bin \
    --lambda 1e-4 \
2>&1 | tee /tmp/physics_92dim_test.log
```

### Step 1.3: Check results
```bash
tail -50 /tmp/physics_92dim_test.log | grep -E "AUC|F1|Precision|Recall"
```

### Expected Outcomes:

**IF PHYSICS HELPS (AUC > 0.71):**
→ Continue to Task 2 (enhance training modules)

**IF NO IMPROVEMENT (AUC ≤ 0.705):**
→ STOP. Physics features don't help. Revert to 80-dim.
→ Document findings and consider different approach

---

## TASK 2: ENHANCE TRAINING MODULES (ONLY IF TASK 1 SUCCEEDS)

### Step 2.1: Implement full reorthogonalization

Read `/mnt/c/Users/Predator/Desktop/PRISM - Codex/foundation/quantum/src/robust_eigen.rs` and adapt:

**Key functions to copy:**
- Lines 240-300: Hermitian symmetrization
- Lines 350-450: Eigendecomposition with preconditioning
- Lines 718-786: Solution validation

Update `crates/prism-gpu/src/training/reorthogonalization.rs`:
```rust
// Replace simple power iteration with robust_eigen approach
pub fn fit(features: &[Vec<f32>]) -> Self {
    // 1. Build 92x92 covariance
    // 2. Symmetrize
    // 3. Eigen with robust solver
    // 4. Validate residual ||Hv - λv||
    // 5. Build whitening matrix W = V·D^{-1/2}
}
```

### Step 2.2: Implement full SA classifier

Read `/mnt/c/Users/Predator/Desktop/PRISM - Codex/foundation/quantum/src/qubo.rs` and adapt lines 70-130:

Update `crates/prism-gpu/src/training/simulated_annealing.rs`:
```rust
pub fn train(...) -> Self {
    // Copy SA loop from qubo.rs solve_cpu_sa()
    // Replace QUBO energy with cross-entropy loss
    // Keep temperature schedule EXACT
    // Keep Metropolis criterion EXACT
}
```

### Step 2.3: Implement two-pass benchmark

Update `crates/prism-gpu/src/training/two_pass.rs`:
```rust
impl TwoPassBenchmark {
    pub fn run(&mut self, ...) -> Result<ComprehensiveReport> {
        // PASS 1: Full benchmark
        // - Extract 92-dim features
        // - Fit whitening
        // - Train SA classifier
        // - Evaluate on test set

        // PASS 2: Top-N refinement
        // - Select top 50 by confidence
        // - Ensemble predictions
        // - Report refined metrics
    }
}
```

---

## TASK 3: FULL INTEGRATION (IF TASK 2 SUCCEEDS)

### Step 3.1: Add residue type parsing

In `mega_fused.rs`, add:
```rust
fn parse_residue_types(pdb_structure: &Structure) -> Vec<i32> {
    // Map residue names to indices 0-19
    // A=0, R=1, N=2, D=3, C=4, Q=5, E=6, G=7, H=8, I=9
    // L=10, K=11, M=12, F=13, P=14, S=15, T=16, W=17, Y=18, V=19
}
```

### Step 3.2: Update kernel signature

In `mega_fused_pocket_kernel.cu`, modify wrapper to accept `residue_types`:
```cuda
extern "C" cudaError_t launch_mega_fused_pocket_detection_with_host_params(
    // ... existing params ...
    const int* d_residue_types,  // ADD THIS [n_residues]
    // ... rest ...
)
```

Replace `nullptr` in Stage 3.6 call with actual `residue_types`.

### Step 3.3: Wire residue_types buffer

In `mega_fused.rs`:
```rust
// Add to BufferPool:
d_residue_types: Option<CudaSlice<i32>>,

// In detect_pockets:
let res_types = parse_residue_types(structure);
self.stream.memcpy_htod(&res_types, d_residue_types)?;
builder.arg(&*d_residue_types);
```

### Step 3.4: Recompile PTX
```bash
CUDA_HOME=/usr/local/cuda-12.6 \
/usr/local/cuda-12.6/bin/nvcc -ptx \
    -o target/ptx/mega_fused_pocket.ptx \
    -arch=sm_86 --std=c++14 -Xcompiler -fPIC \
    crates/prism-gpu/src/kernels/mega_fused_pocket_kernel.cu
```

### Step 3.5: Run full two-pass benchmark
```bash
PRISM_PTX_DIR=./target/ptx \
RUST_LOG=info \
cargo run --release -p prism-gpu --bin two_pass_benchmark -- \
    ./benchmarks/datasets/cryptobench \
    ./results/two_pass
```

---

## VERIFICATION CHECKLIST

After Task 1:
- [ ] 92-dim features extract without errors
- [ ] Physics features [80-91] are non-zero
- [ ] AUC comparison: 92-dim vs 80-dim

After Task 2:
- [ ] Whitening reduces feature correlation
- [ ] SA converges (loss decreasing)
- [ ] Compilation passes

After Task 3:
- [ ] Residue types parsed correctly (0-19 range)
- [ ] PTX recompiles with residue_types signature
- [ ] Two-pass benchmark runs to completion
- [ ] Pass 1 metrics reported
- [ ] Pass 2 metrics reported

---

## DECISION POINTS

### Decision Point 1 (After Task 1):
**IF AUC > 0.71**: Physics helps → Continue to Task 2
**IF AUC ≤ 0.705**: Physics doesn't help → STOP, revert to 80-dim

### Decision Point 2 (After Task 2):
**IF AUC > 0.75**: Training works → Continue to Task 3
**IF AUC ≤ 0.72**: Training marginal → Document as is

### Decision Point 3 (After Task 3):
**IF Pass 1 AUC > 0.78, F1 > 0.25**: Success → Celebrate!
**IF Pass 1 AUC < 0.75**: Document limitations

---

## RECOVERY COMMANDS

Start Session 3:
```bash
cd /mnt/c/Users/Predator/Desktop/PRISM
git checkout phase2-training-modules
git log --oneline -3
```

Verify state:
```bash
grep "TOTAL_COMBINED_FEATURES" crates/prism-gpu/src/mega_fused.rs
ls -la crates/prism-gpu/src/training/
ls -la target/ptx/mega_fused_pocket.ptx
```

Expected:
- `TOTAL_COMBINED_FEATURES: usize = 92`
- 7 files in training/ (benchmark, feature_pipeline, mod, normalization, readout, reorthogonalization, simulated_annealing, two_pass)
- PTX: 527K

---

## CURRENT LIMITATIONS (TO FIX IN SESSION 3)

1. **reorthogonalization.rs**: Uses simple Z-score, not full PCA whitening
2. **simulated_annealing.rs**: Returns default weights, no actual SA loop
3. **two_pass.rs**: No run() method implementation
4. **mega_fused.rs**: residue_types = nullptr (defaults to A=0 for all)

---

## SUCCESS CRITERIA

**Minimum viable (Task 1):**
- 92-dim extracts successfully
- Physics features non-zero
- AUC ≥ 0.71 (improvement over 0.705)

**Good outcome (Task 2):**
- Whitening + SA implemented
- AUC ≥ 0.75
- F1 ≥ 0.15

**Excellent outcome (Task 3):**
- Full two-pass system
- Pass 1: AUC ≥ 0.78, F1 ≥ 0.28
- Pass 2: AUC ≥ 0.85, F1 ≥ 0.55

---

## END OF SESSION 2

Phases 1-2 complete. Ready for Session 3.

**Next session first command:**
```bash
cd /mnt/c/Users/Predator/Desktop/PRISM && git status
```
