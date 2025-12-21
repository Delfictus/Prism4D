# SESSION 2 HANDOFF - TWO-PASS SYSTEM TRAINING MODULES

## SESSION 1 COMPLETION STATUS

✅ **PHASE 1 COMPLETE:** 92-Dim Kernel with Physics Features
- **Commit**: `e15cac5`
- **Tag**: `phase1-92dim-kernel`
- **Branch**: `lbs-unified-50-results`

### Changes Made:
1. Added physics constants (hydrophobicity, charge, volume) - lines 335-366
2. Added Stage 3.6 physics computation function - lines 832-934
3. Updated shared memory for physics_features[32][12] - line 393
4. Updated Stage 6.5 to output 92-dim - lines 1427-1431
5. Added Stage 3.6 call in main kernel - lines 1545-1549
6. Updated TOTAL_COMBINED_FEATURES: 80 → 92
7. Updated mega_fused.rs constant to 92
8. Updated readout_training.rs constant to 92
9. Compiled PTX successfully (527K)
10. Verified Rust build passes

### PTX Status:
- File: `/mnt/c/Users/Predator/Desktop/PRISM/target/ptx/mega_fused_pocket.ptx`
- Size: 527K
- Architecture: sm_86
- Features: 92-dim (48 TDA + 32 base + 12 physics)

---

## PHYSICS FEATURES IMPLEMENTED (Stage 3.6)

| Index | Feature | Formula | Source |
|-------|---------|---------|--------|
| 80 | Entropy production rate | dS/dt from B-factors | thermodynamics.rs |
| 81 | Local hydrophobicity | Kyte-Doolitt scale | Constant array |
| 82 | Neighbor hydrophobicity | Avg neighbors | Simplified |
| 83 | Desolvation cost | hydro × burial | Thermodynamic |
| 84 | Cavity size | Δx (16Å / n_neighbors) | quantum_mechanics.rs |
| 85 | Tunneling accessibility | Δx·Δp | Heisenberg |
| 86 | Energy curvature | 1/r² potential | Hamiltonian |
| 87 | Conservation entropy | H(X) Shannon | information_theory.rs |
| 88 | Mutual information proxy | I(X;Y) | information_theory.rs |
| 89 | Thermodynamic binding | hydro·cons·(1-burial) | Combined |
| 90 | Allosteric potential | bfactor·conservation | Combined |
| 91 | Druggability | cavity·hydro·accessible | Combined |

**Note:** Currently uses `nullptr` for residue_types (defaults to res_type=0).
For Session 3, add actual residue type parsing from PDB.

---

## SESSION 2 TASKS: TRAINING MODULES

### Required Asset Files (verified to exist):
```
/mnt/c/Users/Predator/Desktop/PRISM - Codex/foundation/mathematics/thermodynamics.rs
/mnt/c/Users/Predator/Desktop/PRISM - Codex/foundation/mathematics/information_theory.rs
/mnt/c/Users/Predator/Desktop/PRISM - Codex/foundation/mathematics/quantum_mechanics.rs
/mnt/c/Users/Predator/Desktop/PRISM - Codex/foundation/quantum/src/robust_eigen.rs (29KB)
/mnt/c/Users/Predator/Desktop/PRISM - Codex/foundation/quantum/src/qubo.rs (9KB)
```

### Files to CREATE:

#### 1. `crates/prism-gpu/src/training/mod.rs`
```rust
//! Training modules for PRISM-LBS two-pass system

pub mod reorthogonalization;
pub mod simulated_annealing;
pub mod two_pass;

pub use reorthogonalization::WhiteningParams;
pub use simulated_annealing::{SAClassifier, SAConfig};
pub use two_pass::{TwoPassBenchmark, TwoPassConfig, ComprehensiveReport};
```

#### 2. `crates/prism-gpu/src/training/reorthogonalization.rs`
- Use `robust_eigen.rs` for eigendecomposition (NOT simple power iteration)
- WhiteningParams struct (means, whitening_matrix, eigenvalues)
- fit() method - compute PCA whitening from 92-dim features
- transform() method - apply whitening to new samples
- CRITICAL: Handle 92x92 covariance (may be ill-conditioned)

**Key from robust_eigen.rs to use:**
- Hermitian symmetrization (lines 240-260)
- Preconditioning for condition number > 10^6 (lines 300-350)
- Solution validation via residual (lines 718-786)

#### 3. `crates/prism-gpu/src/training/simulated_annealing.rs`
- Adapt SA loop from `qubo.rs` solve_cpu_sa() (lines 70-130)
- SAClassifier struct (weights[92], bias, threshold)
- train() method - minimize cross-entropy with SA
- Temperature schedule from qubo.rs (cooling_rate)
- Metropolis criterion (EXACT COPY from qubo.rs line 108-115)

**Key from qubo.rs to use:**
- Cooling schedule: temp *= cooling_rate (line 118)
- Acceptance: `delta < 0 || rng.gen::<f64>() < (-delta / temp).exp()` (line 108)
- Perturbation: random weight update (adapt from bit flip)

#### 4. `crates/prism-gpu/src/training/two_pass.rs`
- TwoPassBenchmark struct
- Pass 1: Full CryptoBench with whitening + SA
- Pass 2: Top-N refined with ensemble
- ComprehensiveReport with both sets of metrics
- Pretty-printed report table

### Files to MODIFY:

#### 5. `crates/prism-gpu/src/lib.rs`
Add:
```rust
pub mod training;
pub use training::{TwoPassBenchmark, TwoPassConfig};
```

---

## VERIFICATION COMMANDS FOR SESSION 2

After creating modules:
```bash
# Check structure
ls -la crates/prism-gpu/src/training/

# Verify compilation
cargo build -p prism-gpu --release

# Check module exports
grep "pub use training" crates/prism-gpu/src/lib.rs
```

---

## SESSION 3 TASKS: INTEGRATION + BENCHMARK

1. **Add residue type parsing** (mega_fused.rs)
2. **Pass residue_types to kernel** (update kernel signature)
3. **Create two_pass_benchmark binary**
4. **Run full benchmark**
5. **Report results**

---

## CURRENT BASELINE (DO NOT REGRESS)

| Metric | Value | System |
|--------|-------|--------|
| AUC-ROC | 0.7050 | 80-dim with normalization |
| F1 | 0.0593 | With optimal threshold |
| Precision | 0.0344 | At threshold 0.0656 |
| Recall | 0.2127 | At threshold 0.0656 |

**Target for 92-dim + two-pass:**
- Pass 1: AUC 0.78+, F1 0.28+
- Pass 2: AUC 0.85+, F1 0.55+

---

## CRITICAL NOTES

1. **Use robust_eigen.rs, not simple power iteration** - 92x92 matrices are ill-conditioned
2. **Use qubo.rs SA loop exactly** - temperature schedule and Metropolis are proven
3. **Physics features use default res_type=0** - Session 3 adds actual residue parsing
4. **Z-score normalization already in readout_training.rs** - works with 92-dim

---

## RECOVERY COMMANDS

If Session 2 needs to restart:
```bash
git checkout phase1-92dim-kernel
git log --oneline -3
grep "TOTAL_COMBINED_FEATURES" crates/prism-gpu/src/mega_fused.rs
```

Expected output: `pub const TOTAL_COMBINED_FEATURES: usize = 92;`

---

## END OF SESSION 1

Phase 1 complete. Ready for Session 2.
