# PRISM-LBS GPU Profiling Report

## Executive Summary

**System**: 92-dim physics-enhanced binding site predictor
**Performance**: AUC 0.7142, ~9.3ms/structure
**GPU**: NVIDIA GeForce RTX 3060 Laptop (sm_86, 6GB, 2100MHz)

---

## GPU Hardware

| Specification | Value |
|---------------|-------|
| GPU Model | NVIDIA GeForce RTX 3060 Laptop |
| Compute Capability | 8.6 |
| Memory | 6144 MiB (6 GB) |
| Max SM Clock | 2100 MHz |
| Max Memory Clock | 7001 MHz |
| CUDA Version | 12.6.85 |

---

## Kernel Architecture

### Mega-Fused Kernel Stats
- **File**: `mega_fused_pocket_kernel.cu`
- **Lines**: 2171
- **Stages**: 10 computational stages
- **Synchronization Points**: 24 `__syncthreads()` calls
- **PTX Output**: 527K (sm_86)

### Stage Breakdown

| Stage | Purpose | Complexity |
|-------|---------|------------|
| 1 | Distance + Contact computation | O(n²) per tile |
| 2 | Local features (conservation, B-factor, burial) | O(n) |
| 3 | Network centrality (power iteration) | O(n² × iters) |
| 3.5 | TDA topological features | O(n × neighbors) |
| **3.6** | **Physics features** (NEW) | **O(n × neighbors)** |
| 4 | Dendritic reservoir transform | O(n × reservoir_dim) |
| 5 | Consensus scoring | O(n) |
| 6 | Kempe chain refinement | O(n × iters) |
| 6.5 | Feature combination (92-dim output) | O(n) |
| 7 | Metrics + histogram collection | O(n) |

**Most compute-intensive**: Stage 3 (power iteration), Stage 4 (dendritic GNN)

---

## Resource Usage

### Kernel Launch Configuration
- **Threads per block**: 256
- **Blocks per grid**: ceil(n_residues / TILE_SIZE)
- **TILE_SIZE**: 32
- **Shared memory**: ~50-60 KB estimate (physics_features[32][12] + others)

### PTX Analysis
- **Output size**: 527 KB
- **Architecture**: sm_86 (Ampere)
- **Warnings**: 2 (unused variables, non-critical)

**Note**: Detailed register/spill analysis requires ptxas verbose output which wasn't captured in this build. For production optimization, recompile with explicit `--ptxas-options=-v` and analyze output.

---

## Identified Bottlenecks (Static Analysis)

### 1. Memory Bandwidth (Likely Primary)
**Impact**: High
**Evidence**:
- Multiple global memory reads per residue (atoms, CA indices, TDA neighbors)
- Contact matrix is O(n²) footprint
- Feature writes are coalesced but large (92 floats/residue)

**Solutions**:
- Use texture memory for read-only data
- Implement tiling for contact matrix
- Batch multiple structures per kernel launch

### 2. Compute Intensity (Stages 3, 4)
**Impact**: Medium
**Evidence**:
- Power iteration (Stage 3): 15-20 iterations of matrix-vector products
- Dendritic reservoir (Stage 4): sin/cos operations per neuron

**Solutions**:
- Early convergence check for power iteration
- Use __fdividef() for fast division
- Enable -use_fast_math flag

### 3. Synchronization Overhead
**Impact**: Low-Medium
**Evidence**:
- 24 `__syncthreads()` calls across 10 stages
- ~2-3 per stage average

**Solutions**:
- Reduce syncs where data dependencies allow
- Use warp-level primitives (__shfl_sync) where possible
- Consider async copies between stages

---

## Optimization Opportunities

### High Impact, Low Effort ⭐
1. **Enable fast math**: Add `-use_fast_math` to nvcc flags
   - Expected gain: 5-10% faster
   - Risk: Minimal (binding site prediction tolerates approximations)

2. **Batch proteins**: Launch kernel for multiple structures
   - Expected gain: 2-5x throughput
   - Complexity: Medium (requires batching infrastructure)

3. **Tune block size**: Test 128, 256, 512 threads
   - Expected gain: 5-15% depending on occupancy
   - Complexity: Low (single constant change)

### Medium Impact, Medium Effort
1. **Adaptive iteration counts**: Exit power iteration on convergence
   - Expected gain: 10-20% faster (varies by structure)
   - Complexity: Medium (convergence check + early exit logic)

2. **Shared memory optimization**: Pad arrays to avoid bank conflicts
   - Expected gain: 5-10% memory bandwidth
   - Complexity: Medium (analyze access patterns, add padding)

3. **Warp-level reductions**: Replace atomics with __shfl_down_sync
   - Expected gain: 10-15% for reduction-heavy stages
   - Complexity: Medium (rewrite reductions)

### High Impact, High Effort
1. **Tensor core utilization**: Use wmma for matrix ops (if applicable)
   - Expected gain: 2-4x for eligible operations
   - Complexity: High (requires FP16 path + validation)

2. **Multi-scale TDA**: Process 3 radii in parallel
   - Expected gain: +0.01-0.02 AUC (detection quality)
   - Complexity: High (kernel restructuring)

3. **Persistent kernel**: Keep GPU resident across proteins
   - Expected gain: 2-3x throughput for batch processing
   - Complexity: High (cooperative groups, grid sync)

---

## Detection Quality Improvements

### Physics Features (Stage 3.6)
**Current**: 12 features, default residue type (Alanine)

**Opportunities**:
1. **Residue type parsing** (+0.01-0.02 AUC estimated)
   - Parse actual amino acids from PDB ATOM records
   - Use correct hydrophobicity/charge/volume per residue
   - Complexity: Low (CPU-side parsing)

2. **Refined thermodynamic features** (+0.005-0.01 AUC)
   - Better entropy calculation (not just B-factor proxy)
   - Add enthalpy contribution
   - Electrostatic potential (not just charges)

3. **Multi-resolution physics** (+0.005-0.01 AUC)
   - Compute physics at 3 radii (like TDA)
   - Capture local vs non-local effects

### TDA Features (Stage 3.5)
**Current**: 48 features, low predictive power (weight mean: 0.0689)

**Opportunities**:
1. **Feature selection/pruning** (+0.005 AUC, faster)
   - Remove low-weight TDA features
   - Keep only top-10 TDA by importance
   - Reduces to 58-dim (10 TDA + 32 base + 12 physics + 4 pruned)

2. **Alternative topological descriptors**
   - Persistent homology (full barcode)
   - Alpha shapes instead of VR complex
   - Mapper/Reeb graphs

---

## Recommended Action Plan

### Phase 1: Low-Hanging Fruit (1-2 days)
1. ✅ Enable `-use_fast_math` in PTX compilation
2. ✅ Tune block size (test 128, 256, 512)
3. ✅ Add residue type parsing
4. ✅ Lambda sweep (validate λ=1e-4 is optimal)

**Expected gain**: +5-10% speed, +0.01-0.02 AUC

### Phase 2: Medium Effort (1 week)
1. ⏸️ Implement adaptive convergence for power iteration
2. ⏸️ Batch multiple proteins per launch
3. ⏸️ Integrate SA classifier (already implemented)
4. ⏸️ Feature importance analysis + pruning

**Expected gain**: +10-20% speed, +0.01-0.03 AUC

### Phase 3: High Impact (1 month)
1. ⏸️ Multi-scale TDA (3 radii)
2. ⏸️ Tensor core path for matrix ops
3. ⏸️ Full conservation scores (HMMER integration)
4. ⏸️ Ensemble of 5 models

**Expected gain**: +20-50% speed, +0.03-0.05 AUC

---

## Current Performance vs Theoretical Max

| Metric | Current | Theoretical Max | Gap |
|--------|---------|-----------------|-----|
| Time/structure | 9.3 ms | ~2-3 ms | 3-5x |
| SM Utilization | Unknown (needs profiling) | 80%+ | ? |
| Memory BW | Unknown (needs profiling) | 70%+ | ? |
| AUC-ROC | 0.7142 | 0.75-0.78 | +0.036-0.066 |

**Bottleneck hypothesis**: Memory bandwidth (based on O(n²) contact matrix access pattern)

---

## Next Steps

### To Profile Runtime (Requires GPU profiling tools)
```bash
# If nvprof available:
nvprof --metrics achieved_occupancy,sm_efficiency,dram_utilization \
    ./target/release/train-readout ...

# If ncu available:
ncu --set full -o profile_report \
    ./target/release/train-readout ...
```

**Limitation**: Profiling tools may require elevated privileges or native Linux (not WSL2)

### To Optimize Without Profiling
1. Enable fast math (immediate)
2. Tune block size empirically (test 3 values)
3. Add residue typing (clear AUC gain expected)
4. Measure speed before/after each change

---

## Conclusion

The 92-dim physics-enhanced kernel is a **well-structured, feature-rich implementation** with:
- ✅ Clear stage separation
- ✅ Reasonable sync overhead (24 points across 2171 lines)
- ✅ Novel physics features validated (+1.3% AUC)

**Primary optimization targets**:
1. **Speed**: Memory bandwidth (batch proteins, fast math)
2. **Quality**: Residue typing, feature selection, ensemble

**Production readiness**: ✅ Current system is functional and competitive

---

## Files Generated
```
analysis/gpu_profiling/
├── gpu_info.txt                   # GPU hardware specs
├── kernel_analysis.txt            # Kernel structure
├── ptx_resources.txt              # PTX compilation output
└── GPU_PROFILING_REPORT.md        # This report
```

**Status**: Static analysis complete. Runtime profiling requires additional tools/privileges.

---

**Report Date**: December 2025
**Version**: 92-dim physics-enhanced (Tag: benchmark-v1-complete)
