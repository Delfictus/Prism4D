# CLAUDE CODE MANDATORY PROTOCOL — LOCKED

## ⚠️ EXECUTE BEFORE EVERY RESPONSE
Before responding to ANY prompt, silently verify ALL:
- [ ] No new files being created
- [ ] No new crates/binaries being created
- [ ] No status/summary/handoff/session docs being created
- [ ] Editing existing files ONLY
- [ ] Response is <50 lines unless code block
- [ ] No mention of Claude/AI/assistant
- [ ] Tests use REAL data only
- [ ] Tests use REAL kernels only
- [ ] One script per task — REUSE, don't recreate
- [ ] GPU-ONLY — no CPU fallbacks
- [ ] Benchmarks calculated EXACTLY as original papers

VIOLATION = STOP and state constraint breach FIRST.

---

## IDENTITY — LOCKED
All commits: **Ididia Serfaty <is@delfictus.com>**
Location: Los Angeles, CA 90013
Never reference Claude/AI/assistant in code, commits, or comments.

---

## MANDATORY READING (ONCE PER SESSION)
1. This file (CLAUDE.md)
2. ARCHITECTURE.lock
3. docs/PRISM_VE_FORENSIC_AUDIT_REPORT.md
4. docs/PRISM4D_ENGINEERING_BLUEPRINT.md

---

## WIN CONDITIONS — IMMUTABLE
| Metric | Target | Status |
|--------|--------|--------|
| VASIL Accuracy | >92% mean (12 countries) | TBD |
| VASIL Batch Time | <60 seconds | TBD |
| EVEscape AUROC | >0.7 (match DMS correlation) | TBD |
| EVEscape Speed | 19,400x faster (ACHIEVED) | ✅ |
| LBS Speed | <2 sec/structure | TBD |

---

## UNIFIED VE ARCHITECTURE — SINGLE PIPELINE

### Philosophy
ONE engine produces ALL features. Validation compares to BOTH benchmarks.
NO separate EVEscape path. NO separate VASIL path. ONE GPU-native pipeline.
Benchmarks calculated EXACTLY as original papers specify.

### Pipeline (GPU-ONLY, NO CPU FALLBACK)
```
PDB Structures + Mutations (data/VASIL/ByCountry/*)
         ↓
    Batch Converter/Optimizer
         ↓
    mega_fused_vasil_fluxnet.cu (GPU, L1 cache)
         ↓
    FluxNet RL (Q-learning, 6D state space)
         ↓
┌────────────────────────────────────────────────────────────┐
│              UNIFIED VE OUTPUT (SUPERSET)                  │
│                                                            │
│  EVEscape-comparable features:                             │
│  • Fitness score (EVE-equivalent from sequence evolution)  │
│  • Accessibility score (surface exposure)                  │
│  • Dissimilarity score (antibody binding disruption)       │
│  • ΔΔG binding free energy                                 │
│  • Escape score = fitness × accessibility × dissimilarity  │
│                                                            │
│  VASIL-comparable features:                                │
│  • Fold resistance per epitope class (10 classes)          │
│  • Cross-neutralization P_neut(x,y)                        │
│  • Antibody pharmacokinetics (time-decay)                  │
│  • γy = relative fitness (susceptibles prediction)         │
│  • Per-day rising/falling classification                   │
│                                                            │
│  PRISM-exclusive features:                                 │
│  • Temporal cycle predictions (Stage 8)                    │
│  • Neuromorphic spiking dynamics                           │
│  • Polycentric immunity (Stages 9-10)                      │
│  • Epidemiological projections (Stage 11)                  │
└────────────────────────────────────────────────────────────┘
         ↓
    VALIDATION ROUTER
         ↓
┌─────────────┬─────────────┬─────────────┐
│  EVEscape   │   VASIL     │  Extended   │
│  Validator  │  Validator  │  (PRISM)    │
└─────────────┴─────────────┴─────────────┘
         ↓
    unified_validation_report.json
```

---

## BENCHMARK METHODOLOGY — EXACT REPLICATION REQUIRED

### EVEscape Benchmark (Nature 2023)
**Paper**: Thadani et al., "Learning from prepandemic data to forecast viral escape"

**EXACT METHODOLOGY**:
1. **Escape Score** = P(fitness) × P(accessibility) × P(dissimilarity)
   - Fitness: EVE deep variational autoencoder on historical sequences
   - Accessibility: Surface exposure from structure (Å² accessible)
   - Dissimilarity: Amino acid property change at antibody contact sites

2. **Validation Metric**: AUROC and AUPRC against DMS escape data
   - Compare predicted escape scores vs experimental DMS escape fractions
   - Normalize AUPRC by null model (fraction of observed escapes)
   - DMS data: 836 antibodies across 10 epitope classes

3. **Data Input** (EXACT as paper):
   - Pre-pandemic coronavirus sequences (sarbecoviruses + seasonal)
   - Structure: Spike RBD (PDB structures of antibody complexes)
   - DMS escape fractions from Bloom lab + Xie lab datasets

4. **Speed Comparison**:
   - EVEscape: ~1 mutation/second (CPU-bound)
   - PRISM: >300 structures/second (GPU) = 19,400x faster

### VASIL Benchmark (Nature 2025)
**Paper**: Raharinirina et al., "SARS-CoV-2 evolution on a dynamic immune landscape"

**EXACT METHODOLOGY** (Extended Data Fig 6a):
1. **γy Calculation** (relative fitness):
```
   γy(t) = Sy(t) / Σx Sx(t)
   
   Where:
   Sy(t) = Pop - Σx ∫ πx(s) · I(s) · Pneut(t-s, x, y) ds
   
   Components:
   - Pop = population size
   - πx(s) = frequency of variant x at time s
   - I(s) = incidence (infections) at time s
   - Pneut(t-s, x, y) = neutralization probability (x protects against y)
```

2. **Accuracy Metric** (EXACT):
   - Partition frequency curve πy into RISING (1) and FALLING (-1) DAYS
   - Predict sign(γy) for each day
   - Compare sign(γy) with sign(Δfreq)
   - Exclude negligible changes (<5% relative change)
   - Calculate per-country accuracy
   - Report MEAN across all 12 countries

3. **Cross-Neutralization** (from DMS data):
   - 836 antibodies aggregated into 10 epitope classes (A, B, C, D1, D2, E1, E2, E3, F1, F2, F3)
   - Fold resistance = how much more antibody needed to neutralize mutant
   - Epitope class weights from DMS escape fractions

4. **Antibody Pharmacokinetics**:
   - Rise: 1-2 weeks after antigen exposure
   - Decay: Slow exponential decay
   - Time-dependent protection probability

5. **Data Input** (EXACT as paper):
   - 12 countries: Germany, USA, UK, Japan, Brazil, France, Canada, Denmark, Australia, Sweden, Mexico, South Africa
   - GInPipe incidence reconstruction (φ estimates)
   - GISAID sequence data for variant frequencies
   - Per-country accuracy targets from Extended Data Fig 6a

6. **Target Accuracies by Country** (from paper):
   | Country | VASIL Accuracy |
   |---------|---------------|
   | Germany | 94% |
   | Denmark | 93% |
   | UK | 93% |
   | France | 92% |
   | Sweden | 92% |
   | Canada | 91% |
   | USA | 91% |
   | Japan | 90% |
   | Australia | 90% |
   | Brazil | 89% |
   | Mexico | 88% |
   | South Africa | 87% |
   | **MEAN** | **92%** |

---

## GPU-CENTRIC MANDATE — ABSOLUTE

### NO CPU FALLBACKS — EVER
- All computation on GPU
- All data stays in GPU memory
- L1 cache for hot paths
- No CPU validation paths disguised as "reference"
- No hybrid CPU/GPU modes

### If GPU path fails:
1. STOP
2. Report the failure
3. FIX the GPU path
4. Do NOT fall back to CPU

---

## BANNED ACTIONS — EVERY PROMPT
❌ "Minimal episodes" or "quick test" or "baseline test" — ALWAYS full parameters
❌ Abbreviated runs to "verify wiring" — verify by reading code, not running shortcuts
❌ Any test with reduced parameters (episodes, learning rate, batch size)
❌ "Sanity check" runs — either full benchmark or no run
❌ Create new files (edit existing ONLY)
❌ Create new crates or binaries
❌ Create markdown docs (especially *SESSION*, *STATUS*, *HANDOFF*, *SUMMARY*, *COMPLETE*, *REPORT*)
❌ Mock data or synthetic results
❌ Silent fallbacks that mask failures
❌ Verbose explanations (be TERSE)
❌ Refactor working code without explicit request
❌ Architecture changes without explicit approval
❌ Duplicate existing functionality
❌ Add dependencies without explicit approval
❌ Create new test scripts (edit existing only)
❌ CPU fallbacks of any kind
❌ Separate EVEscape pipeline (use unified)
❌ Separate VASIL pipeline (use unified)
❌ Hardcoded values
❌ Placeholder constants
❌ Erroneous/dummy values
❌ Approximate benchmark calculations (must be EXACT)
❌ Skip any step in benchmark methodology

---

## REQUIRED ACTIONS — EVERY PROMPT
✅ Edit existing files only
✅ Fail loudly with clear error messages
✅ Run `cargo check` before claiming success
✅ Cite specific file:line:column when discussing code
✅ Keep responses concise (<50 lines unless code block)
✅ One solution, not multiple options
✅ Execute, don't explain
✅ GPU-only execution
✅ Real data only
✅ Real kernels only
✅ Benchmark calculations EXACTLY as papers specify

---

## TESTING PROTOCOL — LOCKED

### ONE SCRIPT RULE
- If a test script exists, EDIT IT — never create a new one
- If revising a script, revise IN PLACE — no copies
- Script names are FROZEN: do not rename or duplicate

### REAL DATA ONLY
- VE tests MUST use: data/VASIL/ByCountry/*
- LBS tests MUST use: data/cryptobench_2025_GOLDEN_pockets.csv
- NO synthetic data, NO generated data, NO mock data

### REAL KERNELS ONLY
- VE tests MUST use: mega_fused_vasil_fluxnet.cu
- LBS tests MUST use: mega_fused_pocket_kernel.cu
- NO stub kernels, NO CPU paths

### EXISTING TEST ENTRY POINTS (USE THESE ONLY)
| Purpose | Location | Command |
|---------|----------|---------|
| VE benchmark (unified) | crates/prism-ve-bench/src/bin/train_fluxnet_ve.rs | `cargo run -p prism-ve-bench --bin train-fluxnet-ve` |
| IC50 verification | crates/prism-ve-bench/src/bin/verify_ic50_wiring.rs | `cargo run -p prism-ve-bench --bin verify-ic50-wiring` |
| LBS benchmark | crates/prism-lbs/src/bin/benchmark.rs | `cargo run -p prism-lbs --bin benchmark` |

---

## CURRENT OBJECTIVE
1. Restore VASIL accuracy to 77.4% baseline (current gap: 43.9% → 77.4%)
2. Using UNIFIED pipeline with mega_fused_vasil_fluxnet.cu
3. Calculate γy EXACTLY as VASIL paper (Extended Data Fig 6a methodology)
4. Calculate escape scores EXACTLY as EVEscape paper (fitness × accessibility × dissimilarity)
5. Single run → outputs ALL features → validates against BOTH benchmarks
6. GPU-only. No CPU fallbacks.

---

## VASILExactMetricComputer — REQUIRED IMPLEMENTATION

Must implement EXACTLY:
```rust
// Per Extended Data Fig 6a
fn compute_vasil_accuracy(
    gamma_predictions: &[f32],      // γy for each day
    frequency_changes: &[f32],       // Δfreq for each day
    threshold: f32,                  // 5% relative change threshold
) -> f32 {
    // 1. Partition into rising/falling days
    // 2. Exclude negligible changes (<5% relative)
    // 3. Compare sign(γy) with sign(Δfreq)
    // 4. Return accuracy as fraction correct
}

fn compute_gamma_y(
    population: f32,
    variant_frequencies: &[f32],     // πx(s)
    incidence: &[f32],               // I(s)
    p_neut_matrix: &[[f32]],         // Pneut(t-s, x, y)
    time_points: &[f32],
) -> f32 {
    // Sy(t) = Pop - Σx ∫ πx(s) · I(s) · Pneut(t-s, x, y) ds
    // γy(t) = Sy(t) / Σx Sx(t)
}
```

---

## Q-TABLE REQUIREMENTS
- SAVE Q-table after each episode/generation
- Location: validation_results/fluxnet_ve_qtable.json
- Must persist learning across runs
- Load existing Q-table if present
- Learning rate: 0.3
- Episodes: 100+

---

## CANONICAL KERNEL INVENTORY — DO NOT DUPLICATE

### Mega-Fused Kernels (5)
| Kernel | Purpose |
|--------|---------|
| mega_fused_vasil_fluxnet.cu | UNIFIED VE (EVEscape + VASIL) |
| mega_fused_batch.cu | General batch processing |
| mega_fused_pocket_kernel.cu | LBS pocket detection |
| mega_fused_lbs_complete.cu | LBS complete pipeline |
| prism_lbs_fused.cu | LBS fused operations |

### VE-Swarm Kernels (3)
| Kernel | Function |
|--------|----------|
| ve_swarm_agents.cu | 32-agent swarm inference |
| ve_swarm_dendritic_reservoir.cu | Neuromorphic reservoir |
| ve_swarm_temporal_conv.cu | Temporal convolution |

---

## DATA LOCATIONS — CANONICAL

| Data | Path |
|------|------|
| VASIL 12-country | data/VASIL/ByCountry/ |
| CryptoBench golden | data/cryptobench_2025_GOLDEN_pockets.csv |
| Trained Q-table | validation_results/fluxnet_ve_qtable.json |
| Thresholds | validation_results/fluxnet_threshold_optimized.json |
| PTX kernels | kernels/ptx/*.ptx |

---

## RESPONSE FORMAT — ENFORCED
```
[ACTION]: <what you're doing in ≤10 words>
[FILE]: <path:line if editing>
[CODE]: <code block if any>
[RESULT]: <outcome in ≤10 words>
```

No preamble. No summary. No "Let me explain." Execute.

---

**THIS FILE IS IMMUTABLE. DO NOT MODIFY.**
