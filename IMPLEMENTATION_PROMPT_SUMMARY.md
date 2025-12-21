# Implementation Prompt Merge - Summary

## What I Created

I merged your existing PRISM-VE implementation (77.4% accuracy) with the new "strict benchmark" requirements into a single world-class implementation prompt:

**File:** `WORLD_CLASS_VASIL_BENCHMARK_IMPLEMENTATION_PROMPT.md`

---

## Key Improvements Over Both Original Prompts

### From Your Implementation (Kept & Enhanced)
✅ Real VASIL methodology (92% compliant)  
✅ GPU acceleration (4,772 structures/sec)  
✅ Real DMS data (835 antibodies)  
✅ Zero data leakage verification  
✅ 77.4% verified accuracy baseline  

### From New Prompt (Added)
✅ 75-PK envelope fix (PRIMARY BLOCKER for 92% accuracy)  
✅ Strict mode fail-fast  
✅ Full audit manifest (git hash, SHA256s)  
✅ CI tests (leakage, determinism, golden files)  
✅ Constants centralization  

### Unique to Merged Version
✅ Specific diagnosis of WHY at 77.4% (GPU averages 75 PKs)  
✅ Real temporal data extraction (polycentric enabler)  
✅ Phase-by-phase implementation with test criteria  
✅ Clear success metrics (85% minimum, 90% stretch)  
✅ Forensic-audit-informed priorities  

---

## Implementation Phases (Prioritized by Impact)

### Phase 0: Repository Assessment (2 hours)
**Before any coding** - understand current state
- Read forensic audit, session summary, code
- Produce state report with exact file/line references
- **Gate:** Cannot proceed without complete Phase 0 report

### Phase 1: Fix 75-PK Envelope (6-8 hours) ⭐ HIGHEST PRIORITY
**The accuracy blocker** - GPU averages 75 PKs instead of returning all
- Modify GPU download to return `Vec<[f64; 75]>`
- Implement envelope decision rule (min/max across 75)
- Exclude undecided predictions (envelope crosses zero)
- **Expected Gain:** +5-10% accuracy (77.4% → 82-87%)
- **Test:** Golden file test with Rising/Falling/Undecided cases

### Phase 2: Extract Real Temporal Data (4-6 hours)
**Polycentric enabler** - replace placeholder constants
- Extract `time_since_infection` from phi peaks
- Extract `freq_history_7d` from frequency CSVs
- Extract `current_freq` from batch metadata
- **Expected Gain:** +2-5% accuracy (wave features contribute signal)
- **Test:** Assert non-constant variance, reasonable value ranges

### Phase 3: Strict Mode + Manifest (3-4 hours)
**Reproducibility** - production-grade rigor
- Single CLI: `vasil-benchmark --config bench.toml --strict`
- Fail-fast on placeholders/approximations
- Full manifest: git hash, config hash, data SHA256s
- **Expected Gain:** Scientific credibility, not accuracy
- **Test:** Two runs → identical manifest hashes

### Phase 4: Centralize Constants + CI Tests (3-4 hours)
**Prevent regressions** - single source of truth
- Create `constants.rs` module
- CI test fails if constants duplicated
- Leakage canary, determinism, strict mode tests
- **Expected Gain:** Long-term maintainability
- **Test:** CI pipeline green

### Phase 5: Documentation (2-3 hours)
**Publication-ready** - README_BENCHMARK.md
- One-command reproduction steps
- Scientific guarantees (zero leakage, determinism)
- Known limitations
- Citation info

---

## Critical Differences from Original Prompts

### My Prompt vs. Your Current Implementation

| Aspect | Your Implementation | Merged Prompt |
|--------|---------------------|---------------|
| 75-PK Envelope | Attempted 3×, failed (averages) | **FIX REQUIRED** - Detailed steps to preserve all 75 |
| Temporal Data | Placeholders (constants) | **EXTRACT REQUIRED** - From phi/freq CSVs |
| Strict Mode | Env var gate only | **ADD CLI** - `--strict` flag with fail-fast |
| Manifest | None | **ADD FULL** - Git hash, SHA256s, exclusions |
| Constants | 4 locations | **CENTRALIZE** - Single `constants.rs` |

### My Prompt vs. New "Strict Benchmark" Prompt

| Aspect | New Prompt | Merged Prompt |
|--------|------------|---------------|
| Domain Knowledge | Generic VASIL spec | **PRISM-VE SPECIFIC** - Knows current 77.4% state |
| GPU Pipeline | Not mentioned | **PRESERVE** - 4,772 structs/sec, 75× speedup |
| Polycentric | Not mentioned | **ENABLE** - Real temporal data extraction |
| Prioritization | All phases equal | **PRIORITIZED** - Phase 1 = accuracy blocker |
| Success Metrics | "92% or fail" | **REALISTIC** - 85% minimum, 90% stretch |

---

## Why This Merged Version is Best

### 1. Accuracy-Focused Prioritization
**Problem:** Your implementation stuck at 77.4% despite multiple attempts  
**Root Cause:** GPU averages 75 PK values before CPU receives them  
**Solution:** Phase 1 explicitly fixes this (detailed GPU download code)  
**Impact:** This ONE fix likely → 82-87% accuracy

### 2. Builds on Existing Success
**Problem:** New prompt would require rewriting working GPU pipeline  
**Strength:** Your 92% VASIL compliance, real DMS data, zero leakage  
**Solution:** Keep your foundation, add rigor from new prompt  
**Result:** Best of both worlds

### 3. Realistic Timeline
**New prompt:** Implied "implement everything or fail"  
**Your roadmap:** Multiple competing priorities  
**Merged:** 16-24 hours over 5 days, phased validation  
**Gated:** Cannot proceed to Phase N+1 until Phase N tested

### 4. Scientific Integrity + Performance
**New prompt:** Rigor but no mention of GPU constraints  
**Your impl:** GPU-first but incomplete rigor  
**Merged:** Maintains 4,772 structs/sec WHILE adding strict mode  
**No compromise:** Both speed and integrity

### 5. Actionable from Day 1
**New prompt:** Generic "implement VASIL"  
**Your docs:** "77.4% is ceiling, needs architectural redesign"  
**Merged:** **EXACT FILES/LINES** to change with before/after code  
**Example:** "mega_fused_batch.rs:1910 - change `vec![30.0; n]` to `compute_time_since_infection_from_phi()`"

---

## Success Criteria (Clear Pass/Fail)

### Minimum Viable (Publication-Ready)
- [ ] Accuracy ≥ 85% (within 7pp of VASIL 92%)
- [ ] Strict mode passes (100% VASIL-exact)
- [ ] Zero data leakage (CI-verified)
- [ ] Deterministic (manifest hash identical across runs)
- [ ] Documentation complete (README_BENCHMARK.md)

### Stretch Goal (Match VASIL)
- [ ] Accuracy ≥ 90% (within 2pp of VASIL 92%)
- [ ] All 12 countries functional (no fallback data)
- [ ] Polycentric features contribute (+3-5% gain)
- [ ] Paper-ready figures + tables

---

## How to Use This Prompt

### Step 1: Read the Full Prompt
Open `WORLD_CLASS_VASIL_BENCHMARK_IMPLEMENTATION_PROMPT.md` and read Phase 0.

### Step 2: Execute Phase 0 (Assessment)
Do NOT write code yet. Analyze repository state and produce Phase 0 Report.

### Step 3: Implement Phases 1-4 in Order
Each phase has:
- **Objective** - What it fixes
- **Expected Impact** - Accuracy gain or other benefit
- **Implementation** - Exact code changes with file:line
- **Success Criteria** - How to verify it worked

### Step 4: Test Each Phase Before Proceeding
- Phase 1 → Run benchmark → Verify accuracy improved to 82-87%
- Phase 2 → Check variance > 0 → Verify wave features non-constant
- Phase 3 → Two runs → Verify identical manifest hashes
- Phase 4 → CI green → Verify no leakage/determinism regressions

### Step 5: Document Results (Phase 5)
Update README_BENCHMARK.md with actual accuracy achieved.

---

## Estimated Outcomes

### Conservative (High Confidence)
- **Accuracy:** 82-87% (Phase 1 + 2)
- **Rigor:** Full strict mode, manifest, CI tests
- **Timeline:** 4 days
- **Publication:** Suitable for arXiv/bioRxiv

### Optimistic (Moderate Confidence)
- **Accuracy:** 88-92% (Phase 1 + 2 + parameter tuning)
- **Rigor:** Same as conservative
- **Timeline:** 5 days + 2 days tuning
- **Publication:** Suitable for peer-review journal

### Stretch (Possible but Uncertain)
- **Accuracy:** 92-95% (Beat VASIL if Phase 4 + UK insights synergize)
- **Rigor:** Same + ablation studies
- **Timeline:** 2 weeks
- **Publication:** Nature/Science consideration

---

## Questions This Prompt Answers

### "Why am I at 77.4% instead of 92%?"
**Answer:** GPU averages 75 PK values. You're using mean immunity instead of envelope decision rule. Fix in Phase 1.

### "What should I do first?"
**Answer:** Phase 1 (75-PK envelope). It's your accuracy blocker. Expected +5-10pp gain.

### "How do I know if it's working?"
**Answer:** Each phase has explicit success criteria with tests. Can't fake passing them.

### "What if I can't reach 92%?"
**Answer:** 85% is publication-ready (conservative target). 90% is excellent (stretch goal). Prompt includes realistic ranges.

### "How do I prove scientific integrity?"
**Answer:** Strict mode + manifest. If `strict_mode: true` in manifest, results are trustworthy.

### "What about GPU constraints?"
**Answer:** Prompt preserves all GPU optimizations. No CPU fallbacks. Maintains 4,772 structs/sec.

---

## Files Created

1. **WORLD_CLASS_VASIL_BENCHMARK_IMPLEMENTATION_PROMPT.md** (main prompt)
2. **IMPLEMENTATION_PROMPT_SUMMARY.md** (this file)

---

## Next Steps

**Immediate:** Read Phase 0 of the main prompt and produce Phase 0 Report.

**Do NOT start coding until Phase 0 Report is complete.**

Phase 0 Report template:
```markdown
## Phase 0: Repository State Assessment

### Evaluation Entrypoint
- File: crates/prism-ve-bench/src/main.rs
- Lines: 1006-1109
- Trigger: PRISM_ENABLE_VASIL_METRIC=1
- Current: Single entrypoint ✅

### 75-PK Envelope Status
- Implementation: vasil_exact_metric.rs:148-171
- GPU Output: Vec<f64> (mean only) ❌
- Issue: GPU averages before CPU receives data
- Evidence: SESSION_FINAL_SUMMARY line 153-170

[... complete rest of template ...]
```

Once Phase 0 Report is verified accurate, proceed to Phase 1 implementation.

---

**Good luck! This merged prompt combines the best of both worlds: your domain expertise + production-grade rigor. 🚀**
