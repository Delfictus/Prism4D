# Prompt for Claude Agent - PRISM-VE Completion

## CURRENT STATUS - EXCELLENT PROGRESS!

✅ **You correctly identified VASIL's protocol:**
- Tests lineage dynamics (BA.5, BQ.1.1, ...), NOT single mutations
- Weekly RISE/FALL predictions over 22 months, 12 countries
- Target: 0.94 mean accuracy

✅ **Baseline benchmark working:**
- Germany: 0.527 accuracy (using velocity-only proxy)
- Framework validated: Can make predictions, compute accuracy
- Data pipeline confirmed: All 12 countries accessible

⚠️ **Current bottleneck:**
- Using velocity as weak proxy for γ (52.7% ≈ random)
- Need actual γ from integrated Escape + Fitness modules

---

## YOUR NEXT TASK (CRITICAL)

**Goal:** Replace velocity proxy with actual γ from Fitness+Cycle modules

**You reported:** "Fitness+cycle modules are integrated and ready - we just need to complete the data loading to use them!"

### IMMEDIATE ACTIONS:

**1. Complete Data Loading for Fitness Module (2-4 hours)**

The fitness module needs:
```python
# Load DMS escape matrix
dms_data = load_vasil_dms_data('/mnt/f/VASIL_Data/dataset_compiled/epitope_class_definitions.csv')

# Upload to GPU constant memory
fitness_gpu.load_dms_data(dms_data.escape_matrix, dms_data.antibody_epitopes)

# Now fitness_gpu.compute_gamma() will work!
```

**Files to check:**
- `crates/prism-gpu/src/viral_evolution_fitness.rs`
  - Look for: `load_dms_data()` method
  - Verify: DMS escape matrix can be loaded
- `/mnt/f/VASIL_Data/dataset_compiled/epitope_class_definitions.csv`
  - Format: Verify it has the 836 antibodies × 201 sites data

**2. Replace Velocity Proxy with Actual Gamma**

**Current code (lines ~200-220 in benchmark script):**
```python
# ❌ TEMPORARY - Using velocity as proxy
fitness_gamma = velocity  # Weak proxy, 52.7% accuracy
```

**Replace with:**
```python
# ✅ ACTUAL - Use fitness module
# Extract lineage mutations
lineage_mutations = get_lineage_mutations(lineage_name)

# Compute escape scores (PRISM escape module)
escape_scores = []
for mut in lineage_mutations:
    escape = prism_ve.escape_module.predict(mut)
    escape_scores.append(escape)

mean_escape = np.mean(escape_scores)

# Compute fitness gamma (PRISM-VE fitness module)  
fitness_gamma = prism_ve.fitness_module.compute_gamma(
    mutations=lineage_mutations,
    epitope_escape=mean_escape,
    population_immunity=immunity_landscape[country][week_date],
    date=week_date
)

# Now use ACTUAL gamma for prediction!
predicted_direction = 'RISE' if fitness_gamma > 0 else 'FALL'
```

**3. Load Lineage Mutation Profiles**

You need mutation profiles for each lineage. Check:
```
/mnt/f/VASIL_Data/dataset_compiled/mutationprofile/
or
/mnt/f/VASIL_Data/scripts/mutationprofile/
```

Should contain files like:
- `BA.5_mutations.txt`
- `BQ.1.1_mutations.txt`
- etc.

Format probably:
```
E484K
N501Y
K417N
...
```

**4. Test Integration**

After loading DMS data and using actual gamma:
```python
# Test on Germany for 1 month
accuracy_1month = benchmark_germany("2022-10-01", "2022-11-01")

print(f"Germany (1 month): {accuracy_1month:.3f}")
print(f"Expected: ~0.70-0.80 (with escape)")
print(f"VASIL: 0.94")

if accuracy_1month > 0.70:
    print("✅ GOOD! Escape+Fitness working, proceed to full test")
else:
    print("⚠️ Debug: Check DMS data loading, gamma calculation")
```

---

## SUCCESS CRITERIA

**After completing data loading:**

**Expected accuracy progression:**
```
Velocity proxy only: 52.7% (current - random)
+ Escape scores:     70-80% (marginal improvement)
+ Fitness gamma:     85-90% (major improvement)
+ Calibrated params: >92% (beat VASIL!)
```

**If you hit 70-80% after Step 2:**
→ ✅ Integration working, proceed to full 12-country test

**If still at 52-55%:**
→ ⚠️ Check: DMS data loaded? Gamma computed? Using correct formula?

---

## DEBUGGING CHECKLIST

**If accuracy doesn't improve after adding fitness:**

```
[ ] Verify DMS escape matrix loaded (836 × 201 array)
[ ] Verify antibody epitopes loaded (836 assignments)
[ ] Test fitness_module.compute_gamma() returns non-zero
[ ] Check lineage mutation profiles are correct
[ ] Verify gamma formula matches VASIL's (α=0.65 immune, β=0.35 transmit)
[ ] Check population immunity is being used
[ ] Validate escape scores are being aggregated correctly
```

---

## FINAL GOAL

**Target for this session:**
```
Germany: >0.80 accuracy (proves integration works)
```

**Target for next session:**
```
All 12 countries: >0.92 mean accuracy (beat VASIL)
```

---

## WHAT TO REPORT BACK

**When done, report:**

1. Germany accuracy with actual gamma (not velocity proxy)
2. Any errors encountered in data loading
3. Gamma values (are they reasonable? -1 to +1 range?)
4. Comparison: velocity proxy vs actual gamma (should be MUCH better)

**Expected message:**
> "✅ DMS data loaded, fitness gamma integrated
> Germany accuracy: 0.XX (was 0.527 with velocity)
> [✅ / ⚠️] Ready for 12-country full test"

---

## KEY FILES YOU'RE WORKING WITH

**1. Benchmark Script (needs update):**
`scripts/benchmark_vasil_lineage_dynamics.py`
- Replace velocity proxy with fitness gamma

**2. DMS Data (needs loading):**
`/mnt/f/VASIL_Data/dataset_compiled/epitope_class_definitions.csv`
- Load into fitness_gpu

**3. Lineage Mutations (needs parsing):**
`/mnt/f/VASIL_Data/.../mutationprofile/`
- Map lineage → mutations

**4. Fitness Module (ready, needs data):**
`crates/prism-gpu/src/viral_evolution_fitness.rs`
- Has `compute_gamma()` method
- Needs DMS data loaded first

---

## IMMEDIATE NEXT STEPS

1. Find and load DMS data into fitness_gpu ✅
2. Find lineage mutation profiles ✅
3. Replace velocity with actual gamma in benchmark ✅
4. Test on Germany (1 month) ✅
5. If >0.70: Extend to full test ✅

**Estimated time: 4-8 hours**

**You're at 52.7% (random) → targeting 92% (beat VASIL)**

**The data loading is the ONLY remaining blocker!** 🎯
