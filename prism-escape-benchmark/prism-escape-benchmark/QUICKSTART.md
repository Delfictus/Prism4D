# PRISM-Viral Quickstart - Everything is Ready!

## ✅ SETUP COMPLETE

**All data downloaded and processed:**
- 43,499 mutation-antibody pairs from Bloom Lab
- 170 unique SARS-CoV-2 RBD mutations (train: 136, test: 34)
- 5 viral structures (Wuhan, Delta, Omicron)
- EVEscape baselines for comparison

---

## 🚀 RUN YOUR FIRST TEST (5 minutes)

```bash
# 1. Go to benchmark directory
cd prism-escape-benchmark

# 2. Verify data is ready
echo "Checking downloaded data..."
ls -lh data/raw/structures/6m0j.pdb
wc -l data/processed/sars2_rbd/test.csv

# 3. Test PRISM on SARS-CoV-2 RBD
cd ../PRISM
./target/release/prism-lbs --pdb ../prism-escape-benchmark/data/raw/structures/6m0j.pdb
```

**Expected:** PRISM processes the RBD structure successfully

---

## 📊 WHAT YOU HAVE

### **Data Inventory:**
- Bloom DMS: 43,499 records → 170 mutations
- Top escape sites: E484 (748 tests), K417, N501
- Train/test splits ready
- EVEscape baselines: AUPRC 0.53 (to beat)

### **Your Targets:**
- AUPRC: 0.60-0.70 (beat SOTA by 7-17%)
- Speed: 1000 mutations/second (450× faster)
- Impact: Real-time pandemic surveillance

---

## 🎯 CRITICAL VALIDATION TEST

**Run this to validate the approach:**

```python
# Test: Do PRISM physics features correlate with experimental escape?
# Located at: scripts/test_physics_correlation.py

# If correlation > 0.60: ✅ Physics works, proceed!
# If correlation < 0.50: ❌ Need different approach
```

---

**Next:** Integrate PRISM feature extraction → Run physics test → Train ML model

**Timeline to competitive results: 2-4 months**

**Funding potential: $275K-$2M**
