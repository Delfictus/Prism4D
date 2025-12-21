# PRISM-VE: Complete File Structure and Runtime Mapping

## DIRECTORY STRUCTURE (Complete)

```
prism-ve/
├── crates/                           # Existing Rust code (INHERITED)
│   ├── prism-core/                   # ✅ KEEP AS-IS
│   ├── prism-gpu/                    # ✅ KEEP AS-IS (mega_fused kernel)
│   ├── prism-lbs/                    # ✅ KEEP AS-IS (feature extraction)
│   └── prism-ve/                     # 🆕 NEW CRATE (add this)
│       ├── Cargo.toml
│       ├── src/
│       │   ├── lib.rs                # Main library interface
│       │   ├── escape.rs             # Escape module (wrap existing)
│       │   ├── fitness.rs            # 🆕 Fitness module
│       │   ├── cycle.rs              # 🆕 Cycle module  
│       │   ├── integration.rs        # 🆕 Unified predictor
│       │   └── utils.rs              # Helper functions
│       └── bin/
│           └── prism-ve.rs           # 🆕 CLI binary
│
├── prism-ve-python/                  # 🆕 Python API (NEW)
│   ├── setup.py
│   ├── pyproject.toml
│   ├── prism_ve/
│   │   ├── __init__.py
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── physics_engine.py    # 🆕 Wraps PRISM binary
│   │   │   ├── structure.py         # 🆕 PDB parsing
│   │   │   └── constants.py         # 🆕 AA properties, reference sequences
│   │   ├── modules/
│   │   │   ├── __init__.py
│   │   │   ├── escape.py            # 🆕 Escape prediction
│   │   │   ├── fitness.py           # 🆕 ΔΔG, stability
│   │   │   └── cycle.py             # 🆕 Temporal dynamics
│   │   ├── data/
│   │   │   ├── __init__.py
│   │   │   ├── loaders.py           # 🆕 DMS, GISAID loaders
│   │   │   └── splits.py            # 🆕 Temporal splits
│   │   └── utils/
│   │       ├── __init__.py
│   │       ├── metrics.py           # 🆕 AUPRC, etc.
│   │       └── visualization.py     # 🆕 Plotting
│   └── tests/
│       ├── test_escape.py
│       ├── test_fitness.py
│       └── test_cycle.py
│
├── data/                             # 🆕 Data directory
│   ├── raw/
│   │   ├── dms/                      # DMS datasets
│   │   │   ├── bloom_sars2/          # ✅ HAVE (43K mutations)
│   │   │   ├── doud_influenza/       # ✅ HAVE (10K mutations)
│   │   │   └── dingens_hiv/          # ✅ HAVE (13K mutations)
│   │   ├── structures/               # PDB files
│   │   │   ├── 6m0j.pdb             # ✅ HAVE (SARS-CoV-2 RBD)
│   │   │   ├── 1rv0.pdb             # ✅ HAVE (Influenza HA)
│   │   │   └── 7tfo_env.pdb         # ✅ HAVE (HIV Env)
│   │   └── gisaid/                   # 🆕 NEED TO ADD
│   │       ├── metadata.tsv          # Variant metadata
│   │       ├── sequences.fasta       # Sequences over time
│   │       └── frequencies.csv       # Position frequencies
│   ├── processed/
│   │   ├── sars2_features.npy        # ✅ HAVE (878×92 features)
│   │   ├── influenza_features.npy    # ✅ HAVE (2012×92 features)
│   │   ├── hiv_features.npy          # ✅ HAVE (1594×92 features)
│   │   └── gisaid_trajectories.parquet  # 🆕 Need to create
│   └── models/
│       ├── escape/
│       │   ├── sars2_escape.pkl      # ✅ CAN CREATE (trained model)
│       │   ├── influenza_escape.pkl
│       │   └── hiv_escape.pkl
│       ├── fitness/                  # 🆕 Will create
│       │   └── fitness_predictor.pkl
│       └── cycle/                    # 🆕 Will create
│           └── cycle_detector.pkl
│
├── configs/                          # 🆕 Configuration files
│   ├── prism_config.yaml             # PRISM binary paths, GPU settings
│   ├── model_config.yaml             # XGBoost hyperparameters
│   └── benchmark_config.yaml         # Validation settings
│
├── scripts/                          # 🆕 Execution scripts
│   ├── setup/
│   │   ├── download_gisaid.sh        # 🆕 Download GISAID data
│   │   ├── process_gisaid.py         # 🆕 Build frequency trajectories
│   │   └── prepare_data.py           # Data preprocessing
│   ├── training/
│   │   ├── train_escape.py           # ✅ HAVE (working)
│   │   ├── train_fitness.py          # 🆕 Train fitness module
│   │   └── train_cycle.py            # 🆕 Train cycle detector
│   ├── evaluation/
│   │   ├── benchmark_vs_evescape.py  # ✅ HAVE (3/3 viruses)
│   │   ├── test_fitness.py           # 🆕 Fitness validation
│   │   └── test_cycle.py             # 🆕 Cycle validation
│   └── deployment/
│       ├── build_docker.sh           # Docker container
│       └── deploy_api.py             # REST API server
│
├── notebooks/                        # 🆕 Analysis notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_escape_analysis.ipynb
│   ├── 04_fitness_analysis.ipynb
│   ├── 05_cycle_analysis.ipynb
│   └── 06_integration_demo.ipynb
│
├── docs/                             # Documentation
│   ├── API.md                        # API reference
│   ├── METHODS.md                    # Scientific methods
│   └── TUTORIAL.md                   # User guide
│
└── results/                          # Benchmark outputs
    ├── escape/                       # ✅ HAVE (Nature Methods results)
    ├── fitness/                      # 🆕 Will create
    └── cycle/                        # 🆕 Will create
```

---

## FILES TO CREATE (Priority Order)

### CRITICAL (Week 1-2: Fitness Module)

**1. Python Interface to PRISM Binary**
```
File: prism-ve-python/prism_ve/core/physics_engine.py
Purpose: Wrap existing PRISM binary for feature extraction
Dependencies:
  - ../PRISM/target/release/prism-lbs (existing binary)
  - subprocess, numpy, pandas
Touches: Nothing in main PRISM (read-only access to binary)
```

**2. Fitness Module**
```
File: prism-ve-python/prism_ve/modules/fitness.py
Purpose: Predict ΔΔG, stability, expression from PRISM features
Dependencies:
  - physics_engine.py
  - PRISM features (878×92 NPY files)
  - Optional: Pre-trained ΔΔG model (can train from scratch)
Data Needed:
  - DMS functional scores (ACE2 binding, expression) from Bloom/Starr
  - Optional: PDBbind for ΔΔG validation
```

**3. Fitness Training Script**
```
File: scripts/training/train_fitness.py
Purpose: Train fitness predictor on functional DMS data
Input: Bloom DMS with ACE2 binding scores
Output: models/fitness/fitness_predictor.pkl
```

### IMPORTANT (Week 3-5: Cycle Module)

**4. GISAID Data Integration**
```
Files:
  - scripts/setup/download_gisaid.sh
  - scripts/setup/process_gisaid.py
  
Purpose: Build position-level frequency trajectories
Input: GISAID metadata (download from gisaid.org)
Output: data/processed/gisaid_trajectories.parquet

Format:
  Columns: [position, date, frequency, mutation, variant_name]
  Example: [484, 2021-12-01, 0.05, E484K, Omicron]
```

**5. Cycle Module**
```
File: prism-ve-python/prism_ve/modules/cycle.py
Purpose: Detect evolutionary phase, predict emergence timing
Dependencies:
  - GISAID trajectories (time-series data)
  - Escape scores (from escape module)
  - Fitness scores (from fitness module)
Output: Phase classification, emergence predictions
```

**6. Cycle Training Script**
```
File: scripts/training/train_cycle.py
Purpose: Validate cycle detection on historical data
Input: 
  - GISAID trajectories (2020-2023)
  - Known variants (Alpha, Beta, Delta, Omicron)
Output: Cycle phase classifier
```

### INTEGRATION (Week 6)

**7. Unified Predictor**
```
File: prism-ve-python/prism_ve/integration/predictor.py
Purpose: Combine Escape + Fitness + Cycle
Main API:
  - predict_escape() → escape probabilities
  - predict_fitness() → viability scores
  - predict_emergence() → temporal predictions
```

---

## RUNTIME DEPENDENCIES

### From Existing PRISM (Read-Only)

**Binaries (DON'T MODIFY):**
```
✅ target/release/prism-lbs           # Feature extraction binary
✅ target/ptx/mega_fused_pocket.ptx   # GPU kernel (528KB)
```

**Features (ALREADY EXTRACTED):**
```
✅ prism-escape-benchmark/extracted_features/6m0j_12_COMPLETE.npy
✅ prism-escape-benchmark/extracted_features/influenza_ha.npy
✅ prism-escape-benchmark/extracted_features/hiv_env_7tfo.npy
```

**Trained Models:**
```
✅ Escape models (XGBoost) - can recreate from scripts
```

### New Data Needed

**GISAID (for Cycle Module):**
```
🆕 Download from: https://gisaid.org/
   Registration required (free for academic)
   
Files needed:
  - metadata.tsv (all SARS-CoV-2 sequences, ~15M rows)
  - Filter to: Spike mutations over time
  
Processing:
  scripts/setup/process_gisaid.py
  → data/processed/gisaid_trajectories.parquet
  
Format:
  position | date       | frequency | mutation | variant
  ---------|------------|-----------|----------|--------
  484      | 2021-01-01 | 0.001     | E484K    | Beta
  484      | 2021-02-01 | 0.005     | E484K    | Beta
  484      | 2021-03-01 | 0.012     | E484K    | Beta
  ...
```

**DMS Functional Data (for Fitness Module):**
```
✅ HAVE: Bloom DMS escape scores
🆕 NEED: ACE2 binding scores (from Starr et al.)
🆕 NEED: Expression scores (from Bloom DMS)

Can extract from existing Bloom repo:
  https://github.com/jbloomlab/SARS2_RBD_Ab_escape_maps
  Look for: bind_expr data (ACE2 affinity + expression)
```

---

## FILES THAT TOUCH EXISTING PRISM CODE

### Read-Only (Safe)

**1. Call PRISM Binary**
```
File: prism-ve-python/prism_ve/core/physics_engine.py

Calls:
  subprocess.run([
      '../PRISM/target/release/prism-lbs',
      '--input', pdb_path,
      'extract-features',
      '--output-npy', output_path
  ])

Touches: NOTHING (read-only subprocess call)
Risk: ZERO (can't break PRISM)
```

**2. Load PRISM Features**
```
File: prism-ve-python/prism_ve/modules/*.py

Loads:
  np.load('../PRISM/prism-escape-benchmark/extracted_features/*.npy')

Touches: NOTHING (read-only numpy load)
Risk: ZERO
```

### Modifications (Careful)

**NONE! PRISM-VE is pure Python wrapper around existing PRISM.**

**No Rust code modifications needed.**
**No GPU kernel modifications needed.**

---

## RUNTIME FLOW

### Escape Prediction (Already Working)

```
User → Python API → PRISM Binary → GPU Kernel → Features → XGBoost → Prediction

Files involved:
1. prism_ve/modules/escape.py (NEW Python wrapper)
2. ../PRISM/target/release/prism-lbs (existing binary, read-only)
3. ../PRISM/target/ptx/*.ptx (existing kernels, read-only)
4. Trained XGBoost model (load from .pkl)
```

### Fitness Prediction (NEW)

```
User → Python API → PRISM Features → Fitness Model → ΔΔG Prediction

Files involved:
1. prism_ve/modules/fitness.py (NEW)
2. PRISM features (NPY, read-only)
3. Fitness model (NEW .pkl file)
4. AA property constants (NEW, in constants.py)
```

### Cycle Detection (NEW)

```
User → Python API → GISAID Data → Cycle Detector → Phase + Timing

Files involved:
1. prism_ve/modules/cycle.py (NEW)
2. GISAID trajectories (NEW parquet file)
3. Escape scores (from escape module)
4. Fitness scores (from fitness module)
```

### Integrated Prediction (NEW)

```
User → Unified API → All 3 Modules → Combined Prediction

prism_ve.predict_emergence(mutations, time_horizon="6_months")
  ↓
  1. Get escape scores (escape module)
  2. Get fitness scores (fitness module)
  3. Get cycle phase (cycle module)
  4. Combine: emergence = escape × fitness × cycle_multiplier
  ↓
  Return: {mutation, escape, fitness, phase, timing, emergence_prob}
```

---

## CONFIGURATION FILES

### 1. PRISM Config (prism_config.yaml)
```yaml
prism:
  binary_path: "../PRISM/target/release/prism-lbs"
  ptx_dir: "../PRISM/target/ptx"
  feature_dim: 92
  device: "cuda"
  cache_features: true
  cache_dir: "./cache/features"

structures:
  sars2_rbd: "../PRISM/prism-escape-benchmark/data/raw/structures/6m0j.pdb"
  influenza_ha: "../PRISM/prism-escape-benchmark/data/raw/structures/1rv0.pdb"
  hiv_env: "../PRISM/prism-escape-benchmark/data/raw/structures/7tfo_env.pdb"
```

### 2. Model Config (model_config.yaml)
```yaml
escape:
  model_type: "xgboost"
  max_depth: 4
  learning_rate: 0.1
  n_estimators: 50
  
fitness:
  # ΔΔG prediction
  ddg_model: "physics_based"  # or "ml"
  features: [80, 81, 84, 89, 91]  # Thermodynamic-relevant
  
cycle:
  n_phases: 6
  frequency_threshold_exploring: 0.01
  frequency_threshold_escaped: 0.50
  velocity_threshold_reverting: -0.02
```

---

## DATA FLOW MAP

### Initialization (One-Time)

```
1. Download GISAID data
   scripts/setup/download_gisaid.sh
   → data/raw/gisaid/*.tsv

2. Process GISAID trajectories
   scripts/setup/process_gisaid.py
   → data/processed/gisaid_trajectories.parquet

3. Extract PRISM features (if not done)
   ../PRISM/target/release/prism-lbs extract-features
   → data/processed/*_features.npy

4. Download DMS functional data
   wget Bloom ACE2 binding data
   → data/raw/dms/functional/
```

### Training Phase

```
1. Train Escape Module
   scripts/training/train_escape.py
   Input: Bloom DMS + PRISM features
   Output: models/escape/*.pkl
   
2. Train Fitness Module
   scripts/training/train_fitness.py
   Input: DMS functional scores + PRISM features
   Output: models/fitness/fitness_predictor.pkl
   
3. Validate Cycle Module
   scripts/training/train_cycle.py
   Input: GISAID trajectories + known variants
   Output: Cycle phase validation results
```

### Inference Phase

```
User calls:
  prism_ve.predict_emergence(["E484K", "N501Y"], "6_months")

Internal flow:
  1. Load PRISM features (cached)
  2. Load escape model → predict_escape()
  3. Load fitness model → predict_fitness()
  4. Load GISAID data → detect_phase()
  5. Combine → emergence_probability
  6. Return predictions
```

---

## CRITICAL FILES CHECKLIST

### Must Create (NEW)

**Python Package:**
```
□ prism-ve-python/setup.py
□ prism-ve-python/prism_ve/__init__.py
□ prism-ve-python/prism_ve/core/physics_engine.py
□ prism-ve-python/prism_ve/core/constants.py
□ prism-ve-python/prism_ve/modules/escape.py
□ prism-ve-python/prism_ve/modules/fitness.py
□ prism-ve-python/prism_ve/modules/cycle.py
□ prism-ve-python/prism_ve/integration/predictor.py
□ prism-ve-python/prism_ve/data/loaders.py
□ prism-ve-python/prism_ve/utils/metrics.py
```

**Configuration:**
```
□ configs/prism_config.yaml
□ configs/model_config.yaml
```

**Scripts:**
```
□ scripts/setup/download_gisaid.sh
□ scripts/setup/process_gisaid.py
□ scripts/training/train_fitness.py
□ scripts/training/train_cycle.py
```

**Data:**
```
□ Download GISAID metadata
□ Download DMS functional scores
□ Process GISAID trajectories
```

### Already Have (DON'T RECREATE)

```
✅ PRISM binary (prism-lbs)
✅ GPU kernels (PTX files)
✅ PRISM features (NPY files)
✅ Bloom/Doud/Dingens DMS data
✅ Viral structures (PDB files)
✅ Escape module results (validated)
```

---

## ESTIMATED FILE SIZES

```
Python code:        ~15 files, ~3,000 lines total
GISAID data:        ~500 MB (compressed), ~2 GB (processed)
Models:             ~10 MB (XGBoost + fitness models)
Features:           ~5 MB (NPY files, already have)
Documentation:      ~20 pages
Total new storage:  ~2.5 GB
```

---

## SAFETY GUARANTEES

**PRISM-VE Development CANNOT break PRISM-Viral because:**

1. ✅ Separate worktree (isolated filesystem)
2. ✅ New Python package (no Rust modifications)
3. ✅ Read-only access to PRISM binary
4. ✅ Read-only access to features
5. ✅ Independent git branch

**If PRISM-VE fails:** Just delete worktree, PRISM-Viral unaffected!

---

## NEXT SESSION PLAN

**Start in PRISM-VE worktree:**
```bash
cd /mnt/c/Users/Predator/Desktop/prism-ve

# Session 12: Fitness Module (2 weeks)
# Session 13: Cycle Module (3 weeks)
# Session 14: Integration (1 week)
```

**PRISM-Viral remains safe in main directory for parallel publication!**
