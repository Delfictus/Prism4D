# PRISM Viral Escape Prediction Benchmark Suite

**Objective:** Rigorous evaluation of PRISM against EVEscape (Nature 2023) for viral immune escape prediction.

## Quick Start

```bash
# 1. Download benchmark data
bash scripts/download_data.sh

# 2. Preprocess datasets
python scripts/preprocess.py

# 3. Run benchmark
python scripts/run_benchmark.py --checkpoint path/to/prism.pt

# 4. View results
cat results/prism/benchmark_report.md
```

## Benchmark Datasets

- **SARS-CoV-2 RBD:** Bloom Lab DMS escape maps (120MB, 4000+ mutations)
- **HIV Env:** CATNAP neutralization database (manual download required)
- **Influenza HA:** ProteinGym viral subset
- **EVEscape baselines:** For direct comparison

## Target Performance

| Metric | EVEscape SOTA | PRISM Target | Status |
|--------|---------------|--------------|--------|
| SARS-2 AUPRC | 0.53 | **≥0.60** | ⬜ |
| Top-10% Recall | 0.31 | **≥0.40** | ⬜ |
| HIV AUPRC | 0.32 | **≥0.40** | ⬜ |
| Strain R² | 0.77 | **≥0.82** | ⬜ |

## Citation

If you use this benchmark:
```
PRISM Viral Escape Prediction Benchmark Suite
Built on EVEscape protocol (Nature 622, 818–825, 2023)
https://github.com/YOUR_ORG/prism-escape-benchmark
```

## References

- [EVEscape Paper (Nature 2023)](https://www.nature.com/articles/s41586-023-06617-0)
- [EVEscape GitHub](https://github.com/OATML-Markslab/EVEscape)
- [Bloom Lab DMS](https://github.com/jbloomlab/SARS2_RBD_Ab_escape_maps)
