# VASIL Benchmark Data Download Summary

## Downloaded Data Sizes

$(du -sh data/vasil_benchmark/*/
 2>/dev/null)

**Total: $(du -sh data/vasil_benchmark 2>/dev/null | cut -f1)**

## Key Components

### 1. DMS Antibody Escape Data (128 MB)
- Bloom Lab SARS2_RBD_Ab_escape_maps repository
- VASIL processed escape scores (836 antibodies)
- Antibody → epitope class mappings
- Files: $(ls data/vasil_benchmark/dms/vasil_processed/*.csv 2>/dev/null | wc -l) CSVs

### 2. VASIL Code & Results (290 MB)
- Complete VASIL GitHub repository
- Pre-computed lineage frequencies for 12 countries
- Countries: $(ls -d data/vasil_benchmark/vasil_code/ByCountry/*/ 2>/dev/null | wc -l)
- Key files per country:
  - Daily_Lineages_Freq_1_percent.csv
  - Daily_Lineages_Freq_seq_thres_100.csv
  - Daily_SpikeGroups_Freq.csv

### 3. Surveillance Data (195 MB)
- German RKI sequences
- German wastewater data (AMELAG)
- OWID international COVID data
- GInPipe incidence reconstruction tool

### 4. Protein Structures (19 MB)
- PDB files: $(ls data/vasil_benchmark/structures/*.pdb 2>/dev/null | wc -l) structures
- Spike protein RBD, trimer, antibody complexes
- Used for structural analysis

## Data Ready for Benchmarking

✓ DMS escape scores
✓ VASIL lineage frequencies (2021-2024)
✓ 12 countries with pre-computed results
✓ Protein structures for analysis
✓ Benchmark comparison framework

## Next Steps

1. **Optional**: Download GISAID data (see data/vasil_benchmark/gisaid/DOWNLOAD_INSTRUCTIONS.md)
2. **Optional**: Download UK ONS data (see data/vasil_benchmark/surveillance/uk_ons/)
3. **Run benchmark**: `python scripts/benchmark_vs_vasil.py`
4. **Develop modules**: Start fitness and cycle modules

## Benchmark Target

**VASIL Accuracy: 0.92 (mean across 12 countries)**

Your PRISM-VE implementation must match or exceed this to be competitive.

## Data Coverage

| Dataset | Countries | Time Range | Records |
|---------|-----------|------------|---------|
| VASIL Lineages | 12 | Jul 2021 - Oct 2023 | ~5.6M sequences |
| DMS Escape | Global | 2020-2023 | 836 antibodies |
| Surveillance | Germany | 2020-2024 | Continuous |

