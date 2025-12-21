# VASIL Benchmark Data

This directory contains all data needed to benchmark PRISM-VE against VASIL's published results.

## What is VASIL?

VASIL (Variant-Specific Immunity Landscape) is a model that predicts COVID-19 variant dynamics by combining:
- DMS antibody escape data
- Population immunity levels
- Vaccination campaigns
- Historical infection dynamics

**Their Achievement**: 0.92 accuracy predicting variant rise/fall across 12 countries (2022-2023)

**Our Goal**: Match or exceed 0.92 accuracy with PRISM-VE

## Directory Structure

```
vasil_benchmark/
├── dms/                           # Antibody escape data
│   ├── SARS2_RBD_Ab_escape_maps/  # Bloom Lab raw data
│   └── vasil_processed/           # VASIL's processed version
│       ├── dms_per_ab_per_site.csv    (836 antibodies)
│       └── antibody_mapping.csv        (10 epitope classes)
│
├── vasil_code/                    # VASIL GitHub repository
│   ├── ByCountry/                 # Results for 12 countries
│   │   ├── Germany/
│   │   ├── USA/
│   │   ├── UK/
│   │   ├── Japan/
│   │   ├── Brazil/
│   │   ├── France/
│   │   ├── Canada/
│   │   ├── Denmark/
│   │   ├── Australia/
│   │   ├── Sweden/
│   │   ├── Mexico/
│   │   └── SouthAfrica/
│   └── scripts/                   # VASIL analysis code
│
├── surveillance/
│   ├── germany_sequences/         # RKI public genomic data
│   ├── germany_wastewater/        # AMELAG wastewater surveillance
│   ├── owid_covid_data.csv        # International case data
│   ├── GInPipe/                   # Incidence reconstruction tool
│   └── uk_ons/                    # [MANUAL] UK ONS infection survey
│
├── structures/
│   └── *.pdb                      # 12 protein structures
│
├── gisaid/                        # [OPTIONAL] Raw sequence data
│   └── DOWNLOAD_INSTRUCTIONS.md
│
└── vaccine_efficacy/              # Literature references
    └── SOURCES.md
```

## Key Data Files

### VASIL Lineage Frequencies

Each country directory contains:

```
ByCountry/{Country}/results/
├── Daily_Lineages_Freq_1_percent.csv          # Variants >1% frequency
├── Daily_Lineages_Freq_seq_thres_100.csv      # Variants >100 sequences
├── Daily_SpikeGroups_Freq.csv                 # Grouped by spike mutations
├── epitope_data/
│   ├── dms_per_ab_per_site.csv               # Escape scores
│   └── antibodymapping_greaneyclasses.csv    # Epitope classes
└── Immunological_Landscape_groups/            # Population immunity
```

### DMS Escape Data

Location: `dms/vasil_processed/dms_per_ab_per_site.csv`

Format:
```csv
antibody,epitope_class,site,escape_score
COV2-2050,class1,417,2.45
COV2-2050,class1,484,8.92
...
```

- 836 antibodies
- 10 epitope classes
- Sites 331-531 (RBD region)
- Escape scores: 0 (no escape) to 10+ (strong escape)

## Benchmark Workflow

### 1. Verify Data Download

```bash
# Check downloaded data
du -sh vasil_benchmark/*

# Expected sizes:
# dms/             128 MB
# vasil_code/      290 MB
# surveillance/    195 MB
# structures/       19 MB
# Total:           632 MB
```

### 2. Run Benchmark

```bash
# Basic benchmark (uses pre-computed VASIL frequencies)
python scripts/benchmark_vs_vasil.py

# Specific countries
python scripts/benchmark_vs_vasil.py --countries Germany USA Japan

# With custom output
python scripts/benchmark_vs_vasil.py --output results/my_benchmark.csv
```

### 3. Integrate Your PRISM-VE Model

Edit `scripts/benchmark_vs_vasil.py`:

```python
from prism_pipeline import PRISMPipeline

# Load your model
prism_model = PRISMPipeline.load("models/prism_ve_trained.pt")

# Initialize benchmark
benchmark = VASILBenchmark(
    prism_model=prism_model,
    data_dir="data/vasil_benchmark"
)

# Your model must implement:
# def predict_gamma(self, variant: str, country: str, date: pd.Timestamp) -> float:
#     """
#     Predict growth rate for variant at given date/location.
#
#     Returns:
#         gamma > 0: variant rising
#         gamma < 0: variant falling
#     """
```

## VASIL's Accuracy Metric

From Extended Data Fig 6:

```python
def compute_accuracy(predictions, observations):
    """
    For each (lineage, country, timepoint):
    - Prediction: γ > 0 → RISE, γ < 0 → FALL
    - Observation: frequency increasing → RISE, decreasing → FALL

    Accuracy = correct predictions / total predictions
    """
    correct = sum(pred == obs for pred, obs in zip(predictions, observations))
    return correct / len(predictions)
```

**VASIL Results**:
- Germany: 0.94
- USA: 0.91
- UK: 0.93
- Japan: 0.90
- Brazil: 0.89
- **Mean: 0.92**

## Key Validation Cases

### German Variant Succession (Fig 3a)

Test that your model predicts correct inflection points:

| Variant | Predicted Peak | Actual Peak | Match? |
|---------|----------------|-------------|--------|
| BA.2 | April 2022 | April 2022 | ✅ |
| BA.4/5 | July-Oct 2022 | July-Oct 2022 | ✅ |
| BQ.1.1 | Jan 2023 | Jan 2023 | ✅ |
| XBB.1.5 | Mid-April 2023 | Mid-April 2023 | ✅ |
| EG.5 | Autumn 2023 | Autumn 2023 | ✅ |
| JN.1 | Spring 2024 | Spring 2024 | ✅ |

### Geographic Specificity (Fig 5)

Test that your model explains geographic differences:

| Variant | USA | Germany | Japan | Explanation |
|---------|-----|---------|-------|-------------|
| BA.2.12.1 | >50% | <5% | <5% | Earlier BA.2 wave in GER/JPN created immunity |
| XBB.1.16 | ~15% | ~15% | ~25% | Shorter XBB.1.5 wave in JPN left niche |

### Prospective Prediction (Fig 3b)

Ultimate test - predict future variants:
- Training cutoff: April 16, 2023
- BA.2.86 first detected: July 24, 2023 (3 months later)
- **VASIL correctly predicted BA.2.86 would have growth advantage**

## Data Sources

### Automated Downloads
✅ Bloom Lab DMS: https://github.com/jbloomlab/SARS2_RBD_Ab_escape_maps
✅ VASIL Code: https://github.com/KleistLab/VASIL
✅ GInPipe: https://github.com/KleistLab/GInPipe
✅ RKI Germany: https://github.com/robert-koch-institut/
✅ OWID: https://covid.ourworldindata.org/
✅ PDB Structures: https://files.rcsb.org/

### Manual Downloads
⚠️ GISAID (~5.6M sequences): https://gisaid.org/
⚠️ UK ONS Survey: https://www.ons.gov.uk/

## Citations

If you use this data, cite:

**VASIL**:
> Lemke, H. et al. (2024). "Predicting COVID-19 variant dynamics by cross-neutralization potential."
> *Nature Methods*. DOI: [pending]

**Bloom Lab DMS**:
> Greaney, A.J. et al. (2023). "Mapping mutations to the SARS-CoV-2 RBD that escape antibody binding."
> *Virus Evolution*, 9(1), vead055.

**EVEscape** (similar approach):
> Thadani, N.N. et al. (2023). "Learning from prepandemic data to forecast viral escape."
> *Nature*, 622, 818-825.

## Troubleshooting

### Missing VASIL Frequencies

**Error**: `FileNotFoundError: VASIL frequency file not found`

**Solution**:
```bash
# Re-run download script
bash scripts/download_vasil_complete_benchmark_data.sh

# Or clone manually
git clone --depth 1 https://github.com/KleistLab/VASIL.git data/vasil_benchmark/vasil_code
```

### GISAID Access Required?

**No!** VASIL provides pre-computed frequencies. You can benchmark without GISAID.

### Model Integration

Your PRISM-VE model needs:
1. `predict_gamma(variant, country, date)` method
2. Return positive value for rising variants
3. Return negative value for falling variants

See `scripts/benchmark_vs_vasil.py` for details.

## Ready to Proceed?

Once data is downloaded:

1. ✅ Verify: `ls -lh data/vasil_benchmark/*/`
2. ✅ Test benchmark: `python scripts/benchmark_vs_vasil.py`
3. ✅ Develop fitness module
4. ✅ Develop cycle module
5. ✅ Integrate and re-benchmark

**Target**: Beat VASIL's 0.92 accuracy!
