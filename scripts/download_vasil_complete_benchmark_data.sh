#!/bin/bash
# download_vasil_complete_benchmark_data.sh
# Downloads all data needed to replicate VASIL benchmarks

set -e

BASE_DIR="data/vasil_benchmark"
mkdir -p $BASE_DIR/{dms,gisaid,surveillance,vaccine_efficacy,structures,vasil_code}

echo "=============================================="
echo "VASIL Benchmark Data Download"
echo "Replicating VASIL's exact dataset"
echo "=============================================="

# ============================================
# 1. DMS ESCAPE DATA (Same as EVEscape)
# ============================================
echo ""
echo "[1/8] Downloading DMS escape data..."

# Primary source: Bloom Lab escape maps
if [ ! -d "$BASE_DIR/dms/SARS2_RBD_Ab_escape_maps" ]; then
    git clone --depth 1 \
        https://github.com/jbloomlab/SARS2_RBD_Ab_escape_maps.git \
        $BASE_DIR/dms/SARS2_RBD_Ab_escape_maps
    echo "  ✓ Bloom Lab DMS data cloned"
else
    echo "  ✓ Bloom Lab DMS data already exists"
fi

# VASIL's processed DMS data (836 antibodies, 10 epitope classes)
mkdir -p $BASE_DIR/dms/vasil_processed
if [ ! -f "$BASE_DIR/dms/vasil_processed/dms_per_ab_per_site.csv" ]; then
    wget -q -O $BASE_DIR/dms/vasil_processed/dms_per_ab_per_site.csv \
        "https://raw.githubusercontent.com/KleistLab/VASIL/main/ByCountry/Australia/results/epitope_data/dms_per_ab_per_site.csv" \
        && echo "  ✓ VASIL processed DMS data downloaded" \
        || echo "  ✗ Failed to download VASIL DMS data"
else
    echo "  ✓ VASIL processed DMS data already exists"
fi

# Download antibody class mapping
if [ ! -f "$BASE_DIR/dms/vasil_processed/antibody_mapping.csv" ]; then
    wget -q -O $BASE_DIR/dms/vasil_processed/antibody_mapping.csv \
        "https://raw.githubusercontent.com/KleistLab/VASIL/main/ByCountry/Australia/results/epitope_data/antibodymapping_greaneyclasses.csv" \
        && echo "  ✓ Antibody class mapping downloaded" \
        || echo "  ✗ Failed to download antibody mapping"
else
    echo "  ✓ Antibody mapping already exists"
fi

# ============================================
# 2. VASIL CODE AND PROCESSED DATA
# ============================================
echo ""
echo "[2/8] Downloading VASIL repository..."

if [ ! -d "$BASE_DIR/vasil_code" ]; then
    git clone --depth 1 \
        https://github.com/KleistLab/VASIL.git \
        $BASE_DIR/vasil_code
    echo "  ✓ VASIL repository cloned"
else
    echo "  ✓ VASIL repository already exists"
fi

# This contains:
# - Processed DMS data
# - Epitope class definitions
# - Spike pseudo-group definitions
# - Country-specific results for validation

# ============================================
# 3. GISAID GENOMIC SURVEILLANCE DATA
# ============================================
echo ""
echo "[3/8] GISAID data (requires manual download)..."

mkdir -p $BASE_DIR/gisaid

cat << 'EOF' > $BASE_DIR/gisaid/DOWNLOAD_INSTRUCTIONS.md
# GISAID Data Download Instructions

GISAID requires registration and agreement to terms of use.

## Registration
1. Register at https://gisaid.org/
2. Log in and go to EpiCoV -> Downloads

## Countries Needed (from VASIL paper)
Download metadata.tsv for each country:

| Country | Sequences | Time Range | Primary Use |
|---------|-----------|------------|-------------|
| **Germany** | ~607,000 | Jul 2021 - Jul 2024 | Primary validation |
| Australia | ~150,000 | Oct 2022 - Oct 2023 | International comparison |
| Brazil | ~80,000 | Oct 2022 - Oct 2023 | FE.1 vs EG.5.1 |
| Canada | ~200,000 | Oct 2022 - Oct 2023 | International comparison |
| Denmark | ~300,000 | Oct 2022 - Oct 2023 | BN.1, JN.1 detection |
| France | ~150,000 | Oct 2022 - Oct 2023 | International comparison |
| Japan | ~100,000 | Oct 2022 - Oct 2023 | XBB.1.16 analysis |
| Mexico | ~50,000 | Oct 2022 - Oct 2023 | International comparison |
| South Africa | ~30,000 | Oct 2022 - Oct 2023 | International comparison |
| Sweden | ~50,000 | Oct 2022 - Oct 2023 | XBB.1.16 comparison |
| UK | ~500,000 | Oct 2022 - Oct 2023 | ONS validation |
| USA | ~1,000,000 | Oct 2022 - Oct 2023 | BA.2.12.1 analysis |

**Total: ~5.6 million sequences**

## VASIL's Exact Dataset Reference
DOI: https://doi.org/10.55876/gis8.241022rp
(5,617,986 sequences across all countries)

## Required Fields in metadata.tsv
- Accession ID
- Collection date
- Pango lineage
- Country
- Location (for state-level analysis)

## Alternative: Use VASIL's Pre-processed Frequencies
VASIL provides pre-computed lineage frequencies in their GitHub:
https://github.com/KleistLab/VASIL/tree/main/ByCountry/

Files to use:
- Daily_Lineages_Freq_1_percent.csv
- Daily_Lineages_Freq_seq_thres_100.csv
- Daily_SpikeGroups_Freq.csv

**This alternative allows benchmarking without GISAID access!**

## Download Commands (after GISAID login)
Place downloaded files in: data/vasil_benchmark/gisaid/{country}/metadata.tsv
EOF

echo "  ✓ Created GISAID download instructions"
echo "  → See: $BASE_DIR/gisaid/DOWNLOAD_INSTRUCTIONS.md"

# ============================================
# 4. GERMAN SURVEILLANCE DATA (Public)
# ============================================
echo ""
echo "[4/8] Downloading German surveillance data..."

# German sequence data (Robert Koch Institute - Public)
if [ ! -d "$BASE_DIR/surveillance/germany_sequences" ]; then
    git clone --depth 1 \
        https://github.com/robert-koch-institut/SARS-CoV-2-Sequenzdaten_aus_Deutschland.git \
        $BASE_DIR/surveillance/germany_sequences 2>&1 | grep -E "(Cloning|done)" || echo "  Note: Repository may have been updated"
    echo "  ✓ German RKI sequences cloned"
else
    echo "  ✓ German RKI sequences already exist"
fi

# German wastewater surveillance (AMELAG)
if [ ! -d "$BASE_DIR/surveillance/germany_wastewater" ]; then
    git clone --depth 1 \
        https://github.com/robert-koch-institut/Abwassersurveillance_AMELAG.git \
        $BASE_DIR/surveillance/germany_wastewater 2>&1 | grep -E "(Cloning|done)" || echo "  Note: Repository may have been updated"
    echo "  ✓ German wastewater data cloned"
else
    echo "  ✓ German wastewater data already exists"
fi

# German case numbers (7-day incidence)
if [ ! -f "$BASE_DIR/surveillance/germany_cases.csv" ]; then
    wget -q -O $BASE_DIR/surveillance/germany_cases.csv \
        "https://raw.githubusercontent.com/robert-koch-institut/COVID-19_7-Tage-Inzidenz_in_Deutschland/main/COVID-19-Faelle_7-Tage-Inzidenz_Deutschland.csv" \
        && echo "  ✓ German case data downloaded" \
        || echo "  ✗ German case data failed (repository may have moved)"
else
    echo "  ✓ German case data already exists"
fi

# ============================================
# 5. INTERNATIONAL CASE DATA
# ============================================
echo ""
echo "[5/8] Downloading international case data..."

# Our World in Data COVID dataset
if [ ! -f "$BASE_DIR/surveillance/owid_covid_data.csv" ]; then
    wget -q -O $BASE_DIR/surveillance/owid_covid_data.csv \
        "https://covid.ourworldindata.org/data/owid-covid-data.csv" \
        && echo "  ✓ OWID COVID data downloaded" \
        || echo "  ✗ OWID download failed (check URL)"
else
    echo "  ✓ OWID COVID data already exists"
fi

# UK ONS COVID Infection Survey (for validation)
mkdir -p $BASE_DIR/surveillance/uk_ons
cat << 'EOF' > $BASE_DIR/surveillance/uk_ons/DOWNLOAD_INSTRUCTIONS.md
# UK ONS COVID-19 Infection Survey

This is the ground truth data VASIL used to validate GInPipe incidence estimates.

## Download Location
https://www.ons.gov.uk/peoplepopulationandcommunity/healthandsocialcare/conditionsanddiseases/datasets/coronaviruscovid19infectionsurveydata

## Required Datasets
1. Weekly positive test rates (England, Wales, Scotland, Northern Ireland)
2. Time period: October 2022 - October 2023
3. Format: Excel (.xlsx) or CSV

## Files to Download
- 2023inflectionsurveydata{date}.xlsx
- Look for "Percentage testing positive" tables

## Validation Use
VASIL used ONS data to validate that GInPipe-reconstructed incidence
matched real-world infection rates (Extended Data Fig 2).
EOF

echo "  ✓ Created UK ONS download instructions"

# ============================================
# 6. GINPIPE (Incidence Reconstruction Tool)
# ============================================
echo ""
echo "[6/8] Downloading GInPipe..."

if [ ! -d "$BASE_DIR/surveillance/GInPipe" ]; then
    git clone --depth 1 \
        https://github.com/KleistLab/GInPipe.git \
        $BASE_DIR/surveillance/GInPipe
    echo "  ✓ GInPipe repository cloned"
else
    echo "  ✓ GInPipe already exists"
fi

# ============================================
# 7. VACCINE EFFICACY DATA
# ============================================
echo ""
echo "[7/8] Setting up vaccine efficacy references..."

mkdir -p $BASE_DIR/vaccine_efficacy

cat << 'EOF' > $BASE_DIR/vaccine_efficacy/SOURCES.md
# Vaccine Efficacy Data Sources

VASIL used vaccine efficacy data to calibrate their model.
Data is compiled from literature (Supplementary Tables 4-5).

## Key Studies Referenced

### Wuhan Vaccine vs Delta (Model Calibration)
- Multiple clinical studies summarized in VASIL Supplementary Table 4-5
- Used to establish baseline neutralization-to-efficacy relationship

### Key References
1. **Khoury et al. 2021** (Nat Med)
   - DOI: 10.1038/s41591-021-01377-8
   - Neutralizing antibody levels predictive of protection

2. **Gruell et al. 2022** (Nat Med)
   - DOI: 10.1038/s41591-022-01792-5
   - mRNA booster vs Omicron variants

3. **Multiple VE studies** compiled in VASIL supplementary materials

## Antibody Pharmacokinetics Parameters

From VASIL Supplementary Table 4:

| Parameter | Value | Source |
|-----------|-------|--------|
| t_max (time to peak) | 14-28 days | Refs 45-48 in VASIL |
| t_half (half-life) | 25-69 days | Refs 49-53 in VASIL |
| Peak neutralization | Variant-specific | DMS-derived |

## Usage in VASIL Model

1. **Initial calibration**: Wuhan vaccine → Delta efficacy
2. **Cross-neutralization**: DMS escape scores → fold-reduction in neutralization
3. **Time dynamics**: PK parameters → antibody decay over time
4. **Efficacy prediction**: Neutralization level → VE via Khoury relationship

## Data Files

VASIL does not provide raw VE data, but you can:
1. Extract values from Supplementary Tables 4-5 in the paper
2. Use published VE studies directly (references listed above)
3. Calibrate against reported efficacy ranges

## PRISM-VE Implementation

For benchmarking, you can:
- Use the same PK parameters from VASIL Supplementary Table 4
- Apply Khoury et al. neutralization-to-VE formula
- Validate against reported VE ranges in literature
EOF

echo "  ✓ Created vaccine efficacy source documentation"

# ============================================
# 8. PROTEIN STRUCTURES
# ============================================
echo ""
echo "[8/8] Downloading protein structures..."

mkdir -p $BASE_DIR/structures

# Structures used in VASIL (from their Supplementary Table 1)
PDBS="6M0J 6VXX 6VYB 7BNN 7CAB 7TFO 7PUY 1RVX 5FYL 5EVM 7TY0 7TXZ"

echo "  Downloading ${#PDBS[@]} PDB structures..."
for pdb in $PDBS; do
    if [ ! -f "$BASE_DIR/structures/${pdb}.pdb" ]; then
        wget -q -O $BASE_DIR/structures/${pdb}.pdb \
            "https://files.rcsb.org/download/${pdb}.pdb" \
            && echo "    ✓ ${pdb}.pdb" \
            || echo "    ✗ ${pdb}.pdb failed"
    else
        echo "    ✓ ${pdb}.pdb (already exists)"
    fi
done

# ============================================
# SUMMARY
# ============================================
echo ""
echo "=============================================="
echo "Download Summary"
echo "=============================================="
echo ""

# Count downloaded files
dms_files=$(find $BASE_DIR/dms -type f 2>/dev/null | wc -l)
vasil_files=$(find $BASE_DIR/vasil_code -type f 2>/dev/null | wc -l)
surveillance_files=$(find $BASE_DIR/surveillance -type f 2>/dev/null | wc -l)
structure_files=$(find $BASE_DIR/structures -name "*.pdb" 2>/dev/null | wc -l)

echo "Files downloaded:"
echo "  DMS data: $dms_files files"
echo "  VASIL code: $vasil_files files"
echo "  Surveillance: $surveillance_files files"
echo "  Structures: $structure_files PDB files"
echo ""

# Calculate total size
total_size=$(du -sh $BASE_DIR 2>/dev/null | cut -f1)
echo "Total size: $total_size"
echo ""

echo "=============================================="
echo "Manual Steps Required:"
echo "=============================================="
echo ""
echo "1. GISAID DATA (Optional - VASIL frequencies available)"
echo "   → $BASE_DIR/gisaid/DOWNLOAD_INSTRUCTIONS.md"
echo ""
echo "2. UK ONS DATA (For validation)"
echo "   → $BASE_DIR/surveillance/uk_ons/DOWNLOAD_INSTRUCTIONS.md"
echo ""
echo "3. Vaccine Efficacy (From literature)"
echo "   → $BASE_DIR/vaccine_efficacy/SOURCES.md"
echo ""
echo "=============================================="
echo ""

# Create manifest
find $BASE_DIR -type f \( -name "*.csv" -o -name "*.pdb" -o -name "*.md" \) 2>/dev/null | sort > $BASE_DIR/MANIFEST.txt
echo "File manifest created: $BASE_DIR/MANIFEST.txt"
echo ""

# Create data structure documentation
cat << 'EOF' > $BASE_DIR/DATA_STRUCTURE.md
# VASIL Benchmark Data Structure

## Directory Layout

```
data/vasil_benchmark/
├── dms/                                   # DMS antibody escape data
│   ├── SARS2_RBD_Ab_escape_maps/         # Bloom Lab raw data
│   │   └── processed_data/
│   │       └── escape_data.csv           # 836 antibodies
│   └── vasil_processed/
│       ├── dms_per_ab_per_site.csv       # VASIL's processed version
│       └── antibody_mapping.csv          # Epitope class assignments
│
├── vasil_code/                            # VASIL GitHub repository
│   ├── ByCountry/
│   │   ├── Australia/results/
│   │   ├── Brazil/results/
│   │   ├── Canada/results/
│   │   ├── Denmark/results/
│   │   ├── France/results/
│   │   ├── Germany/results/
│   │   │   ├── Daily_Lineages_Freq_1_percent.csv
│   │   │   ├── Daily_Lineages_Freq_seq_thres_100.csv
│   │   │   └── Daily_SpikeGroups_Freq.csv
│   │   ├── Japan/results/
│   │   ├── Mexico/results/
│   │   ├── SouthAfrica/results/
│   │   ├── Sweden/results/
│   │   ├── UK/results/
│   │   └── USA/results/
│   └── scripts/                           # VASIL analysis code
│
├── gisaid/                                # MANUAL DOWNLOAD
│   ├── DOWNLOAD_INSTRUCTIONS.md
│   └── {country}/metadata.tsv             # ~5.6M sequences total
│
├── surveillance/
│   ├── germany_sequences/                 # RKI public data
│   ├── germany_wastewater/                # AMELAG wastewater
│   ├── germany_cases.csv                  # 7-day incidence
│   ├── owid_covid_data.csv                # Our World in Data
│   ├── uk_ons/                            # MANUAL: ONS survey
│   └── GInPipe/                           # Incidence reconstruction
│
├── vaccine_efficacy/
│   └── SOURCES.md                         # Literature references
│
├── structures/
│   ├── 6M0J.pdb                           # SARS-CoV-2 RBD
│   ├── 6VXX.pdb                           # Spike trimer
│   ├── 6VYB.pdb                           # RBD-ACE2 complex
│   ├── 7BNN.pdb, 7CAB.pdb, ...           # Antibody complexes
│   └── [12 structures total]
│
├── MANIFEST.txt                           # List of all files
└── DATA_STRUCTURE.md                      # This file
```

## Key Files for Benchmarking

### Lineage Frequencies
Location: `vasil_code/ByCountry/{Country}/results/`

Files:
- `Daily_Lineages_Freq_1_percent.csv` - Variants >1% frequency
- `Daily_Lineages_Freq_seq_thres_100.csv` - Variants with >100 sequences
- `Daily_SpikeGroups_Freq.csv` - Grouped by spike mutations

### DMS Escape Data
Location: `dms/vasil_processed/`

Files:
- `dms_per_ab_per_site.csv` - Escape scores per antibody per site
- `antibody_mapping.csv` - Antibody → epitope class mapping

### VASIL Results for Validation
Location: `vasil_code/ByCountry/{Country}/results/`

Files:
- Immunological landscape data
- Predicted vs observed dynamics
- Inflection point timings

## Data Coverage

| Dataset | Time Range | Geographic Scope | Use |
|---------|------------|------------------|-----|
| DMS | 2020-2023 | Global | Antibody escape |
| GISAID* | Jul 2021 - Jul 2024 | 12 countries | Variant frequencies |
| VASIL frequencies | Oct 2022 - Oct 2023 | 12 countries | Direct comparison |
| RKI Germany | 2020-2024 | Germany | Ground truth |
| OWID | 2020-2024 | Global | Case numbers |
| ONS* | 2022-2023 | UK | Infection rates |

*Requires manual download

## Benchmark Targets

From VASIL Extended Data Fig 6:

| Country | VASIL Accuracy | Sequences | Status |
|---------|----------------|-----------|--------|
| Germany | 0.94 | 607,000 | ✓ Frequencies available |
| USA | 0.91 | 1,000,000 | ✓ Frequencies available |
| UK | 0.93 | 500,000 | ✓ Frequencies available |
| Japan | 0.90 | 100,000 | ✓ Frequencies available |
| Brazil | 0.89 | 80,000 | ✓ Frequencies available |
| **Mean** | **0.92** | **5.6M total** | **Target to beat** |
EOF

echo "Data structure documented: $BASE_DIR/DATA_STRUCTURE.md"
echo ""
echo "=============================================="
echo "✓ Setup Complete!"
echo "=============================================="
echo ""
echo "You can now proceed with:"
echo "1. Benchmarking against VASIL's pre-computed frequencies"
echo "2. Developing fitness and cycle modules"
echo "3. Optional: Download GISAID for full replication"
