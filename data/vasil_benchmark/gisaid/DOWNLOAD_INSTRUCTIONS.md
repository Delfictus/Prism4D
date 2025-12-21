# GISAID Data Download Instructions

GISAID (Global Initiative on Sharing All Influenza Data) requires registration and agreement to terms of use.

## Do You Need GISAID?

**NO** - VASIL provides pre-computed lineage frequencies!

You can benchmark PRISM-VE against VASIL using their published frequency data without accessing GISAID.

## If You Want Raw Sequence Data

### Step 1: Register

1. Go to https://gisaid.org/
2. Click "Register"
3. Complete registration form
4. Agree to Data Access Agreement
5. Wait for approval email (usually within 1-2 business days)

### Step 2: Download Data

After approval:

1. Log in to GISAID
2. Navigate to: EpiCoV → Downloads → Download packages
3. Select metadata download

### Countries Needed (from VASIL paper)

| Country | Sequences | Time Range | Use Case |
|---------|-----------|------------|----------|
| Germany | ~607,000 | Jul 2021 - Jul 2024 | Primary validation |
| USA | ~1,000,000 | Oct 2022 - Oct 2023 | BA.2.12.1 analysis |
| UK | ~500,000 | Oct 2022 - Oct 2023 | ONS validation |
| Japan | ~100,000 | Oct 2022 - Oct 2023 | XBB.1.16 analysis |
| Brazil | ~80,000 | Oct 2022 - Oct 2023 | FE.1 vs EG.5.1 |
| France | ~150,000 | Oct 2022 - Oct 2023 | International comparison |
| Canada | ~200,000 | Oct 2022 - Oct 2023 | International comparison |
| Denmark | ~300,000 | Oct 2022 - Oct 2023 | BN.1, JN.1 detection |
| Australia | ~150,000 | Oct 2022 - Oct 2023 | International comparison |
| Sweden | ~50,000 | Oct 2022 - Oct 2023 | XBB.1.16 comparison |
| Mexico | ~50,000 | Oct 2022 - Oct 2023 | International comparison |
| South Africa | ~30,000 | Oct 2022 - Oct 2023 | International comparison |

**Total: ~5.6 million sequences**

### Required Fields

In metadata.tsv:
- Accession ID
- Collection date
- Pango lineage
- Country
- Location (state/province)
- Host age (optional)
- Host gender (optional)

### File Organization

Place downloaded files in:
```
data/vasil_benchmark/gisaid/
├── Germany/
│   └── metadata.tsv
├── USA/
│   └── metadata.tsv
├── UK/
│   └── metadata.tsv
... (etc for each country)
```

## Alternative: Use VASIL's Pre-computed Frequencies

**Recommended approach** - No GISAID account needed!

VASIL provides ready-to-use lineage frequencies:
```
data/vasil_benchmark/vasil_code/ByCountry/{Country}/results/
└── Daily_Lineages_Freq_1_percent.csv
```

This file contains:
- Daily frequency for each lineage
- Filtered to lineages >1% frequency
- Covers full time range (2021-2024 for Germany)
- Already processed and validated

## VASIL's Exact Dataset

DOI: https://doi.org/10.55876/gis8.241022rp

This is the exact GISAID dataset VASIL used (5,617,986 sequences).

## Why Use GISAID?

Only needed if you want to:
1. Replicate VASIL's exact preprocessing pipeline
2. Test alternative lineage filtering thresholds
3. Perform independent frequency calculations
4. Extend analysis beyond VASIL's time range

For benchmarking PRISM-VE dynamics predictions, **VASIL's frequencies are sufficient**.
