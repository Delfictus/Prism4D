#!/bin/bash
set -euo pipefail

# PRISM Viral Escape Benchmark - Data Download Script
# Downloads all required datasets for EVEscape-compatible benchmarking

echo "════════════════════════════════════════════════════════════════"
echo "PRISM VIRAL ESCAPE BENCHMARK - DATA DOWNLOAD"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Base directories
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DIR="$BASE_DIR/data/raw"
STRUCT_DIR="$RAW_DIR/structures"

mkdir -p "$RAW_DIR"/{bloom_dms,evescape,proteingym,structures}
mkdir -p "$BASE_DIR/data/processed"
mkdir -p "$BASE_DIR/data/splits"

cd "$RAW_DIR"

# ============================================================================
# 1. BLOOM LAB SARS-CoV-2 RBD DEEP MUTATIONAL SCANNING
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1. Downloading Bloom Lab SARS-CoV-2 RBD DMS Data"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ ! -d "bloom_dms/SARS2_RBD_Ab_escape_maps" ]; then
    echo "Cloning Bloom Lab DMS repository..."
    git clone --depth 1 https://github.com/jbloomlab/SARS2_RBD_Ab_escape_maps.git bloom_dms/SARS2_RBD_Ab_escape_maps
    echo "✅ Bloom Lab DMS data downloaded"
else
    echo "✅ Bloom Lab DMS data already exists"
fi

# Check what we got
echo ""
echo "Bloom DMS data files:"
find bloom_dms/SARS2_RBD_Ab_escape_maps -name "*.csv" | head -5
echo "..."
echo "Total escape maps: $(find bloom_dms/SARS2_RBD_Ab_escape_maps -name "*escape*.csv" | wc -l)"
echo ""

# ============================================================================
# 2. EVESCAPE BASELINE DATA (for comparison)
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "2. Downloading EVEscape Baseline Data"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# EVEscape benchmark data (if publicly available)
# Note: Some EVEscape data may require manual download
echo "Cloning EVEscape repository for reference..."
if [ ! -d "evescape/EVEscape" ]; then
    git clone --depth 1 https://github.com/OATML-Markslab/EVEscape.git evescape/EVEscape
    echo "✅ EVEscape repository cloned"
else
    echo "✅ EVEscape repository already exists"
fi

# Check for public data
if [ -d "evescape/EVEscape/data" ]; then
    echo "EVEscape data files found:"
    ls -lh evescape/EVEscape/data/ | head -10
else
    echo "⚠️  EVEscape data directory not found in repo"
    echo "   You may need to download separately from: https://marks.hms.harvard.edu/evescape/"
fi
echo ""

# ============================================================================
# 3. SARS-CoV-2 SPIKE RBD STRUCTURES
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "3. Downloading SARS-CoV-2 Spike RBD Structures"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Key structures for benchmarking
STRUCTURES=(
    "6m0j"  # SARS-CoV-2 RBD (original Wuhan-Hu-1)
    "7kmg"  # RBD with antibody (for validation)
    "6m17"  # Full spike trimer
    "7a98"  # Delta variant RBD
    "7t9l"  # Omicron BA.1 RBD
)

for pdb_id in "${STRUCTURES[@]}"; do
    if [ ! -f "$STRUCT_DIR/${pdb_id}.pdb" ]; then
        echo "Downloading ${pdb_id}.pdb..."
        wget -q "https://files.rcsb.org/download/${pdb_id}.pdb" -O "$STRUCT_DIR/${pdb_id}.pdb"
        if [ $? -eq 0 ]; then
            echo "  ✅ ${pdb_id}.pdb ($(du -h "$STRUCT_DIR/${pdb_id}.pdb" | cut -f1))"
        else
            echo "  ❌ Failed to download ${pdb_id}.pdb"
        fi
    else
        echo "  ✅ ${pdb_id}.pdb already exists"
    fi
done
echo ""

# ============================================================================
# 4. PROTEINGYM VIRAL SUBSET (optional - for extended validation)
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "4. ProteinGym Viral Subset (Optional - Large Download)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

read -p "Download ProteinGym (~2GB)? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [ ! -d "proteingym/ProteinGym" ]; then
        echo "Cloning ProteinGym repository..."
        git clone --depth 1 https://github.com/OATML-Markslab/ProteinGym.git proteingym/ProteinGym
        echo "✅ ProteinGym downloaded"
    else
        echo "✅ ProteinGym already exists"
    fi
else
    echo "⏭️  Skipped ProteinGym (not required for initial validation)"
fi
echo ""

# ============================================================================
# 5. VALIDATION: CHECK DOWNLOADS
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "DOWNLOAD SUMMARY"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Count files
BLOOM_CSV=$(find bloom_dms -name "*.csv" 2>/dev/null | wc -l)
STRUCTURES=$(find structures -name "*.pdb" 2>/dev/null | wc -l)

echo ""
echo "✅ Bloom DMS escape maps: $BLOOM_CSV CSV files"
echo "✅ PDB structures: $STRUCTURES files"
echo "✅ EVEscape repo: $([ -d "evescape/EVEscape" ] && echo "Downloaded" || echo "Pending")"
echo ""

# Disk usage
echo "Disk usage:"
du -sh bloom_dms 2>/dev/null || echo "  bloom_dms: Not found"
du -sh structures 2>/dev/null || echo "  structures: Not found"
du -sh evescape 2>/dev/null || echo "  evescape: Not found"
echo ""

# ============================================================================
# 6. CREATE METADATA FILE
# ============================================================================
cat > "$BASE_DIR/data/DATA_MANIFEST.json" << MANIFEST
{
  "download_date": "$(date -Iseconds)",
  "datasets": {
    "bloom_dms": {
      "source": "https://github.com/jbloomlab/SARS2_RBD_Ab_escape_maps",
      "description": "SARS-CoV-2 RBD deep mutational scanning escape maps",
      "n_escape_maps": $BLOOM_CSV,
      "reference": "Greaney et al., Cell Host Microbe 2021"
    },
    "structures": {
      "source": "RCSB PDB",
      "n_structures": $STRUCTURES,
      "pdb_ids": ["6m0j", "7kmg", "6m17", "7a98", "7t9l"]
    },
    "evescape": {
      "source": "https://github.com/OATML-Markslab/EVEscape",
      "reference": "Thadani et al., Nature 2023",
      "baselines": {
        "sars2_auprc": 0.53,
        "sars2_top10_recall": 0.31,
        "strain_r2": 0.77
      }
    }
  }
}
MANIFEST

echo "✅ Data manifest created: data/DATA_MANIFEST.json"
echo ""

# ============================================================================
# 7. NEXT STEPS
# ============================================================================
echo "════════════════════════════════════════════════════════════════"
echo "DOWNLOAD COMPLETE - NEXT STEPS"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "1. Preprocess data:"
echo "   python scripts/preprocess.py"
echo ""
echo "2. Test physics correlation:"
echo "   python scripts/test_physics_correlation.py"
echo ""
echo "3. Run full benchmark:"
echo "   python scripts/run_benchmark.py --checkpoint /path/to/prism.pt"
echo ""
echo "════════════════════════════════════════════════════════════════"
