#!/bin/bash
# Download HIV and Influenza DMS data for multi-virus validation

set -euo pipefail

echo "════════════════════════════════════════════════════════════════"
echo "MULTI-VIRUS VALIDATION - DATA DOWNLOAD"
echo "════════════════════════════════════════════════════════════════"
echo ""

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE_DIR/data/raw"

mkdir -p hiv_env influenza_ha structures

# ════════════════════════════════════════════════════════════════
# HIV ENV DMS DATA
# ════════════════════════════════════════════════════════════════
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1. HIV Env Deep Mutational Scanning"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Bloom lab HIV Env DMS (2024-2025)
cd hiv_env
if [ ! -d "HIV_Envelope_TRO11_DMS" ]; then
    echo "Downloading HIV TRO11 DMS data..."
    git clone --depth 1 https://github.com/dms-vep/HIV_Envelope_TRO11_DMS_3BNC117_10-1074.git
fi

if [ ! -d "HIV_Envelope_BF520_DMS" ]; then
    echo "Downloading HIV BF520 DMS data..."
    git clone --depth 1 https://github.com/dms-vep/HIV_Envelope_BF520_DMS_3BNC117_10-1074.git
fi

echo "✅ HIV Env DMS data downloaded"
cd ..

# ════════════════════════════════════════════════════════════════
# INFLUENZA HA DMS DATA  
# ════════════════════════════════════════════════════════════════
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "2. Influenza HA Deep Mutational Scanning"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check EVEscape bundled data
if [ -d "../evescape/EVEscape/data/experiments/doud2016" ]; then
    echo "✅ Influenza data available in EVEscape (doud2016)"
fi

if [ -d "../evescape/EVEscape/data/experiments/doud2018" ]; then
    echo "✅ Influenza data available in EVEscape (doud2018)"
fi

# ════════════════════════════════════════════════════════════════
# VIRAL PROTEIN STRUCTURES
# ════════════════════════════════════════════════════════════════
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "3. Viral Protein Structures"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

cd structures

# HIV Env structures
HIV_PDBS=("5fuu" "6mlo" "5u7o")
for pdb in "${HIV_PDBS[@]}"; do
    if [ ! -f "${pdb}.pdb" ]; then
        echo "Downloading ${pdb}.pdb (HIV Env)..."
        wget -q "https://files.rcsb.org/download/${pdb}.pdb"
        echo "  ✅ ${pdb}.pdb"
    fi
done

# Influenza HA structures
FLU_PDBS=("1rv0" "4o5n" "5hmg")
for pdb in "${FLU_PDBS[@]}"; do
    if [ ! -f "${pdb}.pdb" ]; then
        echo "Downloading ${pdb}.pdb (Influenza HA)..."
        wget -q "https://files.rcsb.org/download/${pdb}.pdb"
        echo "  ✅ ${pdb}.pdb"
    fi
done

cd "$BASE_DIR"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "DOWNLOAD COMPLETE"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "HIV Env: TRO11 + BF520 strain DMS data"
echo "Influenza: H1/H3 HA DMS data (via EVEscape)"
echo "Structures: 6 viral protein PDBs"
echo ""
echo "Next: Extract features and run benchmarks"
echo "════════════════════════════════════════════════════════════════"
