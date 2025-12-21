#!/bin/bash
# download_all_public_data.sh
# Downloads everything needed to benchmark against VASIL
# NO GISAID ACCOUNT REQUIRED

set -e

BASE_DIR="data/prism_ve_benchmark"
mkdir -p $BASE_DIR

echo "=============================================="
echo "PRISM-VE Benchmark Data Download"
echo "No GISAID account required"
echo "=============================================="

# 1. VASIL Repository (includes processed frequencies)
echo ""
echo "[1/6] Cloning VASIL repository..."
git clone --depth 1 https://github.com/KleistLab/VASIL.git $BASE_DIR/vasil

# 2. Bloom Lab DMS Data
echo ""
echo "[2/6] Cloning Bloom Lab DMS data..."
git clone --depth 1 https://github.com/jbloomlab/SARS2_RBD_Ab_escape_maps.git $BASE_DIR/dms

# 3. German Surveillance (Public)
echo ""
echo "[3/6] Downloading German surveillance..."
mkdir -p $BASE_DIR/surveillance/germany

wget -q -O $BASE_DIR/surveillance/germany/wastewater.csv \
    "https://raw.githubusercontent.com/robert-koch-institut/Abwassersurveillance_AMELAG/main/Abwassersurveillance_AMELAG.csv"

# 4. Our World in Data
echo ""
echo "[4/6] Downloading OWID data..."
wget -q -O $BASE_DIR/surveillance/owid_covid.csv \
    "https://covid.ourworldindata.org/data/owid-covid-data.csv"

# 5. GInPipe
echo ""
echo "[5/6] Cloning GInPipe..."
git clone --depth 1 https://github.com/KleistLab/GInPipe.git $BASE_DIR/GInPipe

# 6. Structures
echo ""
echo "[6/6] Downloading structures..."
mkdir -p $BASE_DIR/structures
for pdb in 6M0J 6VXX 6VYB 7BNN 7CAB 7TFO 7PUY; do
    wget -q -O $BASE_DIR/structures/${pdb}.pdb \
        "https://files.rcsb.org/download/${pdb}.pdb" \
        && echo "  Downloaded ${pdb}.pdb"
done

# Create summary
echo ""
echo "=============================================="
echo "Download Complete!"
echo "=============================================="
echo ""
echo "Directory structure:"
find $BASE_DIR -type d -maxdepth 2 | head -20
echo ""
echo "Key files for benchmarking:"
echo ""
echo "Lineage frequencies (12 countries):"
ls $BASE_DIR/vasil/ByCountry/*/results/lineage_frequencies/*.csv 2>/dev/null | head -5
echo "..."
echo ""
echo "DMS escape data:"
ls $BASE_DIR/dms/processed_data/*.csv 2>/dev/null
echo ""
echo "Total size:"
du -sh $BASE_DIR
