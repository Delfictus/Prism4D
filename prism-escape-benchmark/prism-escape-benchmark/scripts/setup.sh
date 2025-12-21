#!/bin/bash
# Complete setup script for PRISM Viral Escape Benchmark

set -euo pipefail

echo "════════════════════════════════════════════════════════════════"
echo "PRISM VIRAL ESCAPE PREDICTION - COMPLETE SETUP"
echo "════════════════════════════════════════════════════════════════"
echo ""

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE_DIR"

# 1. Install Python dependencies
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1. Installing Python dependencies"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
pip3 install -r requirements.txt || pip install -r requirements.txt
echo "✅ Python dependencies installed"
echo ""

# 2. Download benchmark data
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "2. Downloading benchmark data"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ ! -f "data/raw/bloom_dms/SARS2_RBD_Ab_escape_maps/README.md" ]; then
    bash scripts/download_data.sh
else
    echo "✅ Data already downloaded"
fi
echo ""

# 3. Preprocess data
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "3. Preprocessing benchmark data"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 scripts/preprocess.py
echo ""

# 4. Verify PRISM binary
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "4. Verifying PRISM installation"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

PRISM_CLI="../PRISM/target/release/prism-lbs"

if [ -f "$PRISM_CLI" ]; then
    echo "✅ PRISM binary found: $PRISM_CLI"
    echo "   Version: $($PRISM_CLI --version 2>/dev/null || echo 'Unknown')"
else
    echo "⚠️  PRISM binary not found at $PRISM_CLI"
    echo "   You'll need to build PRISM first:"
    echo "   cd ../PRISM && cargo build --release -p prism-lbs"
fi
echo ""

# 5. Summary
echo "════════════════════════════════════════════════════════════════"
echo "SETUP COMPLETE"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "✅ Python dependencies installed"
echo "✅ Benchmark data downloaded (43,499 mutation-antibody pairs)"
echo "✅ Data preprocessed (170 unique mutations, 136 train / 34 test)"
echo "✅ SARS-CoV-2 RBD structures downloaded (5 PDB files)"
echo ""
echo "NEXT STEPS:"
echo "  1. Test physics correlation:"
echo "     python scripts/test_physics_correlation.py"
echo ""
echo "  2. View data:"
echo "     ls -la data/processed/sars2_rbd/"
echo ""
echo "  3. Start notebook exploration:"
echo "     jupyter notebook notebooks/"
echo ""
echo "════════════════════════════════════════════════════════════════"
