#!/usr/bin/env bash
# PRISM-LBS CryptoBench Hyper-Tuned Benchmark
# Compares against baseline optimal_cryptobench.toml configuration
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRISM_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PRISM_ROOT"

# Configuration paths
INPUT_DIR="benchmarks/datasets/cryptobench/structures/test"
OUTPUT_DIR="benchmarks/results/cryptobench_hyper/pockets"
METRICS_FILE="benchmarks/results/cryptobench_hyper/metrics.json"
PROVENANCE_FILE="benchmarks/results/cryptobench_hyper/PROVENANCE.txt"
GROUND_TRUTH="benchmarks/datasets/cryptobench/dataset.json"
PARALLEL=2  # RTX 3080 10GB can handle 2 parallel; use 1 if OOM

# Environment
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.6}"
export PRISM_PTX_DIR="${PRISM_ROOT}/target/ptx"
export RUST_LOG="${RUST_LOG:-warn}"
export PATH="/home/diddy/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin:/usr/bin:$PATH"

# Hyper-tuned config file
CONFIG_FILE="configs/cryptobench_hyper.toml"

# Display parameters from config (for reference)
PRISM_MIN_POCKET_VOLUME=160.0
PRISM_MAX_POCKET_VOLUME=4800.0
PRISM_DRUGGABILITY_THRESHOLD=0.30
PRISM_MAX_POCKETS=8
PRISM_EXPANSION_DISTANCE=3.5
PRISM_GRID_SPACING=1.8
PRISM_MIN_ALPHA_RADIUS=2.3

echo "=============================================="
echo "PRISM-LBS CryptoBench Hyper-Tuned Benchmark"
echo "=============================================="
echo "Input: $INPUT_DIR"
echo "Output: $OUTPUT_DIR"
echo "Config: $CONFIG_FILE"
echo "Parallel workers: $PARALLEL"
echo ""
echo "Hyper-tuned parameters (from config):"
echo "  min_pocket_volume: $PRISM_MIN_POCKET_VOLUME Å³"
echo "  max_pocket_volume: $PRISM_MAX_POCKET_VOLUME Å³"
echo "  druggability_threshold: $PRISM_DRUGGABILITY_THRESHOLD"
echo "  max_pockets: $PRISM_MAX_POCKETS"
echo "  expansion_distance: $PRISM_EXPANSION_DISTANCE Å"
echo "  grid_spacing: $PRISM_GRID_SPACING Å"
echo "  min_alpha_radius: $PRISM_MIN_ALPHA_RADIUS Å"
echo ""

# Build with optimizations
echo "[1/4] Building release binary..."
export RUSTFLAGS="-C target-cpu=native"
cargo build --release --features cuda -p prism-lbs 2>&1 | tail -5

# Prepare output directory
echo ""
echo "[2/4] Preparing output directory..."
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$(dirname "$PROVENANCE_FILE")"

# Check if input directory exists
if [[ ! -d "$INPUT_DIR" ]] || [[ -z "$(ls -A "$INPUT_DIR" 2>/dev/null)" ]]; then
    echo "Warning: Input directory empty or missing: $INPUT_DIR"
    echo "Trying alternative location..."
    if [[ -d "/tmp/cryptobench_test_structures" ]]; then
        INPUT_DIR="/tmp/cryptobench_test_structures"
        echo "Using: $INPUT_DIR"
    else
        echo "ERROR: No structures found. Please download CryptoBench test set first."
        exit 1
    fi
fi

# Run benchmark with timing
echo ""
echo "[3/4] Running PRISM-LBS benchmark..."
echo "Start: $(date -Iseconds)"

START_TIME=$(date +%s.%N)

CUDA_VISIBLE_DEVICES=0 ./target/release/prism-lbs \
    --input "$INPUT_DIR" \
    --output "$OUTPUT_DIR" \
    --format json \
    --config "$CONFIG_FILE" \
    --gpu-geometry \
    batch --parallel "$PARALLEL" 2>&1

END_TIME=$(date +%s.%N)
ELAPSED=$(echo "$END_TIME - $START_TIME" | bc)

echo "End: $(date -Iseconds)"
echo "Elapsed: ${ELAPSED}s"

# Count outputs
NUM_OUTPUTS=$(ls "$OUTPUT_DIR"/*.json 2>/dev/null | wc -l)
echo "Output files: $NUM_OUTPUTS"

# Evaluate against ground truth
echo ""
echo "[4/4] Evaluating against ground truth..."
if [[ -f "$GROUND_TRUTH" ]] && [[ -f "scripts/compare_cryptobench.py" ]]; then
    python3 scripts/compare_cryptobench.py \
        --predictions "$OUTPUT_DIR" \
        --ground-truth "$GROUND_TRUTH" \
        --structures "$INPUT_DIR" \
        --output "$METRICS_FILE" \
        --verbose
    echo "Metrics saved to: $METRICS_FILE"
else
    echo "Warning: Ground truth or evaluation script not found"
    echo "  Ground truth: $GROUND_TRUTH"
    echo "  Eval script: scripts/compare_cryptobench.py"
fi

# Generate provenance
echo ""
echo "Generating provenance..."
{
    echo "# PRISM-LBS CryptoBench Hyper-Tuned Benchmark Provenance"
    echo "# Generated: $(date -Iseconds)"
    echo ""
    echo "## Command"
    echo "CUDA_VISIBLE_DEVICES=0 ./target/release/prism-lbs \\"
    echo "    --input $INPUT_DIR \\"
    echo "    --output $OUTPUT_DIR \\"
    echo "    --format json \\"
    echo "    batch --parallel $PARALLEL"
    echo ""
    echo "## Hyper-Tuned Parameters"
    echo "min_pocket_volume: $PRISM_MIN_POCKET_VOLUME"
    echo "max_pocket_volume: $PRISM_MAX_POCKET_VOLUME"
    echo "druggability_threshold: $PRISM_DRUGGABILITY_THRESHOLD"
    echo "max_pockets: $PRISM_MAX_POCKETS"
    echo "expansion_distance: $PRISM_EXPANSION_DISTANCE"
    echo "grid_spacing: $PRISM_GRID_SPACING"
    echo "min_alpha_radius: $PRISM_MIN_ALPHA_RADIUS"
    echo ""
    echo "## Git Information"
    echo "commit: $(git rev-parse HEAD 2>/dev/null || echo 'not a git repo')"
    echo "describe: $(git describe --tags --dirty 2>/dev/null || echo 'no tags')"
    echo ""
    echo "## Binary Checksums"
    echo "prism-lbs (sha384): $(sha384sum target/release/prism-lbs 2>/dev/null | cut -d' ' -f1 || echo 'not found')"
    echo "cryptobench_hyper.toml (sha384): $(sha384sum configs/cryptobench_hyper.toml 2>/dev/null | cut -d' ' -f1 || echo 'not found')"
    echo ""
    echo "## GPU Information"
    nvidia-smi --query-gpu=driver_version,name,memory.total --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available"
    echo ""
    echo "## Timing"
    echo "elapsed_sec: $ELAPSED"
    echo "structures_processed: $NUM_OUTPUTS"
    if [[ $NUM_OUTPUTS -gt 0 ]]; then
        echo "ms_per_structure: $(echo "scale=1; $ELAPSED * 1000 / $NUM_OUTPUTS" | bc 2>/dev/null || echo 'N/A')"
    fi
    echo ""
    echo "## Baseline Comparison"
    echo "baseline_config: configs/optimal_cryptobench.toml"
    echo "baseline_top1_50: 82.6%"
    echo "baseline_top1_30: 89.9%"
    echo "baseline_mean_recall: 68.6%"
} > "$PROVENANCE_FILE"

echo "Provenance saved to: $PROVENANCE_FILE"

# Print summary
echo ""
echo "=============================================="
echo "BENCHMARK COMPLETE"
echo "=============================================="
echo "Structures processed: $NUM_OUTPUTS"
echo "Total time: ${ELAPSED}s"
if [[ $NUM_OUTPUTS -gt 0 ]]; then
    echo "Throughput: $(echo "scale=1; $ELAPSED * 1000 / $NUM_OUTPUTS" | bc 2>/dev/null || echo 'N/A') ms/structure"
fi
echo ""
echo "Results:"
echo "  Pockets: $OUTPUT_DIR"
echo "  Metrics: $METRICS_FILE"
echo "  Provenance: $PROVENANCE_FILE"
echo ""

# Show metrics if available
if [[ -f "$METRICS_FILE" ]]; then
    echo "Key Metrics:"
    python3 -c "
import json
with open('$METRICS_FILE') as f:
    m = json.load(f)

print(f\"  Mean Precision: {m.get('mean_precision', 'N/A'):.1%}\")
print(f\"  Mean Recall:    {m.get('mean_recall', 'N/A'):.1%}\")
print(f\"  Mean F1:        {m.get('mean_f1', 'N/A'):.1%}\")
print(f\"  Mean Jaccard:   {m.get('mean_jaccard', 'N/A'):.3f}\")
if m.get('mean_dcc'):
    print(f\"  Mean DCC:       {m['mean_dcc']:.2f} A\")

print()
print('Recall Thresholds (Top-1):')
thresh = m.get('top1_recall_thresholds', {})
for t in ['>=30%', '>=50%', '>=70%']:
    if t in thresh:
        print(f\"  {t}: {thresh[t]:.1%}\")

print()
print('DCC Success Rates:')
dcc = m.get('dcc_success', {})
if '@4A' in dcc:
    print(f\"  DCC <= 4A: {dcc['@4A']:.1%}\")
if '@8A' in dcc:
    print(f\"  DCC <= 8A: {dcc['@8A']:.1%}\")

print()
print('Delta vs Baseline:')
baseline_recall = 0.686
baseline_top1_50 = 0.826
baseline_top1_30 = 0.899
if m.get('mean_recall'):
    delta = m['mean_recall'] - baseline_recall
    print(f\"  Mean Recall: {delta:+.1%}\")
if '>=50%' in thresh:
    delta = thresh['>=50%'] - baseline_top1_50
    print(f\"  Top-1 >= 50%: {delta:+.1%}\")
if '>=30%' in thresh:
    delta = thresh['>=30%'] - baseline_top1_30
    print(f\"  Top-1 >= 30%: {delta:+.1%}\")
" 2>/dev/null || echo "  (metrics parsing failed)"
fi
