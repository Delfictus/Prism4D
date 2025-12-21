#!/usr/bin/env bash
# Code Ocean Run Script for PRISM-LBS
# This script is executed when the capsule runs on Code Ocean
#
# Environment variables available:
#   - CO_CAPSULE_ID: Unique capsule identifier
#   - CO_CPUS: Number of available CPU cores
#   - CO_MEMORY: Available RAM in bytes
#   - CO_GPU: GPU model (if available)

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/../.."
DATA_DIR="${PROJECT_ROOT}/data"
RESULTS_DIR="${PROJECT_ROOT}/results"
BENCHMARK_DIR="${PROJECT_ROOT}/benchmark"

# Create output directories
mkdir -p "${RESULTS_DIR}"/{predictions,metrics,comparisons,figures}

# Log system information
echo "=============================================="
echo "PRISM-LBS Code Ocean Execution"
echo "=============================================="
echo "Date: $(date -Iseconds)"
echo "CPU cores: ${CO_CPUS:-$(nproc)}"
echo "Memory: ${CO_MEMORY:-unknown}"
echo "GPU: ${CO_GPU:-$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')}"
echo ""

# Verify GPU availability
if ! nvidia-smi &>/dev/null; then
    echo "ERROR: No GPU detected. PRISM-LBS requires CUDA GPU."
    exit 1
fi

echo "GPU Memory:"
nvidia-smi --query-gpu=memory.total,memory.free --format=csv
echo ""

# =============================================================================
# Build (if needed)
# =============================================================================

if [[ ! -f "${PROJECT_ROOT}/target/release/prism-lbs" ]]; then
    echo "Building PRISM-LBS..."
    cd "${PROJECT_ROOT}"
    cargo build --release --features cuda -p prism-lbs
fi

export PATH="${PROJECT_ROOT}/target/release:${PATH}"
export PRISM_PTX_PATH="${PROJECT_ROOT}/kernels/ptx"
export PRISM_MODEL_PATH="${PROJECT_ROOT}/models"
export RUST_LOG=info

# =============================================================================
# Download Benchmarks (if needed)
# =============================================================================

if [[ ! -d "${BENCHMARK_DIR}/cryptosite/structures" ]]; then
    echo "Downloading benchmark datasets..."
    python3 "${BENCHMARK_DIR}/download_benchmarks.py" \
        --all \
        --output "${BENCHMARK_DIR}"
fi

# =============================================================================
# Run Validation Suite
# =============================================================================

echo ""
echo "=============================================="
echo "Running Full Validation Suite"
echo "=============================================="

python3 "${BENCHMARK_DIR}/validate_all.py" \
    --all \
    --output "${RESULTS_DIR}/metrics" \
    --compare-baselines \
    --verbose

# =============================================================================
# Generate Publication Figures
# =============================================================================

echo ""
echo "=============================================="
echo "Generating Publication Figures"
echo "=============================================="

# Create figure generation script
cat > "${RESULTS_DIR}/generate_figures.py" << 'PYTHON_EOF'
#!/usr/bin/env python3
"""Generate publication-ready figures from validation results."""
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

# Set publication style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.2

results_dir = Path('/results/metrics')
figures_dir = Path('/results/figures')
figures_dir.mkdir(exist_ok=True)

# Load validation report
report_path = results_dir / 'validation_report.json'
if report_path.exists():
    with open(report_path) as f:
        report = json.load(f)

    # Figure 1: Top-N Success Rates by Benchmark
    fig, ax = plt.subplots(figsize=(8, 5))

    benchmarks = []
    top1_rates = []
    top3_rates = []
    top5_rates = []

    for name, data in report.get('benchmarks', {}).items():
        results = data.get('results', [])
        if results:
            n = len(results)
            benchmarks.append(name)
            top1_rates.append(sum(1 for r in results if r.get('top1_success', False)) / n * 100)
            top3_rates.append(sum(1 for r in results if r.get('top3_success', False)) / n * 100)
            top5_rates.append(sum(1 for r in results if r.get('top5_success', False)) / n * 100)

    x = range(len(benchmarks))
    width = 0.25

    ax.bar([i - width for i in x], top1_rates, width, label='Top-1', color='#2ecc71')
    ax.bar(x, top3_rates, width, label='Top-3', color='#3498db')
    ax.bar([i + width for i in x], top5_rates, width, label='Top-5', color='#9b59b6')

    ax.set_ylabel('Success Rate (%)')
    ax.set_xlabel('Benchmark')
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks)
    ax.legend()
    ax.set_ylim(0, 100)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(figures_dir / 'figure1_topn_success.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(figures_dir / 'figure1_topn_success.png', dpi=300, bbox_inches='tight')
    print("Generated Figure 1: Top-N Success Rates")

    # Figure 2: DCC Distribution
    fig, axes = plt.subplots(1, len(report.get('benchmarks', {})), figsize=(12, 4), sharey=True)
    if len(report.get('benchmarks', {})) == 1:
        axes = [axes]

    for ax, (name, data) in zip(axes, report.get('benchmarks', {}).items()):
        results = data.get('results', [])
        dcc_values = [r.get('dcc_min', 50) for r in results if r.get('dcc_min', 50) < 50]

        ax.hist(dcc_values, bins=20, color='#3498db', edgecolor='white', alpha=0.8)
        ax.axvline(x=4.0, color='#e74c3c', linestyle='--', linewidth=2, label='4Å threshold')
        ax.axvline(x=12.0, color='#f39c12', linestyle='--', linewidth=2, label='12Å threshold')
        ax.set_xlabel('DCC (Å)')
        ax.set_title(name)
        ax.legend(fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[0].set_ylabel('Count')
    plt.tight_layout()
    plt.savefig(figures_dir / 'figure2_dcc_distribution.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(figures_dir / 'figure2_dcc_distribution.png', dpi=300, bbox_inches='tight')
    print("Generated Figure 2: DCC Distribution")

    print("\nAll figures saved to:", figures_dir)
else:
    print("No validation report found. Run validation first.")
PYTHON_EOF

python3 "${RESULTS_DIR}/generate_figures.py" || echo "Figure generation skipped (missing dependencies)"

# =============================================================================
# Summary
# =============================================================================

echo ""
echo "=============================================="
echo "Execution Complete"
echo "=============================================="
echo "Results saved to: ${RESULTS_DIR}"
echo ""
echo "Contents:"
ls -la "${RESULTS_DIR}"
echo ""
echo "Metrics:"
ls -la "${RESULTS_DIR}/metrics" 2>/dev/null || echo "  (none)"
echo ""
echo "Figures:"
ls -la "${RESULTS_DIR}/figures" 2>/dev/null || echo "  (none)"
