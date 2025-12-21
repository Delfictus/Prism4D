#!/usr/bin/env python3
"""
PHASE 0.1: Validate 11-Epitope Rank Assumption via SVD
========================================================

GO/NO-GO Criteria:
- Rank-22 captures >98% variance → ✅ PROCEED to polyalgorithmic
- Rank-22 captures 95-98% variance → ⚠️ PROCEED to single-scale  
- Rank-22 captures <95% variance → ❌ ABORT low-rank approach

This script performs Singular Value Decomposition on VASIL's cross-immunity
matrices to validate that the 11-epitope hypothesis (rank-22) is sufficient.

Expected outcome: ~98-99% variance captured by rank-22 approximation.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.linalg import svd
from typing import Dict, Tuple, List

# Matplotlib is optional - will save CSV only if not available
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠️ Warning: matplotlib not available, skipping plots")

# VASIL data paths
VASIL_BASE = Path("data/VASIL/ByCountry")
COUNTRIES = ["Germany", "USA", "UK", "France", "Canada", "Australia", 
             "Brazil", "Mexico", "Japan", "Sweden", "Denmark", "SouthAfrica"]

# 11 Epitopes (10 RBD + 1 NTD) - CORRECTED from Claude Code's 10
N_EPITOPES = 11
EXPECTED_RANK = 2 * N_EPITOPES  # 22 (each epitope has 2 singular components)


def load_cross_immunity_matrix(country: str) -> Dict[str, np.ndarray]:
    """Load VASIL cross-immunity pickle file."""
    pck_path = VASIL_BASE / country / "results" / "Cross_react_dic_spikegroups_ALL.pck"
    
    if not pck_path.exists():
        # Try alternate path
        pck_path = Path("data/prism_ve_benchmark/vasil/ByCountry") / country / "results" / "Cross_react_dic_spikegroups_ALL.pck"
    
    with open(pck_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"  {country}: {len(data)} variant groups")
    return data


def analyze_svd_rank(matrix: np.ndarray, label: str) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Perform SVD and compute variance explained by different ranks.
    
    Returns:
        singular_values: Sorted singular values
        variance_dict: Variance captured at ranks [11, 22, 50, 100, full]
    """
    # Ensure matrix is square and symmetric
    assert matrix.shape[0] == matrix.shape[1], f"Matrix must be square: {matrix.shape}"
    
    # SVD: M = U @ diag(S) @ Vh
    U, S, Vh = svd(matrix, full_matrices=False)
    
    # Compute total variance (sum of squared singular values)
    total_variance = np.sum(S ** 2)
    
    # Cumulative variance
    cumulative_var = np.cumsum(S ** 2)
    variance_ratios = cumulative_var / total_variance
    
    # Compute variance at key ranks
    ranks_of_interest = [11, 22, 50, 100, len(S)]
    variance_dict = {}
    
    for rank in ranks_of_interest:
        if rank <= len(S):
            var_captured = variance_ratios[rank - 1] * 100
            variance_dict[f"rank_{rank}"] = var_captured
            
            if rank == EXPECTED_RANK:
                print(f"    {label}: Rank-{rank} captures {var_captured:.2f}% variance")
    
    return S, variance_dict


def plot_svd_spectrum(all_singular_values: Dict[str, np.ndarray], output_path: str):
    """Plot singular value decay across countries."""
    if not HAS_MATPLOTLIB:
        print("⚠️ Skipping plot (matplotlib not available)")
        return
    
    plt.figure(figsize=(12, 6))
    
    for country, S in all_singular_values.items():
        # Normalize by first singular value
        normalized_S = S / S[0]
        plt.semilogy(normalized_S[:100], alpha=0.6, label=country)
    
    # Mark rank-11 and rank-22
    plt.axvline(x=10, color='red', linestyle='--', linewidth=2, label='Rank-11 (Single epitope)')
    plt.axvline(x=21, color='green', linestyle='--', linewidth=2, label='Rank-22 (Dual epitope)')
    
    plt.xlabel('Singular Value Index', fontsize=12)
    plt.ylabel('Normalized Singular Value (log scale)', fontsize=12)
    plt.title('SVD Spectrum: VASIL Cross-Immunity Matrices (11 Countries)', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\n✅ Saved spectrum plot: {output_path}")


def main():
    """
    Main validation routine.
    
    GO/NO-GO Decision Logic:
    1. Load cross-immunity matrices from all 12 countries
    2. Perform SVD on each matrix
    3. Compute variance captured by rank-22 approximation
    4. Average across all countries
    5. Make decision based on threshold
    """
    
    print("=" * 70)
    print("PHASE 0.1: 11-EPITOPE RANK VALIDATION (SVD ANALYSIS)")
    print("=" * 70)
    print(f"\nTarget: Rank-{EXPECTED_RANK} (11 epitopes × 2)")
    print(f"GO threshold: >98% variance")
    print(f"CAUTION threshold: 95-98% variance")
    print(f"NO-GO threshold: <95% variance\n")
    
    all_results = []
    all_singular_values = {}
    
    # Process each country
    for country in COUNTRIES:
        print(f"\n[{country}]")
        
        try:
            cross_immunity_dict = load_cross_immunity_matrix(country)
            
            # Filter to only matrix keys (exclude 'variant_list', 'NTD', 'Mutations', etc.)
            matrix_keys = [k for k, v in cross_immunity_dict.items() 
                          if isinstance(v, np.ndarray) and len(v.shape) == 2 and v.shape[0] > 10]
            
            if not matrix_keys:
                print(f"  ❌ No valid matrices found")
                continue
            
            # Use first antibody group matrix (e.g., 'A')
            sample_key = matrix_keys[0]
            matrix = cross_immunity_dict[sample_key]
            
            print(f"  Matrix shape: {matrix.shape}")
            
            # SVD analysis
            S, variance_dict = analyze_svd_rank(matrix, f"{country}/{sample_key}")
            
            all_singular_values[country] = S
            all_results.append({
                'country': country,
                'variant_group': str(sample_key),
                **variance_dict
            })
            
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Aggregate results
    df = pd.DataFrame(all_results)
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(df.to_string(index=False))
    
    # Compute average variance captured at rank-22
    avg_variance_rank22 = df['rank_22'].mean()
    std_variance_rank22 = df['rank_22'].std()
    
    print("\n" + "=" * 70)
    print("RANK-22 VARIANCE ANALYSIS")
    print("=" * 70)
    print(f"Average variance captured: {avg_variance_rank22:.2f}% ± {std_variance_rank22:.2f}%")
    print(f"Min: {df['rank_22'].min():.2f}%")
    print(f"Max: {df['rank_22'].max():.2f}%")
    
    # GO/NO-GO Decision
    print("\n" + "=" * 70)
    print("GO/NO-GO DECISION")
    print("=" * 70)
    
    if avg_variance_rank22 > 98.0:
        decision = "✅ GO - PROCEED TO POLYALGORITHMIC FUSION"
        recommendation = (
            "Rank-22 captures >98% variance. The 11-epitope hypothesis is STRONGLY validated.\n"
            "RECOMMENDATION: Implement polyalgorithmic ontological fusion for 88-92% accuracy target."
        )
    elif avg_variance_rank22 > 95.0:
        decision = "⚠️ CAUTION - PROCEED TO SINGLE-SCALE EPITOPE"
        recommendation = (
            "Rank-22 captures 95-98% variance. The 11-epitope hypothesis is MODERATELY validated.\n"
            "RECOMMENDATION: Implement single-scale epitope approach for 82-86% accuracy target."
        )
    else:
        decision = "❌ NO-GO - ABORT LOW-RANK APPROACH"
        recommendation = (
            "Rank-22 captures <95% variance. The 11-epitope hypothesis is NOT validated.\n"
            "RECOMMENDATION: Fix Claude Code's weighted_avg kernel instead (fallback path)."
        )
    
    print(f"\n{decision}\n")
    print(recommendation)
    
    # Save results
    output_dir = Path("validation_results")
    output_dir.mkdir(exist_ok=True)
    
    csv_path = output_dir / "svd_variance_analysis.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Saved results: {csv_path}")
    
    # Plot spectrum
    plot_path = output_dir / "svd_spectrum.png"
    plot_svd_spectrum(all_singular_values, str(plot_path))
    
    # Save decision
    decision_path = output_dir / "GO_NO_GO_DECISION.txt"
    with open(decision_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("PHASE 0.1: 11-EPITOPE RANK VALIDATION DECISION\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Date: {pd.Timestamp.now()}\n")
        f.write(f"Average Rank-22 Variance: {avg_variance_rank22:.2f}% ± {std_variance_rank22:.2f}%\n\n")
        f.write(f"DECISION: {decision}\n\n")
        f.write(f"RECOMMENDATION:\n{recommendation}\n")
    
    print(f"✅ Saved decision: {decision_path}")
    
    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    
    if avg_variance_rank22 > 98.0:
        print("1. Run: python scripts/calibrate_polyalgorithmic_fusion.py")
        print("2. If calibration passes: Implement polyalgorithmic_p_neut.cu")
        print("3. Target accuracy: 88-92%")
    elif avg_variance_rank22 > 95.0:
        print("1. Run: python scripts/calibrate_gaussian_sigma.py")
        print("2. If calibration passes: Implement epitope_p_neut.cu")
        print("3. Target accuracy: 82-86%")
    else:
        print("1. Fix compute_weighted_avg_susceptibility kernel in gamma_envelope_reduction.cu")
        print("2. Fix variant filter (0.10 → 0.01)")
        print("3. Target accuracy: 77-82%")
    
    return avg_variance_rank22


if __name__ == "__main__":
    main()
