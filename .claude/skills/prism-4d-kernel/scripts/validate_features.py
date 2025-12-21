#!/usr/bin/env python3
"""
PRISM>4D Feature Validation Script

Validates that combined_features output has correct dimensions and non-zero values.
Run after kernel execution to verify Stage 7/8 integration.

Usage:
    python validate_features.py features.bin n_residues
    python validate_features.py --from-rust  # Reads from stdin
"""

import sys
import struct
import numpy as np
from typing import Tuple, List

# Feature layout constants (must match CUDA/Rust)
TOTAL_FEATURES = 101
TDA_FEATURES = 48        # 0-47
RESERVOIR_FEATURES = 32  # 48-79
PHYSICS_FEATURES = 12    # 80-91
FITNESS_FEATURES = 4     # 92-95
CYCLE_FEATURES = 5       # 96-100

# Expected ranges
FEATURE_RANGES = {
    'ddg_binding': (-5.0, 5.0),      # Feature 92
    'ddg_stability': (-5.0, 5.0),    # Feature 93
    'expression': (0.0, 1.0),        # Feature 94
    'transmit': (0.0, 1.0),          # Feature 95
    'phase': (0.0, 5.0),             # Feature 96 (int 0-5)
    'emergence_prob': (0.0, 1.0),    # Feature 97
    'time_to_peak': (0.0, 24.0),     # Feature 98 (months)
    'frequency': (0.0, 1.0),         # Feature 99
    'velocity': (-0.5, 0.5),         # Feature 100
}


def load_features_binary(path: str, n_residues: int) -> np.ndarray:
    """Load features from binary file (f32 little-endian)."""
    with open(path, 'rb') as f:
        data = f.read()
    
    expected_bytes = n_residues * TOTAL_FEATURES * 4
    if len(data) != expected_bytes:
        raise ValueError(f"Expected {expected_bytes} bytes, got {len(data)}")
    
    features = np.frombuffer(data, dtype=np.float32)
    return features.reshape(n_residues, TOTAL_FEATURES)


def validate_dimensions(features: np.ndarray, n_residues: int) -> List[str]:
    """Check array dimensions."""
    errors = []
    
    if features.shape[0] != n_residues:
        errors.append(f"Wrong residue count: {features.shape[0]} vs {n_residues}")
    
    if features.shape[1] != TOTAL_FEATURES:
        errors.append(f"Wrong feature count: {features.shape[1]} vs {TOTAL_FEATURES}")
    
    return errors


def validate_fitness_features(features: np.ndarray) -> Tuple[List[str], List[str]]:
    """Validate Stage 7 fitness features (92-95)."""
    errors = []
    warnings = []
    
    # Extract fitness columns
    ddg_bind = features[:, 92]
    ddg_stab = features[:, 93]
    expression = features[:, 94]
    transmit = features[:, 95]
    
    # Check for all zeros (Stage 7 not integrated)
    if np.allclose(ddg_bind, 0.0):
        errors.append("Feature 92 (ddg_binding) is all zeros - Stage 7 not integrated!")
    
    if np.allclose(ddg_stab, 0.0):
        errors.append("Feature 93 (ddg_stability) is all zeros - Stage 7 not integrated!")
    
    if np.allclose(expression, 0.0):
        errors.append("Feature 94 (expression) is all zeros - Stage 7 not integrated!")
    
    if np.allclose(transmit, 0.0):
        errors.append("Feature 95 (transmit) is all zeros - Stage 7 not integrated!")
    
    # Check ranges
    if not np.all((expression >= 0) & (expression <= 1)):
        warnings.append(f"Feature 94 (expression) out of [0,1]: min={expression.min():.3f}, max={expression.max():.3f}")
    
    if not np.all((transmit >= 0) & (transmit <= 1)):
        warnings.append(f"Feature 95 (transmit) out of [0,1]: min={transmit.min():.3f}, max={transmit.max():.3f}")
    
    # Check for NaN/Inf
    for i, (name, col) in enumerate([('ddg_bind', ddg_bind), ('ddg_stab', ddg_stab), 
                                      ('expression', expression), ('transmit', transmit)]):
        if np.any(np.isnan(col)):
            errors.append(f"Feature {92+i} ({name}) contains NaN values!")
        if np.any(np.isinf(col)):
            errors.append(f"Feature {92+i} ({name}) contains Inf values!")
    
    return errors, warnings


def validate_cycle_features(features: np.ndarray) -> Tuple[List[str], List[str]]:
    """Validate Stage 8 cycle features (96-100)."""
    errors = []
    warnings = []
    
    phase = features[:, 96]
    emergence = features[:, 97]
    time_peak = features[:, 98]
    frequency = features[:, 99]
    velocity = features[:, 100]
    
    # Check for all zeros (Stage 8 not integrated)
    if np.allclose(phase, 0.0) and np.allclose(emergence, 0.0):
        errors.append("Features 96-100 all zeros - Stage 8 not integrated!")
    
    # Phase should be integers 0-5
    unique_phases = np.unique(phase.astype(int))
    invalid_phases = [p for p in unique_phases if p < 0 or p > 5]
    if invalid_phases:
        errors.append(f"Feature 96 (phase) has invalid values: {invalid_phases}")
    
    # Emergence probability should be [0,1]
    if not np.all((emergence >= 0) & (emergence <= 1)):
        warnings.append(f"Feature 97 (emergence_prob) out of [0,1]: min={emergence.min():.3f}, max={emergence.max():.3f}")
    
    # Frequency should be [0,1]
    if not np.all((frequency >= 0) & (frequency <= 1)):
        warnings.append(f"Feature 99 (frequency) out of [0,1]: min={frequency.min():.3f}, max={frequency.max():.3f}")
    
    # Velocity typically [-0.5, 0.5]
    if np.any(np.abs(velocity) > 1.0):
        warnings.append(f"Feature 100 (velocity) unusually large: min={velocity.min():.3f}, max={velocity.max():.3f}")
    
    return errors, warnings


def validate_tda_features(features: np.ndarray) -> Tuple[List[str], List[str]]:
    """Validate TDA features (0-47)."""
    errors = []
    warnings = []
    
    tda = features[:, :48]
    
    # Check for all zeros
    if np.allclose(tda, 0.0):
        errors.append("TDA features (0-47) are all zeros!")
    
    # Check for NaN/Inf
    if np.any(np.isnan(tda)):
        nan_cols = np.where(np.any(np.isnan(tda), axis=0))[0]
        errors.append(f"TDA features contain NaN at columns: {nan_cols.tolist()}")
    
    return errors, warnings


def print_summary(features: np.ndarray):
    """Print feature statistics summary."""
    print("\n=== FEATURE SUMMARY ===")
    print(f"Shape: {features.shape}")
    print(f"Total elements: {features.size}")
    print(f"Non-zero elements: {np.count_nonzero(features)}")
    print(f"NaN count: {np.count_nonzero(np.isnan(features))}")
    print(f"Inf count: {np.count_nonzero(np.isinf(features))}")
    
    print("\n=== STAGE 7 (FITNESS) ===")
    for i, name in enumerate(['ddg_binding', 'ddg_stability', 'expression', 'transmit']):
        col = features[:, 92 + i]
        print(f"  Feature {92+i} ({name}): min={col.min():.4f}, max={col.max():.4f}, mean={col.mean():.4f}")
    
    print("\n=== STAGE 8 (CYCLE) ===")
    for i, name in enumerate(['phase', 'emergence_prob', 'time_to_peak', 'frequency', 'velocity']):
        col = features[:, 96 + i]
        print(f"  Feature {96+i} ({name}): min={col.min():.4f}, max={col.max():.4f}, mean={col.mean():.4f}")
    
    print("\n=== PHASE DISTRIBUTION ===")
    phases = features[:, 96].astype(int)
    phase_names = ['NAIVE', 'EXPLORING', 'ESCAPED', 'COSTLY', 'REVERTING', 'FIXED']
    for p in range(6):
        count = np.sum(phases == p)
        pct = count / len(phases) * 100
        print(f"  Phase {p} ({phase_names[p]}): {count} ({pct:.1f}%)")


def main():
    if len(sys.argv) < 2:
        print("Usage: python validate_features.py <features.bin> <n_residues>")
        print("       python validate_features.py --summary <features.bin> <n_residues>")
        sys.exit(1)
    
    summary_only = False
    args = sys.argv[1:]
    
    if args[0] == '--summary':
        summary_only = True
        args = args[1:]
    
    if len(args) != 2:
        print("Error: Need features file and residue count")
        sys.exit(1)
    
    features_path = args[0]
    n_residues = int(args[1])
    
    print(f"Loading features from {features_path} (n_residues={n_residues})")
    
    try:
        features = load_features_binary(features_path, n_residues)
    except Exception as e:
        print(f"ERROR loading features: {e}")
        sys.exit(1)
    
    all_errors = []
    all_warnings = []
    
    # Run validations
    all_errors.extend(validate_dimensions(features, n_residues))
    
    errors, warnings = validate_tda_features(features)
    all_errors.extend(errors)
    all_warnings.extend(warnings)
    
    errors, warnings = validate_fitness_features(features)
    all_errors.extend(errors)
    all_warnings.extend(warnings)
    
    errors, warnings = validate_cycle_features(features)
    all_errors.extend(errors)
    all_warnings.extend(warnings)
    
    # Print results
    if all_errors:
        print("\n❌ ERRORS:")
        for e in all_errors:
            print(f"  - {e}")
    
    if all_warnings:
        print("\n⚠️  WARNINGS:")
        for w in all_warnings:
            print(f"  - {w}")
    
    if not all_errors and not all_warnings:
        print("\n✅ All validations passed!")
    
    print_summary(features)
    
    sys.exit(1 if all_errors else 0)


if __name__ == '__main__':
    main()
