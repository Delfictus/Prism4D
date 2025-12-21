#!/usr/bin/env python3
"""
Direct feature extraction from PRISM mega-fused output.

Calls prism-lbs in pure-GPU mode and extracts 92-dim combined_features
from MegaFusedOutput (if available in output).
"""

import subprocess
import json
import numpy as np
from pathlib import Path
import tempfile
import sys

def extract_features_from_pdb(pdb_path: Path, prism_binary: Path) -> np.ndarray:
    """
    Extract 92-dim features from a PDB file using PRISM.

    Args:
        pdb_path: Path to PDB structure
        prism_binary: Path to prism-lbs binary

    Returns:
        Feature matrix [n_residues, 92]
    """
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        output_json = Path(f.name)

    try:
        # Run PRISM in pure-GPU mode
        cmd = [
            str(prism_binary),
            "--input", str(pdb_path),
            "--output", str(output_json),
            "--format", "json",
            "--pure-gpu"
        ]

        env = {
            "PRISM_PTX_DIR": "./target/ptx",
            "RUST_LOG": "warn",
        }

        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
            cwd="/mnt/c/Users/Predator/Desktop/PRISM"
        )

        if result.returncode != 0:
            print(f"❌ PRISM failed: {result.stderr}")
            return None

        # Parse JSON output
        with open(output_json) as f:
            data = json.load(f)

        # Check if combined_features exists
        if 'combined_features' in data:
            features = np.array(data['combined_features'])
            print(f"✅ Extracted features: {features.shape}")
            return features
        else:
            print(f"⚠️  No combined_features in JSON output")
            print(f"   Available keys: {list(data.keys())}")
            return None

    finally:
        if output_json.exists():
            output_json.unlink()


if __name__ == "__main__":
    print("="*70)
    print("DIRECT FEATURE EXTRACTION TEST")
    print("="*70)
    print()

    prism_binary = Path("/mnt/c/Users/Predator/Desktop/PRISM/target/release/prism-lbs")
    test_pdb = Path("/mnt/c/Users/Predator/Desktop/PRISM/prism-escape-benchmark/data/raw/structures/6m0j.pdb")

    if not prism_binary.exists():
        print(f"❌ PRISM binary not found: {prism_binary}")
        sys.exit(1)

    if not test_pdb.exists():
        print(f"❌ Test PDB not found: {test_pdb}")
        sys.exit(1)

    print(f"Testing feature extraction on: {test_pdb.name}")
    print()

    features = extract_features_from_pdb(test_pdb, prism_binary)

    if features is not None:
        print(f"\n✅ SUCCESS! Extracted 92-dim features")
        print(f"   Shape: {features.shape}")
        print(f"   Feature range: [{features.min():.4f}, {features.max():.4f}]")
        print(f"\n   Physics features (indices 80-91):")
        for i in range(80, min(92, features.shape[1])):
            print(f"     Feature {i}: mean={features[:,i].mean():.4f}, std={features[:,i].std():.4f}")
    else:
        print("\n❌ Feature extraction failed")
        print("   combined_features not in JSON output")
        print("\n   Current JSON output only has: pockets, n_pockets")
        print("   Need to modify binary to export combined_features")
