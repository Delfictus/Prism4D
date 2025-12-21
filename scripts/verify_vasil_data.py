#!/usr/bin/env python3
"""
verify_vasil_data.py
Verifies that VASIL benchmark data was downloaded correctly
"""

import pandas as pd
from pathlib import Path
import sys

def verify_vasil_data():
    """Verify VASIL repository data"""
    print("\n" + "="*60)
    print("VASIL Benchmark Data Verification")
    print("="*60)

    data_dir = Path("data/prism_ve_benchmark")

    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        return False

    success = True

    # Check VASIL data
    print("\n[1/4] VASIL Lineage Frequencies")
    print("-" * 60)

    vasil_dir = data_dir / "vasil" / "ByCountry"
    countries = ["Germany", "USA", "UK", "Japan", "Brazil", "France",
                 "Canada", "Australia", "Denmark", "Mexico", "SouthAfrica", "Sweden"]

    for country in countries:
        country_dir = vasil_dir / country / "results"

        # Check for lineage frequency files
        lineage_files = [
            "Daily_Lineages_Freq_1_percent.csv",
            "Daily_Lineages_Freq_seq_thres_100.csv",
            "Daily_SpikeGroups_Freq.csv"
        ]

        found_files = []
        for lf in lineage_files:
            if (country_dir / lf).exists():
                found_files.append(lf)

        if found_files:
            # Read one of the files to get stats
            freq_file = country_dir / found_files[0]
            try:
                df = pd.read_csv(freq_file)
                print(f"\n{country}:")
                print(f"  File: {found_files[0]}")
                print(f"  Rows: {len(df):,}")
                if 'date' in df.columns:
                    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
                if 'lineage' in df.columns or 'Lineage' in df.columns:
                    lineage_col = 'lineage' if 'lineage' in df.columns else 'Lineage'
                    print(f"  Lineages: {df[lineage_col].nunique()}")
                print(f"  Columns: {list(df.columns)[:5]}...")
            except Exception as e:
                print(f"\n{country}: ERROR reading file: {e}")
                success = False
        else:
            print(f"\n{country}: WARNING - No lineage frequency files found")

    # Check DMS data
    print("\n\n[2/4] Bloom Lab DMS Escape Data")
    print("-" * 60)

    dms_dir = data_dir / "dms" / "processed_data"
    if dms_dir.exists():
        dms_files = list(dms_dir.glob("*.csv"))
        print(f"Found {len(dms_files)} processed data files:")
        for df_file in sorted(dms_files):
            try:
                df = pd.read_csv(df_file)
                print(f"  {df_file.name}: {len(df):,} rows, {len(df.columns)} columns")
            except Exception as e:
                print(f"  {df_file.name}: ERROR - {e}")
                success = False
    else:
        print("ERROR: DMS processed_data directory not found")
        success = False

    # Check GInPipe
    print("\n\n[3/4] GInPipe Repository")
    print("-" * 60)

    ginpipe_dir = data_dir / "GInPipe"
    if ginpipe_dir.exists():
        scripts_dir = ginpipe_dir / "scripts"
        if scripts_dir.exists():
            script_subdirs = [d for d in scripts_dir.iterdir() if d.is_dir()]
            print(f"GInPipe cloned successfully")
            print(f"  Scripts subdirectories: {len(script_subdirs)}")
            print(f"  Subdirs: {[d.name for d in script_subdirs[:5]]}")
        else:
            print("WARNING: scripts directory not found in GInPipe")
    else:
        print("ERROR: GInPipe directory not found")
        success = False

    # Check PDB structures
    print("\n\n[4/4] PDB Structure Files")
    print("-" * 60)

    structures_dir = data_dir / "structures"
    pdb_ids = ["6M0J", "6VXX", "6VYB", "7BNN", "7CAB", "7TFO", "7PUY"]

    if structures_dir.exists():
        for pdb_id in pdb_ids:
            pdb_file = structures_dir / f"{pdb_id}.pdb"
            if pdb_file.exists():
                size = pdb_file.stat().st_size
                if size > 1000:  # At least 1KB
                    print(f"  {pdb_id}.pdb: {size:,} bytes ✓")
                else:
                    print(f"  {pdb_id}.pdb: {size} bytes - WARNING: file too small")
                    success = False
            else:
                print(f"  {pdb_id}.pdb: MISSING")
                success = False
    else:
        print("ERROR: structures directory not found")
        success = False

    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)

    total_size = sum(f.stat().st_size for f in data_dir.rglob('*') if f.is_file())
    print(f"Total data size: {total_size / (1024**3):.2f} GB")
    print(f"Total files: {len(list(data_dir.rglob('*')))}")

    if success:
        print("\n✓ All verification checks passed!")
        print("\nKey datasets available for benchmarking:")
        print("  - VASIL lineage frequencies (12 countries)")
        print("  - Bloom Lab DMS antibody escape maps")
        print("  - GInPipe analysis pipeline")
        print("  - PDB structures for structural analysis")
        return True
    else:
        print("\n✗ Some verification checks failed")
        return False

if __name__ == "__main__":
    success = verify_vasil_data()
    sys.exit(0 if success else 1)
