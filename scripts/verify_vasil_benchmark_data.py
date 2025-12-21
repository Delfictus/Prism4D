#!/usr/bin/env python3
"""
Verify VASIL benchmark data download completeness.
"""

import pandas as pd
from pathlib import Path
import sys


def verify_vasil_benchmark_data():
    """Verify all VASIL benchmark data is downloaded correctly."""

    print("\n" + "="*80)
    print("VASIL Benchmark Data Verification")
    print("="*80)

    base_dir = Path("data/vasil_benchmark")

    if not base_dir.exists():
        print(f"\n❌ ERROR: Benchmark directory not found: {base_dir}")
        print("Run: bash scripts/download_vasil_complete_benchmark_data.sh")
        return False

    success = True
    checks = []

    # 1. DMS Data
    print("\n[1/5] DMS Antibody Escape Data")
    print("-" * 80)

    dms_processed = base_dir / "dms" / "vasil_processed" / "dms_per_ab_per_site.csv"
    if dms_processed.exists():
        df = pd.read_csv(dms_processed)
        print(f"✓ VASIL DMS data: {len(df):,} rows")
        checks.append(("DMS escape scores", True))
    else:
        print(f"❌ Missing: {dms_processed}")
        success = False
        checks.append(("DMS escape scores", False))

    ab_mapping = base_dir / "dms" / "vasil_processed" / "antibody_mapping.csv"
    if ab_mapping.exists():
        df = pd.read_csv(ab_mapping)
        print(f"✓ Antibody mapping: {len(df):,} antibodies")
        checks.append(("Antibody mapping", True))
    else:
        print(f"❌ Missing: {ab_mapping}")
        success = False
        checks.append(("Antibody mapping", False))

    # 2. VASIL Code & Frequencies
    print("\n[2/5] VASIL Lineage Frequencies")
    print("-" * 80)

    vasil_dir = base_dir / "vasil_code" / "ByCountry"
    countries = [
        "Germany", "USA", "UK", "Japan", "Brazil", "France",
        "Canada", "Denmark", "Australia", "Sweden", "Mexico", "SouthAfrica"
    ]

    found_countries = 0
    for country in countries:
        freq_file = vasil_dir / country / "results" / "Daily_Lineages_Freq_1_percent.csv"
        if freq_file.exists():
            df = pd.read_csv(freq_file, index_col=0)
            found_countries += 1
            print(f"✓ {country:<15} {len(df):>4} dates, {len(df.columns):>3} lineages")
        else:
            print(f"❌ {country:<15} Missing")
            success = False

    if found_countries == 12:
        checks.append(("VASIL frequencies (12 countries)", True))
    else:
        checks.append((f"VASIL frequencies ({found_countries}/12 countries)", False))

    # 3. Protein Structures
    print("\n[3/5] Protein Structures")
    print("-" * 80)

    structures_dir = base_dir / "structures"
    expected_pdbs = ["6M0J", "6VXX", "6VYB", "7BNN", "7CAB", "7TFO", "7PUY",
                     "1RVX", "5FYL", "5EVM", "7TY0", "7TXZ"]

    found_pdbs = 0
    for pdb in expected_pdbs:
        pdb_file = structures_dir / f"{pdb}.pdb"
        if pdb_file.exists():
            size = pdb_file.stat().st_size
            if size > 1000:  # At least 1KB
                found_pdbs += 1
                print(f"✓ {pdb}.pdb ({size/1024:.1f} KB)")
            else:
                print(f"⚠️  {pdb}.pdb ({size} bytes - too small)")
                success = False
        else:
            print(f"❌ {pdb}.pdb missing")
            success = False

    if found_pdbs == len(expected_pdbs):
        checks.append(("PDB structures", True))
    else:
        checks.append((f"PDB structures ({found_pdbs}/{len(expected_pdbs)})", False))

    # 4. Surveillance Data
    print("\n[4/5] Surveillance Data")
    print("-" * 80)

    owid_file = base_dir / "surveillance" / "owid_covid_data.csv"
    if owid_file.exists():
        size = owid_file.stat().st_size / (1024**2)
        print(f"✓ OWID COVID data: {size:.1f} MB")
        checks.append(("OWID surveillance data", True))
    else:
        print(f"❌ Missing: {owid_file}")
        checks.append(("OWID surveillance data", False))
        success = False

    ginpipe_dir = base_dir / "surveillance" / "GInPipe"
    if ginpipe_dir.exists() and (ginpipe_dir / "scripts").exists():
        print(f"✓ GInPipe repository cloned")
        checks.append(("GInPipe", True))
    else:
        print(f"❌ GInPipe missing or incomplete")
        checks.append(("GInPipe", False))
        success = False

    # 5. Documentation
    print("\n[5/5] Documentation")
    print("-" * 80)

    readme = base_dir / "README.md"
    if readme.exists():
        print(f"✓ README.md")
        checks.append(("Documentation", True))
    else:
        print(f"⚠️  README.md missing")

    gisaid_instructions = base_dir / "gisaid" / "DOWNLOAD_INSTRUCTIONS.md"
    if gisaid_instructions.exists():
        print(f"✓ GISAID download instructions")
    else:
        print(f"⚠️  GISAID instructions missing")

    # Summary
    print("\n" + "="*80)
    print("Verification Summary")
    print("="*80)

    total_size = sum(f.stat().st_size for f in base_dir.rglob('*') if f.is_file())
    print(f"\nTotal data size: {total_size / (1024**3):.2f} GB")
    print(f"Total files: {len(list(base_dir.rglob('*')))}")

    print("\nComponent Status:")
    for component, status in checks:
        symbol = "✓" if status else "❌"
        print(f"  {symbol} {component}")

    if success:
        print("\n" + "="*80)
        print("✅ ALL CRITICAL DATA DOWNLOADED SUCCESSFULLY!")
        print("="*80)
        print("\nYou are ready to:")
        print("  1. Run benchmark: python scripts/benchmark_vs_vasil.py")
        print("  2. Develop fitness module")
        print("  3. Develop cycle module")
        print("\nOptional:")
        print("  - Download GISAID data (see data/vasil_benchmark/gisaid/)")
        print("  - Download UK ONS data (see data/vasil_benchmark/surveillance/uk_ons/)")
        return True
    else:
        print("\n" + "="*80)
        print("❌ SOME DATA IS MISSING OR INCOMPLETE")
        print("="*80)
        print("\nRe-run: bash scripts/download_vasil_complete_benchmark_data.sh")
        return False


if __name__ == "__main__":
    success = verify_vasil_benchmark_data()
    sys.exit(0 if success else 1)
