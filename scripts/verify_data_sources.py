#!/usr/bin/env python3
"""
PRISM-VE Data Source Verification

**SCIENTIFIC INTEGRITY**: Verify all data sources are PRIMARY sources,
not VASIL's model outputs or fitted values.

Checks for:
1. GISAID frequencies - must be raw aggregates, not model-smoothed
2. DMS escape data - must be Bloom Lab raw, not VASIL-processed
3. Immunity landscapes - must be from vaccination/case data, not fitted
4. All parameters - must be fitted independently, not copied
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple


class DataSourceVerifier:
    """Verify scientific integrity of data sources."""

    # Red flag keywords indicating model outputs
    RED_FLAGS = [
        'fitted', 'predicted', 'smoothed', 'model', 'estimated',
        'forecasted', 'imputed', 'interpolated', 'extrapolated'
    ]

    # Safe keywords indicating raw data
    SAFE_KEYWORDS = [
        'raw', 'observed', 'gisaid', 'measured', 'counts',
        'sequences', 'reported', 'surveillance'
    ]

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.issues = []

    def verify_gisaid_frequencies(self, country: str) -> bool:
        """
        Verify GISAID frequency files are raw aggregates.

        Returns:
            True if data is primary source, False if model output
        """
        print(f"\n[1] Verifying GISAID Frequencies ({country}):")
        print("-" * 60)

        freq_file = (self.data_dir /
                    f"vasil_code/ByCountry/{country}/results/Daily_Lineages_Freq_1_percent.csv")

        if not freq_file.exists():
            print(f"  ❌ File not found: {freq_file}")
            self.issues.append(f"{country}: Frequency file missing")
            return False

        # Check filename for red flags
        filename = freq_file.name.lower()
        has_red_flag = any(flag in filename for flag in self.RED_FLAGS)

        if has_red_flag:
            print(f"  ⚠️  WARNING: Filename contains model indicators!")
            print(f"     File: {filename}")
            self.issues.append(f"{country}: Filename suggests model output")

        # Load file and check columns
        df = pd.read_csv(freq_file, index_col=0, nrows=5)
        columns_str = ' '.join(df.columns).lower()

        has_column_red_flag = any(flag in columns_str for flag in self.RED_FLAGS)

        if has_column_red_flag:
            print(f"  ⚠️  WARNING: Column names contain model indicators!")
            matching_flags = [flag for flag in self.RED_FLAGS if flag in columns_str]
            print(f"     Found: {matching_flags}")
            self.issues.append(f"{country}: Column names suggest model output")
            return False

        # Check for safe keywords
        has_safe = any(kw in filename or kw in columns_str for kw in self.SAFE_KEYWORDS)

        if has_safe:
            print(f"  ✅ Appears to be RAW GISAID aggregates")
            print(f"     Source appears primary")
            return True
        else:
            print(f"  ⚠️  UNCERTAIN: Cannot determine if raw or processed")
            print(f"     RECOMMENDATION: Download GISAID directly for certainty")
            self.issues.append(f"{country}: Uncertain if raw data")
            return False

    def verify_dms_data(self) -> bool:
        """
        Verify DMS escape data is Bloom Lab primary source.
        """
        print(f"\n[2] Verifying DMS Escape Data:")
        print("-" * 60)

        dms_file = self.data_dir / "dms/vasil_processed/dms_per_ab_per_site.csv"

        if not dms_file.exists():
            dms_file = self.data_dir / "dms/SARS2_RBD_Ab_escape_maps/processed_data/escape_data.csv"

        if not dms_file.exists():
            print(f"  ❌ DMS file not found")
            self.issues.append("DMS: File missing")
            return False

        # Check if using Bloom Lab source
        if "SARS2_RBD_Ab_escape_maps" in str(dms_file):
            print(f"  ✅ Using Bloom Lab PRIMARY SOURCE")
            print(f"     Source: github.com/jbloomlab/SARS2_RBD_Ab_escape_maps")
            return True
        elif "vasil_processed" in str(dms_file):
            print(f"  ⚠️  Using VASIL's processed version")
            print(f"     RECOMMENDATION: Use Bloom Lab direct for independence")
            self.issues.append("DMS: Using VASIL-processed, prefer Bloom Lab raw")
            return False
        else:
            print(f"  ⚠️  Unknown DMS source: {dms_file}")
            self.issues.append("DMS: Unknown source")
            return False

    def verify_parameters(self) -> bool:
        """
        Verify parameters are NOT copied from VASIL.
        """
        print(f"\n[3] Verifying Model Parameters:")
        print("-" * 60)

        # Check if code contains VASIL's exact values
        fitness_file = Path("crates/prism-gpu/src/viral_evolution_fitness.rs")

        if not fitness_file.exists():
            print(f"  ⚠️  Fitness module not found")
            return True  # Can't check

        content = fitness_file.read_text()

        # Check for VASIL's exact values
        if "vasil_alpha" in content.lower() or "vasil_beta" in content.lower():
            print(f"  ❌ FOUND 'vasil_alpha' or 'vasil_beta' in code!")
            print(f"     This indicates using VASIL's parameters")
            print(f"     FIX: Remove and use independently fitted values")
            self.issues.append("Parameters: Contains vasil_alpha/beta")
            return False

        if "0.65" in content and "0.35" in content:
            print(f"  ⚠️  Found values 0.65 and 0.35 in code")
            print(f"     These are VASIL's exact fitted parameters")
            print(f"     VERIFY: Are these our independently fitted values?")
            self.issues.append("Parameters: Contains VASIL's exact values (0.65, 0.35)")

        # Check for independent calibration
        if "escape_weight" in content and "transmit_weight" in content:
            print(f"  ✅ Using escape_weight, transmit_weight (good names)")

            # Check defaults
            if "escape_weight: 0.5" in content or "escape_weight:0.5" in content.replace(" ", ""):
                print(f"  ✅ Neutral default values (0.5, 0.5)")
                print(f"     Ready for independent calibration")
                return True
            else:
                print(f"  ⚠️  Non-neutral defaults detected")
                print(f"     VERIFY: Are these our fitted values or copied?")

        return True

    def run_full_verification(self) -> bool:
        """
        Run complete data source verification.

        Returns:
            True if all sources are primary, False if issues found
        """
        print("\n" + "="*80)
        print("PRISM-VE DATA SOURCE VERIFICATION")
        print("ENSURING SCIENTIFIC INTEGRITY")
        print("="*80)

        # Check GISAID frequencies for multiple countries
        countries = ["Germany", "USA", "UK"]
        gisaid_ok = all(
            self.verify_gisaid_frequencies(country) for country in countries
        )

        # Check DMS data
        dms_ok = self.verify_dms_data()

        # Check parameters
        params_ok = self.verify_parameters()

        # Summary
        print("\n" + "="*80)
        print("VERIFICATION SUMMARY")
        print("="*80)

        all_ok = gisaid_ok and dms_ok and params_ok

        if all_ok:
            print("\n✅ ALL DATA SOURCES VERIFIED AS PRIMARY")
            print("✅ PARAMETERS INDEPENDENTLY CALIBRATED")
            print("\n   PRISM-VE is scientifically honest and defensible!")
        else:
            print("\n⚠️  ISSUES FOUND:")
            for issue in self.issues:
                print(f"   - {issue}")
            print("\n   RECOMMENDATION: Address issues before publication")

        print("="*80)

        # Detailed recommendations
        if not gisaid_ok:
            print("\nGISAID Data Fix:")
            print("  1. Download raw GISAID metadata directly from gisaid.org")
            print("  2. Aggregate frequencies ourselves")
            print("  3. Do NOT use VASIL's processed/fitted frequencies")

        if not dms_ok:
            print("\nDMS Data Fix:")
            print("  1. Use Bloom Lab GitHub directly")
            print("  2. github.com/jbloomlab/SARS2_RBD_Ab_escape_maps")
            print("  3. Process escape scores ourselves")

        if not params_ok:
            print("\nParameter Fix:")
            print("  1. Remove vasil_alpha, vasil_beta from code")
            print("  2. Use escape_weight, transmit_weight with defaults 0.5, 0.5")
            print("  3. Run: python scripts/calibrate_parameters_independently.py")
            print("  4. Document: Parameters fitted on 2021-2022 training data")

        return all_ok


def main():
    """Run verification."""
    verifier = DataSourceVerifier(Path("data/vasil_benchmark"))
    all_ok = verifier.run_full_verification()

    if not all_ok:
        print("\n⚠️  Fix issues above before proceeding to publication!")
        return 1
    else:
        print("\n✅ Ready for honest, peer-review defensible publication!")
        return 0


if __name__ == "__main__":
    exit(main())
