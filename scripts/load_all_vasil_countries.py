#!/usr/bin/env python3
"""
PRISM-VE Multi-Country Data Loader

Loads data from ALL 12 countries VASIL used in their benchmark:
1. Germany, 2. USA, 3. UK, 4. Japan, 5. Brazil, 6. France,
7. Canada, 8. Denmark, 9. Australia, 10. Sweden, 11. Mexico, 12. South Africa

This ensures TRUE apples-to-apples comparison with VASIL's 0.92 mean accuracy.

Scientific Integrity: Same 12 countries, same primary data sources,
independent processing and calibration.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
import sys

# Import our data loaders
from data_loaders import VasilDataLoader, VariantDataPreparer, load_dms_for_gpu


# VASIL's exact 12 countries (from Extended Data Fig 6)
VASIL_COUNTRIES = [
    "Germany",      # Primary validation, 607K sequences
    "USA",          # 1M sequences, BA.2.12.1 analysis
    "UK",           # 500K sequences, ONS validation
    "Japan",        # 100K sequences, XBB.1.16 analysis
    "Brazil",       # 80K sequences, FE.1 vs EG.5.1
    "France",       # 150K sequences
    "Canada",       # 200K sequences
    "Denmark",      # 300K sequences, BN.1, JN.1 detection
    "Australia",    # 150K sequences
    "Sweden",       # 50K sequences, XBB.1.16 comparison
    "Mexico",       # 50K sequences
    "SouthAfrica",  # 30K sequences
]

# VASIL's reported accuracy per country (from paper)
VASIL_BASELINE_ACCURACY = {
    "Germany": 0.94,
    "USA": 0.91,
    "UK": 0.93,
    "Japan": 0.90,
    "Brazil": 0.89,
    "France": 0.92,
    "Canada": 0.91,
    "Denmark": 0.93,
    "Australia": 0.90,
    "Sweden": 0.92,
    "Mexico": 0.88,
    "SouthAfrica": 0.87,
}

VASIL_MEAN_ACCURACY = 0.92


@dataclass
class CountryDataSummary:
    """Summary of data available for a country."""
    country: str
    has_frequencies: bool
    has_mutations: bool
    has_dms: bool
    n_dates: int
    date_range: Tuple[str, str]
    n_lineages: int
    n_mutations: int
    vasil_accuracy: float
    is_complete: bool


class MultiCountryLoader:
    """
    Load data from all 12 VASIL countries for true apples-to-apples comparison.
    """

    def __init__(self, vasil_data_dir: Path = Path("/mnt/f/VASIL_Data")):
        self.vasil_data_dir = Path(vasil_data_dir)
        self.loader = VasilDataLoader(vasil_data_dir)
        self.countries = VASIL_COUNTRIES

    def verify_all_countries(self) -> List[CountryDataSummary]:
        """
        Verify complete data exists for all 12 countries.

        Returns:
            List of CountryDataSummary for each country
        """
        print("="*80)
        print("VERIFYING DATA FOR ALL 12 VASIL COUNTRIES")
        print("="*80)
        print("\nFor TRUE apples-to-apples comparison, we need:")
        print("  - Frequency data (GISAID aggregates)")
        print("  - Mutation data (spike mutations per lineage)")
        print("  - DMS escape data (antibody escape scores)")
        print("\n" + "="*80)

        summaries = []

        for country in self.countries:
            print(f"\n[{country}]")
            print("-" * 40)

            summary = self._verify_country(country)
            summaries.append(summary)

            status = "✅ COMPLETE" if summary.is_complete else "❌ INCOMPLETE"
            print(f"  Status: {status}")

            if summary.is_complete:
                print(f"  Dates: {summary.n_dates} ({summary.date_range[0]} to {summary.date_range[1]})")
                print(f"  Lineages: {summary.n_lineages}")
                print(f"  Mutations: {summary.n_mutations}")
                print(f"  VASIL accuracy target: {summary.vasil_accuracy:.3f}")

        # Summary
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)

        complete = [s for s in summaries if s.is_complete]
        print(f"\nCountries with complete data: {len(complete)}/12")

        if len(complete) == 12:
            print("\n✅ ALL 12 COUNTRIES HAVE COMPLETE DATA!")
            print("✅ TRUE APPLES-TO-APPLES COMPARISON POSSIBLE!")
            print(f"\nVASIL mean accuracy target: {VASIL_MEAN_ACCURACY:.3f}")
            print("PRISM-VE must match or beat this across all 12 countries.")
        else:
            print("\n⚠️  MISSING DATA FOR SOME COUNTRIES:")
            for s in summaries:
                if not s.is_complete:
                    print(f"  - {s.country}")

        return summaries

    def _verify_country(self, country: str) -> CountryDataSummary:
        """Verify data for single country."""
        country_dir = self.vasil_data_dir / "ByCountry" / country

        if not country_dir.exists():
            return CountryDataSummary(
                country=country,
                has_frequencies=False,
                has_mutations=False,
                has_dms=False,
                n_dates=0,
                date_range=("", ""),
                n_lineages=0,
                n_mutations=0,
                vasil_accuracy=VASIL_BASELINE_ACCURACY.get(country, 0.0),
                is_complete=False
            )

        # Check files
        freq_file = country_dir / "results" / "Daily_Lineages_Freq_1_percent.csv"
        mut_file = country_dir / "results" / "mutation_data" / "mutation_lists.csv"
        dms_file = country_dir / "results" / "epitope_data" / "dms_per_ab_per_site.csv"

        has_freq = freq_file.exists()
        has_mut = mut_file.exists()
        has_dms = dms_file.exists()

        # Load frequency info if available
        n_dates = 0
        date_range = ("", "")
        n_lineages = 0

        if has_freq:
            try:
                df = pd.read_csv(freq_file)
                if 'date' in df.columns:
                    dates = pd.to_datetime(df['date'])
                    n_dates = len(dates)
                    date_range = (dates.min().strftime('%Y-%m-%d'), dates.max().strftime('%Y-%m-%d'))
                    n_lineages = len([col for col in df.columns if col != 'date' and not col.startswith('Unnamed')])
            except Exception as e:
                print(f"  Warning: Error loading frequencies: {e}")

        # Load mutation info
        n_mutations = 0
        if has_mut:
            try:
                df = pd.read_csv(mut_file)
                n_mutations = len(df)
            except Exception as e:
                print(f"  Warning: Error loading mutations: {e}")

        is_complete = has_freq and has_mut and has_dms

        return CountryDataSummary(
            country=country,
            has_frequencies=has_freq,
            has_mutations=has_mut,
            has_dms=has_dms,
            n_dates=n_dates,
            date_range=date_range,
            n_lineages=n_lineages,
            n_mutations=n_mutations,
            vasil_accuracy=VASIL_BASELINE_ACCURACY.get(country, 0.0),
            is_complete=is_complete
        )

    def load_all_countries(
        self,
        start_date: str = "2022-10-01",
        end_date: str = "2023-10-01"
    ) -> Dict[str, Dict]:
        """
        Load data from all 12 countries.

        Args:
            start_date: Start date for comparison period
            end_date: End date for comparison period

        Returns:
            Dict mapping country → {frequencies, mutations, preparer}
        """
        print("\n" + "="*80)
        print("LOADING DATA FROM ALL 12 VASIL COUNTRIES")
        print("="*80)
        print(f"\nPeriod: {start_date} to {end_date}")
        print("Countries: 12 (same as VASIL benchmark)")

        all_data = {}

        for country in self.countries:
            print(f"\nLoading {country}...")

            try:
                # Load frequencies
                frequencies = self.loader.load_gisaid_frequencies(
                    country, start_date, end_date
                )

                # Load mutations
                mutations = self.loader.load_variant_mutations(country)

                # Create preparer
                preparer = VariantDataPreparer(self.loader)

                all_data[country] = {
                    'frequencies': frequencies,
                    'mutations': mutations,
                    'preparer': preparer,
                    'vasil_target': VASIL_BASELINE_ACCURACY[country]
                }

                print(f"  ✅ {len(frequencies.dates)} dates, {len(frequencies.lineages)} lineages, {len(mutations)} mutations")

            except Exception as e:
                print(f"  ❌ Error: {e}")
                continue

        print("\n" + "="*80)
        print(f"✅ Loaded {len(all_data)}/12 countries successfully")
        print("="*80)

        if len(all_data) == 12:
            print("\n🎯 READY FOR TRUE APPLES-TO-APPLES VASIL COMPARISON!")
            print(f"   Target: Beat {VASIL_MEAN_ACCURACY:.3f} mean accuracy across all 12 countries")
        else:
            missing = set(self.countries) - set(all_data.keys())
            print(f"\n⚠️  Missing data for: {missing}")

        return all_data


def main():
    """Main verification and loading."""

    # Initialize multi-country loader
    loader = MultiCountryLoader(Path("/mnt/f/VASIL_Data"))

    # Step 1: Verify all countries have data
    print("\nSTEP 1: Verifying Data Coverage")
    summaries = loader.verify_all_countries()

    # Check if all complete
    all_complete = all(s.is_complete for s in summaries)

    if not all_complete:
        print("\n❌ Cannot proceed - missing data for some countries")
        return 1

    # Step 2: Load data from all countries
    print("\n" + "="*80)
    print("STEP 2: Loading All Country Data")

    all_data = loader.load_all_countries(
        start_date="2022-10-01",
        end_date="2023-10-01"
    )

    if len(all_data) != 12:
        print(f"\n❌ Only loaded {len(all_data)}/12 countries")
        return 1

    # Step 3: Summary statistics
    print("\n" + "="*80)
    print("STEP 3: Dataset Summary Statistics")
    print("="*80)

    total_dates = sum(len(data['frequencies'].dates) for data in all_data.values())
    total_lineages = sum(len(data['mutations']) for data in all_data.values())
    total_variants = sum(len(data['frequencies'].lineages) for data in all_data.values())

    print(f"\nAcross all 12 countries:")
    print(f"  Total date points: {total_dates}")
    print(f"  Total unique lineages: {total_lineages}")
    print(f"  Total variant series: {total_variants}")

    print(f"\nTarget accuracy by country:")
    for country in VASIL_COUNTRIES:
        acc = VASIL_BASELINE_ACCURACY[country]
        n_dates = len(all_data[country]['frequencies'].dates)
        print(f"  {country:<15} {acc:.3f} ({n_dates} dates)")

    print(f"\n  {'MEAN':<15} {VASIL_MEAN_ACCURACY:.3f} (VASIL baseline)")

    print("\n" + "="*80)
    print("✅ ALL 12 COUNTRIES READY FOR BENCHMARKING!")
    print("="*80)

    # Save country list for reference
    with open("results/vasil_12_countries.txt", 'w') as f:
        f.write("VASIL Benchmark Countries (12 total)\n")
        f.write("="*60 + "\n\n")
        for i, country in enumerate(VASIL_COUNTRIES, 1):
            summary = [s for s in summaries if s.country == country][0]
            f.write(f"{i:2}. {country:<15} "
                   f"{summary.n_dates:>4} dates, "
                   f"{summary.n_lineages:>4} lineages, "
                   f"{summary.n_mutations:>5} mutations, "
                   f"target={summary.vasil_accuracy:.3f}\n")

        f.write(f"\nMean accuracy target: {VASIL_MEAN_ACCURACY:.3f}\n")
        f.write(f"\nTotal: {total_dates} date points across 12 countries\n")

    print(f"\nCountry list saved to: results/vasil_12_countries.txt")

    return 0


if __name__ == "__main__":
    # Create results directory
    Path("results").mkdir(exist_ok=True)

    sys.exit(main())
