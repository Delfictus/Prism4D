#!/usr/bin/env python3
"""
Compute GISAID Frequency Velocities for PRISM-VE Cycle Module

Processes VASIL frequency data to compute velocity (Δfreq/month) for each lineage.

Input: Daily_Lineages_Freq_1_percent.csv (from VASIL)
Output: Velocity arrays ready for GPU upload

Scientific Integrity: This is OUR processing of the raw frequency aggregates.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List
import argparse


def compute_velocities_for_country(
    freq_file: Path,
    window_days: int = 7
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Compute velocities from VASIL frequency file.

    Args:
        freq_file: Path to Daily_Lineages_Freq_1_percent.csv
        window_days: Window for velocity calculation (default: 7 days = 1 week)

    Returns:
        (frequencies_df, velocity_dict)
        - frequencies_df: Original frequencies with dates
        - velocity_dict: {lineage: velocity_array} for each lineage
    """
    print(f"Loading frequencies from: {freq_file}")

    # Load frequency data
    df = pd.read_csv(freq_file)

    # Set date as index
    if 'date' in df.columns:
        df = df.set_index('date')
        df.index = pd.to_datetime(df.index)
    else:
        # Try reading with first column as index
        df = pd.read_csv(freq_file, index_col=0)
        df.index = pd.to_datetime(df.index)

    # Drop unnamed columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    print(f"  Dates: {len(df)} ({df.index.min()} to {df.index.max()})")
    print(f"  Lineages: {len(df.columns)}")

    # Compute velocities for each lineage
    velocity_dict = {}

    for lineage in df.columns:
        frequencies = df[lineage].values
        dates = df.index

        # Compute velocity (Δfreq per month)
        velocities = np.zeros(len(frequencies), dtype=np.float32)

        for i in range(1, len(frequencies)):
            delta_freq = frequencies[i] - frequencies[i-1]
            delta_days = (dates[i] - dates[i-1]).days

            if delta_days > 0:
                delta_months = delta_days / 30.0
                velocities[i] = delta_freq / delta_months
            else:
                velocities[i] = 0.0

        velocity_dict[lineage] = velocities

    print(f"  Computed velocities for {len(velocity_dict)} lineages")

    return df, velocity_dict


def compute_velocities_all_countries(
    vasil_data_dir: Path = Path("/mnt/f/VASIL_Data"),
    countries: list = None,
    output_dir: Path = Path("data/processed/velocities")
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Compute velocities for all VASIL countries.

    Args:
        vasil_data_dir: Path to VASIL_Data directory
        countries: List of countries (default: all 12)
        output_dir: Where to save computed velocities

    Returns:
        Dict mapping country → {lineage: velocity_array}
    """
    if countries is None:
        countries = [
            "Germany", "USA", "UK", "Japan", "Brazil", "France",
            "Canada", "Denmark", "Australia", "Sweden", "Mexico", "SouthAfrica"
        ]

    output_dir.mkdir(parents=True, exist_ok=True)

    all_velocities = {}

    print("="*80)
    print(f"COMPUTING VELOCITIES FOR ALL {len(countries)} COUNTRIES")
    print("="*80)

    for country in countries:
        print(f"\n[{country}]")

        freq_file = (vasil_data_dir / "ByCountry" / country /
                    "results" / "Daily_Lineages_Freq_1_percent.csv")

        if not freq_file.exists():
            print(f"  ❌ Frequency file not found")
            continue

        try:
            frequencies, velocities = compute_velocities_for_country(freq_file)

            all_velocities[country] = velocities

            # Save to file
            output_file = output_dir / f"{country}_velocities.npz"
            np.savez_compressed(
                output_file,
                **{lineage: vel for lineage, vel in velocities.items()}
            )

            print(f"  ✅ Saved to {output_file}")

        except Exception as e:
            print(f"  ❌ Error: {e}")
            continue

    print("\n" + "="*80)
    print(f"✅ Computed velocities for {len(all_velocities)}/{len(countries)} countries")
    print("="*80)

    return all_velocities


def get_velocity_at_date(
    velocity_dict: Dict[str, np.ndarray],
    frequencies_df: pd.DataFrame,
    lineage: str,
    date: str
) -> float:
    """
    Get velocity for a lineage at a specific date.

    Args:
        velocity_dict: Dict from compute_velocities_for_country()
        frequencies_df: Frequency DataFrame with DatetimeIndex
        lineage: Lineage name
        date: Date string (YYYY-MM-DD)

    Returns:
        Velocity value (Δfreq/month)
    """
    if lineage not in velocity_dict:
        return 0.0

    try:
        # Find date index
        date_pd = pd.to_datetime(date)
        idx = frequencies_df.index.get_loc(date_pd)

        return float(velocity_dict[lineage][idx])

    except (KeyError, IndexError):
        return 0.0


def prepare_gisaid_arrays_for_gpu(
    frequencies_df: pd.DataFrame,
    velocity_dict: Dict[str, np.ndarray],
    date: str,
    min_frequency: float = 0.01
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """
    Prepare GISAID frequency and velocity arrays for GPU upload.

    Args:
        frequencies_df: Frequency DataFrame
        velocity_dict: Velocity dictionary
        date: Date to extract data for
        min_frequency: Minimum frequency threshold

    Returns:
        (lineages, frequency_array, velocity_array)
        - lineages: List of lineage names
        - frequency_array: [n_lineages] current frequencies
        - velocity_array: [n_lineages] velocities (Δfreq/month)
    """
    date_pd = pd.to_datetime(date)

    # Get frequencies at this date
    freq_at_date = frequencies_df.loc[date_pd]

    # Filter to significant lineages
    significant = freq_at_date[freq_at_date > min_frequency]

    lineages = significant.index.tolist()
    frequencies = significant.values.astype(np.float32)

    # Get velocities
    velocities = np.array([
        velocity_dict[lineage][frequencies_df.index.get_loc(date_pd)]
        for lineage in lineages
    ], dtype=np.float32)

    print(f"\nPrepared GPU arrays for {date}:")
    print(f"  Lineages: {len(lineages)}")
    print(f"  Frequency range: {frequencies.min():.3f} - {frequencies.max():.3f}")
    print(f"  Velocity range: {velocities.min():.3f} - {velocities.max():.3f}")
    print(f"  Rising (vel > 0): {(velocities > 0).sum()}")
    print(f"  Falling (vel < 0): {(velocities < 0).sum()}")

    return lineages, frequencies, velocities


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(
        description="Compute GISAID velocities for PRISM-VE cycle module"
    )
    parser.add_argument(
        "--vasil-dir",
        type=str,
        default="/mnt/f/VASIL_Data",
        help="Path to VASIL_Data directory"
    )
    parser.add_argument(
        "--country",
        type=str,
        help="Process single country (default: all 12)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed/velocities",
        help="Output directory for velocity files"
    )

    args = parser.parse_args()

    vasil_dir = Path(args.vasil_dir)
    output_dir = Path(args.output_dir)

    if args.country:
        # Process single country
        freq_file = (vasil_dir / "ByCountry" / args.country /
                    "results" / "Daily_Lineages_Freq_1_percent.csv")

        frequencies, velocities = compute_velocities_for_country(freq_file)

        output_file = output_dir / f"{args.country}_velocities.npz"
        output_dir.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            output_file,
            **{lineage: vel for lineage, vel in velocities.items()}
        )

        print(f"\n✅ Saved velocities to: {output_file}")

    else:
        # Process all countries
        all_velocities = compute_velocities_all_countries(
            vasil_dir, output_dir=output_dir
        )

        print(f"\n✅ Processed {len(all_velocities)} countries")
        print(f"✅ Velocity files saved to: {output_dir}")

    print("\n" + "="*80)
    print("VELOCITY COMPUTATION COMPLETE")
    print("="*80)
    print("\nThese velocities enable:")
    print("  - Cycle phase detection (6 phases)")
    print("  - Emergence probability calculation")
    print("  - Temporal timing predictions")
    print("  - VASIL benchmark comparison")
    print("\nReady for mega_fused Stage 8 (cycle features)!")


if __name__ == "__main__":
    main()
