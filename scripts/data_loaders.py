#!/usr/bin/env python3
"""
PRISM-VE Data Loaders

Loads VASIL benchmark data for fitness module:
1. DMS escape matrix (836 antibodies × 201 RBD sites)
2. GISAID frequency data (lineage frequencies over time)
3. Mutation data (lineage → spike mutations mapping)

Data Source: /mnt/f/VASIL_Data (VASIL's exact dataset)

Scientific Integrity: These are PRIMARY source aggregates, not VASIL model outputs.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import re


@dataclass
class DmsEscapeData:
    """DMS antibody escape data."""
    escape_matrix: np.ndarray        # [836 antibodies × 201 RBD sites]
    antibody_names: List[str]        # [836] antibody names
    antibody_groups: List[str]       # [836] epitope groups (D1, D2, etc.)
    site_positions: np.ndarray       # [201] RBD positions (331-531)

    def get_escape_score(self, antibody_idx: int, site: int) -> float:
        """Get escape score for antibody at RBD site."""
        if site < 331 or site > 531:
            return 0.0
        site_idx = site - 331
        return self.escape_matrix[antibody_idx, site_idx]


@dataclass
class VariantFrequencies:
    """Variant frequency time series."""
    country: str
    lineages: List[str]              # Lineage names
    dates: pd.DatetimeIndex          # Dates
    frequencies: pd.DataFrame        # [dates × lineages]

    def get_frequency(self, lineage: str, date: str) -> float:
        """Get frequency for lineage at date."""
        try:
            return self.frequencies.loc[date, lineage]
        except (KeyError, IndexError):
            return 0.0

    def get_velocity(self, lineage: str, date: str, window_days: int = 7) -> float:
        """Get frequency change rate (Δfreq/month)."""
        try:
            date_pd = pd.to_datetime(date)
            future_date = date_pd + pd.Timedelta(days=window_days)

            freq_now = self.get_frequency(lineage, date)
            freq_future = self.get_frequency(lineage, future_date.strftime('%Y-%m-%d'))

            delta_freq = freq_future - freq_now
            delta_time_months = window_days / 30.0

            return delta_freq / delta_time_months
        except:
            return 0.0


@dataclass
class VariantMutations:
    """Variant spike mutation data."""
    lineage: str
    mutations_rbd: List[str]         # RBD mutations (e.g., ['D614G', 'L452R'])
    mutation_sites: List[int]        # Site positions
    mutation_aas: List[str]          # Amino acid changes

    def has_mutation_at_site(self, site: int) -> bool:
        """Check if variant has mutation at site."""
        return site in self.mutation_sites


class VasilDataLoader:
    """
    Load VASIL benchmark data from /mnt/f/VASIL_Data.

    Provides access to:
    - DMS antibody escape scores
    - GISAID lineage frequencies
    - Variant spike mutations
    """

    def __init__(self, vasil_data_dir: Path = Path("/mnt/f/VASIL_Data")):
        """
        Initialize data loader.

        Args:
            vasil_data_dir: Path to VASIL_Data directory
        """
        self.vasil_data_dir = Path(vasil_data_dir)

        if not self.vasil_data_dir.exists():
            raise FileNotFoundError(
                f"VASIL data directory not found: {vasil_data_dir}\n"
                f"Expected location: /mnt/f/VASIL_Data"
            )

    def load_antibody_epitope_mapping(self, country: str = "Germany") -> Dict[str, str]:
        """
        Load antibody → epitope class mapping.

        Returns:
            Dict mapping antibody name → epitope group (D1, D2, etc.)
        """
        mapping_file = (self.vasil_data_dir / "ByCountry" / country /
                       "results" / "epitope_data" / "antibodymapping_greaneyclasses.csv")

        if not mapping_file.exists():
            raise FileNotFoundError(f"Antibody mapping file not found: {mapping_file}")

        print(f"Loading antibody epitope mapping from: {mapping_file}")
        df = pd.read_csv(mapping_file)

        # Build mapping
        mapping = {}
        for _, row in df.iterrows():
            antibody = row['condition']
            epitope_group = row['group']
            mapping[antibody] = epitope_group

        print(f"  Mapped {len(mapping)} antibodies to epitope groups")

        return mapping

    def load_dms_escape_matrix(self, country: str = "Germany") -> DmsEscapeData:
        """
        Load DMS antibody escape matrix.

        Args:
            country: Country to load from (data is same across countries)

        Returns:
            DmsEscapeData with 836 antibodies × 201 RBD sites
        """
        dms_file = (self.vasil_data_dir / "ByCountry" / country /
                   "results" / "epitope_data" / "dms_per_ab_per_site.csv")

        if not dms_file.exists():
            raise FileNotFoundError(f"DMS file not found: {dms_file}")

        print(f"Loading DMS escape data from: {dms_file}")
        df = pd.read_csv(dms_file)

        # Parse DMS data
        # Format: condition, group, site, mut_escape, IC50
        antibodies = df['condition'].unique()
        groups = df.groupby('condition')['group'].first()
        sites = sorted(df['site'].unique())

        print(f"  Antibodies: {len(antibodies)}")
        print(f"  Sites: {len(sites)} (range: {min(sites)}-{max(sites)})")

        # Build escape matrix
        # Assuming RBD sites 331-531
        n_antibodies = len(antibodies)
        n_sites = 201  # Sites 331-531

        escape_matrix = np.zeros((n_antibodies, n_sites), dtype=np.float32)

        for i, antibody in enumerate(antibodies):
            ab_data = df[df['condition'] == antibody]
            for _, row in ab_data.iterrows():
                site = int(row['site'])
                if 331 <= site <= 531:
                    site_idx = site - 331
                    escape_matrix[i, site_idx] = float(row['mut_escape'])

        return DmsEscapeData(
            escape_matrix=escape_matrix,
            antibody_names=antibodies.tolist(),
            antibody_groups=groups.tolist(),
            site_positions=np.arange(331, 532, dtype=np.int32)
        )

    def load_gisaid_frequencies(
        self,
        country: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> VariantFrequencies:
        """
        Load GISAID lineage frequency data.

        Args:
            country: Country name (e.g., "Germany", "USA")
            start_date: Optional start date filter (YYYY-MM-DD)
            end_date: Optional end date filter (YYYY-MM-DD)

        Returns:
            VariantFrequencies with time series data
        """
        freq_file = (self.vasil_data_dir / "ByCountry" / country /
                    "results" / "Daily_Lineages_Freq_1_percent.csv")

        if not freq_file.exists():
            raise FileNotFoundError(f"Frequency file not found: {freq_file}")

        print(f"Loading GISAID frequencies from: {freq_file}")
        df = pd.read_csv(freq_file)

        # Check if 'date' column exists
        if 'date' in df.columns:
            # Set date as index
            df = df.set_index('date')
            df.index = pd.to_datetime(df.index)
        else:
            # Fallback: try to parse index
            df = pd.read_csv(freq_file, index_col=0)
            df.index = pd.to_datetime(df.index)

        # Drop any unnamed columns
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

        # Filter by date range if specified
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]

        print(f"  Country: {country}")
        print(f"  Dates: {len(df)} ({df.index.min()} to {df.index.max()})")
        print(f"  Lineages: {len(df.columns)}")

        return VariantFrequencies(
            country=country,
            lineages=df.columns.tolist(),
            dates=df.index,
            frequencies=df
        )

    def load_variant_mutations(
        self,
        country: str,
        lineages: Optional[List[str]] = None
    ) -> Dict[str, VariantMutations]:
        """
        Load variant spike mutation data.

        Args:
            country: Country name
            lineages: Optional list of specific lineages to load

        Returns:
            Dict mapping lineage → VariantMutations
        """
        mutation_file = (self.vasil_data_dir / "ByCountry" / country /
                        "results" / "mutation_data" / "mutation_lists.csv")

        if not mutation_file.exists():
            raise FileNotFoundError(f"Mutation file not found: {mutation_file}")

        print(f"Loading variant mutations from: {mutation_file}")
        df = pd.read_csv(mutation_file)

        print(f"  Lineages with mutations: {len(df)}")

        variants = {}
        for _, row in df.iterrows():
            lineage = row['lineage']

            # Filter to requested lineages
            if lineages is not None and lineage not in lineages:
                continue

            # Parse mutations (format: D614G/L452R/P681R/T19R/T478K)
            mutations_str = row['mutated_sites_RBD']
            if pd.isna(mutations_str) or mutations_str == '':
                mutations_rbd = []
            else:
                mutations_rbd = mutations_str.split('/')

            # Parse site positions and amino acid changes
            mutation_sites = []
            mutation_aas = []

            for mut in mutations_rbd:
                if len(mut) < 2:
                    continue

                # Format: D614G → site=614, wt=D, mut=G
                match = re.match(r'([A-Z])(\d+)([A-Z])', mut)
                if match:
                    wt_aa, site_str, mut_aa = match.groups()
                    site = int(site_str)

                    # Only RBD mutations (331-531)
                    if 331 <= site <= 531:
                        mutation_sites.append(site)
                        mutation_aas.append(f"{wt_aa}{site}{mut_aa}")

            variants[lineage] = VariantMutations(
                lineage=lineage,
                mutations_rbd=mutations_rbd,
                mutation_sites=mutation_sites,
                mutation_aas=mutation_aas
            )

        print(f"  Loaded mutations for {len(variants)} lineages")

        return variants

    def load_antibody_epitope_mapping(self, country: str = "Germany") -> Dict[str, str]:
        """
        Load antibody → epitope class mapping.

        Returns:
            Dict mapping antibody name → epitope group (D1, D2, etc.)
        """
        mapping_file = (self.vasil_data_dir / "ByCountry" / country /
                       "results" / "epitope_data" / "antibodymapping_greaneyclasses.csv")

        if not mapping_file.exists():
            raise FileNotFoundError(f"Antibody mapping file not found: {mapping_file}")

        print(f"Loading antibody epitope mapping from: {mapping_file}")
        df = pd.read_csv(mapping_file)

        # Build mapping
        mapping = {}
        for _, row in df.iterrows():
            antibody = row['condition']
            epitope_group = row['group']
            mapping[antibody] = epitope_group

        print(f"  Mapped {len(mapping)} antibodies to epitope groups")

        return mapping


class VariantDataPreparer:
    """
    Prepare variant data for PRISM-VE fitness module.

    Combines DMS, frequencies, and mutations into format needed for GPU.
    """

    def __init__(self, vasil_loader: VasilDataLoader):
        self.loader = vasil_loader

    def prepare_variant_batch(
        self,
        country: str,
        date: str,
        min_frequency: float = 0.03
    ) -> List[Dict]:
        """
        Prepare batch of variants for prediction at given date.

        Args:
            country: Country name
            date: Date (YYYY-MM-DD)
            min_frequency: Minimum frequency threshold (default 3%)

        Returns:
            List of variant data dicts ready for PRISM-VE
        """
        # Load data
        frequencies = self.loader.load_gisaid_frequencies(country)
        mutations = self.loader.load_variant_mutations(country)

        # Get lineages above threshold at this date
        freq_at_date = frequencies.frequencies.loc[date]
        significant_lineages = freq_at_date[freq_at_date > min_frequency].index.tolist()

        print(f"\nPreparing variant batch:")
        print(f"  Date: {date}")
        print(f"  Country: {country}")
        print(f"  Significant lineages (>{min_frequency*100}%): {len(significant_lineages)}")

        # Prepare batch
        batch = []
        for lineage in significant_lineages:
            if lineage not in mutations:
                print(f"  Warning: No mutation data for {lineage}")
                continue

            variant_mut = mutations[lineage]

            # Get frequency and velocity
            freq = frequencies.get_frequency(lineage, date)
            velocity = frequencies.get_velocity(lineage, date)

            batch.append({
                'lineage': lineage,
                'spike_mutations': variant_mut.mutation_sites,
                'mutation_aa': variant_mut.mutation_aas,
                'current_frequency': freq,
                'velocity': velocity,
                'date': date,
                'country': country
            })

        print(f"  Batch size: {len(batch)} variants")

        return batch

    def prepare_training_dataset(
        self,
        country: str,
        start_date: str,
        end_date: str,
        sample_frequency: int = 7  # Sample every N days
    ) -> List[Tuple[str, str, bool]]:
        """
        Prepare training dataset with rise/fall labels.

        Args:
            country: Country name
            start_date: Training start date
            end_date: Training end date
            sample_frequency: Days between samples

        Returns:
            List of (lineage, date, did_rise) tuples
        """
        frequencies = self.loader.load_gisaid_frequencies(
            country, start_date, end_date
        )

        training_data = []

        # For each sampled date
        for i, date in enumerate(frequencies.dates[::sample_frequency]):
            if i + 1 >= len(frequencies.dates[::sample_frequency]):
                break  # Need future data for label

            future_date = frequencies.dates[::sample_frequency][i + 1]

            # For each lineage above threshold
            freq_at_date = frequencies.frequencies.loc[date]
            significant = freq_at_date[freq_at_date > 0.01].index

            for lineage in significant:
                freq_now = frequencies.frequencies.loc[date, lineage]
                freq_future = frequencies.frequencies.loc[future_date, lineage]

                # Label: did it rise?
                if freq_future > freq_now * 1.05:  # 5% threshold
                    did_rise = True
                elif freq_future < freq_now * 0.95:
                    did_rise = False
                else:
                    continue  # Skip stable variants

                training_data.append((lineage, date.strftime('%Y-%m-%d'), did_rise))

        print(f"\nTraining dataset prepared:")
        print(f"  Period: {start_date} to {end_date}")
        print(f"  Samples: {len(training_data)}")
        print(f"  Rising: {sum(1 for _, _, rise in training_data if rise)}")
        print(f"  Falling: {sum(1 for _, _, rise in training_data if not rise)}")

        return training_data


def load_dms_for_gpu(vasil_data_dir: Path = Path("/mnt/f/VASIL_Data")) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load DMS data in format ready for GPU upload.

    Returns:
        (escape_matrix, antibody_epitope_indices)
        - escape_matrix: [836 × 201] float32 array
        - antibody_epitope_indices: [836] int32 array (0-9 for epitope classes)
    """
    loader = VasilDataLoader(vasil_data_dir)
    dms_data = loader.load_dms_escape_matrix()

    # Map epitope groups to indices
    # D1=0, D2=1, D3=2, D4=3, etc.
    epitope_groups = sorted(set(dms_data.antibody_groups))
    group_to_idx = {group: i for i, group in enumerate(epitope_groups)}

    antibody_epitope_indices = np.array(
        [group_to_idx[group] for group in dms_data.antibody_groups],
        dtype=np.int32
    )

    print(f"\nDMS data prepared for GPU:")
    print(f"  Escape matrix shape: {dms_data.escape_matrix.shape}")
    print(f"  Epitope groups: {len(epitope_groups)} ({epitope_groups})")
    print(f"  Antibody indices shape: {antibody_epitope_indices.shape}")

    return dms_data.escape_matrix.astype(np.float32), antibody_epitope_indices


def load_gisaid_for_date(
    country: str,
    date: str,
    vasil_data_dir: Path = Path("/mnt/f/VASIL_Data")
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """
    Load GISAID data for a specific date.

    Returns:
        (lineages, frequencies, velocities)
        - lineages: List of lineage names
        - frequencies: [n_lineages] float32 array
        - velocities: [n_lineages] float32 array (Δfreq/month)
    """
    loader = VasilDataLoader(vasil_data_dir)
    freq_data = loader.load_gisaid_frequencies(country)

    # Get data for this date
    lineages = freq_data.lineages
    frequencies = np.array([
        freq_data.get_frequency(lineage, date) for lineage in lineages
    ], dtype=np.float32)

    velocities = np.array([
        freq_data.get_velocity(lineage, date) for lineage in lineages
    ], dtype=np.float32)

    print(f"\nGISAID data for {country} on {date}:")
    print(f"  Lineages: {len(lineages)}")
    print(f"  Frequencies > 1%: {(frequencies > 0.01).sum()}")
    print(f"  Rising (vel > 0): {(velocities > 0).sum()}")
    print(f"  Falling (vel < 0): {(velocities < 0).sum()}")

    return lineages, frequencies, velocities


def example_usage():
    """Example of using the data loaders."""

    print("="*80)
    print("PRISM-VE Data Loaders - Example Usage")
    print("="*80)

    # Initialize loader
    loader = VasilDataLoader(Path("/mnt/f/VASIL_Data"))

    # 1. Load DMS data
    print("\n[1] Loading DMS Escape Data")
    print("-" * 80)
    dms_data = loader.load_dms_escape_matrix("Germany")

    # Example: Get escape score for antibody 0 at site 484
    escape_score = dms_data.get_escape_score(0, 484)
    print(f"  Example: Antibody '{dms_data.antibody_names[0]}' at site 484: {escape_score:.4f}")

    # 2. Load GISAID frequencies
    print("\n[2] Loading GISAID Frequencies")
    print("-" * 80)
    frequencies = loader.load_gisaid_frequencies("Germany", "2023-01-01", "2023-12-31")

    # Example: Get BA.5 frequency on specific date
    if "BA.5" in frequencies.lineages:
        ba5_freq = frequencies.get_frequency("BA.5", "2023-01-15")
        ba5_vel = frequencies.get_velocity("BA.5", "2023-01-15")
        print(f"  Example: BA.5 on 2023-01-15: freq={ba5_freq:.3f}, velocity={ba5_vel:.4f}/month")

    # 3. Load variant mutations
    print("\n[3] Loading Variant Mutations")
    print("-" * 80)
    mutations = loader.load_variant_mutations("Germany", lineages=["BA.2", "BA.5", "BQ.1.1"])

    for lineage, mut_data in list(mutations.items())[:3]:
        print(f"  {lineage}:")
        print(f"    RBD mutations: {mut_data.mutations_rbd[:5]}...")
        print(f"    Sites: {mut_data.mutation_sites[:5]}...")

    # 4. Prepare variant batch
    print("\n[4] Preparing Variant Batch")
    print("-" * 80)
    preparer = VariantDataPreparer(loader)
    batch = preparer.prepare_variant_batch("Germany", "2023-06-01", min_frequency=0.05)

    if batch:
        print(f"\n  First variant in batch:")
        print(f"    Lineage: {batch[0]['lineage']}")
        print(f"    Frequency: {batch[0]['current_frequency']:.3f}")
        print(f"    Velocity: {batch[0]['velocity']:.4f}/month")
        print(f"    Mutations: {len(batch[0]['spike_mutations'])} RBD sites")

    # 5. Load data for GPU
    print("\n[5] Preparing Data for GPU Upload")
    print("-" * 80)
    escape_matrix, epitope_indices = load_dms_for_gpu()

    print(f"\n  Ready for GPU upload:")
    print(f"    escape_matrix.shape: {escape_matrix.shape}")
    print(f"    epitope_indices.shape: {epitope_indices.shape}")
    print(f"    Memory: {escape_matrix.nbytes / 1024:.1f} KB")

    print("\n" + "="*80)
    print("✅ All data loaders working correctly!")
    print("="*80)


if __name__ == "__main__":
    example_usage()
