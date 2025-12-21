#!/usr/bin/env python3
"""
Data loaders for viral escape prediction benchmarking.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EscapeMutation:
    """Standardized escape mutation record."""
    virus: str                    # sars2, hiv, influenza
    protein: str                  # RBD, Env, HA
    wildtype_seq: str             # Full WT sequence
    position: int                 # 1-indexed position
    wildtype_aa: str              # Original amino acid
    mutant_aa: str                # Mutated amino acid
    escape_score: float           # Continuous escape score (0-1 normalized)
    escape_binary: bool           # Thresholded binary label
    antibody_id: str              # Antibody or serum identifier
    source: str                   # Data source (bloom, catnap, etc.)

    @property
    def mutation_str(self) -> str:
        return f"{self.wildtype_aa}{self.position}{self.mutant_aa}"


class BloomDMSLoader:
    """
    Load Bloom Lab SARS-CoV-2 RBD deep mutational scanning data.

    Reference: https://github.com/jbloomlab/SARS2_RBD_Ab_escape_maps
    Data: Escape fractions for ~4,000 RBD mutations against various antibodies
    """

    # Wuhan-Hu-1 RBD sequence (residues 331-531)
    WUHAN_RBD = (
        "NITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNV"
        "YADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERD"
        "ISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNF"
    )
    RBD_START = 331  # Starting position in Spike protein

    def __init__(self, data_dir: Path):
        """
        Args:
            data_dir: Path to data/raw/bloom_dms directory
        """
        self.data_dir = Path(data_dir)
        self.escape_maps = {}

    def load_all_escape_maps(self) -> pd.DataFrame:
        """
        Load all antibody escape maps into unified DataFrame.

        Returns:
            DataFrame with columns: [mutation, position, wt_aa, mut_aa,
                                    escape_score, antibody, source]
        """
        escape_dir = self.data_dir / "SARS2_RBD_Ab_escape_maps" / "processed_data"

        if not escape_dir.exists():
            # Try alternate location
            escape_dir = self.data_dir / "SARS2_RBD_Ab_escape_maps"

        if not escape_dir.exists():
            raise FileNotFoundError(f"Bloom DMS data not found at {escape_dir}")

        all_mutations = []

        # Find all escape CSV files
        csv_files = list(escape_dir.rglob("*escape*.csv"))
        logger.info(f"Found {len(csv_files)} escape map files")

        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                antibody_id = csv_file.stem.replace("_escape", "").replace("escape_", "")

                # Expected columns: site, wildtype, mutation, escape
                if 'escape' not in df.columns:
                    continue

                for _, row in df.iterrows():
                    position = int(row.get('site', row.get('position', 0)))
                    if position == 0:
                        continue

                    wildtype = row.get('wildtype', row.get('wt', 'X'))
                    mutation = row.get('mutation', row.get('mut', 'X'))
                    escape_score = float(row['escape'])

                    all_mutations.append({
                        'mutation': f"{wildtype}{position}{mutation}",
                        'position': position,
                        'wt_aa': wildtype,
                        'mut_aa': mutation,
                        'escape_score': escape_score,
                        'antibody': antibody_id,
                        'source': 'bloom_dms',
                        'virus': 'sars2',
                        'protein': 'RBD'
                    })

            except Exception as e:
                logger.warning(f"Failed to load {csv_file.name}: {e}")
                continue

        df_all = pd.DataFrame(all_mutations)
        logger.info(f"Loaded {len(df_all)} mutation-antibody pairs from Bloom DMS")

        return df_all

    def get_high_impact_mutations(self, threshold: float = 0.5) -> List[str]:
        """
        Get mutations with high escape scores across multiple antibodies.

        Args:
            threshold: Minimum escape score

        Returns:
            List of high-impact mutations
        """
        df = self.load_all_escape_maps()

        # Group by mutation, compute mean escape across antibodies
        grouped = df.groupby('mutation')['escape_score'].agg(['mean', 'max', 'count'])
        grouped = grouped[grouped['count'] >= 3]  # At least 3 antibodies

        high_impact = grouped[grouped['mean'] >= threshold].index.tolist()

        logger.info(f"Found {len(high_impact)} high-impact mutations (mean escape ≥ {threshold})")

        return high_impact


class EVEscapeLoader:
    """
    Load EVEscape baseline predictions for comparison.

    Reference: https://github.com/OATML-Markslab/EVEscape
    """

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)

    def load_predictions(self, virus: str = "sars2") -> Optional[pd.DataFrame]:
        """
        Load EVEscape predictions if available.

        Args:
            virus: Virus type (sars2, hiv, influenza)

        Returns:
            DataFrame with EVEscape scores or None
        """
        evescape_dir = self.data_dir / "EVEscape"

        # Look for pre-computed scores
        score_files = list(evescape_dir.rglob(f"*{virus}*score*.csv"))

        if not score_files:
            logger.warning(f"No EVEscape baseline scores found for {virus}")
            return None

        df = pd.read_csv(score_files[0])
        logger.info(f"Loaded EVEscape baseline: {len(df)} predictions")

        return df


def create_benchmark_dataset(
    bloom_data: pd.DataFrame,
    min_antibodies: int = 3,
    escape_threshold: float = 0.1
) -> pd.DataFrame:
    """
    Create standardized benchmark dataset from Bloom DMS data.

    Aggregates across antibodies to create robust ground truth labels.

    Args:
        bloom_data: Raw Bloom DMS data
        min_antibodies: Minimum antibodies a mutation must be tested against
        escape_threshold: Threshold for binary escape classification

    Returns:
        Benchmark DataFrame with aggregated labels
    """
    logger.info("Creating benchmark dataset...")

    # Group by mutation, aggregate across antibodies
    grouped = bloom_data.groupby('mutation').agg({
        'position': 'first',
        'wt_aa': 'first',
        'mut_aa': 'first',
        'escape_score': ['mean', 'std', 'max', 'count'],
        'virus': 'first',
        'protein': 'first'
    }).reset_index()

    # Flatten column names
    grouped.columns = ['_'.join(col).strip('_') for col in grouped.columns.values]

    # Filter: Keep mutations tested against ≥ min_antibodies
    grouped = grouped[grouped['escape_score_count'] >= min_antibodies]

    # Create binary label
    grouped['escape_binary'] = (grouped['escape_score_mean'] >= escape_threshold).astype(int)

    # Rename for clarity
    grouped = grouped.rename(columns={
        'escape_score_mean': 'escape_score',
        'escape_score_std': 'escape_std',
        'escape_score_max': 'escape_max',
        'escape_score_count': 'n_antibodies'
    })

    logger.info(f"Benchmark dataset: {len(grouped)} mutations")
    logger.info(f"  Escape rate: {grouped['escape_binary'].mean():.1%}")
    logger.info(f"  Avg antibodies/mutation: {grouped['n_antibodies'].mean():.1f}")

    return grouped


if __name__ == "__main__":
    """Test data loading."""

    # Load Bloom DMS data
    data_dir = Path("data/raw/bloom_dms")
    loader = BloomDMSLoader(data_dir)

    df = loader.load_all_escape_maps()
    print(f"\nLoaded {len(df)} mutation-antibody pairs")
    print(f"Unique mutations: {df['mutation'].nunique()}")
    print(f"Unique antibodies: {df['antibody'].nunique()}")

    # Create benchmark
    benchmark = create_benchmark_dataset(df)
    print(f"\nBenchmark dataset: {len(benchmark)} mutations")
    print(f"Columns: {benchmark.columns.tolist()}")

    # Show high-escape mutations
    print("\nTop 10 high-escape mutations:")
    print(benchmark.nlargest(10, 'escape_score')[['mutation', 'escape_score', 'n_antibodies']])
