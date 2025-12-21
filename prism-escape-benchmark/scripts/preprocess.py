#!/usr/bin/env python3
"""
Preprocess Bloom DMS data into standardized benchmark format.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import numpy as np
from data.loaders import BloomDMSLoader, create_benchmark_dataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    logger.info("Starting data preprocessing...")

    # Paths
    base_dir = Path(__file__).parent.parent
    raw_dir = base_dir / "data" / "raw" / "bloom_dms"
    processed_dir = base_dir / "data" / "processed" / "sars2_rbd"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Load Bloom DMS data
    loader = BloomDMSLoader(raw_dir)
    df_raw = loader.load_all_escape_maps()

    logger.info(f"Loaded {len(df_raw)} mutation-antibody pairs")
    logger.info(f"  Unique mutations: {df_raw['mutation'].nunique()}")
    logger.info(f"  Unique antibodies: {df_raw['antibody'].nunique()}")

    # Save raw data (use CSV if parquet unavailable)
    try:
        df_raw.to_parquet(processed_dir / "raw_escape_data.parquet")
        logger.info(f"Saved raw data: {processed_dir / 'raw_escape_data.parquet'}")
    except ImportError:
        df_raw.to_csv(processed_dir / "raw_escape_data.csv", index=False)
        logger.info(f"Saved raw data (CSV): {processed_dir / 'raw_escape_data.csv'}")

    # Create benchmark dataset (aggregate across antibodies)
    benchmark = create_benchmark_dataset(
        df_raw,
        min_antibodies=3,     # Require at least 3 antibodies
        escape_threshold=0.1  # Binary escape threshold
    )

    # Split into train/test (80/20)
    from sklearn.model_selection import train_test_split

    train, test = train_test_split(
        benchmark,
        test_size=0.2,
        random_state=42,
        stratify=benchmark['escape_binary']
    )

    logger.info(f"\nDataset splits:")
    logger.info(f"  Train: {len(train)} mutations ({train['escape_binary'].sum()} escape)")
    logger.info(f"  Test:  {len(test)} mutations ({test['escape_binary'].sum()} escape)")

    # Save splits (use CSV if parquet unavailable)
    try:
        train.to_parquet(processed_dir / "train.parquet")
        test.to_parquet(processed_dir / "test.parquet")
        benchmark.to_parquet(processed_dir / "full_benchmark.parquet")
        logger.info(f"\n✅ Preprocessing complete (Parquet format)!")
    except ImportError:
        train.to_csv(processed_dir / "train.csv", index=False)
        test.to_csv(processed_dir / "test.csv", index=False)
        benchmark.to_csv(processed_dir / "full_benchmark.csv", index=False)
        logger.info(f"\n✅ Preprocessing complete (CSV format)!")

    logger.info(f"\n✅ Preprocessing complete!")
    logger.info(f"   Train: {processed_dir / 'train.parquet'}")
    logger.info(f"   Test:  {processed_dir / 'test.parquet'}")

    # Print summary statistics
    print("\n" + "="*60)
    print("BENCHMARK DATASET SUMMARY")
    print("="*60)
    print(f"Total mutations: {len(benchmark)}")
    print(f"Escape mutations: {benchmark['escape_binary'].sum()} ({benchmark['escape_binary'].mean():.1%})")
    print(f"Non-escape mutations: {(~benchmark['escape_binary']).sum()}")
    print(f"\nEscape score distribution:")
    print(benchmark['escape_score'].describe())
    print(f"\nTop 10 high-escape mutations:")
    print(benchmark.nlargest(10, 'escape_score')[['mutation', 'escape_score', 'n_antibodies']])


if __name__ == "__main__":
    main()
