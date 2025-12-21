#!/usr/bin/env python3
"""
PRISM-VE Model Wrapper for VASIL Benchmark

Implements predict_gamma(variant, country, date) interface
Loads predictions from Rust-generated predictions_gamma.csv
"""

import pandas as pd
from pathlib import Path


class PRISMVEModel:
    """
    PRISM-VE model wrapper - loads Rust-computed gamma predictions.
    """

    def __init__(self, predictions_file="predictions_gamma.csv"):
        """
        Initialize PRISM-VE model by loading pre-computed predictions.

        Args:
            predictions_file: Path to CSV with gamma_y predictions
        """
        print(f"Loading PRISM-VE predictions from {predictions_file}...")

        if not Path(predictions_file).exists():
            print(f"  WARNING: {predictions_file} not found!")
            print(f"  Run Rust benchmark first to generate predictions.")
            self.cache = {}
            self.n_predictions = 0
            return

        # Load predictions CSV
        df = pd.read_csv(predictions_file)

        # Build lookup cache
        self.cache = {}
        for _, row in df.iterrows():
            key = (row['country'], row['variant'], row['date'])
            self.cache[key] = row['gamma_y']

        self.n_predictions = len(self.cache)
        print(f"  ✅ Loaded {self.n_predictions:,} gamma_y predictions")

    def predict_gamma(self, variant: str, country: str, date: str) -> float:
        """
        Predict relative fitness gamma_y for variant at date in country.

        Args:
            variant: Lineage name (e.g., "BA.5.2.1")
            country: Country name (e.g., "Germany")
            date: Date string "YYYY-MM-DD"

        Returns:
            gamma_y: Relative fitness (>0 = RISE, <0 = FALL)
        """
        key = (country, variant, date)
        return self.cache.get(key, 0.0)  # Return 0 if not found


if __name__ == "__main__":
    # Test
    model = PRISMVEModel()
    gamma = model.predict_gamma("BA.5", "Germany", "2022-10-01")
    print(f"Test prediction: gamma = {gamma}")
