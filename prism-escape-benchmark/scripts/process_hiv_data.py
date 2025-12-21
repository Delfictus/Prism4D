#!/usr/bin/env python3
"""
Process HIV Env DMS data for PRISM-Viral benchmarking.

Data: Dingens 2019 (HIV Env escape from antibodies)
Format: Similar to Bloom DMS (mutation, escape scores)
"""

import pandas as pd
import numpy as np

print("="*80)
print("🦠 PROCESSING HIV ENV DMS DATA")
print("="*80)
print()

# Load HIV Env DMS data (Dingens 2019)
hiv_file = 'prism-escape-benchmark/data/raw/evescape/EVEscape/data/experiments/dingens2019/DMS_Dingens2019a_hiv_env_antibodies_x10.csv'

print(f"Loading: {hiv_file}")
df = pd.read_csv(hiv_file)

print(f"✅ Loaded: {len(df)} records")
print(f"Columns: {df.columns.tolist()[:10]}...")
print()

# Show sample
print("Sample data:")
print(df.head())
print()

# Analyze structure
print("Data structure:")
print(f"  Positions: {df['i'].min()} - {df['i'].max()}")
print(f"  Unique positions: {df['i'].nunique()}")
print(f"  Unique mutations: {len(df)}")
print()

# Aggregate across antibodies (similar to SARS-CoV-2 processing)
print("Aggregating across antibodies...")

# Get escape columns (those with 'mutfracsurvive' in name)
escape_cols = [col for col in df.columns if 'norm_tf_mutfracsurvive' in col or 'max_norm_tf_survive' in col]

# Use max_norm_tf_survive column (already present)
if 'max_norm_tf_survive' in df.columns:
    df['escape_score'] = df['max_norm_tf_survive']
    print(f"✅ Using max_norm_tf_survive as escape score")
else:
    print("⚠️  max_norm_tf_survive not found, computing from summary columns")
    # Use summary columns (mean/median across antibodies)
    summary_cols = [col for col in df.columns if 'summary_' in col and 'mean' in col and 'mutdiffsel' in col]
    if summary_cols:
        print(f"   Found {len(summary_cols)} summary columns, using max")
        df['escape_score'] = df[summary_cols].max(axis=1).abs()  # Use absolute value
        print(f"✅ Computed escape_score from summary columns")
    else:
        print("❌ No suitable escape columns found!")
        import sys
        sys.exit(1)

# Create binary label
threshold = df['escape_score'].median()
df['escape_binary'] = (df['escape_score'] > threshold).astype(int)

print(f"\nEscape score statistics:")
print(f"  Min: {df['escape_score'].min():.2f}")
print(f"  Median: {df['escape_score'].median():.2f}")
print(f"  Max: {df['escape_score'].max():.2f}")
print(f"  Escape rate (>{threshold:.2f}): {df['escape_binary'].mean():.1%}")
print()

# Save processed data
output_file = 'prism-escape-benchmark/data/processed/hiv_env/hiv_mutations.csv'
import os
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Save with position and mutation info
df_out = df[['i', 'wt', 'mut', 'mutation', 'escape_score', 'escape_binary']].copy()
df_out.to_csv(output_file, index=False)

print(f"\n✅ Saved processed data: {output_file}")
print(f"   {len(df)} HIV Env mutations ready for PRISM feature extraction")
print()

# Create summary
print("="*80)
print("HIV ENV DATA READY")
print("="*80)
print(f"Mutations: {len(df)}")
print(f"Position range: {df['i'].min()}-{df['i'].max()}")
print(f"Escape threshold: {threshold:.2f}")
print(f"\nNext: Download HIV Env structure and extract PRISM features")
print("="*80)
