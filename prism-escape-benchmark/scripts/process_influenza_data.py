#!/usr/bin/env python3
"""
Process Influenza HA DMS data for PRISM-Viral benchmarking.

Data: Doud 2018 (H1 HA escape from antibodies)
"""

import pandas as pd
import numpy as np

print("="*80)
print("🦠 PROCESSING INFLUENZA HA DMS DATA")
print("="*80)
print()

# Load Influenza HA DMS data
flu_file = 'prism-escape-benchmark/data/raw/evescape/EVEscape/data/experiments/doud2018/DMS_Doud2018_H1-WSN33_antibodies.csv'

print(f"Loading: {flu_file}")
df = pd.read_csv(flu_file)

print(f"✅ Loaded: {len(df)} records")
print(f"Columns: {df.columns.tolist()[:10]}...")
print()

# Show sample
print("Sample data:")
print(df.head())
print()

# Check for escape columns
escape_cols = [col for col in df.columns if 'survive' in col.lower() or 'escape' in col.lower()]
print(f"Escape-related columns: {len(escape_cols)}")
if escape_cols:
    print(f"  Examples: {escape_cols[:5]}")
print()

# Use max across antibodies
if 'max_norm_survive' in df.columns:
    df['escape_score'] = df['max_norm_survive']
    print("✅ Using max_norm_survive")
elif 'max_norm_tf_survive' in df.columns:
    df['escape_score'] = df['max_norm_tf_survive']
    print("✅ Using max_norm_tf_survive")
else:
    # Find all antibody escape columns
    antibody_cols = [col for col in df.columns if 'antibody_' in col and 'norm_tf_mutfracsurvive' in col]
    if antibody_cols:
        print(f"✅ Found {len(antibody_cols)} antibody columns, computing max")
        df['escape_score'] = df[antibody_cols].max(axis=1).abs()
    else:
        print("❌ No suitable escape columns!")
        import sys
        sys.exit(1)

# Create binary label
threshold = df['escape_score'].median()
df['escape_binary'] = (df['escape_score'] > threshold).astype(int)

print(f"\nEscape score statistics:")
print(f"  Min: {df['escape_score'].min():.2f}")
print(f"  Median: {df['escape_score'].median():.2f}")
print(f"  Max: {df['escape_score'].max():.2f}")
print(f"  Escape rate: {df['escape_binary'].mean():.1%}")
print()

# Save
output_file = 'prism-escape-benchmark/data/processed/influenza_ha/flu_mutations.csv'
import os
os.makedirs(os.path.dirname(output_file), exist_ok=True)

df_out = df[['i', 'wt', 'mut', 'mutation', 'escape_score', 'escape_binary']].copy()
df_out.to_csv(output_file, index=False)

print(f"✅ Saved: {output_file}")
print(f"   {len(df)} Influenza HA mutations ready")
print()

print("="*80)
print("INFLUENZA HA DATA READY")
print("="*80)
print(f"Mutations: {len(df)}")
print(f"Position range: {df['i'].min()}-{df['i'].max()}")
print("\nNext: Download Influenza HA structure and extract features")
print("="*80)
