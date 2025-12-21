#!/usr/bin/env python3
"""
Antibody-Specific Escape Prediction

Trains separate models for each antibody class using Bloom DMS data.
Enables: "E484K escapes VRC01 but not S309"
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
from scipy import stats
import xgboost as xgb
import json

print("="*80)
print("ANTIBODY-SPECIFIC ESCAPE MODELS")
print("="*80)
print()

# Load raw Bloom DMS with per-antibody data
bloom_raw = pd.read_csv('prism-escape-benchmark/data/processed/sars2_rbd/raw_escape_data.csv')

print(f"Raw Bloom DMS: {len(bloom_raw)} mutation-antibody pairs")
print(f"Unique mutations: {bloom_raw['mutation'].nunique()}")
print(f"Unique antibodies: {bloom_raw['antibody'].nunique()}")
print()

# Get antibody list
antibodies = bloom_raw['antibody'].unique()
print(f"Training models for {len(antibodies)} antibodies:")
for ab in antibodies:
    n = (bloom_raw['antibody'] == ab).sum()
    print(f"  {ab}: {n} mutations")
print()

# Load PRISM features (all 12 working!)
features = np.load('prism-escape-benchmark/extracted_features/6m0j_12_COMPLETE.npy')
RBD_START = 331

# Train per-antibody models
antibody_models = {}
antibody_results = {}

for antibody in antibodies:
    print(f"Training model for {antibody}...")

    # Get antibody-specific data
    ab_data = bloom_raw[bloom_raw['antibody'] == antibody].copy()

    # Map to features
    X_list = []
    y_list = []

    for _, row in ab_data.iterrows():
        pos = row['position']
        struct_idx = pos - RBD_START
        if 0 <= struct_idx < features.shape[0]:
            X_list.append(features[struct_idx, :])
            y_list.append(row['escape_score'])

    if len(X_list) < 20:
        print(f"  Skipped (only {len(X_list)} mutations)")
        continue

    X = np.array(X_list)
    y = np.array(y_list)

    threshold = np.median(y)
    y_binary = (y > threshold).astype(int)

    # Simple train/test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
    )

    # Select top features
    correlations = []
    for i in range(X_train.shape[1]):
        if X_train[:, i].std() > 1e-6:
            rho, _ = stats.spearmanr(X_train[:, i], y_train)
            correlations.append((i, abs(rho)))

    correlations.sort(key=lambda x: x[1], reverse=True)
    top_10 = [idx for idx, _ in correlations[:10]]

    # Train
    dtrain = xgb.DMatrix(X_train[:, top_10], label=y_train)
    dtest = xgb.DMatrix(X_test[:, top_10], label=y_test)

    params = {'objective': 'binary:logistic', 'max_depth': 3, 'eta': 0.1}
    model = xgb.train(params, dtrain, num_boost_round=30, verbose_eval=False)

    # Evaluate
    y_pred = model.predict(dtest)
    auprc = average_precision_score(y_test, y_pred)

    antibody_models[antibody] = {
        'model': model,
        'features': top_10,
        'threshold': threshold
    }

    antibody_results[antibody] = {
        'n_mutations': len(X),
        'auprc': float(auprc),
        'selected_features': top_10
    }

    print(f"  {antibody}: AUPRC={auprc:.4f} (n={len(X)})")

print()
print("="*80)
print(f"TRAINED {len(antibody_models)} ANTIBODY-SPECIFIC MODELS")
print("="*80)
print()

# Save results
with open('prism-escape-benchmark/antibody_specific_results.json', 'w') as f:
    json.dump(antibody_results, f, indent=2)

print("✅ Antibody-specific models ready")
print("   Now can predict: 'E484K escapes which antibodies?'")
print()

# Demo: Check E484K against all antibodies
print("EXAMPLE: E484K Antibody-Specific Escape")
print("─"*80)

e484k_pos = 484 - RBD_START
if 0 <= e484k_pos < features.shape[0]:
    e484k_features = features[e484k_pos:e484k_pos+1, :]

    predictions = {}
    for ab_name, ab_model in antibody_models.items():
        model = ab_model['model']
        feat_idx = ab_model['features']

        dtest = xgb.DMatrix(e484k_features[:, feat_idx])
        prob = model.predict(dtest)[0]
        predictions[ab_name] = prob

    # Sort by escape probability
    sorted_abs = sorted(predictions.items(), key=lambda x: x[1], reverse=True)

    print("E484K escape probabilities:")
    for ab, prob in sorted_abs:
        status = "🔴 HIGH" if prob > 0.7 else "🟡 MED" if prob > 0.4 else "🟢 LOW"
        print(f"  {ab:20s}: {prob:.4f} {status}")

print()
print("="*80)
print("ANTIBODY-SPECIFIC ESCAPE: IMPLEMENTED")
print("="*80)
