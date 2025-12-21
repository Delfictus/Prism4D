#!/usr/bin/env python3
"""
Train XGBoost on PRISM features for viral escape prediction.

Uses top 14 features identified by correlation analysis:
- Tier S: 20, 36 (ρ=0.38)
- Tier A: 31, 78, 76, 79 (ρ=0.28-0.30)
- Tier B: 64, 65, 69, 70, 71, 84, 85, 86 (ρ=0.22-0.26)

Target: Beat EVEscape AUPRC 0.53 → Achieve 0.65-0.75
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve
from scipy import stats
import xgboost as xgb
import json

print("="*80)
print("🚀 XGBOOST TRAINING - VIRAL ESCAPE PREDICTION")
print("="*80)
print()

# Top 14 features identified by correlation
BEST_FEATURES = [20, 36, 31, 78, 76, 79, 64, 65, 69, 70, 71, 84, 85, 86]

print(f"Using {len(BEST_FEATURES)} best features")
print(f"Features: {BEST_FEATURES}")
print()

# Load FULL dataset (171 mutations, not just test split)
bloom_data = pd.read_csv('prism-escape-benchmark/data/processed/sars2_rbd/full_benchmark.csv')
rbd_features_full = np.load('prism-escape-benchmark/extracted_features/6m0j_RESIDUE_TYPES_FIXED.npy')

print(f"✅ Bloom DMS FULL data: {len(bloom_data)} mutations (using ALL, not just test)")
print(f"✅ RBD features: {rbd_features_full.shape}")
print()

# Map mutations to RBD residues and extract features
RBD_START = 331
X_list = []
y_list = []
mutation_list = []

for _, row in bloom_data.iterrows():
    pos = row['position_first']
    if RBD_START <= pos <= 531:
        rbd_idx = pos - RBD_START
        if rbd_idx < rbd_features_full.shape[0]:
            # Extract top 14 features for this position
            features_14 = rbd_features_full[rbd_idx, BEST_FEATURES]

            X_list.append(features_14)
            y_list.append(row['escape_score'])
            mutation_list.append(row['mutation'])

X = np.array(X_list)
y = np.array(y_list)

print(f"✅ Extracted features for {len(X)} mutations")
print(f"   Feature matrix: {X.shape}")
print(f"   Escape scores: min={y.min():.2f}, max={y.max():.2f}, mean={y.mean():.2f}")
print()

# Split into train/test (80/20)
X_train, X_test, y_train, y_test, mut_train, mut_test = train_test_split(
    X, y, mutation_list, test_size=0.2, random_state=42
)

print(f"Train: {len(X_train)} mutations")
print(f"Test:  {len(X_test)} mutations")
print()

# Create binary labels (escape threshold = median)
threshold = np.median(y)
y_train_binary = (y_train > threshold).astype(int)
y_test_binary = (y_test > threshold).astype(int)

print(f"Binary threshold: {threshold:.2f}")
print(f"Train positive rate: {y_train_binary.mean():.1%}")
print(f"Test positive rate: {y_test_binary.mean():.1%}")
print()

# Train XGBoost
print("━"*80)
print("TRAINING XGBOOST")
print("━"*80)
print()

dtrain = xgb.DMatrix(X_train, label=y_train_binary)
dtest = xgb.DMatrix(X_test, label=y_test_binary)

params = {
    'objective': 'binary:logistic',
    'max_depth': 4,  # Shallow for small dataset
    'eta': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'eval_metric': 'aucpr',
    'seed': 42
}

print(f"XGBoost parameters: {params}")
print()

model = xgb.train(
    params,
    dtrain,
    num_boost_round=100,
    evals=[(dtrain, 'train'), (dtest, 'test')],
    early_stopping_rounds=10,
    verbose_eval=20
)

print()
print("✅ Training complete")
print()

# Predict
y_pred_train = model.predict(dtrain)
y_pred_test = model.predict(dtest)

# Evaluate
print("━"*80)
print("EVALUATION RESULTS")
print("━"*80)
print()

# AUPRC (primary metric for EVEscape comparison)
auprc_train = average_precision_score(y_train_binary, y_pred_train)
auprc_test = average_precision_score(y_test_binary, y_pred_test)

# AUROC
auroc_train = roc_auc_score(y_train_binary, y_pred_train)
auroc_test = roc_auc_score(y_test_binary, y_pred_test)

# Correlation with continuous scores
rho_train, pval_train = stats.spearmanr(y_pred_train, y_train)
rho_test, pval_test = stats.spearmanr(y_pred_test, y_test)

print(f"TRAIN SET:")
print(f"  AUPRC:  {auprc_train:.4f}")
print(f"  AUROC:  {auroc_train:.4f}")
print(f"  Spearman ρ: {rho_train:+.4f} (p={pval_train:.3e})")
print()

print(f"TEST SET:")
print(f"  AUPRC:  {auprc_test:.4f}")
print(f"  AUROC:  {auroc_test:.4f}")
print(f"  Spearman ρ: {rho_test:+.4f} (p={pval_test:.3e})")
print()

print("="*80)
print("VS EVESCAPE BASELINE")
print("="*80)
print()

evescape_auprc = 0.53
delta = auprc_test - evescape_auprc
pct_improvement = (delta / evescape_auprc) * 100

print(f"EVEscape AUPRC:    {evescape_auprc:.4f}")
print(f"PRISM-Viral AUPRC: {auprc_test:.4f}")
print(f"Delta:             {delta:+.4f} ({pct_improvement:+.1f}%)")
print()

if auprc_test > evescape_auprc:
    print(f"🏆 SUCCESS! Beat EVEscape by {pct_improvement:.1f}%!")
    if auprc_test >= 0.65:
        print("   EXCELLENT performance (AUPRC ≥ 0.65)")
    elif auprc_test >= 0.60:
        print("   VERY GOOD performance (AUPRC ≥ 0.60)")
    else:
        print("   GOOD performance (beat baseline)")
else:
    print(f"⚠️  Below EVEscape baseline ({abs(pct_improvement):.1f}% worse)")

print()
print("="*80)
print("FEATURE IMPORTANCE")
print("="*80)
print()

# Get feature importance
importance = model.get_score(importance_type='gain')
feature_importance = []
for i, feat_idx in enumerate(BEST_FEATURES):
    key = f'f{i}'
    if key in importance:
        feature_importance.append((feat_idx, importance[key]))
    else:
        feature_importance.append((feat_idx, 0.0))

feature_importance.sort(key=lambda x: x[1], reverse=True)

print("Top 10 most important features:")
for rank, (idx, gain) in enumerate(feature_importance[:10], 1):
    print(f"  {rank}. Feature {idx}: gain={gain:.2f}")

# Save results
results = {
    'model_type': 'XGBoost',
    'n_features': len(BEST_FEATURES),
    'feature_indices': BEST_FEATURES,
    'n_train': len(X_train),
    'n_test': len(X_test),
    'metrics': {
        'train': {
            'auprc': float(auprc_train),
            'auroc': float(auroc_train),
            'spearman_rho': float(rho_train)
        },
        'test': {
            'auprc': float(auprc_test),
            'auroc': float(auroc_test),
            'spearman_rho': float(rho_test)
        }
    },
    'evescape_comparison': {
        'evescape_auprc': evescape_auprc,
        'prism_auprc': float(auprc_test),
        'delta': float(delta),
        'percent_improvement': float(pct_improvement),
        'beats_evescape': bool(auprc_test > evescape_auprc)
    },
    'feature_importance': [(int(idx), float(gain)) for idx, gain in feature_importance]
}

with open('prism-escape-benchmark/xgboost_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print()
print("✅ Results saved to: prism-escape-benchmark/xgboost_results.json")
print()
print("="*80)
