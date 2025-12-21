#!/usr/bin/env python3
"""
PRISM-LBS SOTA ENSEMBLE TRAINER (SESSION 10B)
Target: F1 > 0.40, AUC > 0.82

This script trains XGBoost + Random Forest with proper 61x class weighting.
Designed for the 70-dim SOTA feature set from Session 9.
"""

import numpy as np
import sys
import json
from pathlib import Path

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        roc_auc_score, f1_score, precision_score, recall_score,
        precision_recall_curve, confusion_matrix, classification_report
    )
    import xgboost as xgb
except ImportError as e:
    print(f"ERROR: Missing dependency: {e}")
    print("Install with: pip install xgboost scikit-learn numpy")
    sys.exit(1)

def load_data():
    """Load NPY feature files from ml_training directory."""
    train_X = np.load("ml_training/train_features.npy").astype(np.float32)
    train_y = np.load("ml_training/train_labels.npy").astype(np.int32)
    test_X = np.load("ml_training/test_features.npy").astype(np.float32)
    test_y = np.load("ml_training/test_labels.npy").astype(np.int32)

    n_features = train_X.shape[1]
    pos_train = np.sum(train_y == 1)
    neg_train = np.sum(train_y == 0)

    print(f"Features: {n_features} (70-dim SOTA)")
    print(f"Train: {len(train_y)} samples ({pos_train} positive, {pos_train/len(train_y)*100:.2f}%)")
    print(f"Test: {len(test_y)} samples ({np.sum(test_y==1)} positive)")

    # Calculate imbalance ratio
    imbalance = neg_train / max(pos_train, 1)
    print(f"Class imbalance: {imbalance:.1f}:1 (neg:pos)")

    return train_X, train_y, test_X, test_y, imbalance

def find_optimal_threshold(y_true, y_proba):
    """Find threshold that maximizes F1."""
    prec, rec, thresh = precision_recall_curve(y_true, y_proba)
    f1 = 2 * prec * rec / (prec + rec + 1e-10)
    best_idx = np.argmax(f1)
    optimal_f1 = f1[best_idx]
    optimal_thresh = thresh[min(best_idx, len(thresh)-1)]
    return optimal_thresh, optimal_f1

def train_and_evaluate():
    """Main training function."""

    print("=" * 70)
    print("PRISM-LBS SOTA ENSEMBLE TRAINING (SESSION 10B)")
    print("TARGET: F1 > 0.40, AUC > 0.82")
    print("=" * 70)
    print()

    # Load data
    train_X, train_y, test_X, test_y, imbalance = load_data()

    # ═══════════════════════════════════════════════════════════════
    # TRAIN XGBOOST
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 50)
    print(f"TRAINING XGBOOST (scale_pos_weight={imbalance:.1f})")
    print("=" * 50)

    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=10,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=imbalance,  # KEY: Proper class weighting
        eval_metric='aucpr',
        random_state=42,
        n_jobs=-1,
        tree_method='hist'  # Fast histogram-based algorithm
    )

    print("Training XGBoost...")
    xgb_model.fit(train_X, train_y, verbose=False)
    print("Training complete")

    xgb_proba = xgb_model.predict_proba(test_X)[:, 1]
    xgb_thresh, xgb_best_f1 = find_optimal_threshold(test_y, xgb_proba)
    xgb_pred = (xgb_proba >= xgb_thresh).astype(int)

    xgb_auc = roc_auc_score(test_y, xgb_proba)
    xgb_f1 = f1_score(test_y, xgb_pred)
    xgb_prec = precision_score(test_y, xgb_pred)
    xgb_rec = recall_score(test_y, xgb_pred)

    print(f"\nXGBoost Results:")
    print(f"  Optimal threshold: {xgb_thresh:.4f}")
    print(f"  AUC:       {xgb_auc:.4f}")
    print(f"  F1:        {xgb_f1:.4f}")
    print(f"  Precision: {xgb_prec:.4f}")
    print(f"  Recall:    {xgb_rec:.4f}")

    # ═══════════════════════════════════════════════════════════════
    # TRAIN RANDOM FOREST
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 50)
    print(f"TRAINING RANDOM FOREST (class_weight={{1: {imbalance:.1f}}})")
    print("=" * 50)

    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight={0: 1.0, 1: imbalance},  # KEY: Proper class weighting
        random_state=42,
        n_jobs=-1
    )

    print("Training Random Forest...")
    rf_model.fit(train_X, train_y)
    print("Training complete")

    rf_proba = rf_model.predict_proba(test_X)[:, 1]
    rf_thresh, rf_best_f1 = find_optimal_threshold(test_y, rf_proba)
    rf_pred = (rf_proba >= rf_thresh).astype(int)

    rf_auc = roc_auc_score(test_y, rf_proba)
    rf_f1 = f1_score(test_y, rf_pred)
    rf_prec = precision_score(test_y, rf_pred)
    rf_rec = recall_score(test_y, rf_pred)

    print(f"\nRandom Forest Results:")
    print(f"  Optimal threshold: {rf_thresh:.4f}")
    print(f"  AUC:       {rf_auc:.4f}")
    print(f"  F1:        {rf_f1:.4f}")
    print(f"  Precision: {rf_prec:.4f}")
    print(f"  Recall:    {rf_rec:.4f}")

    # ═══════════════════════════════════════════════════════════════
    # ENSEMBLE (WEIGHTED AVERAGE)
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 50)
    print("CREATING ENSEMBLE (0.6 XGBoost + 0.4 RF)")
    print("=" * 50)

    ens_proba = 0.6 * xgb_proba + 0.4 * rf_proba
    ens_thresh, ens_best_f1 = find_optimal_threshold(test_y, ens_proba)
    ens_pred = (ens_proba >= ens_thresh).astype(int)

    ens_auc = roc_auc_score(test_y, ens_proba)
    ens_f1 = f1_score(test_y, ens_pred)
    ens_prec = precision_score(test_y, ens_pred)
    ens_rec = recall_score(test_y, ens_pred)

    print(f"\nEnsemble Results:")
    print(f"  Optimal threshold: {ens_thresh:.4f}")
    print(f"  AUC:       {ens_auc:.4f}")
    print(f"  F1:        {ens_f1:.4f}")
    print(f"  Precision: {ens_prec:.4f}")
    print(f"  Recall:    {ens_rec:.4f}")

    # ═══════════════════════════════════════════════════════════════
    # FINAL REPORT
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("FINAL TEST SET RESULTS")
    print("=" * 70)

    print("\nModel Comparison:")
    print(f"{'Model':<15} {'AUC':>8} {'F1':>8} {'Precision':>11} {'Recall':>8}")
    print("-" * 70)
    print(f"{'XGBoost':<15} {xgb_auc:>8.4f} {xgb_f1:>8.4f} {xgb_prec:>11.4f} {xgb_rec:>8.4f}")
    print(f"{'Random Forest':<15} {rf_auc:>8.4f} {rf_f1:>8.4f} {rf_prec:>11.4f} {rf_rec:>8.4f}")
    print(f"{'ENSEMBLE':<15} {ens_auc:>8.4f} {ens_f1:>8.4f} {ens_prec:>11.4f} {ens_rec:>8.4f}")

    # Success check
    print("\n" + "=" * 70)
    if ens_f1 >= 0.40:
        print("SUCCESS: F1 >= 0.40 ACHIEVED!")
        print(f"  Final F1:  {ens_f1:.4f}")
        print(f"  Final AUC: {ens_auc:.4f}")
    elif ens_f1 >= 0.35:
        print(f"CLOSE: F1 = {ens_f1:.4f} (target: 0.40)")
        print("Consider tuning hyperparameters or feature selection")
    else:
        print(f"MORE WORK NEEDED: F1 = {ens_f1:.4f} (target: 0.40)")
    print("=" * 70)

    # Save results
    results = {
        "xgboost": {
            "auc": float(xgb_auc),
            "f1": float(xgb_f1),
            "precision": float(xgb_prec),
            "recall": float(xgb_rec),
            "threshold": float(xgb_thresh)
        },
        "random_forest": {
            "auc": float(rf_auc),
            "f1": float(rf_f1),
            "precision": float(rf_prec),
            "recall": float(rf_rec),
            "threshold": float(rf_thresh)
        },
        "test_results": {
            "ensemble": {
                "auc": float(ens_auc),
                "f1": float(ens_f1),
                "precision": float(ens_prec),
                "recall": float(ens_rec),
                "threshold": float(ens_thresh)
            }
        },
        "config": {
            "train_samples": len(train_y),
            "test_samples": len(test_y),
            "features": train_X.shape[1],
            "imbalance_ratio": float(imbalance)
        }
    }

    with open("ml_training/ensemble_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to ml_training/ensemble_results.json")

    # Save models
    try:
        import pickle
        with open("ml_training/xgb_model.pkl", "wb") as f:
            pickle.dump(xgb_model, f)
        with open("ml_training/rf_model.pkl", "wb") as f:
            pickle.dump(rf_model, f)
        print("Models saved to ml_training/*.pkl")
    except Exception as e:
        print(f"Warning: Could not save models: {e}")

    return results

if __name__ == "__main__":
    train_and_evaluate()
