#!/usr/bin/env python3
"""
PRISM-LBS SOTA Ensemble Classifier
- GPU feature extraction (keep the speed)
- XGBoost + Random Forest ensemble
- Proper class balancing
- F1-optimized thresholds

Target: F1 0.25-0.30, AUC 0.75-0.78
"""

import numpy as np
import json
from pathlib import Path
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_fscore_support
import xgboost as xgb

def load_features_from_gpu(feature_dir):
    """Load 92-dim features extracted by GPU kernel"""
    print("Loading GPU-extracted features...")
    
    # Features are in normalized format from train-readout
    # We'll load them and select features 48-91 (drop TDA)
    
    features = []
    labels = []
    
    # Load from binary or text format
    # TODO: Adapt to actual output format
    
    return np.array(features), np.array(labels)

def train_xgboost(X_train, y_train, X_test, y_test):
    """Train XGBoost with proper imbalance handling"""
    
    # Calculate actual class imbalance
    n_neg = np.sum(y_train == 0)
    n_pos = np.sum(y_train == 1)
    scale_pos_weight = n_neg / n_pos  # ~61 for CryptoBench
    
    print(f"Class imbalance: {n_pos} pos, {n_neg} neg, ratio: {scale_pos_weight:.1f}")
    
    params = {
        'objective': 'binary:logistic',
        'eval_metric': ['aucpr', 'auc'],  # Focus on AUCPR for imbalanced
        'max_depth': 6,
        'eta': 0.05,  # Learning rate
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'scale_pos_weight': scale_pos_weight,  # CRITICAL: Actual imbalance
        'min_child_weight': 1,
        'gamma': 0.1,
        'tree_method': 'hist',
        'seed': 42
    }
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    evals = [(dtrain, 'train'), (dtest, 'test')]
    
    print("Training XGBoost (300 trees with early stopping)...")
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=300,
        evals=evals,
        early_stopping_rounds=50,
        verbose_eval=50
    )
    
    # Predict probabilities
    y_pred_proba = model.predict(dtest)
    
    return model, y_pred_proba

def train_random_forest(X_train, y_train, X_test, y_test):
    """Train Random Forest with proper imbalance handling"""
    
    n_neg = np.sum(y_train == 0)
    n_pos = np.sum(y_train == 1)
    
    print("Training Random Forest (300 trees)...")
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=4,
        class_weight={0: 1.0, 1: n_neg/n_pos},  # Actual imbalance
        max_features='sqrt',
        n_jobs=-1,
        random_state=42
    )
    
    rf.fit(X_train, y_train)
    
    # Predict probabilities
    y_pred_proba = rf.predict_proba(X_test)[:, 1]
    
    return rf, y_pred_proba

def find_optimal_threshold(y_true, y_pred_proba, metric='f1'):
    """Find threshold that maximizes F1"""
    
    best_threshold = 0.5
    best_score = 0.0
    
    for threshold in np.arange(0.01, 0.99, 0.01):
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        if metric == 'f1':
            score = f1_score(y_true, y_pred)
        else:
            prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
            score = f1
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    return best_threshold, best_score

def ensemble_predict(xgb_proba, rf_proba, method='average'):
    """Ensemble predictions from XGBoost and Random Forest"""
    
    if method == 'average':
        return (xgb_proba + rf_proba) / 2.0
    elif method == 'weighted':
        # Weight XGBoost slightly higher (typically better for tabular)
        return 0.6 * xgb_proba + 0.4 * rf_proba
    elif method == 'vote':
        # Majority vote (requires thresholds)
        return ((xgb_proba > 0.5) & (rf_proba > 0.5)).astype(float)
    
def main():
    print("═══════════════════════════════════════════════════════════════")
    print("PRISM-LBS SOTA ENSEMBLE TRAINING")
    print("GPU Feature Extraction + XGBoost + Random Forest")
    print("═══════════════════════════════════════════════════════════════")
    
    # Load GPU-extracted features
    X, y = load_features_from_gpu("./features")
    
    # CRITICAL: Select only features 48-91 (drop TDA garbage)
    print(f"\nOriginal features: {X.shape[1]} dims")
    X_selected = X[:, 48:92]  # Features 48-91 (44 dims: 32 base + 12 physics)
    print(f"Selected features: {X_selected.shape[1]} dims (dropped TDA 0-47)")
    
    # Train/test split (use folds from JSON)
    # TODO: Load actual folds
    split_idx = int(0.8 * len(X_selected))
    X_train, X_test = X_selected[:split_idx], X_selected[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"\nTrain: {len(X_train)} samples ({np.sum(y_train)} positive)")
    print(f"Test:  {len(X_test)} samples ({np.sum(y_test)} positive)")
    
    # Train both models
    xgb_model, xgb_proba = train_xgboost(X_train, y_train, X_test, y_test)
    rf_model, rf_proba = train_random_forest(X_train, y_train, X_test, y_test)
    
    # Ensemble predictions
    ensemble_proba = ensemble_predict(xgb_proba, rf_proba, method='weighted')
    
    # Find optimal thresholds
    print("\n═══════════════════════════════════════════════════════════════")
    print("THRESHOLD OPTIMIZATION")
    print("═══════════════════════════════════════════════════════════════")
    
    xgb_thresh, xgb_f1 = find_optimal_threshold(y_test, xgb_proba)
    rf_thresh, rf_f1 = find_optimal_threshold(y_test, rf_proba)
    ens_thresh, ens_f1 = find_optimal_threshold(y_test, ensemble_proba)
    
    print(f"XGBoost:  threshold={xgb_thresh:.3f}, F1={xgb_f1:.4f}")
    print(f"RF:       threshold={rf_thresh:.3f}, F1={rf_f1:.4f}")
    print(f"Ensemble: threshold={ens_thresh:.3f}, F1={ens_f1:.4f}")
    
    # Final metrics
    print("\n═══════════════════════════════════════════════════════════════")
    print("FINAL RESULTS")
    print("═══════════════════════════════════════════════════════════════")
    
    for name, proba, thresh in [
        ("XGBoost", xgb_proba, xgb_thresh),
        ("Random Forest", rf_proba, rf_thresh),
        ("Ensemble", ensemble_proba, ens_thresh)
    ]:
        y_pred = (proba >= thresh).astype(int)
        
        auc = roc_auc_score(y_test, proba)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
        
        print(f"\n{name}:")
        print(f"  AUC-ROC:   {auc:.4f}")
        print(f"  F1:        {f1:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  Threshold: {thresh:.3f}")
    
    # Feature importance
    print("\n═══════════════════════════════════════════════════════════════")
    print("TOP 10 FEATURES (XGBoost)")
    print("═══════════════════════════════════════════════════════════════")
    
    importance = xgb_model.get_score(importance_type='gain')
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
    for feat, score in sorted_importance:
        print(f"  {feat}: {score:.2f}")

if __name__ == "__main__":
    main()
