#!/usr/bin/env python3
"""
EVEscape-compatible evaluation metrics for viral escape prediction.

Reference: Thadani et al., Nature 622, 818–825 (2023)
"""

import numpy as np
from scipy import stats
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    matthews_corrcoef,
    r2_score,
)
from typing import Dict, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class EscapeMetricsResult:
    """Container for escape prediction metrics."""
    # Primary metrics (EVEscape benchmarks)
    auprc: float                  # Area under precision-recall curve
    auprc_normalized: float       # Normalized by null model (AUPRC / baseline)
    auroc: float                  # Area under ROC curve
    spearman_rho: float          # Spearman correlation
    spearman_pval: float         # P-value for correlation

    # Secondary metrics
    mcc: float                    # Matthews correlation coefficient
    top_10_recall: float         # Recall in top 10% predictions
    top_10_precision: float      # Precision in top 10%
    top_5_recall: float          # Recall in top 5%

    # Calibration
    ece: float                    # Expected calibration error

    # Sample info
    n_total: int
    n_positive: int
    imbalance_ratio: float

    def to_dict(self) -> Dict:
        return self.__dict__

    def __str__(self) -> str:
        return f"""
Escape Prediction Metrics:
─────────────────────────────────────────────────
PRIMARY METRICS:
  AUPRC:           {self.auprc:.4f}
  AUPRC (norm):    {self.auprc_normalized:.2f}x
  AUROC:           {self.auroc:.4f}
  Spearman ρ:      {self.spearman_rho:.4f} (p={self.spearman_pval:.2e})

SECONDARY METRICS:
  MCC:             {self.mcc:.4f}
  Top-10% Recall:  {self.top_10_recall:.4f}
  Top-10% Precision: {self.top_10_precision:.4f}
  ECE:             {self.ece:.4f}

DATASET:
  Total samples:   {self.n_total}
  Positive:        {self.n_positive} ({self.n_positive/self.n_total:.1%})
  Imbalance:       {self.imbalance_ratio:.1f}:1
─────────────────────────────────────────────────
"""


class EscapeMetrics:
    """
    Compute evaluation metrics for viral escape prediction.

    Implements EVEscape benchmark protocol (Nature 2023):
    - AUPRC normalized by class imbalance
    - Spearman correlation for continuous predictions
    - Top-decile recall for high-risk mutation identification
    """

    def compute_all(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_true_continuous: np.ndarray = None
    ) -> EscapeMetricsResult:
        """
        Compute full suite of escape prediction metrics.

        Args:
            y_true: Binary escape labels (0/1)
            y_pred: Predicted escape probabilities [0, 1]
            y_true_continuous: Optional continuous escape scores for Spearman

        Returns:
            EscapeMetricsResult with all metrics
        """
        if y_true_continuous is None:
            y_true_continuous = y_true.astype(float)

        # Primary metrics
        auprc = average_precision_score(y_true, y_pred)
        baseline = np.mean(y_true)  # Null model AUPRC
        auprc_normalized = auprc / baseline if baseline > 0 else 0.0

        auroc = roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.5

        spearman_rho, spearman_pval = stats.spearmanr(y_true_continuous, y_pred)

        # Secondary metrics
        y_pred_binary = (y_pred >= 0.5).astype(int)
        mcc = matthews_corrcoef(y_true, y_pred_binary)

        top_10_recall = self._top_k_recall(y_true, y_pred, k=0.10)
        top_10_precision = self._top_k_precision(y_true, y_pred, k=0.10)
        top_5_recall = self._top_k_recall(y_true, y_pred, k=0.05)

        ece = self._expected_calibration_error(y_true, y_pred)

        # Sample info
        n_total = len(y_true)
        n_positive = int(np.sum(y_true))
        imbalance_ratio = (n_total - n_positive) / max(n_positive, 1)

        return EscapeMetricsResult(
            auprc=auprc,
            auprc_normalized=auprc_normalized,
            auroc=auroc,
            spearman_rho=spearman_rho,
            spearman_pval=spearman_pval,
            mcc=mcc,
            top_10_recall=top_10_recall,
            top_10_precision=top_10_precision,
            top_5_recall=top_5_recall,
            ece=ece,
            n_total=n_total,
            n_positive=n_positive,
            imbalance_ratio=imbalance_ratio
        )

    def _top_k_recall(self, y_true: np.ndarray, y_pred: np.ndarray, k: float) -> float:
        """
        Recall in top-k% of predictions.

        EVEscape target: >30% of escape mutations in top 10%
        """
        n_top = max(1, int(len(y_pred) * k))
        top_indices = np.argsort(y_pred)[-n_top:]

        n_captured = np.sum(y_true[top_indices])
        n_total_positive = np.sum(y_true)

        return n_captured / n_total_positive if n_total_positive > 0 else 0.0

    def _top_k_precision(self, y_true: np.ndarray, y_pred: np.ndarray, k: float) -> float:
        """Precision in top-k% of predictions."""
        n_top = max(1, int(len(y_pred) * k))
        top_indices = np.argsort(y_pred)[-n_top:]

        return np.mean(y_true[top_indices])

    def _expected_calibration_error(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """Expected Calibration Error - measures probability calibration."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            in_bin = (y_pred > bin_boundaries[i]) & (y_pred <= bin_boundaries[i + 1])
            prop_in_bin = np.mean(in_bin)

            if prop_in_bin > 0:
                avg_confidence = np.mean(y_pred[in_bin])
                avg_accuracy = np.mean(y_true[in_bin])
                ece += np.abs(avg_confidence - avg_accuracy) * prop_in_bin

        return ece


def compare_to_evescape(
    prism_metrics: EscapeMetricsResult,
    virus: str = "sars2"
) -> Dict:
    """
    Compare PRISM results to EVEscape baselines.

    EVEscape baselines (Nature 2023):
    - SARS-CoV-2: AUPRC 0.53, Top-10% recall 0.31
    - HIV: AUPRC 0.32
    - Influenza: AUPRC 0.28
    """
    baselines = {
        "sars2": {"auprc": 0.53, "top_10_recall": 0.31, "auroc": 0.85},
        "hiv": {"auprc": 0.32},
        "influenza": {"auprc": 0.28},
    }

    baseline = baselines.get(virus, {})

    comparison = {
        "prism": prism_metrics.to_dict(),
        "evescape": baseline,
        "deltas": {}
    }

    # Compute deltas
    for metric, baseline_val in baseline.items():
        prism_val = getattr(prism_metrics, metric, None)
        if prism_val is not None:
            delta = prism_val - baseline_val
            comparison["deltas"][metric] = {
                "absolute": delta,
                "relative": (delta / baseline_val * 100) if baseline_val != 0 else 0
            }

    return comparison


if __name__ == "__main__":
    """Test metrics computation."""

    # Simulate data
    np.random.seed(42)
    n_samples = 1000
    n_positive = 100

    y_true = np.array([1] * n_positive + [0] * (n_samples - n_positive))
    y_pred = np.random.beta(2, 5, n_samples)  # Skewed towards low scores

    # Add signal to positives
    y_pred[y_true == 1] += np.random.beta(5, 2, n_positive) * 0.5

    y_pred = np.clip(y_pred, 0, 1)

    # Compute metrics
    metrics = EscapeMetrics()
    result = metrics.compute_all(y_true, y_pred, y_true.astype(float))

    print(result)

    # Compare to EVEscape
    comparison = compare_to_evescape(result, "sars2")
    print("\n" + "="*60)
    print("COMPARISON TO EVESCAPE")
    print("="*60)

    for metric, delta_info in comparison["deltas"].items():
        print(f"{metric:20s}: {delta_info['absolute']:+.4f} ({delta_info['relative']:+.1f}%)")
