"""
Evaluation metrics for credit scoring experiments.

Consolidates all metric computation including:
- Standard classification metrics (accuracy, precision, recall, F1, AUC, etc.)
- Reject inference metrics (kickout, rejection rates), inspired by
  Kozodoi et al. 2019; our Kickout differs from theirs, which is defined as a
  two-model comparison, and measures a single model's rejection precision on
  one evaluation pool (see paper, Sec. 3)

Label convention: By default assumes 1 = default (bad), 0 = no default (good).
Use positive_is_good=True if your labels are reversed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
)


# =============================================================================
# Standard Evaluation Metrics
# =============================================================================

@dataclass
class EvalMetrics:
    """Container for standard evaluation metrics."""
    accuracy: float
    precision: float
    recall: float
    
    def to_dict(self) -> dict[str, float]:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
        }


def evaluate_predictions(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    zero_division: float = 0.0,
) -> EvalMetrics:
    """Compute standard classification metrics."""
    return EvalMetrics(
        accuracy=accuracy_score(y_true, y_pred),  # type: ignore
        precision=precision_score(y_true, y_pred, zero_division=zero_division),  # type: ignore
        recall=recall_score(y_true, y_pred, zero_division=zero_division),  # type: ignore
    )


def get_confusion_matrix(
    y_true: ArrayLike,
    y_pred: ArrayLike,
) -> np.ndarray:
    """Return confusion matrix as 2x2 array: [[TN, FP], [FN, TP]]."""
    return confusion_matrix(y_true, y_pred)


# =============================================================================
# Comprehensive Metrics (for model evaluation)
# =============================================================================

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    positive_is_good: bool = False,
) -> dict[str, float]:
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        y_proba: Predicted probabilities for positive class. If None,
            probability-based metrics (AUC, Brier, etc.) are skipped.
        positive_is_good: Label convention. For credit scoring where
            1=default (bad), use False (default).
    
    Returns:
        Dictionary of metric names to values.
    """
    metrics: dict[str, float] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    } # type: ignore
    
    # Probability-based metrics
    if y_proba is not None:
        metrics.update({
            "auc": roc_auc_score(y_true, y_proba),
            "avg_precision": average_precision_score(y_true, y_proba),
            "brier": brier_score_loss(y_true, y_proba),
        }) # type: ignore
    
    # Reject inference metrics
    cm = confusion_matrix(y_true, y_pred)
    kickout_result = compute_kickout_from_cm(cm, positive_is_good=positive_is_good)
    metrics.update({
        "kickout": kickout_result.kickout,
        "rejection_rate": kickout_result.rejection_rate,
        "bad_rejection_rate": kickout_result.bad_rejection_rate,
        "good_rejection_rate": kickout_result.good_rejection_rate,
    })
    
    return metrics


# =============================================================================
# Kickout Metric
# =============================================================================

@dataclass
class KickoutResult:
    """Results from kickout metric computation."""
    kickout: float              # KK score in [-1, 1]
    n_rejected: int             # Total rejected
    n_bad_rejected: int         # Bad payers rejected (good outcome)
    n_good_rejected: int        # Good payers rejected (bad outcome)
    rejection_rate: float       # Fraction of population rejected
    bad_rejection_rate: float   # P(rejected | bad)
    good_rejection_rate: float  # P(rejected | good)
    
    def to_dict(self) -> dict[str, float]:
        return {
            "kickout": self.kickout,
            "n_rejected": self.n_rejected,
            "n_bad_rejected": self.n_bad_rejected,
            "n_good_rejected": self.n_good_rejected,
            "rejection_rate": self.rejection_rate,
            "bad_rejection_rate": self.bad_rejection_rate,
            "good_rejection_rate": self.good_rejection_rate,
        }


def compute_kickout_from_cm(
    confusion_matrix: np.ndarray,
    positive_is_good: bool = False,
) -> KickoutResult:
    """
    Compute kickout metric from a confusion matrix.
    
    Assumes standard sklearn format: [[TN, FP], [FN, TP]]
    
    "Rejected" = predicted to default (denied the loan).
    Kickout measures whether rejected applicants are disproportionately
    bad (KK > 0, desirable) or good (KK < 0, undesirable).
    
    Args:
        confusion_matrix: sklearn confusion matrix [[TN, FP], [FN, TP]].
        positive_is_good: If True, label 1 means "good" (no default).
            If False, label 1 means "bad" (default).
    
    Returns:
        KickoutResult with metric and component counts.
    """
    cm = np.asarray(confusion_matrix)
    if cm.shape != (2, 2):
        raise ValueError(f"Expected 2x2 confusion matrix, got {cm.shape}")
    
    tn, fp = cm[0, 0], cm[0, 1]
    fn, tp = cm[1, 0], cm[1, 1]
    
    if positive_is_good:
        # Label 1 = good, Label 0 = bad
        # Reject = predicted 0 (predicted bad, denied loan)
        # Rejected: TN (truly bad, predicted bad) + FN (truly good, predicted bad)
        n_bad_rejected = tn
        n_good_rejected = fn
        n_good = fn + tp
        n_bad = tn + fp
    else:
        # Label 1 = bad (default), Label 0 = good (no default)
        # Reject = predicted 1 (predicted bad, denied loan)
        # Rejected: TP (truly bad, predicted bad) + FP (truly good, predicted bad)
        n_bad_rejected = tp
        n_good_rejected = fp
        n_good = tn + fp
        n_bad = fn + tp
    
    n_rejected = n_bad_rejected + n_good_rejected
    
    if n_rejected == 0:
        kickout = 0.0
    else:
        kickout = (n_bad_rejected - n_good_rejected) / n_rejected
    
    n_total = cm.sum()
    
    return KickoutResult(
        kickout=float(kickout),
        n_rejected=int(n_rejected),
        n_bad_rejected=int(n_bad_rejected),
        n_good_rejected=int(n_good_rejected),
        rejection_rate=n_rejected / n_total if n_total > 0 else 0.0,
        bad_rejection_rate=n_bad_rejected / n_bad if n_bad > 0 else 0.0,
        good_rejection_rate=n_good_rejected / n_good if n_good > 0 else 0.0,
    )


# =============================================================================
# Sanity Check
# =============================================================================

def _sanity_check():
    """Verify kickout interpretation with credit scoring label convention."""
    # Scenario: 100 applicants, 20 defaulters, 80 non-defaulters
    # Model rejects 25: 15 defaulters, 10 non-defaulters
    cm = np.array([[70, 10], [5, 15]])
    result = compute_kickout_from_cm(cm, positive_is_good=False)
    
    assert result.n_bad_rejected == 15, f"Expected 15, got {result.n_bad_rejected}"
    assert result.n_good_rejected == 10, f"Expected 10, got {result.n_good_rejected}"
    assert abs(result.kickout - 0.2) < 1e-6, f"Expected 0.2, got {result.kickout}"
    assert abs(result.bad_rejection_rate - 0.75) < 1e-6
    assert abs(result.good_rejection_rate - 0.125) < 1e-6
    
    print("✓ Sanity check passed")
    return result


if __name__ == "__main__":
    _sanity_check()