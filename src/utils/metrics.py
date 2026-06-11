"""
metrics.py — Evaluation metrics for MedAlert.
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    average_precision_score, confusion_matrix, classification_report,
)
from loguru import logger


def compute_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray,
                    threshold: float = 0.5) -> dict:
    """Compute full suite of classification metrics."""
    y_pred = (y_pred_proba >= threshold).astype(int)

    metrics = {
        "auc_roc": roc_auc_score(y_true, y_pred_proba),
        "auc_pr": average_precision_score(y_true, y_pred_proba),
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": _specificity(y_true, y_pred),
        "threshold": threshold,
    }

    logger.info("=" * 50)
    logger.info("Evaluation Results:")
    for k, v in metrics.items():
        if isinstance(v, float):
            logger.info(f"  {k:<20}: {v:.4f}")
    logger.info("=" * 50)

    return metrics


def _specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0


def find_optimal_threshold(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """Find threshold that maximizes F1 score."""
    thresholds = np.arange(0.1, 0.9, 0.01)
    f1s = [f1_score(y_true, (y_pred_proba >= t).astype(int), zero_division=0)
           for t in thresholds]
    best_threshold = thresholds[np.argmax(f1s)]
    logger.info(f"Optimal threshold: {best_threshold:.2f} (F1: {max(f1s):.4f})")
    return float(best_threshold)


def log_classification_report(y_true: np.ndarray, y_pred: np.ndarray):
    report = classification_report(y_true, y_pred,
                                   target_names=["Non-Serious", "Serious"])
    logger.info(f"\nClassification Report:\n{report}")
    return report