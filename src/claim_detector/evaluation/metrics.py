"""Consistent classification metrics with explicit single-class behavior."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

MetricValue: TypeAlias = float | int | None


def binary_classification_metrics(
    labels: Sequence[int] | np.ndarray,
    predictions: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray | None = None,
) -> dict[str, MetricValue]:
    y_true = np.asarray(labels, dtype=int)
    y_pred = np.asarray(predictions, dtype=int)
    if y_true.ndim != 1 or y_pred.ndim != 1 or len(y_true) != len(y_pred):
        raise ValueError("labels and predictions must be one-dimensional and equally sized")
    if len(y_true) == 0:
        raise ValueError("metrics require at least one example")
    if not np.isin(y_true, [0, 1]).all() or not np.isin(y_pred, [0, 1]).all():
        raise ValueError("labels and predictions must be binary")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics: dict[str, MetricValue] = {
        "samples": len(y_true),
        "prevalence": float(y_true.mean()),
        "positive_prediction_rate": float(y_pred.mean()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "claim_precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "claim_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "claim_f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "pr_auc": None,
        "roc_auc": None,
    }
    if scores is not None:
        y_score = np.asarray(scores, dtype=float)
        if y_score.ndim != 1 or len(y_score) != len(y_true):
            raise ValueError("scores must be one-dimensional and match labels")
        if len(np.unique(y_true)) == 2:
            metrics["pr_auc"] = float(average_precision_score(y_true, y_score))
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
    return metrics
