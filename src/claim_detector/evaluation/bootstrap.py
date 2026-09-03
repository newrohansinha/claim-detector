"""Deterministic stratified bootstrap confidence intervals."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from claim_detector.evaluation.metrics import MetricValue, binary_classification_metrics

DEFAULT_INTERVAL_METRICS = (
    "accuracy",
    "claim_precision",
    "claim_recall",
    "claim_f1",
    "macro_f1",
)


def bootstrap_intervals(
    labels: Sequence[int] | np.ndarray,
    predictions: Sequence[int] | np.ndarray,
    *,
    repetitions: int = 2_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    if repetitions < 2:
        raise ValueError("repetitions must be at least 2")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")

    y_true = np.asarray(labels, dtype=int)
    y_pred = np.asarray(predictions, dtype=int)
    if y_true.ndim != 1 or y_pred.ndim != 1 or len(y_true) != len(y_pred):
        raise ValueError("labels and predictions must be one-dimensional and equally sized")
    if len(y_true) == 0:
        raise ValueError("bootstrap requires at least one example")

    class_indices = [np.flatnonzero(y_true == label) for label in np.unique(y_true)]
    random = np.random.default_rng(seed)
    sampled: dict[str, list[float]] = {name: [] for name in DEFAULT_INTERVAL_METRICS}
    for _ in range(repetitions):
        indices = np.concatenate(
            [random.choice(group, size=len(group), replace=True) for group in class_indices]
        )
        metrics = binary_classification_metrics(y_true[indices], y_pred[indices])
        for name in DEFAULT_INTERVAL_METRICS:
            value: MetricValue = metrics[name]
            if not isinstance(value, int | float):
                raise TypeError(f"Bootstrap metric {name} was unexpectedly non-numeric")
            sampled[name].append(float(value))

    alpha = 1 - confidence
    return {
        name: {
            "low": float(np.quantile(values, alpha / 2)),
            "high": float(np.quantile(values, 1 - alpha / 2)),
        }
        for name, values in sampled.items()
    }


def evaluated_binary_predictions(
    labels: Sequence[int] | np.ndarray,
    predictions: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    *,
    bootstrap_repetitions: int = 2_000,
) -> dict[str, object]:
    return {
        "metrics": binary_classification_metrics(labels, predictions, scores),
        "confidence_intervals_95": bootstrap_intervals(
            labels,
            predictions,
            repetitions=bootstrap_repetitions,
        ),
        "bootstrap_repetitions": bootstrap_repetitions,
    }
