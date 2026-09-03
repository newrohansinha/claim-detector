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


def paired_metric_difference_interval(
    labels: Sequence[int] | np.ndarray,
    reference_predictions: Sequence[int] | np.ndarray,
    candidate_predictions: Sequence[int] | np.ndarray,
    *,
    metric: str,
    repetitions: int = 2_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    if metric not in {"macro_f1", "claim_recall"}:
        raise ValueError(f"Unsupported paired-bootstrap metric: {metric}")
    if repetitions < 2:
        raise ValueError("repetitions must be at least 2")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")

    y_true = np.asarray(labels, dtype=int)
    reference = np.asarray(reference_predictions, dtype=int)
    candidate = np.asarray(candidate_predictions, dtype=int)
    if any(values.ndim != 1 for values in (y_true, reference, candidate)):
        raise ValueError("paired-bootstrap inputs must be one-dimensional")
    if not len(y_true) or len(reference) != len(y_true) or len(candidate) != len(y_true):
        raise ValueError("paired-bootstrap inputs must be non-empty and equally sized")

    class_indices = [np.flatnonzero(y_true == label) for label in np.unique(y_true)]
    random = np.random.default_rng(seed)
    differences = []
    for _ in range(repetitions):
        indices = np.concatenate(
            [random.choice(group, size=len(group), replace=True) for group in class_indices]
        )
        reference_value = binary_classification_metrics(y_true[indices], reference[indices])[metric]
        candidate_value = binary_classification_metrics(y_true[indices], candidate[indices])[metric]
        if not isinstance(reference_value, int | float) or not isinstance(
            candidate_value, int | float
        ):
            raise TypeError(f"Paired-bootstrap metric {metric} was unexpectedly non-numeric")
        differences.append(float(candidate_value) - float(reference_value))

    alpha = 1 - confidence
    return {
        "low": float(np.quantile(differences, alpha / 2)),
        "high": float(np.quantile(differences, 1 - alpha / 2)),
    }
