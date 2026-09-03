from __future__ import annotations

import pytest

from claim_detector.evaluation.metrics import binary_classification_metrics


def test_binary_metrics_match_hand_calculation() -> None:
    metrics = binary_classification_metrics(
        labels=[0, 0, 1, 1],
        predictions=[0, 1, 0, 1],
        scores=[0.1, 0.8, 0.4, 0.9],
    )

    assert metrics["accuracy"] == 0.5
    assert metrics["claim_precision"] == 0.5
    assert metrics["claim_recall"] == 0.5
    assert metrics["claim_f1"] == 0.5
    assert metrics["macro_f1"] == 0.5
    assert metrics["true_negatives"] == 1
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 1
    assert metrics["true_positives"] == 1
    assert metrics["pr_auc"] == pytest.approx(0.8333333333)
    assert metrics["roc_auc"] == 0.75


def test_single_class_does_not_report_auc() -> None:
    metrics = binary_classification_metrics([1, 1], [1, 0], [0.8, 0.2])

    assert metrics["claim_recall"] == 0.5
    assert metrics["pr_auc"] is None
    assert metrics["roc_auc"] is None


@pytest.mark.parametrize(
    ("labels", "predictions", "message"),
    [
        ([], [], "at least one"),
        ([0, 1], [0], "equally sized"),
        ([0, 2], [0, 1], "binary"),
    ],
)
def test_invalid_metric_inputs_fail(
    labels: list[int], predictions: list[int], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        binary_classification_metrics(labels, predictions)
