import numpy as np
import pytest

from claim_detector.evaluation.calibration import (
    apply_temperature,
    binary_log_loss,
    calibration_metrics,
    choose_review_threshold,
    evaluate_review_policy,
    fit_temperature,
)


def test_temperature_scaling_reduces_overconfident_log_loss() -> None:
    labels = [1, 0, 1, 0]
    probabilities = [0.99, 0.9, 0.1, 0.01]

    temperature = fit_temperature(labels, probabilities)
    calibrated = apply_temperature(probabilities, temperature)

    assert temperature > 1
    assert binary_log_loss(labels, calibrated) < binary_log_loss(labels, probabilities)
    assert (calibrated >= 0.5).tolist() == (np.asarray(probabilities) >= 0.5).tolist()


def test_calibration_metrics_have_expected_perfect_values() -> None:
    metrics = calibration_metrics([0, 1], [0.0, 1.0], bins=2)

    assert metrics["brier_score"] == pytest.approx(0, abs=1e-12)
    assert metrics["expected_calibration_error"] == pytest.approx(1e-7)


def test_review_threshold_is_selected_only_from_eligible_predictions() -> None:
    policy = choose_review_threshold(
        [0, 1, 1, 0, 1],
        [0.01, 0.99, 0.9, 0.6, 0.55],
        target_risk=0,
        minimum_records=2,
    )

    assert policy["accepted_records"] == 3
    assert policy["confidence_threshold"] == pytest.approx(0.9)
    evaluation = evaluate_review_policy([0, 1, 1], [0.01, 0.99, 0.4], 0.9)
    assert evaluation["accepted_records"] == 2
    assert evaluation["empirical_risk"] == 0


def test_probability_validation_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="between zero and one"):
        apply_temperature([1.2], 1.0)
    with pytest.raises(ValueError, match="positive"):
        apply_temperature([0.5], 0)
