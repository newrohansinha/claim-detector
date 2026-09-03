from __future__ import annotations

import pytest

from claim_detector.evaluation.bootstrap import bootstrap_intervals


def test_bootstrap_is_deterministic_and_contains_observed_accuracy() -> None:
    labels = [0, 0, 0, 1, 1, 1]
    predictions = [0, 0, 1, 1, 1, 0]

    first = bootstrap_intervals(labels, predictions, repetitions=100, seed=7)
    second = bootstrap_intervals(labels, predictions, repetitions=100, seed=7)

    assert first == second
    assert first["accuracy"]["low"] <= 4 / 6
    assert first["accuracy"]["high"] >= 4 / 6


@pytest.mark.parametrize(
    ("repetitions", "confidence", "message"),
    [(1, 0.95, "repetitions"), (10, 1.0, "confidence")],
)
def test_bootstrap_rejects_invalid_configuration(
    repetitions: int, confidence: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        bootstrap_intervals(
            [0, 1],
            [0, 1],
            repetitions=repetitions,
            confidence=confidence,
        )
