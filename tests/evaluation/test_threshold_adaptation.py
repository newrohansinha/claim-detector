from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from claim_detector.evaluation.threshold_adaptation import (
    operating_metrics,
    run_adaptation_study,
    select_macro_f1_threshold,
)


def test_threshold_selection_recovers_separated_classes() -> None:
    threshold, score = select_macro_f1_threshold(
        [0, 0, 1, 1],
        [0.60, 0.70, 0.80, 0.90],
    )

    assert threshold == pytest.approx(0.75)
    assert score == pytest.approx(1.0)


def test_operating_metrics_report_both_class_recalls() -> None:
    metrics = operating_metrics([0, 0, 1, 1], [0, 1, 1, 1])

    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["claim_recall"] == pytest.approx(1.0)
    assert metrics["non_claim_recall"] == pytest.approx(0.5)
    assert metrics["positive_prediction_rate"] == pytest.approx(0.75)


def _prediction_file(path: Path, probabilities: list[float]) -> Path:
    labels = [0, 1] * (len(probabilities) // 2)
    frame = pd.DataFrame(
        {
            "source_record_id": [f"checkthat:{index}" for index in range(len(labels))],
            "source": "checkthat",
            "normalized_text_hash": [f"hash-{index}" for index in range(len(labels))],
            "label": labels,
            "claim_probability": probabilities,
        }
    )
    frame.to_csv(path, index=False)
    return path


def test_adaptation_study_is_paired_deterministic_and_disjoint(tmp_path: Path) -> None:
    records = 40
    labels = [0, 1] * (records // 2)
    first = _prediction_file(
        tmp_path / "bert.csv",
        [0.60 + 0.01 * label + 0.001 * index for index, label in enumerate(labels)],
    )
    second = _prediction_file(
        tmp_path / "tfidf.csv",
        [0.40 + 0.15 * label + 0.001 * index for index, label in enumerate(labels)],
    )
    paths = {"bert": first, "tfidf": second}

    results, trials, aligned = run_adaptation_study(
        paths,
        budgets=[10],
        repetitions=5,
        seed=7,
    )
    repeated_results, repeated_trials, _ = run_adaptation_study(
        paths,
        budgets=[10],
        repetitions=5,
        seed=7,
    )

    assert len(aligned) == records
    assert len(trials) == 10
    assert set(trials["evaluation_records"]) == {30}
    assert set(trials["adaptation_records"]) == {10}
    assert trials.groupby(["adaptation_budget", "repetition"])["model"].nunique().eq(2).all()
    pd.testing.assert_frame_equal(trials, repeated_trials)
    assert results["adaptation_results"] == repeated_results["adaptation_results"]


def test_threshold_selection_rejects_single_class_adaptation_data() -> None:
    with pytest.raises(ValueError, match="both labels"):
        select_macro_f1_threshold([1, 1], [0.7, 0.8])
