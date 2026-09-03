from __future__ import annotations

import json

import pandas as pd
import pytest

from claim_detector.data.download import PROJECT_ROOT
from claim_detector.evaluation.metrics import binary_classification_metrics

pytestmark = pytest.mark.real_data


def test_saved_tfidf_metrics_recompute_from_saved_predictions() -> None:
    report_dir = PROJECT_ROOT / "reports" / "generated" / "tfidf_baseline"
    results = json.loads((report_dir / "metrics.json").read_text(encoding="utf-8"))

    for name, evaluation in results["evaluations"].items():
        prediction_metadata = evaluation.get("predictions")
        if prediction_metadata is None:
            continue
        predictions = pd.read_csv(PROJECT_ROOT / prediction_metadata["path"])
        recomputed = binary_classification_metrics(
            predictions["label"].to_numpy(),
            predictions["prediction"].to_numpy(),
            predictions["claim_probability"].to_numpy(),
        )
        for metric, value in recomputed.items():
            if value is None:
                assert evaluation["metrics"][metric] is None
            else:
                assert evaluation["metrics"][metric] == pytest.approx(value)
        assert "text" not in predictions.columns
        if name.startswith("holdout_"):
            assert evaluation["held_out_source_integrity"] is True
            assert set(evaluation["train_sources"]).isdisjoint(evaluation["test_sources"])


def test_saved_source_probe_metrics_recompute_from_predictions() -> None:
    report_dir = PROJECT_ROOT / "reports" / "generated" / "source_probe"
    results = json.loads((report_dir / "metrics.json").read_text(encoding="utf-8"))
    predictions = pd.read_csv(report_dir / "predictions.csv")

    accuracy = predictions["source"].eq(predictions["predicted_source"]).mean()
    assert len(predictions) == results["samples"] == 12_996
    assert results["text_source_probe"]["accuracy"] == pytest.approx(accuracy)
    assert "text" not in predictions.columns
