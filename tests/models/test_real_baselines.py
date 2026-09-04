from __future__ import annotations

import json

import pandas as pd
import pytest

from claim_detector.data.download import PROJECT_ROOT, digest_file
from claim_detector.evaluation.metrics import binary_classification_metrics
from claim_detector.evaluation.threshold_adaptation import run_adaptation_study

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


def test_saved_bert_metrics_recompute_from_predictions() -> None:
    report_dir = PROJECT_ROOT / "reports" / "generated" / "bert_mixed"
    results = json.loads((report_dir / "metrics.json").read_text(encoding="utf-8"))

    assert results["configuration"]["model_revision"] == (
        "86b5e0934494bd15c9632b12f734a8a67f723594"
    )
    assert results["training"]["selected_epoch"] == 2
    for name in ("validation", "calibration", "mixed_paper_test", "external_checkthat"):
        prediction_path = PROJECT_ROOT / results["evaluations"][name]["predictions"]["path"]
        predictions = pd.read_csv(prediction_path)
        recomputed = binary_classification_metrics(
            predictions["label"].to_numpy(),
            predictions["prediction"].to_numpy(),
            predictions["claim_probability"].to_numpy(),
        )
        for metric, value in recomputed.items():
            if value is None:
                assert results["evaluations"][name]["metrics"][metric] is None
            else:
                assert results["evaluations"][name]["metrics"][metric] == pytest.approx(value)
        assert "text" not in predictions.columns


def test_saved_bert_source_holdouts_are_traceable_and_leakage_free() -> None:
    report_dir = PROJECT_ROOT / "reports" / "generated" / "bert_heldout"
    results = json.loads((report_dir / "metrics.json").read_text(encoding="utf-8"))

    for source, run in results["runs"].items():
        evaluation = run["evaluation"]
        prediction_path = PROJECT_ROOT / evaluation["predictions"]["path"]
        predictions = pd.read_csv(prediction_path)
        size, sha256 = digest_file(prediction_path)
        assert size == evaluation["predictions"]["bytes"]
        assert sha256 == evaluation["predictions"]["sha256"]
        assert evaluation["held_out_source_integrity"] is True
        assert source not in run["training"]["fit_sources"]
        assert source not in run["training"]["validation_sources"]
        assert set(predictions["source"]) == {source}
        assert "text" not in predictions.columns

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

    comparison = results["frozen_test_comparison"]
    assert comparison["evaluation_split"] == "paper_test"
    for source, source_comparison in comparison["sources"].items():
        assert source in results["runs"]
        assert source_comparison["samples"] > 0
        assert source_comparison["absolute_change"] < 0
        assert source_comparison["absolute_change_confidence_interval_95"]["high"] < 0


def test_saved_threshold_adaptation_is_traceable_and_recomputable() -> None:
    report_dir = PROJECT_ROOT / "reports" / "generated" / "threshold_adaptation"
    results = json.loads((report_dir / "metrics.json").read_text(encoding="utf-8"))
    trials_path = PROJECT_ROOT / results["trials"]["path"]
    trial_bytes, trial_sha256 = digest_file(trials_path)
    trials = pd.read_csv(trials_path)

    assert trial_bytes == results["trials"]["bytes"]
    assert trial_sha256 == results["trials"]["sha256"]
    assert results["analysis_status"].startswith("Post-hoc exploratory")
    assert len(trials) == 2 * 5 * 2_000
    assert "text" not in trials.columns

    prediction_paths = {}
    for model, metadata in results["inputs"].items():
        prediction_path = PROJECT_ROOT / metadata["path"]
        prediction_paths[model] = prediction_path
        size, sha256 = digest_file(prediction_path)
        predictions = pd.read_csv(prediction_path)
        recomputed = binary_classification_metrics(
            predictions["label"].to_numpy(),
            predictions["claim_probability"].ge(0.5).astype(int).to_numpy(),
            predictions["claim_probability"].to_numpy(),
        )
        assert size == metadata["bytes"]
        assert sha256 == metadata["sha256"]
        for metric, value in recomputed.items():
            reported = results["full_target_diagnostics"][model]["fixed_threshold_metrics"][
                metric
            ]
            if value is None:
                assert reported is None
            else:
                assert reported == pytest.approx(value)

        for budget, budget_results in results["adaptation_results"][model].items():
            subset = trials[
                trials["model"].eq(model)
                & trials["adaptation_budget"].eq(int(budget))
            ]
            reported_macro = budget_results["metrics_across_repetitions"]["macro_f1"]
            assert len(subset) == 2_000
            assert reported_macro["median"] == pytest.approx(subset["macro_f1"].median())
            assert reported_macro["central_95_percent_low"] == pytest.approx(
                subset["macro_f1"].quantile(0.025)
            )
            assert reported_macro["central_95_percent_high"] == pytest.approx(
                subset["macro_f1"].quantile(0.975)
            )

    regenerated, regenerated_trials, _ = run_adaptation_study(prediction_paths)
    pd.testing.assert_frame_equal(
        trials,
        regenerated_trials,
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )
    assert regenerated["adaptation_results"] == results["adaptation_results"]
