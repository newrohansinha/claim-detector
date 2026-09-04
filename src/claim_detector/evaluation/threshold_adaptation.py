"""Measure how many labeled target examples repair an out-of-domain decision threshold."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedShuffleSplit

from claim_detector.data.download import PROJECT_ROOT, digest_file
from claim_detector.evaluation.metrics import binary_classification_metrics

DEFAULT_BERT_PREDICTIONS = (
    PROJECT_ROOT / "reports" / "generated" / "bert_calibration" / "predictions"
    / "external_checkthat.csv"
)
DEFAULT_TFIDF_PREDICTIONS = (
    PROJECT_ROOT / "reports" / "generated" / "tfidf_baseline" / "predictions"
    / "external_checkthat.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "generated" / "threshold_adaptation"
DEFAULT_BUDGETS = (10, 25, 50, 100, 200)
DEFAULT_REPETITIONS = 2_000
DEFAULT_SEED = 42
MODEL_NAMES = {"bert": "BERT", "tfidf": "TF-IDF"}
SUMMARY_METRICS = (
    "selected_threshold",
    "macro_f1",
    "fixed_threshold_macro_f1",
    "macro_f1_improvement",
    "accuracy",
    "claim_recall",
    "non_claim_recall",
    "positive_prediction_rate",
)


def _validated_arrays(
    labels: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(labels, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    if y_true.ndim != 1 or scores.ndim != 1 or len(y_true) != len(scores) or not len(y_true):
        raise ValueError("Labels and probabilities must be non-empty, aligned vectors")
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("Labels must be binary")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Probabilities must be finite values between zero and one")
    return y_true, scores


def operating_metrics(
    labels: Sequence[int] | np.ndarray,
    predictions: Sequence[int] | np.ndarray,
) -> dict[str, float | int]:
    y_true = np.asarray(labels, dtype=int)
    y_pred = np.asarray(predictions, dtype=int)
    if y_true.ndim != 1 or y_pred.ndim != 1 or len(y_true) != len(y_pred) or not len(y_true):
        raise ValueError("Labels and predictions must be non-empty, aligned vectors")
    if not np.isin(y_true, [0, 1]).all() or not np.isin(y_pred, [0, 1]).all():
        raise ValueError("Labels and predictions must be binary")

    true_positive = int(((y_true == 1) & (y_pred == 1)).sum())
    false_positive = int(((y_true == 0) & (y_pred == 1)).sum())
    false_negative = int(((y_true == 1) & (y_pred == 0)).sum())
    true_negative = int(((y_true == 0) & (y_pred == 0)).sum())

    def f1(true: int, false_pos: int, false_neg: int) -> float:
        denominator = 2 * true + false_pos + false_neg
        return 2 * true / denominator if denominator else 0.0

    return {
        "records": len(y_true),
        "accuracy": (true_positive + true_negative) / len(y_true),
        "claim_recall": true_positive / (true_positive + false_negative),
        "non_claim_recall": true_negative / (true_negative + false_positive),
        "macro_f1": (
            f1(true_positive, false_positive, false_negative)
            + f1(true_negative, false_negative, false_positive)
        )
        / 2,
        "positive_prediction_rate": float(y_pred.mean()),
        "true_negatives": true_negative,
        "false_positives": false_positive,
        "false_negatives": false_negative,
        "true_positives": true_positive,
    }


def threshold_candidates(probabilities: Sequence[float] | np.ndarray) -> np.ndarray:
    scores = np.asarray(probabilities, dtype=float)
    if scores.ndim != 1 or not len(scores):
        raise ValueError("Threshold selection requires a non-empty probability vector")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Probabilities must be finite values between zero and one")
    unique = np.unique(scores)
    midpoints = (unique[:-1] + unique[1:]) / 2
    return np.unique(np.concatenate(([0.0], midpoints, [1.0])))


def select_macro_f1_threshold(
    labels: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
) -> tuple[float, float]:
    """Select a threshold on adaptation data, with a deterministic median tie break."""
    y_true, scores = _validated_arrays(labels, probabilities)
    if len(np.unique(y_true)) != 2:
        raise ValueError("Threshold selection requires both labels")
    candidates = threshold_candidates(scores)
    predictions = scores[:, np.newaxis] >= candidates[np.newaxis, :]
    positive = y_true == 1
    negative = ~positive
    true_positive = predictions[positive].sum(axis=0)
    false_negative = positive.sum() - true_positive
    false_positive = predictions[negative].sum(axis=0)
    true_negative = negative.sum() - false_positive
    positive_denominator = 2 * true_positive + false_positive + false_negative
    negative_denominator = 2 * true_negative + false_negative + false_positive
    positive_f1 = np.divide(
        2 * true_positive,
        positive_denominator,
        out=np.zeros_like(true_positive, dtype=float),
        where=positive_denominator != 0,
    )
    negative_f1 = np.divide(
        2 * true_negative,
        negative_denominator,
        out=np.zeros_like(true_negative, dtype=float),
        where=negative_denominator != 0,
    )
    values = (positive_f1 + negative_f1) / 2
    best_indices = np.flatnonzero(np.isclose(values, values.max(), rtol=0, atol=1e-15))
    selected_index = int(best_indices[len(best_indices) // 2])
    return float(candidates[selected_index]), float(values[selected_index])


def _align_predictions(paths: Mapping[str, Path]) -> pd.DataFrame:
    required = {
        "source_record_id",
        "source",
        "normalized_text_hash",
        "label",
        "claim_probability",
    }
    aligned: pd.DataFrame | None = None
    identity = ["source_record_id", "source", "normalized_text_hash", "label"]
    for model, path in paths.items():
        frame = pd.read_csv(path)
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{model} predictions are missing columns: {sorted(missing)}")
        if frame["source_record_id"].duplicated().any():
            raise ValueError(f"{model} predictions contain duplicate record identifiers")
        selected = frame[[*identity, "claim_probability"]].rename(
            columns={"claim_probability": f"{model}_probability"}
        )
        if aligned is None:
            aligned = selected
        else:
            aligned = aligned.merge(selected, on=identity, how="inner", validate="one_to_one")

    if aligned is None:
        raise ValueError("At least one prediction artifact is required")
    expected_rows = {
        len(pd.read_csv(path, usecols=["source_record_id"])) for path in paths.values()
    }
    if len(expected_rows) != 1 or len(aligned) != expected_rows.pop():
        raise ValueError("Prediction artifacts are not aligned on the same labeled records")
    if set(aligned["source"]) != {"checkthat"}:
        raise ValueError("Threshold adaptation is restricted to CheckThat target records")
    return aligned.sort_values("source_record_id").reset_index(drop=True)


def _summary(values: pd.Series) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "median": float(values.median()),
        "central_95_percent_low": float(values.quantile(0.025)),
        "central_95_percent_high": float(values.quantile(0.975)),
    }


def _report_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def run_adaptation_study(
    prediction_paths: Mapping[str, Path],
    *,
    budgets: Sequence[int] = DEFAULT_BUDGETS,
    repetitions: int = DEFAULT_REPETITIONS,
    seed: int = DEFAULT_SEED,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    if repetitions < 2:
        raise ValueError("Repetitions must be at least two")
    if not budgets or len(set(budgets)) != len(budgets):
        raise ValueError("Adaptation budgets must be non-empty and unique")

    aligned = _align_predictions(prediction_paths)
    labels = aligned["label"].to_numpy(dtype=int)
    if len(np.unique(labels)) != 2:
        raise ValueError("Target data must contain both labels")
    if any(budget < 2 or budget >= len(labels) for budget in budgets):
        raise ValueError("Each adaptation budget must be at least two and smaller than the target")

    full_evaluations: dict[str, Any] = {}
    for model in prediction_paths:
        scores = aligned[f"{model}_probability"].to_numpy(dtype=float)
        fixed = binary_classification_metrics(labels, scores >= 0.5, scores)
        oracle_threshold, oracle_adaptation_score = select_macro_f1_threshold(labels, scores)
        oracle = binary_classification_metrics(labels, scores >= oracle_threshold, scores)
        full_evaluations[model] = {
            "display_name": MODEL_NAMES.get(model, model),
            "fixed_threshold": 0.5,
            "fixed_threshold_metrics": fixed,
            "oracle_diagnostic": {
                "interpretation": (
                    "Upper-bound diagnostic selected and evaluated on all target labels; "
                    "not a deployable performance estimate."
                ),
                "threshold": oracle_threshold,
                "adaptation_macro_f1": oracle_adaptation_score,
                "metrics": oracle,
            },
        }

    records: list[dict[str, float | int | str]] = []
    for budget in sorted(budgets):
        splitter = StratifiedShuffleSplit(
            n_splits=repetitions,
            train_size=budget,
            random_state=seed + budget,
        )
        for repetition, (adaptation_indices, evaluation_indices) in enumerate(
            splitter.split(np.zeros(len(labels)), labels)
        ):
            adaptation_labels = labels[adaptation_indices]
            evaluation_labels = labels[evaluation_indices]
            for model in prediction_paths:
                scores = aligned[f"{model}_probability"].to_numpy(dtype=float)
                threshold, adaptation_macro_f1 = select_macro_f1_threshold(
                    adaptation_labels, scores[adaptation_indices]
                )
                adapted_metrics = operating_metrics(
                    evaluation_labels, scores[evaluation_indices] >= threshold
                )
                fixed_metrics = operating_metrics(
                    evaluation_labels, scores[evaluation_indices] >= 0.5
                )
                records.append(
                    {
                        "model": model,
                        "adaptation_budget": budget,
                        "repetition": repetition,
                        "adaptation_records": len(adaptation_indices),
                        "adaptation_claims": int(adaptation_labels.sum()),
                        "adaptation_non_claims": int(
                            len(adaptation_labels) - adaptation_labels.sum()
                        ),
                        "evaluation_records": len(evaluation_indices),
                        "selected_threshold": threshold,
                        "adaptation_macro_f1": adaptation_macro_f1,
                        "macro_f1": float(adapted_metrics["macro_f1"]),
                        "fixed_threshold_macro_f1": float(fixed_metrics["macro_f1"]),
                        "macro_f1_improvement": float(
                            adapted_metrics["macro_f1"] - fixed_metrics["macro_f1"]
                        ),
                        "accuracy": float(adapted_metrics["accuracy"]),
                        "claim_recall": float(adapted_metrics["claim_recall"]),
                        "non_claim_recall": float(adapted_metrics["non_claim_recall"]),
                        "positive_prediction_rate": float(
                            adapted_metrics["positive_prediction_rate"]
                        ),
                    }
                )

    trials = pd.DataFrame.from_records(records)
    budget_results: dict[str, Any] = {}
    for model in prediction_paths:
        model_results: dict[str, Any] = {}
        for budget in sorted(budgets):
            subset = trials[
                trials["model"].eq(model) & trials["adaptation_budget"].eq(budget)
            ]
            model_results[str(budget)] = {
                "adaptation_claims_per_split": int(subset["adaptation_claims"].iloc[0]),
                "adaptation_non_claims_per_split": int(subset["adaptation_non_claims"].iloc[0]),
                "evaluation_records_per_split": int(subset["evaluation_records"].iloc[0]),
                "metrics_across_repetitions": {
                    metric: _summary(subset[metric]) for metric in SUMMARY_METRICS
                },
            }
        budget_results[model] = model_results

    inputs = {}
    for model, path in prediction_paths.items():
        size, sha256 = digest_file(path)
        inputs[model] = {
            "path": _report_path(path),
            "bytes": size,
            "sha256": sha256,
            "probability": (
                "in-domain temperature-scaled" if model == "bert" else "uncalibrated"
            ),
        }

    results = {
        "schema_version": 1,
        "experiment": "labeled_target_threshold_adaptation",
        "research_question": (
            "Under domain shift, did class separation fail, or did the learned operating "
            "threshold stop transferring?"
        ),
        "analysis_status": (
            "Post-hoc exploratory follow-up motivated by the frozen CheckThat evaluation; "
            "not a preregistered or untouched-target estimate."
        ),
        "target": "CheckThat English COVID-19 tweets",
        "target_records": len(aligned),
        "target_claims": int(labels.sum()),
        "target_non_claims": int(len(labels) - labels.sum()),
        "protocol": {
            "adaptation_budgets": sorted(budgets),
            "repetitions_per_budget": repetitions,
            "split_method": "repeated stratified random holdout",
            "selection_metric": "macro_f1",
            "evaluation_rule": (
                "Select a threshold only on the labeled adaptation subset, then score it on "
                "the disjoint remainder. Model weights never change."
            ),
            "tie_break": "median threshold among exact macro-F1 maximizers",
            "summary_interval": (
                "2.5th to 97.5th percentile across repeated adaptation/evaluation splits; "
                "not a population confidence interval"
            ),
            "seed": seed,
        },
        "inputs": inputs,
        "full_target_diagnostics": full_evaluations,
        "adaptation_results": budget_results,
    }
    return results, trials, aligned


def write_report(
    results: dict[str, Any],
    trials: pd.DataFrame,
    aligned: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    trials_path = output_dir / "trials.csv.gz"
    trials.to_csv(
        trials_path,
        index=False,
        compression={"method": "gzip", "compresslevel": 9, "mtime": 0},
    )
    trial_bytes, trial_sha256 = digest_file(trials_path)
    results["trials"] = {
        "path": str(trials_path.relative_to(PROJECT_ROOT)),
        "bytes": trial_bytes,
        "sha256": trial_sha256,
        "contains_sentence_text": False,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(results, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    colors = {"bert": "#176B87", "tfidf": "#C06C3E"}
    labels = aligned["label"].to_numpy(dtype=int)
    for model in MODEL_NAMES:
        scores = aligned[f"{model}_probability"].to_numpy(dtype=float)
        false_positive_rate, true_positive_rate, _ = roc_curve(labels, scores)
        auc = results["full_target_diagnostics"][model]["fixed_threshold_metrics"]["roc_auc"]
        axes[0].plot(
            false_positive_rate,
            true_positive_rate,
            color=colors[model],
            linewidth=2.3,
            label=f"{MODEL_NAMES[model]} (AUC {auc:.3f})",
        )
    axes[0].plot([0, 1], [0, 1], color="#777777", linestyle="--", linewidth=1)
    axes[0].set(
        xlabel="False-positive rate",
        ylabel="True-positive rate",
        title="A. Ranking remains useful on CheckThat",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axes[0].legend(frameon=False, loc="lower right")

    budgets = sorted(trials["adaptation_budget"].unique())
    for model in MODEL_NAMES:
        medians = []
        lows = []
        highs = []
        for budget in budgets:
            values = trials.loc[
                trials["model"].eq(model) & trials["adaptation_budget"].eq(budget),
                "macro_f1",
            ]
            medians.append(float(values.median()))
            lows.append(float(values.quantile(0.025)))
            highs.append(float(values.quantile(0.975)))
        fixed = results["full_target_diagnostics"][model]["fixed_threshold_metrics"]["macro_f1"]
        x_values = np.asarray([0, *budgets])
        y_values = np.asarray([fixed, *medians])
        axes[1].plot(
            x_values,
            y_values,
            marker="o",
            linewidth=2.3,
            color=colors[model],
            label=MODEL_NAMES[model],
        )
        axes[1].fill_between(
            budgets,
            lows,
            highs,
            color=colors[model],
            alpha=0.15,
            linewidth=0,
        )
    axes[1].set(
        xlabel="Labeled CheckThat examples used to set the threshold",
        ylabel="Macro F1 on disjoint remainder",
        title="B. A small target sample repairs the threshold",
        ylim=(0.35, 0.76),
    )
    axes[1].set_xticks([0, *budgets])
    axes[1].legend(frameon=False, loc="lower right")

    for axis in axes:
        axis.grid(alpha=0.2)
    figure.suptitle("BERT retains useful separation, but its 50% cutoff fails on CheckThat")
    figure.tight_layout()
    figure.savefig(output_dir / "threshold_adaptation.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bert-predictions", type=Path, default=DEFAULT_BERT_PREDICTIONS)
    parser.add_argument("--tfidf-predictions", type=Path, default=DEFAULT_TFIDF_PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    paths = {"bert": args.bert_predictions, "tfidf": args.tfidf_predictions}
    results, trials, aligned = run_adaptation_study(
        paths,
        repetitions=args.repetitions,
        seed=args.seed,
    )
    write_report(results, trials, aligned, args.output_dir)
    print(json.dumps(results, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
