"""Fit and evaluate temperature scaling on reserved BERT predictions."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit

from claim_detector.data.download import PROJECT_ROOT, digest_file

DEFAULT_BERT_REPORT_DIR = PROJECT_ROOT / "reports" / "generated" / "bert_mixed"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "generated" / "bert_calibration"
EPSILON = 1e-7
CALIBRATION_BINS = 15
TARGET_REVIEW_RISK = 0.05
MINIMUM_ACCEPTED_RECORDS = 100


def _arrays(
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
    return y_true, np.clip(scores, EPSILON, 1 - EPSILON)


def binary_log_loss(
    labels: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
) -> float:
    y_true, scores = _arrays(labels, probabilities)
    return float(-np.mean(y_true * np.log(scores) + (1 - y_true) * np.log1p(-scores)))


def apply_temperature(
    probabilities: Sequence[float] | np.ndarray, temperature: float
) -> np.ndarray:
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be positive and finite")
    _, scores = _arrays(np.zeros(len(probabilities), dtype=int), probabilities)
    logits = np.log(scores) - np.log1p(-scores)
    return cast(np.ndarray, expit(logits / temperature))


def fit_temperature(
    labels: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
) -> float:
    y_true, scores = _arrays(labels, probabilities)
    logits = np.log(scores) - np.log1p(-scores)

    def objective(log_temperature: float) -> float:
        calibrated = expit(logits / np.exp(log_temperature))
        return binary_log_loss(y_true, calibrated)

    result = minimize_scalar(
        objective,
        bounds=(np.log(0.05), np.log(20.0)),
        method="bounded",
        options={"xatol": 1e-10},
    )
    if not result.success:
        raise RuntimeError(f"Temperature optimization failed: {result.message}")
    return float(np.exp(result.x))


def calibration_metrics(
    labels: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    *,
    bins: int = CALIBRATION_BINS,
) -> dict[str, Any]:
    if bins < 2:
        raise ValueError("Calibration requires at least two bins")
    y_true, scores = _arrays(labels, probabilities)
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.minimum(np.digitize(scores, boundaries[1:-1]), bins - 1)
    records = []
    weighted_gap = 0.0
    maximum_gap = 0.0
    for index in range(bins):
        mask = assignments == index
        count = int(mask.sum())
        mean_probability = float(scores[mask].mean()) if count else None
        observed_rate = float(y_true[mask].mean()) if count else None
        gap = abs(cast(float, mean_probability) - cast(float, observed_rate)) if count else None
        if gap is not None:
            weighted_gap += count / len(scores) * gap
            maximum_gap = max(maximum_gap, gap)
        records.append(
            {
                "lower": float(boundaries[index]),
                "upper": float(boundaries[index + 1]),
                "records": count,
                "mean_claim_probability": mean_probability,
                "observed_claim_rate": observed_rate,
                "absolute_gap": gap,
            }
        )
    return {
        "records": len(scores),
        "negative_log_likelihood": binary_log_loss(y_true, scores),
        "brier_score": float(np.mean((scores - y_true) ** 2)),
        "expected_calibration_error": weighted_gap,
        "maximum_calibration_error": maximum_gap,
        "bins": records,
    }


def prediction_confidence(probabilities: Sequence[float] | np.ndarray) -> np.ndarray:
    scores = np.asarray(probabilities, dtype=float)
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Probabilities must be finite values between zero and one")
    return cast(np.ndarray, np.maximum(scores, 1 - scores))


def choose_review_threshold(
    labels: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    *,
    target_risk: float = TARGET_REVIEW_RISK,
    minimum_records: int = MINIMUM_ACCEPTED_RECORDS,
) -> dict[str, float | int]:
    if not 0 <= target_risk < 1:
        raise ValueError("Target risk must be in [0, 1)")
    y_true, scores = _arrays(labels, probabilities)
    if minimum_records < 1 or minimum_records > len(y_true):
        raise ValueError("Minimum records must fit within the calibration sample")
    predictions = (scores >= 0.5).astype(int)
    confidence = prediction_confidence(scores)
    best: dict[str, float | int] | None = None
    for threshold in np.unique(confidence):
        accepted = confidence >= threshold
        accepted_records = int(accepted.sum())
        if accepted_records < minimum_records:
            continue
        risk = float(np.mean(predictions[accepted] != y_true[accepted]))
        if risk <= target_risk and (best is None or accepted_records > best["accepted_records"]):
            best = {
                "confidence_threshold": float(threshold),
                "accepted_records": accepted_records,
                "coverage": accepted_records / len(y_true),
                "empirical_risk": risk,
                "target_risk": target_risk,
                "minimum_records": minimum_records,
            }
    if best is None:
        raise ValueError("No confidence threshold meets the requested empirical risk")
    return best


def evaluate_review_policy(
    labels: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    threshold: float,
) -> dict[str, float | int | None]:
    y_true, scores = _arrays(labels, probabilities)
    confidence = prediction_confidence(scores)
    accepted = confidence >= threshold
    accepted_records = int(accepted.sum())
    if not accepted_records:
        risk = None
    else:
        risk = float(np.mean((scores[accepted] >= 0.5).astype(int) != y_true[accepted]))
    return {
        "records": len(y_true),
        "accepted_records": accepted_records,
        "reviewed_records": int((~accepted).sum()),
        "coverage": accepted_records / len(y_true),
        "empirical_risk": risk,
        "confidence_threshold": threshold,
    }


def _model_sha(metrics: Mapping[str, Any]) -> str:
    for item in metrics["artifact"]["files"]:
        if Path(item["path"]).name == "model.safetensors":
            return str(item["sha256"])
    raise ValueError("Mixed BERT metrics do not contain a model.safetensors digest")


def calibrate_predictions(bert_report_dir: Path, output_dir: Path) -> dict[str, Any]:
    prediction_dir = bert_report_dir / "predictions"
    paths = {
        "calibration": prediction_dir / "calibration.csv",
        "mixed_paper_test": prediction_dir / "mixed_paper_test.csv",
        "external_checkthat": prediction_dir / "external_checkthat.csv",
    }
    frames = {name: pd.read_csv(path) for name, path in paths.items()}
    fit_frame = frames["calibration"]
    temperature = fit_temperature(
        fit_frame["label"].to_numpy(), fit_frame["claim_probability"].to_numpy()
    )

    output_prediction_dir = output_dir / "predictions"
    output_prediction_dir.mkdir(parents=True, exist_ok=True)
    evaluations: dict[str, Any] = {}
    calibrated_frames: dict[str, pd.DataFrame] = {}
    for name, frame in frames.items():
        calibrated = frame.copy()
        calibrated["raw_claim_probability"] = calibrated["claim_probability"]
        calibrated["claim_probability"] = apply_temperature(
            calibrated["raw_claim_probability"].to_numpy(), temperature
        )
        calibrated["confidence"] = prediction_confidence(calibrated["claim_probability"].to_numpy())
        calibrated["prediction"] = (calibrated["claim_probability"] >= 0.5).astype(int)
        calibrated_frames[name] = calibrated
        path = output_prediction_dir / f"{name}.csv"
        calibrated.to_csv(path, index=False)
        size, sha = digest_file(path)
        evaluations[name] = {
            "raw": calibration_metrics(
                frame["label"].to_numpy(), frame["claim_probability"].to_numpy()
            ),
            "temperature_scaled": calibration_metrics(
                calibrated["label"].to_numpy(), calibrated["claim_probability"].to_numpy()
            ),
            "predictions": {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "bytes": size,
                "sha256": sha,
            },
        }

    review_policy = choose_review_threshold(
        calibrated_frames["calibration"]["label"].to_numpy(),
        calibrated_frames["calibration"]["claim_probability"].to_numpy(),
    )
    threshold = float(review_policy["confidence_threshold"])
    for name, frame in calibrated_frames.items():
        evaluations[name]["review_policy"] = evaluate_review_policy(
            frame["label"].to_numpy(), frame["claim_probability"].to_numpy(), threshold
        )

    source_metrics = json.loads((bert_report_dir / "metrics.json").read_text(encoding="utf-8"))
    model_sha = _model_sha(source_metrics)
    calibration_source_bytes, calibration_source_sha = digest_file(paths["calibration"])
    calibrator = {
        "schema_version": 1,
        "method": "temperature_scaling",
        "temperature": temperature,
        "fit_split": "calibration",
        "fit_records": len(fit_frame),
        "calibration_bins": CALIBRATION_BINS,
        "model_sha256": model_sha,
        "source_predictions": {
            "path": str(paths["calibration"].relative_to(PROJECT_ROOT)),
            "bytes": calibration_source_bytes,
            "sha256": calibration_source_sha,
        },
        "review_policy": review_policy,
        "review_policy_interpretation": (
            "Threshold selected for <=5% empirical error on the reserved calibration split; "
            "it is evaluated, not assumed, on frozen test and external data."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "calibrator.json").write_text(
        json.dumps(calibrator, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {
        "schema_version": 1,
        "calibrator": calibrator,
        "evaluations": evaluations,
    }


def write_report(results: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(results, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    names = {
        "calibration": "Calibration split",
        "mixed_paper_test": "Frozen mixed test",
        "external_checkthat": "CheckThat transfer",
    }
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True, sharey=True)
    for axis, (name, title) in zip(axes, names.items(), strict=True):
        evaluation = results["evaluations"][name]
        for condition, label, marker in (
            ("raw", "Raw", "o"),
            ("temperature_scaled", "Temperature-scaled", "s"),
        ):
            populated = [row for row in evaluation[condition]["bins"] if row["records"]]
            axis.plot(
                [row["mean_claim_probability"] for row in populated],
                [row["observed_claim_rate"] for row in populated],
                marker=marker,
                label=label,
            )
        axis.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
        axis.set_title(title)
        axis.set_xlabel("Mean predicted claim probability")
    axes[0].set_ylabel("Observed claim rate")
    axes[-1].legend(frameon=False)
    figure.suptitle("Confidence calibration does not guarantee transfer calibration")
    figure.tight_layout()
    figure.savefig(output_dir / "reliability_diagram.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bert-report-dir", type=Path, default=DEFAULT_BERT_REPORT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    results = calibrate_predictions(args.bert_report_dir, args.output_dir)
    write_report(results, args.output_dir)
    print(json.dumps(results, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
