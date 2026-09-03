"""Train size- and class-prior-matched controls for the source-holdout experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from transformers import AutoTokenizer

from claim_detector.data.download import PROJECT_ROOT, digest_file
from claim_detector.data.prepare import DEFAULT_PROCESSED_DIR
from claim_detector.evaluation.bootstrap import (
    evaluated_binary_predictions,
    paired_metric_difference_interval,
)
from claim_detector.models.bert import (
    MODEL_CONFIG,
    _git_state,
    validate_development_splits,
)
from claim_detector.models.bert_training import train_selected_classifier
from claim_detector.models.bert_transfer import HELD_OUT_SOURCES, source_holdout_frames
from claim_detector.models.runtime import MODEL_ID, MODEL_REVISION, DeviceName, resolve_device

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "generated" / "bert_matched_control"
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "bert_matched_control"
HELDOUT_PREDICTION_DIR = PROJECT_ROOT / "reports" / "generated" / "bert_heldout" / "predictions"
METRIC_BY_SOURCE = {
    "claimbuster": "macro_f1",
    "policlaim": "macro_f1",
    "averitec": "claim_recall",
}


def _counts(frame: pd.DataFrame) -> dict[int, int]:
    return {int(cast(int, key)): int(value) for key, value in frame["label"].value_counts().items()}


def grouped_label_sample(
    frame: pd.DataFrame,
    target_counts: Mapping[int, int],
    *,
    seed: int,
) -> pd.DataFrame:
    """Sample whole normalized-text groups while matching each requested label count exactly."""
    random = np.random.default_rng(seed)
    selected_hashes: set[str] = set()
    for label, target in sorted(target_counts.items()):
        candidates = frame[frame["label"].eq(label)]
        groups = candidates.groupby("normalized_text_hash", sort=True).size()
        if int(groups.sum()) < target:
            raise ValueError(
                f"Only {int(groups.sum())} rows available for label {label}; need {target}"
            )
        ordered_hashes = groups.index.to_numpy(copy=True)
        random.shuffle(ordered_hashes)
        remaining = target
        for normalized_hash in ordered_hashes:
            group_size = int(groups.loc[normalized_hash])
            if group_size <= remaining:
                selected_hashes.add(str(normalized_hash))
                remaining -= group_size
            if remaining == 0:
                break
        if remaining:
            raise ValueError(
                f"Could not preserve groups and match label {label}; short by {remaining}"
            )

    sample = frame[frame["normalized_text_hash"].isin(selected_hashes)].copy()
    sample = sample.sort_index().reset_index(drop=True)
    if _counts(sample) != dict(target_counts):
        raise AssertionError(
            f"Matched sample has {_counts(sample)}, expected {dict(target_counts)}"
        )
    return sample


def matched_control_frames(
    composite: pd.DataFrame,
    source: str,
    *,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build the matched condition without exposing any frozen-test duplicate text."""
    heldout_fit, heldout_validation, _ = source_holdout_frames(composite, source)
    evaluation = composite[
        composite["paper_split"].eq("test") & composite["source"].eq(source)
    ].reset_index(drop=True)
    evaluation_hashes = set(evaluation["normalized_text_hash"])

    development = composite[composite["paper_split"].eq("train")]
    candidate_fit = development[
        development["development_split"].eq("fit")
        & ~development["normalized_text_hash"].isin(evaluation_hashes)
    ]
    candidate_validation = development[
        development["development_split"].eq("validation")
        & ~development["normalized_text_hash"].isin(evaluation_hashes)
    ]
    fit = grouped_label_sample(candidate_fit, _counts(heldout_fit), seed=seed)
    validation = grouped_label_sample(
        candidate_validation,
        _counts(heldout_validation),
        seed=seed + 1,
    )

    if source not in set(fit["source"]) or source not in set(validation["source"]):
        raise ValueError(f"Matched control failed to include {source} in model selection")
    if set(fit["normalized_text_hash"]) & evaluation_hashes:
        raise ValueError(f"Frozen-test text leaked into the {source} matched fit sample")
    if set(validation["normalized_text_hash"]) & evaluation_hashes:
        raise ValueError(f"Frozen-test text leaked into the {source} matched validation sample")

    selected_ids = sorted([*fit["source_record_id"].astype(str), *validation["source_record_id"]])
    selection_sha = hashlib.sha256("\n".join(selected_ids).encode()).hexdigest()
    matching = {
        "seed": seed,
        "sampling_unit": "normalized_text_hash",
        "fit_records": {"heldout": len(heldout_fit), "matched": len(fit)},
        "validation_records": {
            "heldout": len(heldout_validation),
            "matched": len(validation),
        },
        "fit_label_counts": {
            "heldout": _counts(heldout_fit),
            "matched": _counts(fit),
        },
        "validation_label_counts": {
            "heldout": _counts(heldout_validation),
            "matched": _counts(validation),
        },
        "target_source_fit_records": int(fit["source"].eq(source).sum()),
        "target_source_validation_records": int(validation["source"].eq(source).sum()),
        "excluded_frozen_test_overlap_candidates": int(
            development["normalized_text_hash"].isin(evaluation_hashes).sum()
        ),
        "selected_record_ids_sha256": selection_sha,
    }
    return fit, validation, evaluation, matching


def _aligned_heldout_predictions(source: str, control: pd.DataFrame) -> pd.DataFrame:
    heldout_path = HELDOUT_PREDICTION_DIR / f"holdout_{source}.csv"
    if not heldout_path.exists():
        raise FileNotFoundError("Run the source-held-out experiment before its matched control")
    heldout = pd.read_csv(heldout_path).set_index("source_record_id")
    ordered_ids = control["source_record_id"].tolist()
    if not set(ordered_ids).issubset(heldout.index):
        raise ValueError(f"Held-out predictions do not cover the {source} frozen test set")
    return heldout.loc[ordered_ids].reset_index()


def train_matched_controls(
    processed_dir: Path,
    output_dir: Path,
    artifact_dir: Path,
    *,
    requested_device: DeviceName = "auto",
) -> dict[str, Any]:
    composite = pd.read_csv(processed_dir / "composite.csv")
    validate_development_splits(composite)
    device = resolve_device(requested_device)
    tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
        MODEL_ID,
        revision=MODEL_REVISION,
        use_fast=True,
    )
    seed = cast(int, MODEL_CONFIG["seed"])
    runs: dict[str, Any] = {}
    for source in HELD_OUT_SOURCES:
        fit, validation, evaluation, matching = matched_control_frames(
            composite,
            source,
            seed=seed,
        )
        predictions, training, tokenization, artifact = train_selected_classifier(
            fit,
            validation,
            evaluation,
            tokenizer=tokenizer,
            artifact_dir=artifact_dir / source,
            device=device,
            log_context={"control_for_source": source},
        )
        prediction_path = output_dir / "predictions" / f"matched_{source}.csv"
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(prediction_path, index=False)
        prediction_bytes, prediction_sha = digest_file(prediction_path)

        heldout = _aligned_heldout_predictions(source, predictions)
        if not predictions["label"].equals(heldout["label"]):
            raise ValueError(f"Matched and held-out labels do not align for {source}")
        metric = METRIC_BY_SOURCE[source]
        control_evaluation = evaluated_binary_predictions(
            predictions["label"].to_numpy(),
            predictions["prediction"].to_numpy(),
            predictions["claim_probability"].to_numpy(),
        )
        heldout_evaluation = evaluated_binary_predictions(
            heldout["label"].to_numpy(),
            heldout["prediction"].to_numpy(),
            heldout["claim_probability"].to_numpy(),
        )
        control_value = float(cast(dict[str, Any], control_evaluation["metrics"])[metric])
        heldout_value = float(cast(dict[str, Any], heldout_evaluation["metrics"])[metric])
        runs[source] = {
            "metric": metric,
            "matching": matching,
            "training": training,
            "tokenization": tokenization,
            "artifact": artifact,
            "predictions": {
                "path": str(prediction_path.relative_to(PROJECT_ROOT)),
                "bytes": prediction_bytes,
                "sha256": prediction_sha,
            },
            "matched_source_included": control_evaluation,
            "source_heldout": heldout_evaluation,
            "absolute_change": heldout_value - control_value,
            "absolute_change_confidence_interval_95": paired_metric_difference_interval(
                predictions["label"].to_numpy(),
                predictions["prediction"].to_numpy(),
                heldout["prediction"].to_numpy(),
                metric=metric,
            ),
        }

    dataset_path = processed_dir / "dataset_manifest.json"
    dataset_bytes, dataset_sha = digest_file(dataset_path)
    return {
        "schema_version": 1,
        "experiment": "bert_source_exposure_size_and_class_prior_matched",
        "control_definition": (
            "For each target source, match the held-out fit and validation row counts and label "
            "counts exactly, preserve normalized-text groups, restore target-source exposure, "
            "and exclude text hashes present in that source's frozen test set."
        ),
        "configuration": MODEL_CONFIG,
        "git": _git_state(),
        "dataset_manifest": {
            "path": str(dataset_path.relative_to(PROJECT_ROOT)),
            "bytes": dataset_bytes,
            "sha256": dataset_sha,
        },
        "runs": runs,
    }


def write_report(results: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(results, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    labels = {
        "claimbuster": "ClaimBuster — macro F1",
        "policlaim": "PoliClaim — macro F1",
        "averitec": "AVeriTeC — claim recall",
    }
    rows = []
    for source, run in results["runs"].items():
        metric = run["metric"]
        rows.extend(
            [
                {
                    "evaluation": labels[source],
                    "condition": "Matched · source included",
                    "score": run["matched_source_included"]["metrics"][metric],
                },
                {
                    "evaluation": labels[source],
                    "condition": "Matched · source held out",
                    "score": run["source_heldout"]["metrics"][metric],
                },
            ]
        )
    sns.set_theme(style="whitegrid", context="talk")
    figure, axis = plt.subplots(figsize=(11, 6.5))
    sns.barplot(data=pd.DataFrame(rows), x="score", y="evaluation", hue="condition", ax=axis)
    axis.set(
        xlim=(0, 1),
        xlabel="Score",
        ylabel="",
        title="Source effect after matching size and class prior",
    )
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, title=None)
    figure.savefig(output_dir / "matched_control_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    results = train_matched_controls(
        args.processed_dir,
        args.output_dir,
        args.artifact_dir,
        requested_device=cast(DeviceName, args.device),
    )
    write_report(results, args.output_dir)
    print(json.dumps(results, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
