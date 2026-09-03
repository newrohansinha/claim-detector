"""Train fresh BERT classifiers with each composite source held out in turn."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
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
from claim_detector.models.runtime import MODEL_ID, MODEL_REVISION, DeviceName, resolve_device

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "generated" / "bert_heldout"
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "bert_heldout"
BERT_MIXED_PREDICTIONS = (
    PROJECT_ROOT / "reports" / "generated" / "bert_mixed" / "predictions" / "mixed_paper_test.csv"
)
HELD_OUT_SOURCES = ("claimbuster", "policlaim", "averitec")


def source_holdout_frames(
    composite: pd.DataFrame,
    held_out_source: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if held_out_source not in HELD_OUT_SOURCES:
        raise ValueError(f"Unknown held-out source: {held_out_source}")
    remaining = composite[composite["source"].ne(held_out_source)]
    fit = remaining[remaining["development_split"].eq("fit")].reset_index(drop=True)
    validation = remaining[remaining["development_split"].eq("validation")].reset_index(drop=True)
    test = composite[composite["source"].eq(held_out_source)].reset_index(drop=True)
    if held_out_source in set(fit["source"]) | set(validation["source"]):
        raise ValueError(f"Held-out source leaked into model selection: {held_out_source}")
    if set(test["source"]) != {held_out_source}:
        raise ValueError(f"Evaluation set is not isolated to {held_out_source}")
    if fit["label"].nunique() != 2 or validation["label"].nunique() != 2:
        raise ValueError(f"Training and validation must remain binary for {held_out_source}")
    return fit, validation, test


def _score_summary(probabilities: pd.Series) -> dict[str, float]:
    return {
        "minimum": float(probabilities.min()),
        "p05": float(probabilities.quantile(0.05)),
        "median": float(probabilities.median()),
        "p95": float(probabilities.quantile(0.95)),
        "maximum": float(probabilities.max()),
    }


def train_heldout_source(
    composite: pd.DataFrame,
    held_out_source: str,
    tokenizer: Any,
    output_dir: Path,
    artifact_root: Path,
    device: Any,
) -> dict[str, Any]:
    fit, validation, test = source_holdout_frames(composite, held_out_source)
    artifact_dir = artifact_root / held_out_source
    test_predictions, training, tokenization, artifact = train_selected_classifier(
        fit,
        validation,
        test,
        tokenizer=tokenizer,
        artifact_dir=artifact_dir,
        device=device,
        log_context={"held_out_source": held_out_source},
    )
    prediction_path = output_dir / "predictions" / f"holdout_{held_out_source}.csv"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    test_predictions.to_csv(prediction_path, index=False)
    prediction_bytes, prediction_sha = digest_file(prediction_path)

    evaluation = {
        "test_records": len(test),
        "test_sources": sorted(map(str, test["source"].unique())),
        "held_out_source_integrity": held_out_source not in set(fit["source"])
        and held_out_source not in set(validation["source"]),
        "score_summary": _score_summary(test_predictions["claim_probability"]),
        "predictions": {
            "path": str(prediction_path.relative_to(PROJECT_ROOT)),
            "bytes": prediction_bytes,
            "sha256": prediction_sha,
        },
        **evaluated_binary_predictions(
            test_predictions["label"].to_numpy(),
            test_predictions["prediction"].to_numpy(),
            test_predictions["claim_probability"].to_numpy(),
        ),
    }
    return {
        "held_out_source": held_out_source,
        "training": training,
        "tokenization": {
            "fit": tokenization["fit"],
            "validation": tokenization["validation"],
            "test": tokenization["evaluation"],
        },
        "artifact": artifact,
        "evaluation": evaluation,
    }


def frozen_test_comparison(
    composite: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Any]:
    if not BERT_MIXED_PREDICTIONS.exists():
        raise FileNotFoundError("Train the mixed BERT model before comparing transfer")
    paper_test = composite[composite["paper_split"].eq("test")]
    mixed_predictions = pd.read_csv(BERT_MIXED_PREDICTIONS)
    metric_by_source = {
        "claimbuster": "macro_f1",
        "policlaim": "macro_f1",
        "averitec": "claim_recall",
    }
    comparison: dict[str, Any] = {}
    for source, metric in metric_by_source.items():
        test_ids = set(paper_test.loc[paper_test["source"].eq(source), "source_record_id"])
        included = mixed_predictions[mixed_predictions["source"].eq(source)].set_index(
            "source_record_id"
        )
        heldout_path = output_dir / "predictions" / f"holdout_{source}.csv"
        heldout = pd.read_csv(heldout_path)
        heldout = heldout[heldout["source_record_id"].isin(test_ids)].set_index("source_record_id")
        ordered_ids = sorted(test_ids)
        if set(included.index) != test_ids or set(heldout.index) != test_ids:
            raise ValueError(f"Frozen test records do not align for {source}")
        included = included.loc[ordered_ids]
        heldout = heldout.loc[ordered_ids]
        if not included["label"].equals(heldout["label"]):
            raise ValueError(f"Frozen test labels do not align for {source}")

        included_evaluation = evaluated_binary_predictions(
            included["label"].to_numpy(),
            included["prediction"].to_numpy(),
            included["claim_probability"].to_numpy(),
        )
        heldout_evaluation = evaluated_binary_predictions(
            heldout["label"].to_numpy(),
            heldout["prediction"].to_numpy(),
            heldout["claim_probability"].to_numpy(),
        )
        included_value = float(cast(dict[str, Any], included_evaluation["metrics"])[metric])
        heldout_value = float(cast(dict[str, Any], heldout_evaluation["metrics"])[metric])
        comparison[source] = {
            "samples": len(ordered_ids),
            "metric": metric,
            "source_included": included_evaluation,
            "source_heldout": heldout_evaluation,
            "absolute_change": heldout_value - included_value,
            "absolute_change_confidence_interval_95": paired_metric_difference_interval(
                included["label"].to_numpy(),
                included["prediction"].to_numpy(),
                heldout["prediction"].to_numpy(),
                metric=metric,
            ),
        }
    return {
        "evaluation_split": "paper_test",
        "record_alignment": "source_record_id",
        "sources": comparison,
    }


def train_source_holdouts(
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
    runs = {
        source: train_heldout_source(
            composite,
            source,
            tokenizer,
            output_dir,
            artifact_dir,
            device,
        )
        for source in HELD_OUT_SOURCES
    }
    comparison = frozen_test_comparison(composite, output_dir)
    dataset_path = processed_dir / "dataset_manifest.json"
    dataset_bytes, dataset_sha = digest_file(dataset_path)
    return {
        "schema_version": 1,
        "experiment": "bert_leave_one_source_out",
        "configuration": MODEL_CONFIG,
        "git": _git_state(),
        "dataset_manifest": {
            "path": str(dataset_path.relative_to(PROJECT_ROOT)),
            "bytes": dataset_bytes,
            "sha256": dataset_sha,
        },
        "frozen_test_comparison": comparison,
        "runs": runs,
    }


def write_report(results: dict[str, Any], output_dir: Path) -> None:
    display = {
        "claimbuster": ("ClaimBuster — macro F1", "macro_f1"),
        "policlaim": ("PoliClaim — macro F1", "macro_f1"),
        "averitec": ("AVeriTeC — claim recall", "claim_recall"),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(results, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    rows = []
    for source, (label, _) in display.items():
        comparison = results["frozen_test_comparison"]["sources"][source]
        metric = comparison["metric"]
        rows.append(
            {
                "evaluation": label,
                "condition": "BERT · source included",
                "score": float(comparison["source_included"]["metrics"][metric]),
            }
        )
        rows.append(
            {
                "evaluation": label,
                "condition": "BERT · source held out",
                "score": float(comparison["source_heldout"]["metrics"][metric]),
            }
        )

    sns.set_theme(style="whitegrid", context="talk")
    figure, axis = plt.subplots(figsize=(11, 6.5))
    sns.barplot(data=pd.DataFrame(rows), x="score", y="evaluation", hue="condition", ax=axis)
    axis.set(
        xlim=(0, 1),
        xlabel="Score",
        ylabel="",
        title="Source-held-out transfer gap",
    )
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, title=None)
    figure.savefig(
        output_dir / "source_heldout_comparison.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument(
        "--comparison-only",
        action="store_true",
        help="Reuse trained checkpoints and regenerate the matched-source comparison",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.comparison_only:
        results = json.loads((args.output_dir / "metrics.json").read_text(encoding="utf-8"))
        composite = pd.read_csv(args.processed_dir / "composite.csv")
        results.pop("mixed_reference", None)
        results.pop("transfer_comparison", None)
        results["frozen_test_comparison"] = frozen_test_comparison(composite, args.output_dir)
    else:
        results = train_source_holdouts(
            args.processed_dir,
            args.output_dir,
            args.artifact_dir,
            requested_device=cast(DeviceName, args.device),
        )
    write_report(results, args.output_dir)
    print(json.dumps(results, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
