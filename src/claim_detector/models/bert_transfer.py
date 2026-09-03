"""Train fresh BERT classifiers with each composite source held out in turn."""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from claim_detector.data.download import PROJECT_ROOT, digest_file
from claim_detector.data.prepare import DEFAULT_PROCESSED_DIR
from claim_detector.evaluation.bootstrap import evaluated_binary_predictions
from claim_detector.evaluation.metrics import binary_classification_metrics
from claim_detector.models.bert import (
    MODEL_CONFIG,
    MODEL_ID,
    MODEL_REVISION,
    DeviceName,
    _artifact_manifest,
    _git_state,
    build_loader,
    encode_frame,
    linear_warmup_decay,
    predict_loader,
    prediction_frame,
    resolve_device,
    set_reproducible_seed,
    train_epoch,
    validate_development_splits,
)

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "generated" / "bert_heldout"
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "bert_heldout"
MIXED_ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "bert_mixed"
TFIDF_METRICS_PATH = PROJECT_ROOT / "reports" / "generated" / "tfidf_baseline" / "metrics.json"
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


def _label_counts(frame: pd.DataFrame) -> dict[str, int]:
    keys = frame["source"].astype(str) + "|" + frame["label"].astype(str)
    return {str(key): int(count) for key, count in keys.value_counts().sort_index().items()}


def _score_summary(probabilities: pd.Series) -> dict[str, float]:
    return {
        "minimum": float(probabilities.min()),
        "p05": float(probabilities.quantile(0.05)),
        "median": float(probabilities.median()),
        "p95": float(probabilities.quantile(0.95)),
        "maximum": float(probabilities.max()),
    }


def _clear_device_cache(device: torch.device) -> None:
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def train_heldout_source(
    composite: pd.DataFrame,
    held_out_source: str,
    tokenizer: Any,
    output_dir: Path,
    artifact_root: Path,
    device: torch.device,
) -> dict[str, Any]:
    seed = cast(int, MODEL_CONFIG["seed"])
    max_length = cast(int, MODEL_CONFIG["max_length"])
    set_reproducible_seed(seed)
    fit, validation, test = source_holdout_frames(composite, held_out_source)
    fit_dataset, fit_tokenization = encode_frame(fit, tokenizer, max_length=max_length)
    validation_dataset, validation_tokenization = encode_frame(
        validation,
        tokenizer,
        max_length=max_length,
    )
    test_dataset, test_tokenization = encode_frame(test, tokenizer, max_length=max_length)
    fit_loader = build_loader(
        fit_dataset,
        batch_size=cast(int, MODEL_CONFIG["train_batch_size"]),
        shuffle=True,
        seed=seed,
    )
    validation_loader = build_loader(
        validation_dataset,
        batch_size=cast(int, MODEL_CONFIG["evaluation_batch_size"]),
        shuffle=False,
        seed=seed,
    )
    test_loader = build_loader(
        test_dataset,
        batch_size=cast(int, MODEL_CONFIG["evaluation_batch_size"]),
        shuffle=False,
        seed=seed,
    )

    print(json.dumps({"held_out_source": held_out_source, "status": "training"}), flush=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        num_labels=2,
        id2label={0: "not_claim", 1: "claim"},
        label2id={"not_claim": 0, "claim": 1},
    ).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=cast(float, MODEL_CONFIG["learning_rate"]),
        weight_decay=cast(float, MODEL_CONFIG["weight_decay"]),
    )
    total_steps = len(fit_loader) * cast(int, MODEL_CONFIG["epochs"])
    warmup_steps = int(total_steps * cast(float, MODEL_CONFIG["warmup_ratio"]))
    scheduler = linear_warmup_decay(
        optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
    )

    artifact_dir = artifact_root / held_out_source
    artifact_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, int | float]] = []
    best_macro_f1 = -1.0
    selected_epoch = 0
    started_at = datetime.now(UTC)
    for epoch in range(1, cast(int, MODEL_CONFIG["epochs"]) + 1):
        epoch_started = time.perf_counter()
        train_loss = train_epoch(
            model,
            fit_loader,
            optimizer,
            scheduler,
            device,
            max_gradient_norm=cast(float, MODEL_CONFIG["max_gradient_norm"]),
        )
        probabilities = predict_loader(model, validation_loader, device)
        validation_predictions = (probabilities >= 0.5).astype(int)
        metrics = binary_classification_metrics(
            validation["label"].to_numpy(),
            validation_predictions,
            probabilities,
        )
        epoch_seconds = time.perf_counter() - epoch_started
        macro_f1 = float(cast(float, metrics["macro_f1"]))
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "epoch_seconds": epoch_seconds,
            "training_records_per_second": len(fit) / epoch_seconds,
            "validation_accuracy": float(cast(float, metrics["accuracy"])),
            "validation_claim_f1": float(cast(float, metrics["claim_f1"])),
            "validation_macro_f1": macro_f1,
        }
        history.append(record)
        print(
            json.dumps({"held_out_source": held_out_source, **record}, sort_keys=True), flush=True
        )
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            selected_epoch = epoch
            model.save_pretrained(artifact_dir, safe_serialization=True)
            tokenizer.save_pretrained(artifact_dir)

    del model, optimizer, scheduler
    _clear_device_cache(device)
    selected_model = AutoModelForSequenceClassification.from_pretrained(artifact_dir).to(device)
    test_probabilities = predict_loader(selected_model, test_loader, device)
    test_predictions = prediction_frame(test, test_probabilities)
    prediction_path = output_dir / "predictions" / f"holdout_{held_out_source}.csv"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    test_predictions.to_csv(prediction_path, index=False)
    prediction_bytes, prediction_sha = digest_file(prediction_path)

    training = {
        "device": str(device),
        "fit_records": len(fit),
        "validation_records": len(validation),
        "fit_sources": sorted(map(str, fit["source"].unique())),
        "validation_sources": sorted(map(str, validation["source"].unique())),
        "source_label_counts": _label_counts(pd.concat([fit, validation])),
        "total_optimizer_steps": total_steps,
        "warmup_steps": warmup_steps,
        "selected_epoch": selected_epoch,
        "selection_metric": "validation_macro_f1",
        "history": history,
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
    }
    (artifact_dir / "training_metadata.json").write_text(
        json.dumps(training, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
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
    artifact = _artifact_manifest(artifact_dir)
    del selected_model
    _clear_device_cache(device)
    return {
        "held_out_source": held_out_source,
        "training": training,
        "tokenization": {
            "fit": fit_tokenization,
            "validation": validation_tokenization,
            "test": test_tokenization,
        },
        "artifact": artifact,
        "evaluation": evaluation,
    }


def evaluate_mixed_model_by_source(
    composite: pd.DataFrame,
    tokenizer: Any,
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    if not MIXED_ARTIFACT_DIR.exists():
        raise FileNotFoundError("Train the mixed BERT model before running transfer comparison")
    dataset, tokenization = encode_frame(
        composite.reset_index(drop=True),
        tokenizer,
        max_length=cast(int, MODEL_CONFIG["max_length"]),
    )
    loader = build_loader(
        dataset,
        batch_size=cast(int, MODEL_CONFIG["evaluation_batch_size"]),
        shuffle=False,
        seed=cast(int, MODEL_CONFIG["seed"]),
    )
    model = AutoModelForSequenceClassification.from_pretrained(MIXED_ARTIFACT_DIR).to(device)
    probabilities = predict_loader(model, loader, device)
    predictions = prediction_frame(composite.reset_index(drop=True), probabilities)
    path = output_dir / "predictions" / "mixed_model_all_sources.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(path, index=False)
    prediction_bytes, prediction_sha = digest_file(path)
    source_evaluations = {
        str(source): {
            "score_summary": _score_summary(group["claim_probability"]),
            **evaluated_binary_predictions(
                group["label"].to_numpy(),
                group["prediction"].to_numpy(),
                group["claim_probability"].to_numpy(),
            ),
        }
        for source, group in predictions.groupby("source")
    }
    del model
    _clear_device_cache(device)
    return {
        "model_artifact": str(MIXED_ARTIFACT_DIR.relative_to(PROJECT_ROOT)),
        "records": len(composite),
        "tokenization": tokenization,
        "predictions": {
            "path": str(path.relative_to(PROJECT_ROOT)),
            "bytes": prediction_bytes,
            "sha256": prediction_sha,
        },
        "source_evaluations": source_evaluations,
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
    mixed_reference = evaluate_mixed_model_by_source(composite, tokenizer, output_dir, device)
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
        "mixed_reference": mixed_reference,
        "runs": runs,
    }


def _result_metric(results: dict[str, Any], source: str, metric: str) -> float:
    value = results["runs"][source]["evaluation"]["metrics"][metric]
    if not isinstance(value, int | float):
        raise TypeError(f"Expected numeric metric for {source}.{metric}")
    return float(value)


def write_report(results: dict[str, Any], output_dir: Path) -> None:
    display = {
        "claimbuster": ("ClaimBuster — macro F1", "macro_f1"),
        "policlaim": ("PoliClaim — macro F1", "macro_f1"),
        "averitec": ("AVeriTeC — claim recall", "claim_recall"),
    }
    results["transfer_comparison"] = {
        source: {
            "metric": metric,
            "mixed_model": float(
                results["mixed_reference"]["source_evaluations"][source]["metrics"][metric]
            ),
            "source_heldout_model": _result_metric(results, source, metric),
            "absolute_change": _result_metric(results, source, metric)
            - float(results["mixed_reference"]["source_evaluations"][source]["metrics"][metric]),
        }
        for source, (_, metric) in display.items()
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(results, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    tfidf = json.loads(TFIDF_METRICS_PATH.read_text(encoding="utf-8"))
    rows = []
    for source, (label, metric) in display.items():
        rows.append(
            {
                "evaluation": label,
                "condition": "BERT · source included",
                "score": float(
                    results["mixed_reference"]["source_evaluations"][source]["metrics"][metric]
                ),
            }
        )
        rows.append(
            {
                "evaluation": label,
                "condition": "BERT · source held out",
                "score": _result_metric(results, source, metric),
            }
        )
        rows.append(
            {
                "evaluation": label,
                "condition": "TF-IDF · source held out",
                "score": float(tfidf["evaluations"][f"holdout_{source}"]["metrics"][metric]),
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
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, title=None)
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
        device = resolve_device(cast(DeviceName, args.device))
        tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
            MODEL_ID,
            revision=MODEL_REVISION,
            use_fast=True,
        )
        results["mixed_reference"] = evaluate_mixed_model_by_source(
            composite,
            tokenizer,
            args.output_dir,
            device,
        )
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
