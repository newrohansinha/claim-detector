"""Shared training loop for controlled BERT fine-tuning experiments."""

from __future__ import annotations

import gc
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd
import torch
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification

from claim_detector.evaluation.metrics import binary_classification_metrics
from claim_detector.models.bert import (
    MODEL_CONFIG,
    MODEL_ID,
    MODEL_REVISION,
    _artifact_manifest,
    build_loader,
    encode_frame,
    linear_warmup_decay,
    predict_loader,
    prediction_frame,
    set_reproducible_seed,
    train_epoch,
)


def _source_label_counts(frame: pd.DataFrame) -> dict[str, int]:
    keys = frame["source"].astype(str) + "|" + frame["label"].astype(str)
    return {str(key): int(value) for key, value in keys.value_counts().sort_index().items()}


def clear_device_cache(device: torch.device) -> None:
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def train_selected_classifier(
    fit: pd.DataFrame,
    validation: pd.DataFrame,
    evaluation: pd.DataFrame,
    *,
    tokenizer: Any,
    artifact_dir: Path,
    device: torch.device,
    log_context: dict[str, str],
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, dict[str, int | float]], dict[str, Any]]:
    """Train, select on validation macro F1, then score one untouched evaluation frame."""
    if fit["label"].nunique() != 2 or validation["label"].nunique() != 2:
        raise ValueError("Fit and validation frames must both contain two labels")
    if set(fit["normalized_text_hash"]) & set(validation["normalized_text_hash"]):
        raise ValueError("Duplicate text hashes cross fit and validation")

    seed = cast(int, MODEL_CONFIG["seed"])
    max_length = cast(int, MODEL_CONFIG["max_length"])
    set_reproducible_seed(seed)
    frames = {"fit": fit, "validation": validation, "evaluation": evaluation}
    datasets = {}
    tokenization = {}
    for name, frame in frames.items():
        datasets[name], tokenization[name] = encode_frame(frame, tokenizer, max_length=max_length)

    fit_loader = build_loader(
        datasets["fit"],
        batch_size=cast(int, MODEL_CONFIG["train_batch_size"]),
        shuffle=True,
        seed=seed,
    )
    validation_loader = build_loader(
        datasets["validation"],
        batch_size=cast(int, MODEL_CONFIG["evaluation_batch_size"]),
        shuffle=False,
        seed=seed,
    )
    evaluation_loader = build_loader(
        datasets["evaluation"],
        batch_size=cast(int, MODEL_CONFIG["evaluation_batch_size"]),
        shuffle=False,
        seed=seed,
    )

    print(json.dumps({**log_context, "status": "training"}, sort_keys=True), flush=True)
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
        metrics = binary_classification_metrics(
            validation["label"].to_numpy(),
            (probabilities >= 0.5).astype(int),
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
        print(json.dumps({**log_context, **record}, sort_keys=True), flush=True)
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            selected_epoch = epoch
            model.save_pretrained(artifact_dir, safe_serialization=True)
            tokenizer.save_pretrained(artifact_dir)

    del model, optimizer, scheduler
    clear_device_cache(device)
    selected_model = AutoModelForSequenceClassification.from_pretrained(artifact_dir).to(device)
    probabilities = predict_loader(selected_model, evaluation_loader, device)
    predictions = prediction_frame(evaluation, probabilities)

    training = {
        "device": str(device),
        "fit_records": len(fit),
        "validation_records": len(validation),
        "fit_sources": sorted(map(str, fit["source"].unique())),
        "validation_sources": sorted(map(str, validation["source"].unique())),
        "fit_source_label_counts": _source_label_counts(fit),
        "validation_source_label_counts": _source_label_counts(validation),
        "fit_label_counts": {
            str(key): int(value) for key, value in fit["label"].value_counts().sort_index().items()
        },
        "validation_label_counts": {
            str(key): int(value)
            for key, value in validation["label"].value_counts().sort_index().items()
        },
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
    artifact = _artifact_manifest(artifact_dir)
    del selected_model
    clear_device_cache(device)
    return predictions, training, tokenization, artifact
