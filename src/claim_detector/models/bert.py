"""Fine-tune and evaluate the pre-registered BERT claim detector."""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import random
import subprocess
import time
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import transformers
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from claim_detector.data.download import PROJECT_ROOT, digest_file
from claim_detector.data.prepare import DEFAULT_PROCESSED_DIR
from claim_detector.evaluation.bootstrap import evaluated_binary_predictions
from claim_detector.evaluation.metrics import binary_classification_metrics
from claim_detector.models.runtime import MODEL_ID, MODEL_REVISION, DeviceName, resolve_device

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "generated" / "bert_mixed"
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "bert_mixed"
TFIDF_METRICS_PATH = PROJECT_ROOT / "reports" / "generated" / "tfidf_baseline" / "metrics.json"

MODEL_CONFIG: dict[str, Any] = {
    "model_id": MODEL_ID,
    "model_revision": MODEL_REVISION,
    "max_length": 128,
    "epochs": 3,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "train_batch_size": 16,
    "evaluation_batch_size": 32,
    "max_gradient_norm": 1.0,
    "seed": 42,
}


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def validate_development_splits(composite: pd.DataFrame) -> None:
    train = composite[composite["paper_split"].eq("train")]
    expected = {"fit", "validation", "calibration"}
    actual = set(train["development_split"].astype(str).unique())
    if actual != expected:
        raise ValueError(f"Expected development splits {sorted(expected)}, got {sorted(actual)}")

    hash_sets = {
        split: set(train.loc[train["development_split"].eq(split), "normalized_text_hash"])
        for split in expected
    }
    for left, right in (
        ("fit", "validation"),
        ("fit", "calibration"),
        ("validation", "calibration"),
    ):
        overlap = hash_sets[left] & hash_sets[right]
        if overlap:
            raise ValueError(f"Duplicate text hashes cross {left}/{right}: {len(overlap)}")


def tokenizer_length_statistics(lengths: Sequence[int], max_length: int) -> dict[str, int | float]:
    values = np.asarray(lengths, dtype=int)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("Tokenizer lengths must be a non-empty one-dimensional sequence")
    return {
        "records": len(values),
        "minimum": int(values.min()),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "maximum": int(values.max()),
        "truncated_records": int((values > max_length).sum()),
        "truncated_fraction": float((values > max_length).mean()),
    }


def encode_frame(
    frame: pd.DataFrame,
    tokenizer: Any,
    *,
    max_length: int,
) -> tuple[TensorDataset, dict[str, int | float]]:
    texts = frame["text"].astype(str).tolist()
    untruncated = tokenizer(
        texts,
        add_special_tokens=True,
        padding=False,
        truncation=False,
        return_length=True,
    )
    length_stats = tokenizer_length_statistics(untruncated["length"], max_length)
    encoded = tokenizer(
        texts,
        add_special_tokens=True,
        max_length=max_length,
        padding="max_length",
        return_attention_mask=True,
        return_tensors="pt",
        truncation=True,
    )
    labels = torch.as_tensor(frame["label"].to_numpy(dtype=np.int64), dtype=torch.long)
    dataset = TensorDataset(encoded["input_ids"], encoded["attention_mask"], labels)
    return dataset, length_stats


def build_loader(
    dataset: TensorDataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader[tuple[torch.Tensor, ...]]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
    )


def linear_warmup_decay(
    optimizer: AdamW,
    *,
    total_steps: int,
    warmup_steps: int,
) -> LambdaLR:
    if total_steps <= 0 or not 0 <= warmup_steps < total_steps:
        raise ValueError("Warmup and total steps must satisfy 0 <= warmup < total")

    def multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        remaining = total_steps - step
        decay_steps = total_steps - warmup_steps
        return max(0.0, float(remaining) / float(decay_steps))

    return LambdaLR(optimizer, multiplier)


def train_epoch(
    model: Any,
    loader: DataLoader[tuple[torch.Tensor, ...]],
    optimizer: AdamW,
    scheduler: LambdaLR,
    device: torch.device,
    *,
    max_gradient_norm: float,
) -> float:
    model.train()
    total_loss = 0.0
    examples = 0
    for step, (input_ids, attention_mask, labels) in enumerate(loader, start=1):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = output.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
        optimizer.step()
        scheduler.step()
        batch_size = int(labels.shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size
        examples += batch_size
        if step % 100 == 0 or step == len(loader):
            print(
                json.dumps(
                    {
                        "training_batch": step,
                        "training_batches": len(loader),
                        "mean_loss": total_loss / examples,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    return total_loss / examples


def predict_loader(
    model: Any,
    loader: DataLoader[tuple[torch.Tensor, ...]],
    device: torch.device,
) -> np.ndarray:
    model.eval()
    probabilities: list[np.ndarray] = []
    with torch.inference_mode():
        for input_ids, attention_mask, _ in loader:
            output = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
            )
            batch_probabilities = torch.softmax(output.logits.float(), dim=1)[:, 1]
            probabilities.append(batch_probabilities.cpu().numpy())
    return np.concatenate(probabilities)


def prediction_frame(frame: pd.DataFrame, probabilities: np.ndarray) -> pd.DataFrame:
    if len(frame) != len(probabilities):
        raise ValueError("A probability is required for every record")
    result = frame[["source_record_id", "source", "normalized_text_hash", "label"]].copy()
    result["claim_probability"] = probabilities
    result["prediction"] = (probabilities >= 0.5).astype(int)
    return result


def _git_state() -> dict[str, str | bool | None]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"revision": revision, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"revision": None, "dirty": None}


def _artifact_manifest(artifact_dir: Path) -> dict[str, Any]:
    files: list[dict[str, int | str]] = []
    for path in sorted(item for item in artifact_dir.rglob("*") if item.is_file()):
        size, sha256 = digest_file(path)
        files.append(
            {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "bytes": size,
                "sha256": sha256,
            }
        )
    return {"path": str(artifact_dir.relative_to(PROJECT_ROOT)), "files": files}


def _save_predictions(
    name: str,
    predictions: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Any]:
    path = output_dir / "predictions" / f"{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(path, index=False)
    size, sha256 = digest_file(path)
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "bytes": size,
        "sha256": sha256,
    }


def _evaluation(predictions: pd.DataFrame) -> dict[str, object]:
    return evaluated_binary_predictions(
        predictions["label"].to_numpy(),
        predictions["prediction"].to_numpy(),
        predictions["claim_probability"].to_numpy(),
    )


def _batches(
    datasets: dict[str, TensorDataset],
    *,
    batch_size: int,
    seed: int,
) -> Iterator[tuple[str, DataLoader[tuple[torch.Tensor, ...]]]]:
    for name, dataset in datasets.items():
        yield name, build_loader(dataset, batch_size=batch_size, shuffle=False, seed=seed)


def train_mixed_model(
    processed_dir: Path,
    output_dir: Path,
    artifact_dir: Path,
    *,
    requested_device: DeviceName = "auto",
) -> dict[str, Any]:
    started_at = datetime.now(UTC)
    seed = cast(int, MODEL_CONFIG["seed"])
    max_length = cast(int, MODEL_CONFIG["max_length"])
    set_reproducible_seed(seed)
    device = resolve_device(requested_device)

    composite = pd.read_csv(processed_dir / "composite.csv")
    checkthat = pd.read_csv(processed_dir / "checkthat.csv")
    validate_development_splits(composite)
    frames = {
        "fit": composite[composite["development_split"].eq("fit")].reset_index(drop=True),
        "validation": composite[composite["development_split"].eq("validation")].reset_index(
            drop=True
        ),
        "calibration": composite[composite["development_split"].eq("calibration")].reset_index(
            drop=True
        ),
        "mixed_paper_test": composite[composite["paper_split"].eq("test")].reset_index(drop=True),
        "external_checkthat": checkthat.reset_index(drop=True),
    }

    tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
        MODEL_ID,
        revision=MODEL_REVISION,
        use_fast=True,
    )
    datasets: dict[str, TensorDataset] = {}
    tokenization: dict[str, dict[str, int | float]] = {}
    for name, frame in frames.items():
        datasets[name], tokenization[name] = encode_frame(
            frame,
            tokenizer,
            max_length=max_length,
        )

    train_loader = build_loader(
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
    total_steps = len(train_loader) * cast(int, MODEL_CONFIG["epochs"])
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
    for epoch in range(1, cast(int, MODEL_CONFIG["epochs"]) + 1):
        epoch_started = time.perf_counter()
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            max_gradient_norm=cast(float, MODEL_CONFIG["max_gradient_norm"]),
        )
        validation_probabilities = predict_loader(model, validation_loader, device)
        validation_predictions = (validation_probabilities >= 0.5).astype(int)
        validation_metrics = binary_classification_metrics(
            frames["validation"]["label"].to_numpy(),
            validation_predictions,
            validation_probabilities,
        )
        macro_f1 = float(cast(float, validation_metrics["macro_f1"]))
        epoch_seconds = time.perf_counter() - epoch_started
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "epoch_seconds": epoch_seconds,
                "training_records_per_second": len(frames["fit"]) / epoch_seconds,
                "validation_accuracy": float(cast(float, validation_metrics["accuracy"])),
                "validation_claim_f1": float(cast(float, validation_metrics["claim_f1"])),
                "validation_macro_f1": macro_f1,
            }
        )
        print(json.dumps(history[-1], sort_keys=True), flush=True)
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            selected_epoch = epoch
            model.save_pretrained(artifact_dir, safe_serialization=True)
            tokenizer.save_pretrained(artifact_dir)

    del optimizer, scheduler, model
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
    model = AutoModelForSequenceClassification.from_pretrained(artifact_dir).to(device)
    evaluations: dict[str, Any] = {}
    for name, loader in _batches(
        {key: value for key, value in datasets.items() if key != "fit"},
        batch_size=cast(int, MODEL_CONFIG["evaluation_batch_size"]),
        seed=seed,
    ):
        probabilities = predict_loader(model, loader, device)
        predictions = prediction_frame(frames[name], probabilities)
        prediction_metadata = _save_predictions(name, predictions, output_dir)
        evaluations[name] = {
            "predictions": prediction_metadata,
            **_evaluation(predictions),
        }

    test_predictions = pd.read_csv(
        PROJECT_ROOT / evaluations["mixed_paper_test"]["predictions"]["path"]
    )
    paper_train_hashes = set(
        composite.loc[composite["paper_split"].eq("train"), "normalized_text_hash"]
    )
    clean_mask = ~test_predictions["normalized_text_hash"].isin(paper_train_hashes)
    clean_predictions = test_predictions[clean_mask].reset_index(drop=True)
    evaluations["mixed_paper_test_without_train_duplicates"] = {
        "removed_records": int((~clean_mask).sum()),
        **_evaluation(clean_predictions),
    }

    training_metadata = {
        "device": str(device),
        "fit_records": len(frames["fit"]),
        "validation_records": len(frames["validation"]),
        "calibration_records_reserved": len(frames["calibration"]),
        "total_optimizer_steps": total_steps,
        "warmup_steps": warmup_steps,
        "selected_epoch": selected_epoch,
        "selection_metric": "validation_macro_f1",
        "history": history,
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
    }
    (artifact_dir / "training_metadata.json").write_text(
        json.dumps(training_metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    dataset_manifest_path = processed_dir / "dataset_manifest.json"
    dataset_bytes, dataset_sha = digest_file(dataset_manifest_path)
    results = {
        "schema_version": 1,
        "model": "bert_base_uncased_mixed",
        "decision_threshold": 0.5,
        "configuration": MODEL_CONFIG,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "git": _git_state(),
        "dataset_manifest": {
            "path": str(dataset_manifest_path.relative_to(PROJECT_ROOT)),
            "bytes": dataset_bytes,
            "sha256": dataset_sha,
        },
        "training": training_metadata,
        "tokenization": tokenization,
        "artifact": _artifact_manifest(artifact_dir),
        "evaluations": evaluations,
    }
    return results


def _metric(results: dict[str, Any], evaluation: str, name: str) -> float:
    value = results["evaluations"][evaluation]["metrics"][name]
    if not isinstance(value, int | float):
        raise TypeError(f"Expected numeric metric {evaluation}.{name}")
    return float(value)


def write_report(results: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(results, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    rows = []
    for evaluation in ("mixed_paper_test", "external_checkthat"):
        rows.extend(
            [
                {
                    "evaluation": {
                        "mixed_paper_test": "Mixed test",
                        "external_checkthat": "CheckThat transfer",
                    }[evaluation],
                    "model": "BERT",
                    "metric": {"claim_f1": "Claim F1", "macro_f1": "Macro F1"}[metric],
                    "score": _metric(results, evaluation, metric),
                }
                for metric in ("claim_f1", "macro_f1")
            ]
        )
    if TFIDF_METRICS_PATH.exists():
        tfidf = json.loads(TFIDF_METRICS_PATH.read_text(encoding="utf-8"))
        for evaluation in ("mixed_paper_test", "external_checkthat"):
            rows.extend(
                [
                    {
                        "evaluation": {
                            "mixed_paper_test": "Mixed test",
                            "external_checkthat": "CheckThat transfer",
                        }[evaluation],
                        "model": "TF-IDF",
                        "metric": {"claim_f1": "Claim F1", "macro_f1": "Macro F1"}[metric],
                        "score": _metric(tfidf, evaluation, metric),
                    }
                    for metric in ("claim_f1", "macro_f1")
                ]
            )

    plot_frame = pd.DataFrame(rows)
    sns.set_theme(style="whitegrid", context="talk")
    chart = sns.catplot(
        data=plot_frame,
        x="score",
        y="model",
        hue="metric",
        col="evaluation",
        kind="bar",
        height=5,
        aspect=1.1,
    )
    chart.set(xlim=(0, 1), xlabel="Score", ylabel="")
    chart.set_titles("{col_name}")
    if chart.legend is not None:
        chart.legend.set_title("")
    chart.figure.suptitle("Claim F1 alone hides transfer failure", y=1.05)
    chart.figure.savefig(output_dir / "bert_tfidf_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(chart.figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    results = train_mixed_model(
        args.processed_dir,
        args.output_dir,
        args.artifact_dir,
        requested_device=cast(DeviceName, args.device),
    )
    write_report(results, args.output_dir)
    print(json.dumps(results, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
