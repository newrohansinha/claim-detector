"""Generate an evidence-backed audit of the prepared research datasets."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from claim_detector.data.download import PROJECT_ROOT, digest_file
from claim_detector.data.prepare import DEFAULT_PROCESSED_DIR
from claim_detector.evaluation.metrics import binary_classification_metrics

DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports" / "generated" / "data_audit"


def source_majority_diagnostic(frame: pd.DataFrame) -> dict[str, Any]:
    train = frame[frame["paper_split"].eq("train")]
    test = frame[frame["paper_split"].eq("test")].copy()
    majority_by_source = train.groupby("source")["label"].mean().ge(0.5).astype(int).to_dict()
    test["prediction"] = test["source"].map(majority_by_source)
    if test["prediction"].isna().any():
        raise ValueError("Test data contains a source absent from training")
    metrics = binary_classification_metrics(test["label"].to_numpy(), test["prediction"].to_numpy())
    return {
        "description": "Uses held-out source identity only; sentence text is ignored.",
        "majority_label_by_source": majority_by_source,
        "metrics": metrics,
    }


def _records_by_source(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    result = {}
    for source, group in frame.groupby("source"):
        word_counts = group["text"].str.split().str.len()
        character_counts = group["text"].str.len()
        result[str(source)] = {
            "records": len(group),
            "positive": int(group["label"].sum()),
            "negative": int((1 - group["label"]).sum()),
            "positive_rate": float(group["label"].mean()),
            "words_median": float(word_counts.median()),
            "words_p95": float(word_counts.quantile(0.95)),
            "characters_median": float(character_counts.median()),
            "characters_p95": float(character_counts.quantile(0.95)),
        }
    return result


def _duplicate_audit(frame: pd.DataFrame) -> dict[str, Any]:
    groups = frame.groupby("normalized_text_hash", sort=False)
    sizes = groups.size()
    duplicate_hashes = set(sizes[sizes > 1].index)
    train_hashes = set(frame.loc[frame["paper_split"].eq("train"), "normalized_text_hash"])
    test_hashes = set(frame.loc[frame["paper_split"].eq("test"), "normalized_text_hash"])
    overlap = train_hashes & test_hashes
    return {
        "normalized_duplicate_groups": len(duplicate_hashes),
        "records_in_normalized_duplicate_groups": int(
            frame["normalized_text_hash"].isin(duplicate_hashes).sum()
        ),
        "conflicting_label_groups": int(groups["label"].nunique().gt(1).sum()),
        "cross_source_duplicate_groups": int(groups["source"].nunique().gt(1).sum()),
        "paper_train_test_overlap_groups": len(overlap),
        "paper_train_test_overlap_records": int(frame["normalized_text_hash"].isin(overlap).sum()),
        "paper_train_test_overlap_by_source": {
            str(source): int(count)
            for source, count in (
                frame[frame["normalized_text_hash"].isin(overlap)]
                .groupby("source")["normalized_text_hash"]
                .nunique()
                .items()
            )
        },
    }


def audit_datasets(processed_dir: Path) -> dict[str, Any]:
    composite_path = processed_dir / "composite.csv"
    checkthat_path = processed_dir / "checkthat.csv"
    manifest_path = processed_dir / "dataset_manifest.json"
    composite = pd.read_csv(composite_path)
    checkthat = pd.read_csv(checkthat_path)
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    manifest_size, manifest_sha = digest_file(manifest_path)

    return {
        "schema_version": 1,
        "input_manifest": {
            "path": str(manifest_path.relative_to(PROJECT_ROOT)),
            "bytes": manifest_size,
            "sha256": manifest_sha,
        },
        "integrity": manifest["records"],
        "composite_by_source": _records_by_source(composite),
        "checkthat": _records_by_source(checkthat)["checkthat"],
        "duplicates": _duplicate_audit(composite),
        "source_majority_diagnostic": source_majority_diagnostic(composite),
    }


def _format_metric(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_markdown(audit: dict[str, Any], output_path: Path) -> None:
    lines = [
        "# Dataset Audit",
        "",
        "This report is generated from checksum-verified upstream data. It contains measured",
        "properties, not planned or simulated values.",
        "",
        "## Source composition",
        "",
        "| Source | Records | Positive | Negative | Positive rate | Median words | P95 words |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for source, values in audit["composite_by_source"].items():
        lines.append(
            f"| {source} | {values['records']} | {values['positive']} | "
            f"{values['negative']} | {values['positive_rate']:.3f} | "
            f"{values['words_median']:.1f} | {values['words_p95']:.1f} |"
        )

    duplicates = audit["duplicates"]
    lines.extend(
        [
            "",
            "## Integrity findings",
            "",
            f"- Upstream composite records: {audit['integrity']['composite_total_upstream']}.",
            f"- Usable composite records: {audit['integrity']['composite_valid']}.",
            f"- Invalid records: {audit['integrity']['composite_invalid']}.",
            f"- Normalized duplicate groups: {duplicates['normalized_duplicate_groups']}.",
            f"- Conflicting-label duplicate groups: {duplicates['conflicting_label_groups']}.",
            f"- Cross-source duplicate groups: {duplicates['cross_source_duplicate_groups']}.",
            "- Normalized hashes crossing the paper train/test boundary: "
            f"{duplicates['paper_train_test_overlap_groups']}.",
            "- Records belonging to those cross-boundary groups: "
            f"{duplicates['paper_train_test_overlap_records']}.",
            "",
            "The frozen paper split is retained for comparability. Derived fit, validation, and",
            "calibration partitions keep normalized duplicate groups together.",
            "",
            "## Source-majority diagnostic",
            "",
            "This diagnostic reads only source identity and predicts that source's training-set",
            "majority label. It does not inspect sentence text and is not deployable.",
            "",
            "| Metric | Value |",
            "|---|---:|",
        ]
    )
    for metric, value in audit["source_majority_diagnostic"]["metrics"].items():
        lines.append(f"| {metric} | {_format_metric(value)} |")
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_figures(processed_dir: Path, output_dir: Path) -> None:
    composite = pd.read_csv(processed_dir / "composite.csv")
    sns.set_theme(style="whitegrid", context="talk")

    counts = composite.groupby(["source", "label"]).size().rename("records").reset_index()
    counts["label"] = counts["label"].map({0: "Not claim", 1: "Claim"})
    figure, axis = plt.subplots(figsize=(10, 6))
    sns.barplot(data=counts, x="source", y="records", hue="label", ax=axis)
    axis.set(title="Composite label distribution by source", xlabel="Source", ylabel="Records")
    figure.tight_layout()
    figure.savefig(output_dir / "source_label_distribution.png", dpi=180)
    plt.close(figure)

    lengths = composite[["source", "text"]].copy()
    lengths["words"] = lengths["text"].str.split().str.len()
    upper = lengths["words"].quantile(0.99)
    figure, axis = plt.subplots(figsize=(10, 6))
    sns.boxplot(
        data=lengths[lengths["words"].le(upper)],
        x="source",
        y="words",
        hue="source",
        legend=False,
        ax=axis,
    )
    axis.set(
        title="Sentence lengths by source (up to global 99th percentile)",
        xlabel="Source",
        ylabel="Whitespace-delimited words",
    )
    figure.tight_layout()
    figure.savefig(output_dir / "source_length_distribution.png", dpi=180)
    plt.close(figure)


def write_audit(audit: dict[str, Any], processed_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "data_audit.json").open("w", encoding="utf-8") as stream:
        json.dump(audit, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    write_markdown(audit, output_dir / "data_audit.md")
    write_figures(processed_dir, output_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REPORT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    audit = audit_datasets(args.processed_dir)
    write_audit(audit, args.processed_dir, args.output_dir)
    print(json.dumps(audit, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
