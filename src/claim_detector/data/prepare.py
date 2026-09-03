"""Reconstruct the paper dataset while preserving record provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from claim_detector.data.download import DEFAULT_OUTPUT_DIR, PROJECT_ROOT, digest_file

DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
POLICLAIM_FILES = (
    "AL2003_G4_1.xlsx",
    "CT2014_G4_1.xlsx",
    "DE1999_G4_1.xlsx",
    "DE2021_G4_1.xlsx",
    "IN2001_G4_1.xlsx",
    "IN2011_G4_1.xlsx",
    "KY2018_G4_1.xlsx",
    "US2016_G4_1.xlsx",
)
EXPECTED_SOURCE_COUNTS = {"policlaim": 1953, "averitec": 3068, "claimbuster": 7976}
EXPECTED_PAPER_SPLIT_COUNTS = {"train": 10397, "test": 2600}


class DataValidationError(RuntimeError):
    """Raised when the upstream schema or reconstructed split is invalid."""


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(normalized.split())


def text_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode()).hexdigest()


def require_columns(frame: pd.DataFrame, columns: set[str], source: str) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise DataValidationError(f"{source} is missing required columns: {sorted(missing)}")


def load_policlaim(raw_dir: Path) -> pd.DataFrame:
    frames = []
    for filename in POLICLAIM_FILES:
        file_path = raw_dir / "PoliClaim" / filename
        frame = pd.read_excel(file_path)
        require_columns(frame, {"SENTENCES", "golden"}, str(file_path))
        selected = frame[["SENTENCES", "golden"]].copy()
        selected.columns = ["text", "label"]
        selected["source"] = "policlaim"
        selected["source_record_id"] = [f"{Path(filename).stem}:{index}" for index in frame.index]
        frames.append(selected)
    return pd.concat(frames, ignore_index=True)


def load_averitec(raw_dir: Path) -> pd.DataFrame:
    file_path = raw_dir / "AVeriTeC" / "train.json"
    frame = pd.read_json(file_path)
    require_columns(frame, {"claim"}, str(file_path))
    selected = frame[["claim"]].rename(columns={"claim": "text"}).copy()
    selected["label"] = 1
    selected["source"] = "averitec"
    selected["source_record_id"] = [f"averitec:{index}" for index in frame.index]
    return selected


def load_claimbuster(raw_dir: Path) -> pd.DataFrame:
    file_path = raw_dir / "Claimbuster" / "full.json"
    frame = pd.read_json(file_path)
    require_columns(frame, {"sentence_id", "text", "label"}, str(file_path))
    selected = frame[["text", "label", "sentence_id"]].copy()
    selected["source"] = "claimbuster"
    selected["source_record_id"] = selected.pop("sentence_id").map(
        lambda value: f"claimbuster:{value}"
    )
    return selected


def reconstruct_paper_split(composite: pd.DataFrame, raw_dir: Path) -> pd.DataFrame:
    shuffled = composite.sample(frac=1, random_state=42).reset_index(drop=True)
    train_count = EXPECTED_PAPER_SPLIT_COUNTS["train"]
    shuffled["paper_split"] = np.where(shuffled.index < train_count, "train", "test")

    for split in ("train", "test"):
        reference = pd.read_csv(raw_dir / "ours" / f"{split}.csv")
        require_columns(reference, {"text", "label"}, f"paper {split} reference")
        actual = shuffled[shuffled["paper_split"] == split]
        actual_text = actual["text"].fillna("").astype(str).to_numpy()
        expected_text = reference["text"].fillna("").astype(str).to_numpy()
        actual_label = actual["label"].astype(int).to_numpy()
        expected_label = reference["label"].astype(int).to_numpy()
        if not np.array_equal(actual_text, expected_text) or not np.array_equal(
            actual_label, expected_label
        ):
            raise DataValidationError(
                f"Reconstructed {split} split does not match the pinned paper artifact"
            )
    return shuffled


def assign_development_splits(composite: pd.DataFrame) -> pd.DataFrame:
    result = composite.copy()
    result["development_split"] = "test"
    train_mask = result["paper_split"].eq("train") & result["is_valid"]
    train = result.loc[train_mask]
    strata = train["source"].astype(str) + ":" + train["label"].astype(str)
    splitter = StratifiedGroupKFold(n_splits=13, shuffle=True, random_state=42)
    folds = np.full(len(train), -1, dtype=int)
    for fold, (_, fold_indices) in enumerate(
        splitter.split(train, y=strata, groups=train["normalized_text_hash"])
    ):
        folds[fold_indices] = fold
    if np.any(folds < 0):
        raise DataValidationError("Not every training record received a development fold")
    split_names = np.where(
        folds == 0,
        "validation",
        np.where(folds == 1, "calibration", "fit"),
    )
    result.loc[train_mask, "development_split"] = split_names
    return result


def build_composite(raw_dir: Path) -> pd.DataFrame:
    composite = pd.concat(
        [load_policlaim(raw_dir), load_averitec(raw_dir), load_claimbuster(raw_dir)],
        ignore_index=True,
    )
    actual_counts = composite["source"].value_counts().to_dict()
    if actual_counts != EXPECTED_SOURCE_COUNTS:
        raise DataValidationError(
            f"Unexpected source counts: expected {EXPECTED_SOURCE_COUNTS}, got {actual_counts}"
        )
    if not composite["label"].isin([0, 1]).all():
        raise DataValidationError("Composite labels must be binary 0/1")

    composite = reconstruct_paper_split(composite, raw_dir)
    text_is_string = composite["text"].map(lambda value: isinstance(value, str))
    has_content = composite["text"].fillna("").astype(str).str.strip().ne("")
    composite["is_valid"] = text_is_string & has_content
    composite["validation_issue"] = np.where(composite["is_valid"], "", "empty_or_non_string")
    composite["normalized_text_hash"] = composite["text"].map(
        lambda value: text_hash(value) if isinstance(value, str) and value.strip() else ""
    )
    return assign_development_splits(composite)


def build_checkthat(raw_dir: Path) -> pd.DataFrame:
    file_path = raw_dir / "CheckThat" / "CT22_english_1B_claim_dev_test.tsv"
    frame = pd.read_csv(file_path, sep="\t")
    require_columns(
        frame,
        {"tweet_id", "tweet_text", "class_label", "topic"},
        str(file_path),
    )
    selected = frame[["tweet_text", "class_label", "tweet_id", "topic"]].copy()
    selected.columns = ["text", "label", "source_record_id", "topic"]
    selected["source"] = "checkthat"
    selected["source_record_id"] = selected["source_record_id"].map(
        lambda value: f"checkthat:{value}"
    )
    selected["normalized_text_hash"] = selected["text"].map(text_hash)
    if len(selected) != 911 or not selected["label"].isin([0, 1]).all():
        raise DataValidationError("CheckThat must contain 911 binary-labeled rows")
    return selected[
        ["text", "label", "source", "source_record_id", "topic", "normalized_text_hash"]
    ]


def _json_counts(frame: pd.DataFrame, columns: list[str]) -> dict[str, int]:
    counts = frame.groupby(columns, dropna=False).size()
    return {
        "|".join(map(str, key if isinstance(key, tuple) else (key,))): int(value)
        for key, value in counts.items()
    }


def write_outputs(
    composite: pd.DataFrame, checkthat: pd.DataFrame, output_dir: Path
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    invalid = composite[~composite["is_valid"]].copy()
    valid = composite[composite["is_valid"]].drop(columns=["validation_issue"]).copy()

    output_files = {
        "composite": output_dir / "composite.csv",
        "checkthat": output_dir / "checkthat.csv",
        "invalid_records": output_dir / "invalid_records.csv",
    }
    valid.to_csv(output_files["composite"], index=False)
    checkthat.to_csv(output_files["checkthat"], index=False)
    invalid.to_csv(output_files["invalid_records"], index=False)

    train_hashes = set(valid.loc[valid["paper_split"].eq("train"), "normalized_text_hash"])
    test_hashes = set(valid.loc[valid["paper_split"].eq("test"), "normalized_text_hash"])
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "paper_random_seed": 42,
        "records": {
            "composite_total_upstream": len(composite),
            "composite_valid": len(valid),
            "composite_invalid": len(invalid),
            "checkthat": len(checkthat),
        },
        "source_label_counts": _json_counts(valid, ["source", "label"]),
        "paper_split_counts": _json_counts(valid, ["paper_split"]),
        "development_split_counts": _json_counts(valid, ["development_split"]),
        "normalized_hash_overlap_between_paper_train_test": len(train_hashes & test_hashes),
        "outputs": {},
    }
    for name, output_path in output_files.items():
        size, sha256 = digest_file(output_path)
        metadata["outputs"][name] = {
            "path": output_path.name,
            "bytes": size,
            "sha256": sha256,
        }

    manifest_path = output_dir / "dataset_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return metadata


def prepare(raw_dir: Path, output_dir: Path) -> dict[str, Any]:
    composite = build_composite(raw_dir)
    checkthat = build_checkthat(raw_dir)
    return write_outputs(composite, checkthat, output_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    metadata = prepare(args.raw_dir, args.output_dir)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
