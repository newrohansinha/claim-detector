"""Train real TF-IDF baselines across mixed and source-held-out evaluations."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from claim_detector.data.download import PROJECT_ROOT, digest_file
from claim_detector.data.prepare import DEFAULT_PROCESSED_DIR
from claim_detector.evaluation.bootstrap import evaluated_binary_predictions

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "generated" / "tfidf_baseline"
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "tfidf_baseline"
HELD_OUT_SOURCES = ("claimbuster", "policlaim", "averitec")
MODEL_CONFIG: dict[str, Any] = {
    "vectorizer": {
        "ngram_range": [1, 2],
        "min_df": 2,
        "max_df": 0.995,
        "max_features": 100_000,
        "sublinear_tf": True,
        "strip_accents": "unicode",
    },
    "classifier": {
        "C": 1.0,
        "max_iter": 2_000,
        "random_state": 42,
    },
}


def build_model() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=tuple(MODEL_CONFIG["vectorizer"]["ngram_range"]),
                    min_df=MODEL_CONFIG["vectorizer"]["min_df"],
                    max_df=MODEL_CONFIG["vectorizer"]["max_df"],
                    max_features=MODEL_CONFIG["vectorizer"]["max_features"],
                    sublinear_tf=MODEL_CONFIG["vectorizer"]["sublinear_tf"],
                    strip_accents=MODEL_CONFIG["vectorizer"]["strip_accents"],
                ),
            ),
            ("classifier", LogisticRegression(**MODEL_CONFIG["classifier"])),
        ]
    )


def predict_frame(model: Pipeline, frame: pd.DataFrame) -> pd.DataFrame:
    probabilities = model.predict_proba(frame["text"].astype(str))[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    result = frame[["source_record_id", "source", "normalized_text_hash", "label"]].copy()
    result["claim_probability"] = probabilities
    result["prediction"] = predictions
    return result


def _evaluate(predictions: pd.DataFrame) -> dict[str, object]:
    return evaluated_binary_predictions(
        predictions["label"].to_numpy(),
        predictions["prediction"].to_numpy(),
        predictions["claim_probability"].to_numpy(),
    )


def fit_and_record(
    name: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    output_dir: Path,
    artifact_dir: Path,
) -> tuple[Pipeline, dict[str, Any], pd.DataFrame]:
    source_overlap = set(test["source"]) & set(train["source"])
    if name.startswith("holdout_"):
        if source_overlap:
            raise ValueError(f"Held-out evaluation {name} leaked sources: {sorted(source_overlap)}")
        held_out_integrity = True
    else:
        held_out_integrity = None

    model = build_model()
    model.fit(train["text"].astype(str), train["label"].astype(int))
    predictions = predict_frame(model, test)

    model_path = artifact_dir / name / "model.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    model_bytes, model_sha = digest_file(model_path)

    prediction_path = output_dir / "predictions" / f"{name}.csv"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(prediction_path, index=False)
    prediction_bytes, prediction_sha = digest_file(prediction_path)

    evaluation = {
        "train_records": len(train),
        "test_records": len(test),
        "train_sources": sorted(map(str, train["source"].unique())),
        "test_sources": sorted(map(str, test["source"].unique())),
        "held_out_source_integrity": held_out_integrity,
        "model_artifact": {
            "path": str(model_path.relative_to(PROJECT_ROOT)),
            "bytes": model_bytes,
            "sha256": model_sha,
        },
        "predictions": {
            "path": str(prediction_path.relative_to(PROJECT_ROOT)),
            "bytes": prediction_bytes,
            "sha256": prediction_sha,
        },
        **_evaluate(predictions),
    }
    return model, evaluation, predictions


def run_baselines(
    processed_dir: Path,
    output_dir: Path,
    artifact_dir: Path,
) -> dict[str, Any]:
    composite = pd.read_csv(processed_dir / "composite.csv")
    checkthat = pd.read_csv(processed_dir / "checkthat.csv")
    output_dir.mkdir(parents=True, exist_ok=True)

    paper_train = composite[composite["paper_split"].eq("train")]
    paper_test = composite[composite["paper_split"].eq("test")]
    mixed_model, mixed_evaluation, mixed_predictions = fit_and_record(
        "mixed_paper_test",
        paper_train,
        paper_test,
        output_dir,
        artifact_dir,
    )

    train_hashes = set(paper_train["normalized_text_hash"])
    clean_mask = ~mixed_predictions["normalized_text_hash"].isin(train_hashes)
    clean_predictions = mixed_predictions[clean_mask]

    evaluations: dict[str, Any] = {
        "mixed_paper_test": mixed_evaluation,
        "mixed_paper_test_without_train_duplicates": {
            "removed_records": int((~clean_mask).sum()),
            **_evaluate(clean_predictions),
        },
    }

    for held_out_source in HELD_OUT_SOURCES:
        train = composite[composite["source"].ne(held_out_source)]
        test = composite[composite["source"].eq(held_out_source)]
        _, evaluation, _ = fit_and_record(
            f"holdout_{held_out_source}",
            train,
            test,
            output_dir,
            artifact_dir,
        )
        evaluations[f"holdout_{held_out_source}"] = evaluation

    checkthat_predictions = predict_frame(mixed_model, checkthat)
    checkthat_path = output_dir / "predictions" / "external_checkthat.csv"
    checkthat_predictions.to_csv(checkthat_path, index=False)
    prediction_bytes, prediction_sha = digest_file(checkthat_path)
    evaluations["external_checkthat"] = {
        "train_records": len(paper_train),
        "test_records": len(checkthat),
        "train_sources": sorted(map(str, paper_train["source"].unique())),
        "test_sources": ["checkthat"],
        "predictions": {
            "path": str(checkthat_path.relative_to(PROJECT_ROOT)),
            "bytes": prediction_bytes,
            "sha256": prediction_sha,
        },
        **_evaluate(checkthat_predictions),
    }

    return {
        "schema_version": 1,
        "model": "tfidf_logistic_regression",
        "decision_threshold": 0.5,
        "configuration": MODEL_CONFIG,
        "evaluations": evaluations,
    }


def write_report(results: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(results, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    results = run_baselines(args.processed_dir, args.output_dir, args.artifact_dir)
    write_report(results, args.output_dir)
    print(json.dumps(results, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
