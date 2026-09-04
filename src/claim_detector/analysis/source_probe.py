"""Measure how accurately sentence text reveals its source dataset."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from claim_detector.data.download import PROJECT_ROOT
from claim_detector.data.prepare import DEFAULT_PROCESSED_DIR
from claim_detector.models.tfidf import MODEL_CONFIG

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "generated" / "source_probe"
SOURCE_ORDER = ("averitec", "claimbuster", "policlaim")


def build_source_model() -> Pipeline:
    vectorizer = MODEL_CONFIG["vectorizer"]
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=tuple(vectorizer["ngram_range"]),
                    min_df=vectorizer["min_df"],
                    max_df=vectorizer["max_df"],
                    max_features=vectorizer["max_features"],
                    sublinear_tf=vectorizer["sublinear_tf"],
                    strip_accents=vectorizer["strip_accents"],
                ),
            ),
            (
                "classifier",
                LogisticRegression(C=1.0, max_iter=2_000, random_state=42),
            ),
        ]
    )


def top_source_features(model: Pipeline, count: int = 20) -> dict[str, list[dict[str, Any]]]:
    vectorizer: TfidfVectorizer = model.named_steps["tfidf"]
    classifier: LogisticRegression = model.named_steps["classifier"]
    feature_names = np.asarray(vectorizer.get_feature_names_out())
    result = {}
    for class_index, source in enumerate(classifier.classes_):
        indices = np.argsort(classifier.coef_[class_index])[-count:][::-1]
        result[str(source)] = [
            {
                "feature": str(feature_names[index]),
                "coefficient": float(classifier.coef_[class_index, index]),
            }
            for index in indices
        ]
    return result


def run_source_probe(processed_dir: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    frame = pd.read_csv(processed_dir / "composite.csv")
    model = build_source_model()
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    predictions = cross_val_predict(
        model,
        frame["text"].astype(str),
        frame["source"].astype(str),
        groups=frame["normalized_text_hash"],
        cv=splitter,
        method="predict",
        n_jobs=-1,
    )
    labels = frame["source"].astype(str).to_numpy()
    majority_source = str(frame["source"].mode().iat[0])
    majority_predictions = np.full(len(frame), majority_source)
    matrix = confusion_matrix(labels, predictions, labels=SOURCE_ORDER)

    model.fit(frame["text"].astype(str), frame["source"].astype(str))
    results = {
        "schema_version": 1,
        "task": "predict dataset source from sentence text",
        "protocol": "5-fold stratified group cross-validation by normalized text hash",
        "samples": len(frame),
        "sources": list(SOURCE_ORDER),
        "majority_source": majority_source,
        "majority_baseline": {
            "accuracy": float(accuracy_score(labels, majority_predictions)),
            "macro_f1": float(f1_score(labels, majority_predictions, average="macro")),
        },
        "text_source_probe": {
            "accuracy": float(accuracy_score(labels, predictions)),
            "macro_f1": float(f1_score(labels, predictions, average="macro")),
        },
        "confusion_matrix": matrix.tolist(),
        "top_features": top_source_features(model),
    }
    prediction_frame = frame[["source_record_id", "source", "normalized_text_hash"]].copy()
    prediction_frame["predicted_source"] = predictions
    return results, prediction_frame


def write_source_probe(
    results: dict[str, Any],
    predictions: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(results, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    predictions.to_csv(output_dir / "predictions.csv", index=False)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    results, predictions = run_source_probe(args.processed_dir)
    write_source_probe(results, predictions, args.output_dir)
    print(json.dumps(results, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
