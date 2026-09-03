from __future__ import annotations

import json

import pandas as pd
import pytest

from claim_detector.data.prepare import DEFAULT_PROCESSED_DIR

pytestmark = pytest.mark.real_data


def test_real_composite_matches_pinned_source_counts() -> None:
    composite = pd.read_csv(DEFAULT_PROCESSED_DIR / "composite.csv")

    assert len(composite) == 12_996
    assert composite.groupby(["source", "label"]).size().to_dict() == {
        ("averitec", 1): 3067,
        ("claimbuster", 0): 5982,
        ("claimbuster", 1): 1994,
        ("policlaim", 0): 799,
        ("policlaim", 1): 1154,
    }


def test_development_splits_are_duplicate_isolated() -> None:
    composite = pd.read_csv(DEFAULT_PROCESSED_DIR / "composite.csv")
    development = composite[composite["paper_split"].eq("train")]

    split_counts = development.groupby("normalized_text_hash")["development_split"].nunique()
    assert split_counts.max() == 1


def test_manifest_records_paper_split_contamination() -> None:
    manifest = json.loads(
        (DEFAULT_PROCESSED_DIR / "dataset_manifest.json").read_text(encoding="utf-8")
    )

    assert manifest["records"] == {
        "checkthat": 911,
        "composite_invalid": 1,
        "composite_total_upstream": 12997,
        "composite_valid": 12996,
    }
    assert manifest["normalized_hash_overlap_between_paper_train_test"] == 18
