from __future__ import annotations

import pandas as pd

from claim_detector.data.prepare import assign_development_splits, normalize_text, text_hash


def test_normalization_is_conservative_and_stable() -> None:
    assert normalize_text("  The  CAFÉ\nopened. ") == "the café opened."
    assert text_hash("The  claim") == text_hash("the claim")


def test_development_split_keeps_duplicate_groups_together() -> None:
    rows = []
    for source in ("claimbuster", "policlaim"):
        for label in (0, 1):
            for index in range(30):
                text = f"{source} label {label} example {index}"
                rows.append(
                    {
                        "text": text,
                        "label": label,
                        "source": source,
                        "paper_split": "train",
                        "is_valid": True,
                        "normalized_text_hash": text_hash(text),
                    }
                )
    rows.append({**rows[0], "text": "CLAIMBUSTER  LABEL 0 EXAMPLE 0"})
    frame = assign_development_splits(pd.DataFrame(rows))

    grouped_split_counts = frame.groupby("normalized_text_hash")["development_split"].nunique()
    assert grouped_split_counts.max() == 1
    assert set(frame["development_split"]) == {"fit", "validation", "calibration"}
