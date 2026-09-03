import pandas as pd
import pytest

from claim_detector.models.bert_control import grouped_label_sample, matched_control_frames


def _frame() -> pd.DataFrame:
    rows = []
    for label in (0, 1):
        for index in range(8):
            rows.append(
                {
                    "source_record_id": f"a-{label}-{index}",
                    "source": "a" if index < 4 else "b",
                    "label": label,
                    "normalized_text_hash": f"hash-{label}-{index}",
                    "paper_split": "train",
                    "development_split": "fit",
                }
            )
    return pd.DataFrame(rows)


def test_grouped_label_sample_matches_counts_and_is_deterministic() -> None:
    frame = _frame()
    first = grouped_label_sample(frame, {0: 3, 1: 5}, seed=42)
    second = grouped_label_sample(frame, {0: 3, 1: 5}, seed=42)

    assert first["source_record_id"].tolist() == second["source_record_id"].tolist()
    assert first["label"].value_counts().to_dict() == {1: 5, 0: 3}


def test_grouped_label_sample_keeps_duplicate_groups_together() -> None:
    frame = _frame()
    duplicate = frame.iloc[[0]].copy()
    duplicate["source_record_id"] = "duplicate"
    frame = pd.concat([frame, duplicate], ignore_index=True)

    sample = grouped_label_sample(frame, {0: 4, 1: 3}, seed=7)
    selected_duplicate_group = sample["normalized_text_hash"].eq("hash-0-0").sum()
    assert selected_duplicate_group in {0, 2}


def test_grouped_label_sample_rejects_impossible_request() -> None:
    with pytest.raises(ValueError, match="available"):
        grouped_label_sample(_frame(), {0: 20, 1: 1}, seed=42)


@pytest.mark.real_data
def test_real_matched_controls_match_heldout_counts_and_include_source() -> None:
    composite = pd.read_csv("data/processed/composite.csv")
    for source in ("claimbuster", "policlaim", "averitec"):
        fit, validation, evaluation, matching = matched_control_frames(composite, source, seed=42)
        assert matching["fit_records"]["heldout"] == len(fit)
        assert matching["validation_records"]["heldout"] == len(validation)
        assert source in set(fit["source"])
        assert source in set(validation["source"])
        assert not set(evaluation["normalized_text_hash"]) & set(fit["normalized_text_hash"])
