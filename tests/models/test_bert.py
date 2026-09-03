from __future__ import annotations

import pandas as pd
import pytest
import torch
from torch.optim import AdamW

from claim_detector.models.bert import (
    linear_warmup_decay,
    prediction_frame,
    tokenizer_length_statistics,
    validate_development_splits,
)


def test_tokenizer_length_statistics_reports_truncation() -> None:
    statistics = tokenizer_length_statistics([2, 10, 128, 129, 200], max_length=128)

    assert statistics["records"] == 5
    assert statistics["maximum"] == 200
    assert statistics["truncated_records"] == 2
    assert statistics["truncated_fraction"] == pytest.approx(0.4)


def test_prediction_frame_excludes_raw_text() -> None:
    source = pd.DataFrame(
        {
            "text": ["A claim.", "Thanks."],
            "source_record_id": ["test:1", "test:2"],
            "source": ["test", "test"],
            "normalized_text_hash": ["a", "b"],
            "label": [1, 0],
        }
    )

    predictions = prediction_frame(source, torch.tensor([0.75, 0.25]).numpy())

    assert "text" not in predictions.columns
    assert predictions["prediction"].tolist() == [1, 0]


def test_development_split_validation_rejects_hash_leakage() -> None:
    composite = pd.DataFrame(
        {
            "paper_split": ["train", "train", "train"],
            "development_split": ["fit", "validation", "calibration"],
            "normalized_text_hash": ["shared", "shared", "unique"],
        }
    )

    with pytest.raises(ValueError, match="Duplicate text hashes"):
        validate_development_splits(composite)


def test_linear_schedule_warms_up_and_decays() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = AdamW([parameter], lr=1.0)
    scheduler = linear_warmup_decay(optimizer, total_steps=10, warmup_steps=2)

    rates = []
    for _ in range(10):
        optimizer.step()
        scheduler.step()
        rates.append(optimizer.param_groups[0]["lr"])

    assert max(rates) <= 1.0
    assert rates[-1] == pytest.approx(0.0)
