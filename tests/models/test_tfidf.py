from __future__ import annotations

import pandas as pd

from claim_detector.models.tfidf import build_model, predict_frame


def test_tfidf_model_produces_traceable_probabilities() -> None:
    train = pd.DataFrame(
        {
            "text": [
                "The city has ten schools.",
                "The company earned five dollars.",
                "Thank you very much.",
                "Please sit down now.",
            ],
            "label": [1, 1, 0, 0],
        }
    )
    test = pd.DataFrame(
        {
            "source_record_id": ["test:1"],
            "source": ["test"],
            "normalized_text_hash": ["hash"],
            "text": ["The city earned ten dollars."],
            "label": [1],
        }
    )
    model = build_model()
    model.fit(train["text"], train["label"])

    predictions = predict_frame(model, test)

    assert list(predictions.columns) == [
        "source_record_id",
        "source",
        "normalized_text_hash",
        "label",
        "claim_probability",
        "prediction",
    ]
    assert 0 <= predictions.loc[0, "claim_probability"] <= 1
    assert predictions.loc[0, "prediction"] in {0, 1}
