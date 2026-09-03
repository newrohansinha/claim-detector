from __future__ import annotations

import pandas as pd

from claim_detector.analysis.audit import source_majority_diagnostic


def test_source_majority_diagnostic_never_reads_text() -> None:
    frame = pd.DataFrame(
        [
            {"text": "same", "label": 0, "source": "a", "paper_split": "train"},
            {"text": "same", "label": 0, "source": "a", "paper_split": "train"},
            {"text": "same", "label": 1, "source": "b", "paper_split": "train"},
            {"text": "same", "label": 1, "source": "b", "paper_split": "train"},
            {"text": "same", "label": 0, "source": "a", "paper_split": "test"},
            {"text": "same", "label": 1, "source": "b", "paper_split": "test"},
        ]
    )

    result = source_majority_diagnostic(frame)

    assert result["majority_label_by_source"] == {"a": 0, "b": 1}
    assert result["metrics"]["accuracy"] == 1.0
