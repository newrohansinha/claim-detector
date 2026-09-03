"""Runtime configuration loaded from explicit environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from claim_detector.data.download import PROJECT_ROOT
from claim_detector.models.runtime import DeviceName

DEFAULT_MODEL_DIR = PROJECT_ROOT / "artifacts" / "bert_mixed"
DEFAULT_CALIBRATOR_PATH = (
    PROJECT_ROOT / "reports" / "generated" / "bert_calibration" / "calibrator.json"
)


@dataclass(frozen=True)
class ApiSettings:
    model_dir: Path = DEFAULT_MODEL_DIR
    calibrator_path: Path = DEFAULT_CALIBRATOR_PATH
    device: DeviceName = "auto"
    max_request_bytes: int = 16_384

    @classmethod
    def from_environment(cls) -> ApiSettings:
        device = os.environ.get("CLAIM_DEVICE", "auto")
        if device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError(f"Invalid CLAIM_DEVICE: {device}")
        max_request_bytes = int(os.environ.get("CLAIM_MAX_REQUEST_BYTES", "16384"))
        if max_request_bytes < 2_048:
            raise ValueError("CLAIM_MAX_REQUEST_BYTES must be at least 2048")
        return cls(
            model_dir=Path(os.environ.get("CLAIM_MODEL_DIR", DEFAULT_MODEL_DIR)),
            calibrator_path=Path(os.environ.get("CLAIM_CALIBRATOR_PATH", DEFAULT_CALIBRATOR_PATH)),
            device=cast(DeviceName, device),
            max_request_bytes=max_request_bytes,
        )
