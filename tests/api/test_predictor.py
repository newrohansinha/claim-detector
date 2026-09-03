import json
from pathlib import Path

import pytest

from claim_detector.api.predictor import BertClaimPredictor, temperature_scale_probability


def test_runtime_temperature_scaling_preserves_decision_and_softens_confidence() -> None:
    assert 0.5 < temperature_scale_probability(0.9, 2.0) < 0.9
    assert 0.1 < temperature_scale_probability(0.1, 2.0) < 0.5


@pytest.mark.parametrize("probability", [-0.1, 1.1, float("nan")])
def test_runtime_temperature_scaling_rejects_invalid_probability(probability: float) -> None:
    with pytest.raises(ValueError, match="Probability"):
        temperature_scale_probability(probability, 1.0)


def test_artifact_loader_rejects_missing_model_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Model artifact directory"):
        BertClaimPredictor.from_artifacts(
            tmp_path / "missing",
            tmp_path / "calibrator.json",
            requested_device="cpu",
        )


def test_artifact_loader_rejects_incomplete_calibrator(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    calibrator = tmp_path / "calibrator.json"
    calibrator.write_text(json.dumps({"method": "temperature_scaling"}), encoding="utf-8")

    with pytest.raises(ValueError, match="missing fields"):
        BertClaimPredictor.from_artifacts(model_dir, calibrator, requested_device="cpu")


def test_artifact_loader_rejects_checkpoint_calibrator_mismatch(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model.safetensors").write_bytes(b"not-the-calibrated-checkpoint")
    calibrator = tmp_path / "calibrator.json"
    calibrator.write_text(
        json.dumps(
            {
                "method": "temperature_scaling",
                "temperature": 1.5,
                "model_sha256": "0" * 64,
                "max_tokens": 128,
                "review_policy": {"confidence_threshold": 0.8},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not match"):
        BertClaimPredictor.from_artifacts(model_dir, calibrator, requested_device="cpu")
