"""Artifact-validated, calibrated BERT inference."""

from __future__ import annotations

import json
import math
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from claim_detector.data.download import digest_file
from claim_detector.models.runtime import MODEL_ID, DeviceName, resolve_device

PROBABILITY_EPSILON = 1e-7


def temperature_scale_probability(probability: float, temperature: float) -> float:
    if not 0 <= probability <= 1 or not math.isfinite(probability):
        raise ValueError("Probability must be finite and between zero and one")
    if temperature <= 0 or not math.isfinite(temperature):
        raise ValueError("Temperature must be finite and positive")
    clipped = min(max(probability, PROBABILITY_EPSILON), 1 - PROBABILITY_EPSILON)
    log_odds = math.log(clipped) - math.log1p(-clipped)
    return 1 / (1 + math.exp(-log_odds / temperature))


@dataclass(frozen=True)
class Prediction:
    is_claim: bool
    confidence: float
    claim_probability: float
    review_recommended: bool
    model_version: str

    def as_dict(self) -> dict[str, bool | float | str]:
        return cast(dict[str, bool | float | str], asdict(self))


@dataclass(frozen=True)
class ModelInfo:
    model_id: str
    model_version: str
    calibration_method: str
    temperature: float
    review_confidence_threshold: float
    maximum_input_characters: int

    def as_dict(self) -> dict[str, float | int | str]:
        return cast(dict[str, float | int | str], asdict(self))


class ClaimPredictor(Protocol):
    @property
    def info(self) -> ModelInfo: ...

    def predict(self, sentence: str) -> Prediction: ...


class BertClaimPredictor:
    """One model instance loaded at process startup and guarded against oversubscription."""

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        device: torch.device,
        temperature: float,
        review_threshold: float,
        maximum_input_characters: int,
        max_tokens: int,
        model_version: str,
        calibration_method: str,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._device = device
        self._temperature = temperature
        self._review_threshold = review_threshold
        self._max_tokens = max_tokens
        self._lock = threading.Lock()
        self._info = ModelInfo(
            model_id=MODEL_ID,
            model_version=model_version,
            calibration_method=calibration_method,
            temperature=temperature,
            review_confidence_threshold=review_threshold,
            maximum_input_characters=maximum_input_characters,
        )

    @classmethod
    def from_artifacts(
        cls,
        model_dir: Path,
        calibrator_path: Path,
        *,
        requested_device: DeviceName = "auto",
        maximum_input_characters: int = 2_000,
    ) -> BertClaimPredictor:
        if not model_dir.is_dir():
            raise FileNotFoundError(f"Model artifact directory does not exist: {model_dir}")
        if not calibrator_path.is_file():
            raise FileNotFoundError(f"Calibrator does not exist: {calibrator_path}")
        calibration = json.loads(calibrator_path.read_text(encoding="utf-8"))
        required = {"method", "temperature", "model_sha256", "max_tokens", "review_policy"}
        missing = required - set(calibration)
        if missing:
            raise ValueError(f"Calibrator is missing fields: {sorted(missing)}")

        model_path = model_dir / "model.safetensors"
        _, actual_sha = digest_file(model_path)
        expected_sha = str(calibration["model_sha256"])
        if actual_sha != expected_sha:
            raise ValueError(
                "Model artifact does not match the calibrated checkpoint: "
                f"expected {expected_sha}, got {actual_sha}"
            )

        device = resolve_device(requested_device)
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)  # type: ignore[no-untyped-call]
        model = AutoModelForSequenceClassification.from_pretrained(
            model_dir, local_files_only=True
        ).to(device)
        model.eval()
        return cls(
            model=model,
            tokenizer=tokenizer,
            device=device,
            temperature=float(calibration["temperature"]),
            review_threshold=float(calibration["review_policy"]["confidence_threshold"]),
            maximum_input_characters=maximum_input_characters,
            max_tokens=int(calibration["max_tokens"]),
            model_version=actual_sha[:12],
            calibration_method=str(calibration["method"]),
        )

    @property
    def info(self) -> ModelInfo:
        return self._info

    def predict(self, sentence: str) -> Prediction:
        encoded = self._tokenizer(
            sentence,
            add_special_tokens=True,
            max_length=self._max_tokens,
            return_attention_mask=True,
            return_tensors="pt",
            truncation=True,
        )
        with self._lock, torch.inference_mode():
            output = self._model(
                input_ids=encoded["input_ids"].to(self._device),
                attention_mask=encoded["attention_mask"].to(self._device),
            )
            raw_probability = float(torch.softmax(output.logits.float(), dim=1)[0, 1].cpu())
        claim_probability = temperature_scale_probability(raw_probability, self._temperature)
        is_claim = claim_probability >= 0.5
        confidence = claim_probability if is_claim else 1 - claim_probability
        return Prediction(
            is_claim=is_claim,
            confidence=confidence,
            claim_probability=claim_probability,
            review_recommended=confidence < self._review_threshold,
            model_version=self._info.model_version,
        )
