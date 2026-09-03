import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from claim_detector.api.app import create_app
from claim_detector.api.config import ApiSettings
from claim_detector.api.predictor import BertClaimPredictor, ModelInfo, Prediction


class FixedPredictor:
    @property
    def info(self) -> ModelInfo:
        return ModelInfo(
            model_id="test-model",
            model_version="abc123",
            calibration_method="temperature_scaling",
            temperature=1.5,
            review_confidence_threshold=0.8,
            maximum_input_characters=2_000,
        )

    def predict(self, sentence: str) -> Prediction:
        assert sentence == sentence.strip()
        return Prediction(
            is_claim=True,
            confidence=0.91,
            claim_probability=0.91,
            review_recommended=False,
            model_version="abc123",
        )


@pytest.fixture
def client() -> TestClient:
    app = create_app(predictor=FixedPredictor())
    with TestClient(app) as test_client:
        yield test_client


def test_predict_contract_and_response_headers(client: TestClient) -> None:
    response = client.post(
        "/v1/predict",
        json={"sentence": "  The Empire State Building is in New York.  "},
        headers={"X-Request-ID": "request-123"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "is_claim": True,
        "confidence": 0.91,
        "claim_probability": 0.91,
        "review_recommended": False,
        "model_version": "abc123",
    }
    assert response.headers["x-request-id"] == "request-123"
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"


@pytest.mark.parametrize(
    "body",
    [
        {"sentence": ""},
        {"sentence": "   "},
        {"sentence": 42},
        {"sentence": "valid", "unexpected": True},
        {"sentence": "x" * 2_001},
    ],
)
def test_predict_rejects_invalid_payloads(client: TestClient, body: dict[str, object]) -> None:
    assert client.post("/v1/predict", json=body).status_code == 422


def test_request_size_limit_is_enforced() -> None:
    settings = ApiSettings(max_request_bytes=2_048)
    with TestClient(create_app(settings=settings, predictor=FixedPredictor())) as client:
        response = client.post("/v1/predict", json={"sentence": "x" * 3_000})
    assert response.status_code == 413


def test_health_and_model_metadata(client: TestClient) -> None:
    assert client.get("/health/live").json() == {"status": "live"}
    assert client.get("/health/ready").json() == {"status": "ready"}
    assert client.get("/v1/model").json() == {
        "model_id": "test-model",
        "model_version": "abc123",
        "calibration_method": "temperature_scaling",
        "temperature": 1.5,
        "review_confidence_threshold": 0.8,
        "maximum_input_characters": 2_000,
    }


@pytest.mark.model_artifact
def test_real_artifact_loads_and_returns_calibrated_prediction() -> None:
    model_dir = Path("artifacts/bert_mixed")
    calibrator = Path("reports/generated/bert_calibration/calibrator.json")
    if not model_dir.exists():
        pytest.skip("Locally trained BERT artifact is not present")
    predictor = BertClaimPredictor.from_artifacts(model_dir, calibrator, requested_device="cpu")

    result = predictor.predict("Water freezes at zero degrees Celsius at standard pressure.")

    assert isinstance(result.is_claim, bool)
    assert 0.5 <= result.confidence <= 1
    assert 0 <= result.claim_probability <= 1
    calibration = json.loads(calibrator.read_text(encoding="utf-8"))
    assert result.model_version == calibration["model_sha256"][:12]
