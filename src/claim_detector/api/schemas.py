"""Versioned request and response contracts."""

from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_SENTENCE_CHARACTERS = 2_000


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    sentence: str = Field(min_length=1, max_length=MAX_SENTENCE_CHARACTERS)

    @field_validator("sentence")
    @classmethod
    def sentence_must_contain_text(cls, sentence: str) -> str:
        sentence = sentence.strip()
        if not sentence:
            raise ValueError("sentence must contain non-whitespace text")
        return sentence


class PredictionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    is_claim: bool
    confidence: float = Field(ge=0.5, le=1.0)
    claim_probability: float = Field(ge=0.0, le=1.0)
    review_recommended: bool
    model_version: str


class HealthResponse(BaseModel):
    status: str


class ModelInfoResponse(BaseModel):
    model_id: str
    model_version: str
    calibration_method: str
    temperature: float
    review_confidence_threshold: float
    maximum_input_characters: int
