"""FastAPI application factory."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Annotated, Any
from uuid import uuid4

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware

from claim_detector.api.config import ApiSettings
from claim_detector.api.predictor import BertClaimPredictor, ClaimPredictor
from claim_detector.api.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)


class RequestPolicyMiddleware(BaseHTTPMiddleware):
    """Reject oversized declared bodies and attach safe response headers."""

    def __init__(self, app: Any, *, max_request_bytes: int) -> None:
        super().__init__(app)
        self.max_request_bytes = max_request_bytes

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                if int(content_length) > self.max_request_bytes:
                    return JSONResponse(
                        status_code=413, content={"detail": "Request body too large"}
                    )
            except ValueError:
                return JSONResponse(status_code=400, content={"detail": "Invalid Content-Length"})
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Request-ID"] = request.headers.get("X-Request-ID", str(uuid4()))[:128]
        return response


def get_predictor(request: Request) -> ClaimPredictor:
    return request.app.state.predictor  # type: ignore[no-any-return]


PredictorDependency = Annotated[ClaimPredictor, Depends(get_predictor)]


def create_app(
    *,
    settings: ApiSettings | None = None,
    predictor: ClaimPredictor | None = None,
) -> FastAPI:
    configuration = settings or ApiSettings.from_environment()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if predictor is None:
            app.state.predictor = BertClaimPredictor.from_artifacts(
                configuration.model_dir,
                configuration.calibrator_path,
                requested_device=configuration.device,
            )
        else:
            app.state.predictor = predictor
        yield

    app = FastAPI(
        title="Claim Detector API",
        version="1.0.0",
        description="Sentence-level factual claim detection; this service does not verify truth.",
        lifespan=lifespan,
    )
    app.add_middleware(RequestPolicyMiddleware, max_request_bytes=configuration.max_request_bytes)

    @app.get("/health/live", response_model=HealthResponse, tags=["health"])
    def live() -> HealthResponse:
        return HealthResponse(status="live")

    @app.get("/health/ready", response_model=HealthResponse, tags=["health"])
    def ready(request: Request) -> HealthResponse | JSONResponse:
        if not hasattr(request.app.state, "predictor"):
            return JSONResponse(status_code=503, content={"status": "not_ready"})
        return HealthResponse(status="ready")

    @app.get("/v1/model", response_model=ModelInfoResponse, tags=["model"])
    def model_info(service: PredictorDependency) -> ModelInfoResponse:
        return ModelInfoResponse.model_validate(service.info.as_dict())

    @app.post("/v1/predict", response_model=PredictionResponse, tags=["prediction"])
    def predict_sentence(
        payload: PredictionRequest,
        service: PredictorDependency,
    ) -> PredictionResponse:
        return PredictionResponse.model_validate(service.predict(payload.sentence).as_dict())

    return app


app = create_app()
