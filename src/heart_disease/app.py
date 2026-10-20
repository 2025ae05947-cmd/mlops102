from __future__ import annotations

import logging
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from heart_disease.data_pipeline import FEATURE_COLUMNS, load_model_bundle, predict_from_bundle

logger = logging.getLogger("heart_disease_api")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Heart Disease Prediction API", version="1.0.0")
MODEL_PATH = Path(__file__).resolve().parents[2] / "artifacts" / "best_model.joblib"

REQUEST_COUNT = Counter("heart_disease_requests_total", "Total number of API requests")
REQUEST_LATENCY = Histogram("heart_disease_request_latency_seconds", "Request latency in seconds")
PREDICTION_COUNT = Counter("heart_disease_predictions_total", "Total model predictions made")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start
    REQUEST_COUNT.inc()
    REQUEST_LATENCY.observe(duration)
    logger.info("%s %s %s in %.4fs", request.method, request.url.path, response.status_code, duration)
    return response


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics")
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict")
def predict(payload: dict[str, float]) -> dict[str, object]:
    start = time.perf_counter()
    try:
        bundle = load_model_bundle(MODEL_PATH)
        result = predict_from_bundle(payload, bundle)
        PREDICTION_COUNT.inc()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        REQUEST_LATENCY.observe(time.perf_counter() - start)
    return {"status": "success", **result}


@app.get("/features")
def features() -> dict[str, list[str]]:
    return {"required_features": FEATURE_COLUMNS}
