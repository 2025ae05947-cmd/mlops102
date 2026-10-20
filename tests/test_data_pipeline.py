from __future__ import annotations

from fastapi.testclient import TestClient

from heart_disease.app import app
from heart_disease.data_pipeline import FEATURE_COLUMNS, build_model_pipeline, load_dataset, predict_from_bundle


def test_dataset_and_model_pipeline_work() -> None:
    df = load_dataset()
    assert FEATURE_COLUMNS + ["target"] == list(df.columns)
    assert set(df["target"].unique()).issubset({0, 1})
    model = build_model_pipeline("logistic_regression")
    model.fit(df[FEATURE_COLUMNS], df["target"])
    sample = df.iloc[0].to_dict()
    result = predict_from_bundle(sample, {"model": model, "feature_columns": FEATURE_COLUMNS})
    assert set(result.keys()) == {"prediction", "probability", "risk_level"}
    assert result["prediction"] in {0, 1}
    assert 0.0 <= result["probability"] <= 1.0


def test_prediction_endpoint_returns_success_response() -> None:
    client = TestClient(app)
    payload = {
        "age": 63,
        "sex": 1,
        "cp": 3,
        "trestbps": 145,
        "chol": 233,
        "fbs": 1,
        "restecg": 0,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 2.3,
        "slope": 0,
        "ca": 0,
        "thal": 1,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["prediction"] in {0, 1}
    assert 0.0 <= body["probability"] <= 1.0
