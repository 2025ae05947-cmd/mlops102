from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "best_model.joblib"
FEATURE_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]
TARGET_COLUMN = "target"


def _ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def download_or_prepare_data(data_path: str | Path | None = None) -> Path:
    """Load the Cleveland heart disease dataset from the bundled archive or a local CSV."""
    _ensure_dirs()
    target = Path(data_path) if data_path is not None else DATA_DIR / "heart_disease.csv"
    if target.exists():
        return target

    archive_candidates = [
        ROOT / "heart+disease.zip",
        ROOT / "heart_disease.zip",
    ]
    archive_file = next((candidate for candidate in archive_candidates if candidate.exists()), None)
    if archive_file is None:
        raise FileNotFoundError(
            "No dataset archive found. Place 'heart+disease.zip' in the project root or provide a CSV path."
        )

    with zipfile.ZipFile(archive_file) as archive:
        if "processed.cleveland.data" in archive.namelist():
            csv_name = "processed.cleveland.data"
        else:
            csv_name = next((name for name in archive.namelist() if name.endswith(".data") and "cleveland" in name.lower()), None)
            if csv_name is None:
                raise FileNotFoundError("No Cleveland dataset file was found inside the zip archive.")

        raw_text = archive.read(csv_name).decode("utf-8")

    csv_lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    with open(target, "w", encoding="utf-8") as csv_file:
        csv_file.write(",".join(FEATURE_COLUMNS + [TARGET_COLUMN]) + "\n")
        for line in csv_lines:
            csv_file.write(line.rstrip() + "\n")

    return target


def load_dataset(dataset_path: str | Path | None = None) -> pd.DataFrame:
    cleaned_path = download_or_prepare_data(dataset_path)
    df = pd.read_csv(cleaned_path)
    df.columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    df = df.replace("?", np.nan)
    for col in FEATURE_COLUMNS + [TARGET_COLUMN]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[TARGET_COLUMN] = df[TARGET_COLUMN].apply(lambda value: 1 if pd.notna(value) and value > 0 else 0)
    df = df.dropna(axis=0, how="any").reset_index(drop=True)
    return df


def build_model_pipeline(model_name: str) -> Pipeline:
    if model_name == "logistic_regression":
        estimator = LogisticRegression(max_iter=2000, random_state=42)
    elif model_name == "random_forest":
        estimator = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", estimator),
        ]
    )


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }


def train_models(output_dir: str | Path | None = None) -> dict[str, Any]:
    _ensure_dirs()
    df = load_dataset()
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    candidate_results: list[dict[str, Any]] = []
    candidate_names = ["logistic_regression", "random_forest"]
    for candidate in candidate_names:
        pipe = build_model_pipeline(candidate)
        pipe.fit(X_train, y_train)
        metrics = evaluate_model(pipe, X_test, y_test)
        candidate_results.append({"name": candidate, "model": pipe, "metrics": metrics})

    best = max(candidate_results, key=lambda item: item["metrics"]["roc_auc"])
    best_model = best["model"]
    best_metrics = best["metrics"]

    # Save the artifact in a serializable form for API inference.
    bundle = {"model": best_model, "feature_columns": FEATURE_COLUMNS, "target_column": TARGET_COLUMN}
    target_path = Path(output_dir) if output_dir is not None else MODEL_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)
    dump(bundle, target_path)

    # MLflow logging for experiment tracking.
    mlflow.set_experiment("heart-disease-classification")
    with mlflow.start_run(run_name=f"best_{best['name']}") as run:
        mlflow.log_params({
            "model": best["name"],
            "test_size": 0.2,
            "random_state": 42,
            "stratify": True,
        })
        mlflow.log_metrics(best_metrics)
        mlflow.sklearn.log_model(
            best_model,
            artifact_path="model",
            skops_trusted_types=["numpy.dtype"],
        )
        mlflow.log_artifact(str(target_path), artifact_path="artifacts")
        with open(ROOT / "artifacts" / "training_summary.json", "w", encoding="utf-8") as handle:
            json.dump({
                "run_id": run.info.run_id,
                "best_model": best["name"],
                "metrics": best_metrics,
                "feature_columns": FEATURE_COLUMNS,
            }, handle, indent=2)
        mlflow.log_artifact(str(ROOT / "artifacts" / "training_summary.json"), artifact_path="artifacts")

    return {
        "best_model": f"best_model_{best['name']}",
        "best_model_name": best["name"],
        "metrics": best_metrics,
        "artifact_path": str(target_path),
        "candidate_results": [{item["name"]: item["metrics"]} for item in candidate_results],
    }


def load_model_bundle(model_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(model_path) if model_path is not None else MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(f"Model bundle not found at {path}. Train the model first.")
    return load(path)


def predict_from_bundle(payload: dict[str, Any], model_bundle: dict[str, Any] | None = None) -> dict[str, Any]:
    bundle = model_bundle or load_model_bundle()
    model = bundle["model"]
    columns = bundle.get("feature_columns", FEATURE_COLUMNS)
    missing = [column for column in columns if column not in payload]
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    row = pd.DataFrame([payload], columns=columns)
    prediction = int(model.predict(row)[0])
    probability = float(model.predict_proba(row)[0, 1])
    return {
        "prediction": prediction,
        "probability": probability,
        "risk_level": "High risk" if prediction == 1 else "Low risk",
    }
