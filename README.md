# Heart Disease Prediction — MLOps Assignment 1

**Repository:** https://github.com/2025ae05947-cmd/mlops102

This project demonstrates an end-to-end machine-learning workflow for binary heart-disease classification using the Cleveland subset of the UCI Heart Disease dataset. It includes dataset preparation, exploratory data analysis, model training and comparison, MLflow tracking, a FastAPI prediction service, automated checks, Docker packaging, local Kubernetes deployment with `kind`, and Prometheus monitoring.

> **Important:** This is an educational machine-learning project, not a medical device. Its predictions are not a diagnosis or a substitute for advice from a qualified healthcare professional.

## Contents

- [Project overview](#project-overview)
- [Repository layout](#repository-layout)
- [Requirements](#requirements)
- [Setup and data preparation](#setup-and-data-preparation)
- [Exploratory analysis and model training](#exploratory-analysis-and-model-training)
- [Experiment tracking](#experiment-tracking)
- [Run the API locally](#run-the-api-locally)
- [API reference](#api-reference)
- [Run with Docker](#run-with-docker)
- [Deploy to local Kubernetes](#deploy-to-local-kubernetes)
- [Continuous integration and tests](#continuous-integration-and-tests)
- [Screenshots, report, and demo](#screenshots-report-and-demo)
- [Limitations](#limitations)

## Project overview

The workflow is:

```text
heart+disease.zip
       |
       v
Dataset preparation and cleaning
       |
       +----> Exploratory data analysis
       |
       v
Logistic Regression + Random Forest
       |
       v
ROC-AUC model selection + MLflow tracking
       |
       v
artifacts/best_model.joblib
       |
       v
FastAPI service (/predict, /health, /features, /metrics)
       |
       +----> Docker image
       |
       +----> Local kind Kubernetes deployment
                    |
                    v
                 Prometheus
```

The training pipeline reads the Cleveland dataset from the supplied archive, treats `?` as missing data, converts columns to numeric values, maps the original disease severity values `0` through `4` to a binary target (`0` = no disease, `1` = disease present), and drops rows containing missing values. It uses a reproducible stratified train/test split with a fixed random seed.

Two scikit-learn pipelines are fitted and compared:

- Logistic Regression
- Random Forest Classifier

Both use median imputation and standard scaling. The model with the higher ROC-AUC on the held-out test set is saved as `artifacts/best_model.joblib`. Accuracy, precision, recall, and ROC-AUC are logged for the selected model.

## Repository layout

```text
.
├── .github/workflows/ci.yml       # GitHub Actions checks
├── artifacts/                     # Trained model and generated EDA/training outputs
├── data/                          # Prepared dataset CSV (generated)
├── docs/
│   ├── assignment-demo.mp4        # Short silent Docker/Kubernetes demo
│   └── report.md                  # Project report and repository link
├── kubernetes/
│   ├── deployment.yaml            # API Deployment and Service
│   └── prometheus.yaml            # Prometheus config, Deployment, and Service
├── screenshots/                   # API, Docker, Kubernetes, and Prometheus evidence
├── scripts/
│   ├── create_demo_video.swift    # Rebuild the demo video from screenshots
│   ├── download_data.py           # Prepare the dataset from the supplied archive
│   └── eda.py                     # Generate exploratory analysis outputs
├── src/heart_disease/
│   ├── app.py                     # FastAPI application and Prometheus metrics
│   ├── data_pipeline.py           # Data, training, evaluation, and prediction logic
│   └── train_model.py             # Training command-line entry point
├── tests/
│   └── test_data_pipeline.py      # Focused pipeline and API tests
├── Dockerfile
├── pyproject.toml
└── requirements.txt
```

## Requirements

For local development:

- Python 3.11 or newer
- The supplied `heart+disease.zip` dataset archive
- Git

For container or Kubernetes workflows, also install Docker and `kubectl`. The Kubernetes steps below use `kind` and `cloud-provider-kind` and are written for macOS with Homebrew.

## Setup and data preparation

Clone the repository and enter its directory:

```bash
git clone https://github.com/2025ae05947-cmd/mlops102.git
cd mlops102
```

Create and activate a virtual environment, then install the project and development dependency:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Place the supplied archive at the repository root with the name `heart+disease.zip`. The data pipeline also recognizes `heart_disease.zip`. Prepare the CSV and confirm its location:

```bash
python scripts/download_data.py
```

The prepared file is written to `data/heart_disease.csv`. If neither supported archive name is present, the data preparation command reports that the archive could not be found.

## Exploratory analysis and model training

Generate the EDA summary and correlation heatmap:

```bash
python scripts/eda.py
```

The script prints a data preview and descriptive statistics and saves its figures under `artifacts/`.

Train and compare both candidate models:

```bash
python -m heart_disease.train_model
```

Training reads the prepared dataset, creates a stratified 80/20 train/test split with `random_state=42`, evaluates both classifiers, selects the candidate with the higher test ROC-AUC, and saves:

- `artifacts/best_model.joblib` — serialized model and feature metadata used by the API
- `artifacts/training_summary.json` — selected model, test metrics, and MLflow run ID

Training requires the dataset archive to be present at the project root. Run training before starting the API or building the Docker image, because the service and image need the trained model artifact.

## Experiment tracking

The training pipeline creates or uses the MLflow experiment `heart-disease-classification` and logs the selected model's parameters, evaluation metrics, model, and training artifacts. MLflow's default local tracking store is used unless an MLflow tracking URI is configured in the environment.

To inspect local runs created with the default file-based store:

```bash
mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
```

Then open <http://localhost:5000>. Keep the server process running while using the UI.

## Run the API locally

After training has generated `artifacts/best_model.joblib`, start the FastAPI application:

```bash
uvicorn heart_disease.app:app --host 127.0.0.1 --port 8000
```

Interactive API documentation is available at <http://localhost:8000/docs>.

## API reference

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Returns a simple service health response |
| `GET` | `/features` | Lists the required model input feature names |
| `POST` | `/predict` | Returns a binary prediction, probability, and risk label |
| `GET` | `/metrics` | Exposes Prometheus-format request, prediction, and latency metrics |

The prediction endpoint expects a JSON object with all 13 model features:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
    "thal": 1
  }'
```

A successful response has this shape:

```json
{
  "status": "success",
  "prediction": 1,
  "probability": 0.51,
  "risk_level": "High risk"
}
```

The probability is the model's estimated probability for the positive class. A missing model artifact returns HTTP `503`; a request missing required features returns HTTP `400`.

Quick endpoint checks:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/features
curl http://localhost:8000/metrics
```

## Run with Docker

First ensure Docker is running, and that training has created `artifacts/best_model.joblib`. The Docker build copies the `artifacts/` directory into the image.

Build and start the API:

```bash
docker build -t heart-disease-api:latest .
docker run --rm --name heart-disease-api -p 8001:8000 heart-disease-api:latest
```

The API is available on the host at <http://localhost:8001>. In a second terminal, verify it:

```bash
curl http://localhost:8001/health
curl http://localhost:8001/features
curl http://localhost:8001/metrics
```

Use the same `/predict` request shown in the [API reference](#api-reference), replacing port `8000` with `8001`, or open <http://localhost:8001/docs>.

Stop the container with `Ctrl+C` in the terminal where it is running. If it was started in detached mode, use:

```bash
docker stop heart-disease-api
```

## Deploy to local Kubernetes

This project has been deployed to a **local** `kind` cluster. This is not a public-cloud deployment. The instructions below assume Docker is running and the trained API image is available locally.

Install the tools on macOS with Homebrew if needed:

```bash
brew install kind kubectl cloud-provider-kind
```

Create the cluster and load the local image into it:

```bash
kind create cluster --name heart-disease
kind load docker-image heart-disease-api:latest --name heart-disease
kubectl config use-context kind-heart-disease
```

Apply the API and Prometheus resources:

```bash
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/prometheus.yaml
```

The API Deployment requests two replicas. Prometheus is configured to scrape the API's Kubernetes Service on port 80 every 15 seconds. To provide LoadBalancer integration for the local cluster, run this in its own terminal and leave it running:

```bash
cloud-provider-kind --enable-lb-port-mapping=false
```

Wait for the workloads and inspect their status:

```bash
kubectl rollout status deployment/heart-disease-api
kubectl rollout status deployment/prometheus
kubectl get deployments,pods,services -o wide
```

For reliable access from macOS, forward the Kubernetes Service ports to localhost. Run each command in a separate terminal and leave both active:

```bash
kubectl port-forward service/heart-disease-api 8080:80
```

```bash
kubectl port-forward service/prometheus 9090:9090
```

Verify the Kubernetes-served API at <http://localhost:8080/health> and <http://localhost:8080/docs>. Prometheus is available at <http://localhost:9090>; open <http://localhost:9090/targets> and confirm that the `heart-disease-api` target is `UP`.

To remove the local deployment when finished:

```bash
kubectl delete -f kubernetes/prometheus.yaml
kubectl delete -f kubernetes/deployment.yaml
kind delete cluster --name heart-disease
```

## Continuous integration and tests

The workflow in `.github/workflows/ci.yml` runs on pushes and pull requests targeting `main` or `master`, and can also be started manually from the GitHub Actions tab. It:

1. Sets up Python 3.11.
2. Installs the project and development dependency.
3. Runs Ruff on `src`, `tests`, and `scripts`.
4. Runs the training pipeline.
5. Runs the test suite with `pytest -q`.
6. Attempts to upload generated artifacts and MLflow files.

Run the same focused checks locally:

```bash
ruff check src tests scripts
pytest -q
```

The tests cover loading and using the data/model pipeline and a successful prediction request against the FastAPI test client.

## Screenshots, report, and demo

The repository includes deployment evidence under `screenshots/`, including Docker API responses, Kubernetes API responses, and Prometheus target/graph views. See [screenshots/README.md](screenshots/README.md) for the list and descriptions.

The written report is [docs/report.md](docs/report.md). The short demo is [docs/assignment-demo.mp4](docs/assignment-demo.mp4); it is silent and assembled from Docker/Kubernetes screenshots. If you update the screenshots and have Swift installed, recreate it with:

```bash
swift scripts/create_demo_video.swift screenshots docs/assignment-demo.mp4
```

The screenshots and demo document the local Docker and `kind` environment. A hosted GitHub Actions run and a public-cloud deployment are separate from this local evidence.
