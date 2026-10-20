# Heart Disease MLOps Project

This repository contains an end-to-end MLOps workflow for the Heart Disease UCI dataset. The project includes data acquisition, exploratory analysis, model training, experiment tracking with MLflow, automated tests, Docker packaging, Kubernetes deployment, and a FastAPI serving layer.

## Project goals
- Train and compare multiple classification models for heart disease prediction.
- Log experiment metadata, metrics, and artifacts with MLflow.
- Package the best model for reproducible inference.
- Serve the model via a REST API.
- Provide deployment and CI/CD configuration for a production-ready workflow.

## Local setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py
python scripts/eda.py
python -m heart_disease.train_model
pytest -q
```

## Run the API locally
```bash
uvicorn heart_disease.app:app --host 0.0.0.0 --port 8000
```

Example request:
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

## Docker
```bash
docker build -t heart-disease-api .
docker run --rm -p 8000:8000 heart-disease-api
```

## Deployment
The project includes a Kubernetes manifest in `kubernetes/deployment.yaml` for deploying the API to a cluster.

## Report
A summary report is provided in `docs/report.md`.
