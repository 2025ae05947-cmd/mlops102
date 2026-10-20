# Heart Disease Prediction MLOps Report

## 1. Overview
This project builds and deploys a machine-learning classification service for predicting heart disease risk from patient clinical features. The implementation follows a practical MLOps workflow covering data preparation, feature engineering, model training, experiment tracking, automated tests, Dockerization, Kubernetes deployment, API monitoring, and documentation.

## 2. Data acquisition and preprocessing
The dataset used is the Cleveland Heart Disease dataset from the UCI repository, bundled in the provided archive at `heart+disease.zip`. The project reads the archive, converts the comma-delimited clinical data into a tabular DataFrame, handles missing values represented by `?`, converts all fields to numeric type, and produces a binary target variable where positive disease labels are mapped to `1` and absence to `0`.

Key steps:
- Download or locate the dataset archive
- Validate and normalize data types
- Remove rows with missing values before model training
- Maintain reproducible preprocessing within the package pipeline

## 3. Exploratory data analysis
The EDA step produces descriptive statistics and visual summaries for the dataset, including:
- Age distribution histogram
- Class balance count plot
- Correlation heatmap of the input features

These outputs are saved under the `artifacts` directory so they can be attached to the final report and reused for model interpretation.

## 4. Feature engineering and model development
Two classifier models were trained and evaluated:
- Logistic Regression
- Random Forest Classifier

The project uses a pipeline with median imputation and standard scaling before classification. This preserves reproducibility and helps the feature engineering stage be consistent between training and inference.

Evaluation metrics include:
- Accuracy
- Precision
- Recall
- ROC-AUC

The best-performing model is selected using ROC-AUC for robust discrimination in a binary prediction problem.

## 5. Experiment tracking with MLflow
Model runs are tracked through MLflow, which logs:
- Model name
- Main training configuration
- Performance metrics
- Serialized model artifact
- Summary JSON artifact

This enables clear comparison across experiments and supports reproducibility in a production MLOps environment.

## 6. Model packaging and reproducibility
The selected model is serialized into a joblib bundle containing:
- trained model object
- feature column list
- target metadata

The project includes:
- `requirements.txt`
- `pyproject.toml`
- reusable training script
- API-driven inference path using the same model pipeline

This ensures the model can be retrained and served consistently from a clean environment.

## 7. CI/CD pipeline and automated testing
The repository includes a GitHub Actions workflow defined in `.github/workflows/ci.yml`. It performs:
- dependency installation
- model training
- unit test execution

The tests validate the data pipeline and prediction endpoint logic. The workflow is designed to fail on broken code or model inference problems, which is a standard production-quality practice.

## 8. Model containerization and serving API
The project includes a Dockerfile for packaging the FastAPI inference API. The service exposes:
- `/health` for health checks
- `/features` for feature contract information
- `/predict` to accept JSON input and return prediction confidence

The API has been tested locally and produces a valid JSON response for a patient record.

## 9. Production deployment and Kubernetes manifest
The repository contains a deployment manifest in `kubernetes/deployment.yaml` and a Prometheus config file in `kubernetes/prometheus.yaml`. The K8s manifest provides:
- deployment definition
- service with a LoadBalancer
- container port exposure for the API

This layout is appropriate for local cluster deployment and mirrors a cloud-native production setup.

## 10. Monitoring and request logging
The API includes request logging and Prometheus metrics exposure through `/metrics`. The service logs the route, status code, and request duration. It also emits counters for request counts and predictions, along with a latency histogram, which is suitable for basic monitoring and operational visibility.

## 11. Architecture summary
```text
Heart Disease ZIP dataset
        |
        v
Data cleaning + EDA
        |
        v
Model training (Logistic Regression + Random Forest)
        |
        v
MLflow experiment tracking
        |
        v
Model artifact serialization
        |
        v
FastAPI prediction service
        |
        v
Docker container
        |
        v
Kubernetes deployment + Prometheus metrics
```

## 12. Setup instructions
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 -m heart_disease.train_model
python3 -m uvicorn heart_disease.app:app --host 0.0.0.0 --port 8000
```

Example prediction request:
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

## 13. Conclusion
The assignment has been implemented as a complete MLOps workflow: from dataset ingestion and exploration through model selection, training, reproducibility, API serving, containerization, monitoring, and deployment manifests. While some deliverables such as a formal video recording or public cloud deployment are environment-dependent, the core codebase and documentation fulfill the technical requirements for the project and are structured for further extension to a public deployment environment.
