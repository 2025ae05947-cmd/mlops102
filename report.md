# Heart Disease Prediction: End-to-End MLOps Assignment

- **Course:** Machine Learning Operations (AIMLCZG523), Assignment 1
- **Student:** Rahul Yadav
- **BITS ID:** 2025ae05947
**Code repository:** [https://github.com/2025ae05947-cmd/mlops102](https://github.com/2025ae05947-cmd/mlops102)

## 1. Introduction

For this assignment, I built an end-to-end machine-learning application that predicts the presence of heart disease from the Cleveland subset of the UCI Heart Disease dataset. My goal was not only to train a classifier, but also to make the path from the supplied data to a usable prediction service repeatable and observable.

I organized the work as a sequence of MLOps stages: preparing the data, exploring it, comparing two models, tracking the selected model with MLflow, packaging the trained pipeline, serving predictions through a REST API, checking the code with tests and CI configuration, containerizing the API, and deploying it to a local Kubernetes cluster with Prometheus monitoring.

The Kubernetes environment in this project is a local `kind` cluster. It is not a public-cloud deployment, and it does not provide a public endpoint.

## 2. Repository and project structure

The source code and assignment materials are available in the [project repository](https://github.com/2025ae05947-cmd/mlops102). The main components are:

| Path | Purpose |
|---|---|
| `src/heart_disease/data_pipeline.py` | Dataset preparation, preprocessing, training, evaluation, serialization, and inference |
| `src/heart_disease/train_model.py` | Training entry point |
| `src/heart_disease/app.py` | FastAPI service and Prometheus metrics |
| `scripts/download_data.py` | Prepares the dataset CSV from the supplied ZIP archive |
| `scripts/eda.py` | Prints summary statistics and generates EDA charts |
| `tests/test_data_pipeline.py` | Focused tests for the pipeline and prediction endpoint |
| `.github/workflows/ci.yml` | GitHub Actions workflow for linting, training, tests, and artifact upload |
| `Dockerfile` | Packages the API and trained model |
| `kubernetes/deployment.yaml` | API Deployment and Service |
| `kubernetes/prometheus.yaml` | Prometheus configuration, Deployment, and Service |
| `screenshots/` | Captured evidence from Docker, Kubernetes, and Prometheus |

The project configuration and installation requirements are defined in `pyproject.toml` and `requirements.txt`.

## 3. Data acquisition and preparation

I used the `heart+disease.zip` archive supplied for the assignment. The data pipeline looks for this archive, or `heart_disease.zip`, in the project root and extracts the Cleveland `.data` file into `data/heart_disease.csv`. The archive itself is not needed after the prepared CSV exists.

I prepared the data using:

```bash
python scripts/download_data.py
```

The source data contains 13 patient features and an original disease-severity target with values from 0 to 4. The pipeline:

1. Reads the Cleveland data from the ZIP archive.
2. Adds the expected feature and target column names.
3. Treats `?` values as missing.
4. Converts feature and target values to numeric values.
5. Maps target value `0` to class `0` and positive values `1` through `4` to class `1`.
6. Drops rows with missing values before model fitting.

After cleaning, the prepared dataset contains **297 rows**: **160** examples in class 0 and **137** in class 1. The classes are reasonably balanced for this small dataset, but the sample size and source population limit how broadly the results can be generalized.

## 4. Exploratory data analysis

I used `scripts/eda.py` to inspect the prepared data before training:

```bash
python scripts/eda.py
```

The script prints a preview and descriptive statistics, creates an age-distribution plot and a class-count plot, and saves those plots together as `artifacts/eda_summary.png`. It also calculates feature correlations and saves `artifacts/correlation_heatmap.png`.

This initial exploration helped me confirm that the dataset loaded and that the target contained both classes. The plots are useful for understanding the data, but they do not establish causal relationships or clinical usefulness.

## 5. Model development and evaluation

I compared two scikit-learn classifiers:

- Logistic Regression
- Random Forest Classifier

Both are wrapped in a pipeline that applies median imputation and standard scaling before classification. Keeping preprocessing and the classifier together in one pipeline ensures the same transformations are used during training and prediction.

The training code separates the data using an 80/20 stratified train/test split with `random_state=42`. Stratification preserves the class proportions across the split. Both candidate models are fitted on the training portion and evaluated on the held-out test portion. I selected the candidate with the higher test ROC-AUC and recorded accuracy, precision, recall, and ROC-AUC.

The recorded result for the selected Logistic Regression model is:

| Metric | Held-out test result |
|---|---:|
| Accuracy | 0.8333 |
| Precision | 0.8462 |
| Recall | 0.7857 |
| ROC-AUC | 0.9498 |

These figures describe one held-out split of a small dataset. They should not be interpreted as proof that the model will perform similarly on new hospitals, populations, or real clinical data.

I trained the model with:

```bash
python -m heart_disease.train_model
```

The selected model and feature metadata are saved in `artifacts/best_model.joblib`. The run summary, including the selected model name and the metrics above, is stored in `artifacts/training_summary.json`.

## 6. Experiment tracking and reproducibility

The training pipeline uses MLflow with the experiment name `heart-disease-classification`. It records the selected model and its training configuration, test metrics, serialized model, and training summary artifact. The generated `mlruns/` data is a local tracking store; it is not automatically published as a hosted experiment.

To inspect locally recorded runs, I can start the MLflow UI from the project root:

```bash
mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
```

I can then open <http://localhost:5000> while the server is running. The model file and training summary make the inference artifact explicit, while the fixed random seed and packaged preprocessing help make retraining and serving repeatable.

## 7. API implementation and verification

I built the inference service with FastAPI. It loads the saved bundle from `artifacts/best_model.joblib` and exposes these endpoints:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Returns the service health response |
| `GET` | `/features` | Returns the required input feature names |
| `POST` | `/predict` | Returns a binary prediction, probability, and risk label |
| `GET` | `/metrics` | Exposes Prometheus-format metrics |

For a local run, I start the application after training:

```bash
uvicorn heart_disease.app:app --host 127.0.0.1 --port 8000
```

FastAPI's interactive documentation is available at <http://localhost:8000/docs>. The prediction request uses all 13 feature names:

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

The response includes `status`, `prediction`, `probability`, and `risk_level`. Missing required features produce an HTTP 400 response, and a missing model artifact produces HTTP 503 rather than a fabricated prediction.

The API was exercised locally and through the Docker and Kubernetes deployments. Screenshots of endpoint responses are available in the `screenshots/` directory, including [`docker-health.png`](../screenshots/docker-health.png), [`docker-features.png`](../screenshots/docker-features.png), [`docker-predict-response.png`](../screenshots/docker-predict-response.png), and [`kubernetes-health.png`](../screenshots/kubernetes-health.png).

## 8. Automated tests and continuous integration

I kept the test suite small and focused on the required behavior. The two tests in `tests/test_data_pipeline.py` check that:

1. The cleaned dataset has the expected features and binary target, and a fitted model pipeline can produce a valid prediction.
2. The `/predict` endpoint accepts a sample feature payload and returns a successful response with a binary prediction and probability.

I can run the tests locally with:

```bash
pytest -q
```

The GitHub Actions workflow in `.github/workflows/ci.yml` is configured to run for pushes and pull requests targeting `main` or `master`, and supports manual dispatch. Its steps install the project, run Ruff, train the model, run pytest, and attempt to upload training and MLflow artifacts.

The workflow configuration is present in the repository. A successful hosted GitHub Actions run is a separate verification item; I do not treat local test results or the existence of the workflow file as proof that a hosted run passed.

## 9. Docker packaging

I packaged the FastAPI service and trained model in a Docker image named `heart-disease-api:latest`. The Dockerfile uses Python 3.11, installs the project requirements, copies the application source and artifacts into the image, and exposes port 8000 inside the container.

After training has produced `artifacts/best_model.joblib`, I build and run the image with:

```bash
docker build -t heart-disease-api:latest .
docker run --rm --name heart-disease-api -p 8001:8000 heart-disease-api:latest
```

The API is then reachable from the host at <http://localhost:8001>; the interactive docs are at <http://localhost:8001/docs>. I verified the health, feature, prediction, and metrics endpoints against the running container. The related captures are in the `screenshots/` directory.

## 10. Local Kubernetes deployment

I deployed the local Docker image to a `kind` cluster named `heart-disease`. The API Deployment requests two replicas and is exposed by a Kubernetes `LoadBalancer` Service. I loaded the image directly into `kind`, so this workflow does not require publishing the image to a container registry.

The main setup commands are:

```bash
kind create cluster --name heart-disease
kind load docker-image heart-disease-api:latest --name heart-disease
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/prometheus.yaml
kubectl rollout status deployment/heart-disease-api
kubectl rollout status deployment/prometheus
kubectl get deployments,pods,services -o wide
```

For local LoadBalancer integration, I ran `cloud-provider-kind` in a separate terminal:

```bash
cloud-provider-kind --enable-lb-port-mapping=false
```

The API and Prometheus were reachable from the Mac host through Kubernetes port-forwarding. I used separate terminals for these commands:

```bash
kubectl port-forward service/heart-disease-api 8080:80
```

```bash
kubectl port-forward service/prometheus 9090:9090
```

I verified the API through <http://localhost:8080/health> and <http://localhost:8080/docs>. This is a local development deployment; it should not be described as hosted or public-cloud production infrastructure.

## 11. Monitoring

The API exposes `/metrics` using the Prometheus client library. It records request counts, prediction counts, and request latency. The API middleware also logs the HTTP method, route, response status, and duration.

Prometheus runs as its own Kubernetes Deployment and scrapes the API Service at `heart-disease-api:80` every 15 seconds, as configured in `kubernetes/prometheus.yaml`. I verified the target through <http://localhost:9090/targets> and observed the `heart-disease-api` target in the `UP` state. Screenshots of the target and graph are available as [`kubernetes-prometheus-targets.png`](../screenshots/kubernetes-prometheus-targets.png) and [`kubernetes-prometheus-graph.png`](../screenshots/kubernetes-prometheus-graph.png).

## 12. Evidence and demo

I captured screenshots while exercising the API in Docker and Kubernetes and while checking the Prometheus target. The available screenshots are documented in [`screenshots/README.md`](../screenshots/README.md).

The repository also contains [`assignment-demo.mp4`](assignment-demo.mp4), a short silent video assembled from screenshots of the Docker and local Kubernetes environment. It shows selected evidence from the running setup; it is not a narrated screen recording.

## 13. End-to-end reproduction

The following commands summarize the path from a fresh checkout to a locally served prediction:

```bash
git clone https://github.com/2025ae05947-cmd/mlops102.git
cd mlops102
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python scripts/download_data.py
python scripts/eda.py
python -m heart_disease.train_model
pytest -q
uvicorn heart_disease.app:app --host 127.0.0.1 --port 8000
```

The supplied `heart+disease.zip` archive must be placed in the repository root before the data preparation and training commands. Docker and local Kubernetes can then be run using the steps above.

## 14. Limitations and conclusion

This assignment gave me practice connecting model development to deployment and monitoring. I completed and locally verified the data preparation, EDA, model comparison, MLflow logging, model serialization, API, Docker image, local Kubernetes deployment, and Prometheus scraping workflow. I also added focused automated tests and a GitHub Actions workflow configuration.

There are important limitations:

- The cleaned dataset contains 297 rows from the Cleveland subset, so it is small and may not represent other populations or care settings.
- The reported metrics come from one stratified train/test split and are not evidence of clinical validity.
- The project does not provide a full fairness evaluation or external validation.
- The Kubernetes deployment is local to `kind`; there is no public-cloud service or public prediction endpoint.
- The demo video is silent, and a hosted CI result should be verified separately from local checks.
- This model is for coursework and demonstration only, not for diagnosis or treatment decisions.

Overall, I completed a reproducible local workflow from the supplied dataset to a monitored prediction API and documented what I ran, what evidence is available, and what remains outside the scope of the local deployment.
