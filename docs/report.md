# Heart Disease Prediction: End-to-End MLOps Assignment

- **Course:** Machine Learning Operations (AIMLCZG523), Assignment 1
- **Student:** Rahul Yadav
- **BITS ID:** 2025ae05947

**Code repository:** [https://github.com/2025ae05947-cmd/mlops102](https://github.com/2025ae05947-cmd/mlops102)

## Executive summary

I implemented and locally exercised a complete path from a supplied heart-disease dataset to a prediction API running in a container and a local Kubernetes cluster. I cleaned the Cleveland data, generated a small set of EDA charts, compared Logistic Regression and Random Forest on one reproducible held-out split, saved the selected pipeline, recorded a selected-model run with MLflow, and served predictions with FastAPI. I then built and ran the Docker image, deployed two API replicas to `kind`, and confirmed that Prometheus could scrape the API.

The held-out results favored different models depending on the metric: Random Forest had higher accuracy, precision, and recall, while Logistic Regression had the higher ROC-AUC. I selected Logistic Regression because ROC-AUC was the implemented selection criterion. The dataset is small and this evaluation used one split, so the results are illustrative rather than a claim of clinical performance.

The implementation is not a public-cloud service. Some assignment criteria are only partially addressed: the current training code does not run cross-validation or hyperparameter search, MLflow records the selected run rather than a run for each candidate, and a successful hosted GitHub Actions run has not been confirmed. I call out these limits explicitly rather than presenting them as completed work.

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

### Design decisions

I chose a compact Python package with separate responsibilities rather than putting data loading, model training, and HTTP serving into a single script:

- `data_pipeline.py` is the shared implementation for loading, training, evaluation, model serialization, and prediction.
- `train_model.py` is a small command entry point that calls the training function.
- `app.py` loads the saved artifact and exposes HTTP routes; it does not train a model when the service starts.
- `scripts/` contains tasks that are useful to run separately, such as data preparation and EDA.
- `kubernetes/` contains manifests that can be applied using `kubectl`.

The model, preprocessing steps, feature list, and target name are bundled together so inference uses the same fitted transformations as training. I selected `kind` for deployment because the assignment permits local Kubernetes and the environment was available locally; this avoids implying cloud hosting or introducing cloud credentials and costs.

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

The source Cleveland file contains 303 records. Six records with at least one missing value are excluded by the current `dropna` step, leaving 297 complete cases. The resulting class counts correspond to approximately 53.9% class 0 and 46.1% class 1. I used complete-case filtering for this assignment's small dataset; an alternative would be to retain missing rows and impute them, but that would need to be evaluated and documented as a separate data-preparation choice.

The feature columns used by the model are:

| Feature | Meaning |
|---|---|
| `age` | Age in years |
| `sex` | Dataset's coded sex value |
| `cp` | Chest-pain type code |
| `trestbps` | Resting blood pressure |
| `chol` | Serum cholesterol |
| `fbs` | Fasting blood sugar indicator |
| `restecg` | Resting electrocardiographic result code |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina indicator |
| `oldpeak` | ST depression induced by exercise relative to rest |
| `slope` | Slope of the peak exercise ST segment |
| `ca` | Number of major vessels colored by fluoroscopy |
| `thal` | Thalassemia test result code |

These are dataset codes and measurements, not a clinical interpretation of an individual patient. The application does not attempt to turn the output into medical advice.

## 4. Exploratory data analysis

I used `scripts/eda.py` to inspect the prepared data before training:

```bash
python scripts/eda.py
```

The script prints a preview and descriptive statistics, creates an age-distribution plot and a class-count plot, and saves those plots together as `artifacts/eda_summary.png`. It also calculates pairwise correlations between the 13 feature columns and saves `artifacts/correlation_heatmap.png`.

This initial exploration helped me confirm that the dataset loaded and that the target contained both classes. The class-count plot makes the relatively balanced target visible; the age histogram shows the age distribution in this sample; and the heatmap offers a compact view of linear feature correlations. Correlation is not causation, and these plots do not establish clinical usefulness. The current EDA script is intentionally limited: it does not produce every possible feature/outcome plot or a comprehensive data-quality report.

## 5. Model development and evaluation

I compared two scikit-learn classifiers:

- Logistic Regression
- Random Forest Classifier

Both are wrapped in a pipeline that applies median imputation and standard scaling before classification. Keeping preprocessing and the classifier together in one pipeline ensures the same transformations are used during training and prediction.

The training code separates the data using an 80/20 stratified train/test split with `random_state=42`. This produces 237 training rows and 60 held-out test rows. Stratification preserves the class proportions: the test set has 32 class-0 and 28 class-1 examples.

Both candidate models use the same pipeline structure:

1. `SimpleImputer(strategy="median")` provides a numeric fallback for missing feature values.
2. `StandardScaler` standardizes feature values.
3. The estimator produces a binary class prediction and class probability.

In the current cleaning path, rows with missing values are removed before this pipeline is fitted, so the imputer is a defensive part of the reusable pipeline rather than the mechanism currently responsible for retaining incomplete source rows. All supplied feature columns are numeric, including coded categorical values; this implementation does not one-hot encode those codes.

The Logistic Regression estimator uses `max_iter=2000` and `random_state=42`. The Random Forest uses 300 trees, `class_weight="balanced"`, and `random_state=42`. I fitted each estimator once on the training split and evaluated it on the held-out split. I did not perform hyperparameter search or cross-validation in the current training code; the metrics below are single-split results, not cross-validation averages.

| Model | Accuracy | Precision | Recall | ROC-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression (selected) | 0.8333 | 0.8462 | 0.7857 | 0.9498 |
| Random Forest | 0.8667 | 0.8846 | 0.8214 | 0.9364 |

The selected Logistic Regression model has ROC-AUC 0.9498, higher than the Random Forest's 0.9364, which is why it was selected. The Random Forest has higher accuracy, precision, and recall on this split. This trade-off is important: selection depends on the metric chosen in advance, and another selection objective could choose differently. At a threshold of 0.5, the selected model's confusion matrix on the 60 test rows is:

Accuracy is the proportion of all test records classified correctly; precision is the proportion of predicted-positive records that are positive; recall is the proportion of actual-positive records detected; and ROC-AUC measures how well the model ranks positive records above negative records across thresholds. These metrics summarize different behavior and do not by themselves determine whether a model is safe for clinical use.

| Actual / predicted | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 28 | 4 |
| Actual 1 | 6 | 22 |

Thus, on this test split, the model correctly classified 28 class-0 and 22 class-1 records, with four false positives and six false negatives. For a medical application, false negatives and false positives have different consequences; threshold selection and clinical risk review would require domain expertise and substantially stronger validation.

I trained the model with:

```bash
python -m heart_disease.train_model
```

The selected model and feature metadata are saved in `artifacts/best_model.joblib`. The run summary, including the selected model name and the metrics above, is stored in `artifacts/training_summary.json`. Reproducibility controls include the fixed split seed, stratification, explicit estimator settings, saved feature order, and serialized preprocessing-plus-model pipeline.

## 6. Experiment tracking and reproducibility

The training pipeline uses MLflow with the experiment name `heart-disease-classification`. In the recorded local run, the tracking backend is SQLite at `mlflow.db`. The recorded run ID is `5b5c589217c24a41b9414746f0354e17`, with run name `best_logistic_regression` and status `FINISHED`. The run logs the selected model name, test split size, random seed, stratification flag, the four test metrics, the serialized model, and the training summary.

The current implementation logs the selected candidate, not an individual MLflow run for every candidate. It does not log cross-validation results or EDA/model-evaluation plots. Those are gaps relative to the assignment's request to log parameters, metrics, artifacts, and plots for all runs. The local SQLite database is machine-local evidence and is not a hosted tracking service.

To inspect the SQLite-backed tracking store from the project root:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
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

The response includes `status`, `prediction`, `probability`, and `risk_level`. `prediction` is 0 or 1, `probability` is the estimated probability of class 1, and `risk_level` is a label derived from the predicted class. Missing required features produce an HTTP 400 response, and a missing model artifact produces HTTP 503 rather than a fabricated prediction. The current endpoint checks feature presence but does not enforce clinical ranges or additional domain validation; callers should not assume such checks are implemented.

The API was exercised locally and through the Docker and Kubernetes deployments. Screenshots of endpoint responses are available in the `screenshots/` directory, including [`docker-health.png`](../screenshots/docker-health.png), [`docker-features.png`](../screenshots/docker-features.png), [`docker-predict-response.png`](../screenshots/docker-predict-response.png), and [`kubernetes-health.png`](../screenshots/kubernetes-health.png).

## 8. Automated tests and continuous integration

I kept the test suite small and focused on core behavior. The two tests in `tests/test_data_pipeline.py` check that:

1. The cleaned dataset has the expected features and binary target, and a fitted model pipeline can produce a valid prediction.
2. The `/predict` endpoint accepts a sample feature payload and returns a successful response with a binary prediction and probability.

These tests are smoke-level checks for the pipeline and successful prediction path. They do not cover every invalid input, model metric, deployment behavior, or monitoring counter.

I can run the tests locally with:

```bash
pytest -q
```

The focused local test run passed both tests. This local result confirms those checks in the local environment; it does not establish that the hosted GitHub Actions workflow completed successfully.

The GitHub Actions workflow in `.github/workflows/ci.yml` is configured to run for pushes and pull requests targeting `main` or `master`, and supports manual dispatch. Its steps install the project, run Ruff, train the model, run pytest, and attempt to upload `artifacts/`, `mlruns/`, and `mlflow.db` as a workflow artifact, warning rather than failing if those files are unavailable.

The workflow configuration is present in the repository, and the lint/test commands can also be run locally. A successful hosted GitHub Actions run is a separate verification item; I do not treat local test results or the existence of the workflow file as proof that a hosted run passed. The workflow trains before running tests, so the prediction endpoint test can use the model artifact generated in that workflow run.

## 9. Docker packaging

I packaged the FastAPI service and trained model in a Docker image named `heart-disease-api:latest`. The Dockerfile uses Python 3.11, installs the project requirements, copies the application source and artifacts into the image, and exposes port 8000 inside the container.

After training has produced `artifacts/best_model.joblib`, I build and run the image with:

```bash
docker build -t heart-disease-api:latest .
docker run --rm --name heart-disease-api -p 8001:8000 heart-disease-api:latest
```

The API is then reachable from the host at <http://localhost:8001>; the interactive docs are at <http://localhost:8001/docs>. I verified the health, feature, prediction, and metrics endpoints against the running container. The related captures include [`docker-health.png`](../screenshots/docker-health.png), [`docker-features.png`](../screenshots/docker-features.png), [`docker-predict-response.png`](../screenshots/docker-predict-response.png), and [`docker-metrics.png`](../screenshots/docker-metrics.png).

The container uses the model artifact already present at build time; it does not train a model on startup. The required order is therefore data preparation, training, image build, then container run. A newly trained model requires rebuilding the image before redeployment.

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

The API pod receives `PYTHONPATH=/app/src`, uses the locally loaded image with `imagePullPolicy: IfNotPresent`, and listens on container port 8000. The Service maps port 80 to target port 8000. The manifests do not currently define an Ingress, persistent volumes, resource limits, readiness/liveness probes, or an autoscaler; this is a small assignment deployment rather than a hardened production cluster.

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

This is basic service-level telemetry: it demonstrates that Prometheus can reach the API metrics endpoint and collect the exported counters and histogram. It is not alerting, dashboard-based incident response, or long-term monitoring, and the local Prometheus data is not a durable production monitoring setup.

## 12. Architecture and end-to-end workflow

The training and request-serving paths are separate. Training produces the artifact; the API loads it to serve requests. Prometheus observes the running API rather than participating in prediction.

```text
Training path
-------------
heart+disease.zip
  -> prepare CSV and clean records
  -> EDA
  -> stratified train/test split
  -> fit Logistic Regression and Random Forest
  -> compare held-out metrics, select by ROC-AUC
  -> MLflow selected-model run + joblib artifact

Serving and operations path
---------------------------
joblib model bundle
  -> FastAPI application
  -> Docker image
  -> kind Deployment (2 API replicas)
  -> Kubernetes Service (port 80 -> container port 8000)
  -> Prometheus scrape (every 15 seconds)
```

This structure made it possible to test the model/API path locally first, then reuse the same API container in Docker and Kubernetes. The cluster used for the assignment is local; the workflow does not claim a public cloud endpoint.

## 13. Evidence and demo

I captured screenshots while exercising the API in Docker and Kubernetes and while checking the Prometheus target. The available screenshots are documented in [`screenshots/README.md`](../screenshots/README.md).

The repository also contains [`assignment-demo.mp4`](assignment-demo.mp4), a short silent video assembled from screenshots of the Docker and local Kubernetes environment. It shows selected evidence from the running setup; it is not a narrated screen recording.

| Evidence | What it demonstrates |
|---|---|
| [`docker-health.png`](../screenshots/docker-health.png) | Containerized API health route |
| [`docker-features.png`](../screenshots/docker-features.png) | Required feature contract |
| [`docker-predict-response.png`](../screenshots/docker-predict-response.png) | Sample prediction through Docker |
| [`docker-metrics.png`](../screenshots/docker-metrics.png) | Metrics endpoint in the container |
| [`kubernetes-health.png`](../screenshots/kubernetes-health.png) | Kubernetes-served health route |
| [`kubernetes-predict-response.png`](../screenshots/kubernetes-predict-response.png) | Kubernetes-served prediction |
| [`kubernetes-prometheus-targets.png`](../screenshots/kubernetes-prometheus-targets.png) | Prometheus target status |
| [`kubernetes-prometheus-graph.png`](../screenshots/kubernetes-prometheus-graph.png) | Prometheus query for API target health |

## 14. End-to-end reproduction

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

For Docker, after the model has been generated, build and run:

```bash
docker build -t heart-disease-api:latest .
docker run --rm --name heart-disease-api -p 8001:8000 heart-disease-api:latest
```

For the local Kubernetes route, follow the cluster steps in Section 10, ensure the `cloud-provider-kind` process is running, and start both port forwards in separate terminals.

## 15. Assignment coverage and remaining work

| Assignment area | What this project provides | Evidence / status |
|---|---|---|
| Data acquisition and EDA | ZIP-based data preparation, missing-value cleanup, age and class plots, feature correlation heatmap | Implemented in `scripts/` and `data_pipeline.py`; generated figures under `artifacts/` |
| Feature engineering and models | Numeric imputation/scaling pipeline; Logistic Regression and Random Forest; four test metrics | Both candidates evaluated on one stratified split |
| Cross-validation and tuning | Not implemented in the current training script | Current reported metrics are single held-out split metrics; no hyperparameter search |
| Experiment tracking | MLflow selected-model run with parameters, metrics, model, and summary | Local SQLite run verified; candidates are not logged separately and plots are not logged |
| Packaging and reproducibility | Joblib bundle with preprocessing pipeline and feature metadata; requirements and package config | `artifacts/best_model.joblib`, `requirements.txt`, `pyproject.toml` |
| CI and automated testing | GitHub Actions workflow for Ruff, training, tests, and artifact upload; two focused tests | Workflow file exists; hosted successful run not confirmed |
| Containerization | FastAPI Docker image and local endpoint checks | Docker image was built and exercised locally |
| Deployment | Two-replica API and Prometheus deployed to local `kind` with Services | Local cluster; no public-cloud deployment |
| Monitoring | Request logging and Prometheus metrics scrape | Target was observed `UP`; no alerting or persistent monitoring |
| Documentation and evidence | This report, README, screenshots, and short demo video | Video is silent; PDF export is a separate deliverable |

This table is included so the implementation can be assessed against the assignment requirements without implying that every requested extension is present.

## 16. Limitations and conclusion

This assignment gave me practice connecting model development to deployment and monitoring. I implemented and locally verified data preparation, basic EDA, model comparison, selected-model MLflow logging, model serialization, API serving, Docker packaging, a local Kubernetes deployment, Prometheus scraping, focused automated tests, and a CI workflow configuration.

There are important limitations:

- The cleaned dataset contains 297 rows from the Cleveland subset, so it is small and may not represent other populations or care settings.
- The reported metrics come from one stratified train/test split. Cross-validation, hyperparameter search, confidence intervals, and external validation are not implemented.
- The current MLflow run tracks the selected model; the candidate comparison is not logged as separate runs and plots are not logged as MLflow artifacts.
- The EDA implementation produces a small number of plots and does not cover every feature/outcome relationship.
- The project does not provide a full fairness evaluation or external validation.
- The Kubernetes deployment is local to `kind`; there is no public-cloud service or public prediction endpoint.
- The demo video is silent, and a successful hosted CI result should be verified separately from the workflow configuration and local checks.
- The service metrics and deployment manifests are educational examples and do not provide production-grade alerting, resilience, or security controls.
- This model is for coursework and demonstration only, not for diagnosis or treatment decisions.

Overall, I completed a reproducible local workflow from the supplied dataset to a monitored prediction API. I documented the actual model results, the local deployment evidence, and the remaining gaps so that the report reflects both the work completed and the work that would be needed to make the pipeline more rigorous and production-ready.
