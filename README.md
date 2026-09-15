# 📊 Customer Churn Prediction — MLOps Pipeline

[![CI/CD Pipeline](https://github.com/dusvanth/customer-churn-mlops/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/dusvanth/customer-churn-mlops/actions/workflows/ci-cd.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue)](https://mlflow.org/)
[![DVC](https://img.shields.io/badge/DVC-Data%20Versioning-purple)](https://dvc.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-009688)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED)](https://www.docker.com/)

A **production-grade MLOps pipeline** that automates training and scalable deployment of a telecom customer churn prediction model. Tests **5+ ML models** (Logistic Regression, Random Forest, XGBoost, LightGBM) achieving **~85% test accuracy**, with integrated experiment tracking, data versioning, CI/CD automation, and containerized REST API deployment.

---

## 🎯 Highlights

- **End-to-end ML pipeline**: Data preprocessing → Multi-model training → Evaluation → Serving
- **5+ models compared**: Logistic Regression, Random Forest, XGBoost, LightGBM
- **MLflow integration**: Full experiment tracking with metrics, parameters, and model registry
- **DVC versioning**: Reproducible data and model artifact versioning
- **GitHub Actions CI/CD**: Automated linting, testing, training, and Docker builds
- **FastAPI REST API**: Real-time churn predictions with Swagger documentation
- **Docker deployment**: Containerized with multi-stage builds, deployed on AWS EC2

---

## 📂 Project Structure

```
customer-churn-mlops/
├── .github/
│   └── workflows/
│       └── ci-cd.yml               # GitHub Actions CI/CD pipeline
├── .dvc/                            # DVC configuration & cache
│   └── config
├── data/
│   ├── raw/
│   │   └── telco_customer_churn.csv.dvc   # DVC-tracked raw dataset
│   └── processed/
│       └── cleaned_telco.csv              # Preprocessed output (DVC-tracked)
├── src/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── app.py                   # FastAPI REST API server
│   │   └── schemas.py               # Pydantic request/response models
│   ├── data/
│   │   ├── __init__.py
│   │   └── preprocess.py            # Data cleaning & feature engineering
│   └── models/
│       ├── __init__.py
│       ├── train.py                 # Single model training (Logistic Regression)
│       ├── train_all_models.py      # Multi-model training & comparison
│       └── evaluate.py              # Model evaluation & reporting
├── reports/
│   └── metrics.json                 # Model comparison metrics
├── Dockerfile                       # Multi-stage Docker build
├── docker-compose.yml               # Docker Compose for deployment
├── .dockerignore
├── .gitignore
├── .dvcignore
├── dvc.yaml                         # DVC pipeline definition
├── dvc.lock                         # DVC pipeline state
├── model.pkl                        # Best trained model (DVC-tracked)
├── requirements.txt                 # Full development dependencies
├── requirements-prod.txt            # Production-only dependencies
└── README.md
```

---

## 🛠️ Tech Stack

| Category                | Tool / Library                       | Purpose                                       |
| ----------------------- | ------------------------------------ | --------------------------------------------- |
| **Language**            | Python 3.11                          | Core programming language                     |
| **ML Models**           | scikit-learn, XGBoost, LightGBM     | Model training & evaluation                   |
| **Data Processing**     | pandas, NumPy                        | Data manipulation & numerical computing       |
| **Experiment Tracking** | MLflow                               | Metric logging, model comparison & registry   |
| **Data Versioning**     | DVC                                  | Dataset & model artifact versioning           |
| **API Framework**       | FastAPI + Uvicorn                    | REST API for real-time predictions            |
| **CI/CD**               | GitHub Actions                       | Automated lint, test, train & deploy          |
| **Containerization**    | Docker + Docker Compose              | Reproducible deployment                       |
| **Cloud Hosting**       | AWS EC2                              | Production deployment                         |

---

## 🏗️ Pipeline Architecture

```
┌──────────────┐     ┌──────────────────┐     ┌──────────────────────┐     ┌──────────────┐
│  Raw Data    │────▶│  Preprocessing   │────▶│  Model Training      │────▶│  Evaluation  │
│  (CSV/DVC)   │     │  preprocess.py   │     │  train_all_models.py │     │  evaluate.py │
└──────────────┘     └──────────────────┘     └──────────────────────┘     └──────────────┘
                                                       │                          │
                                                       ▼                          ▼
                                              ┌──────────────────┐     ┌──────────────────┐
                                              │  MLflow Tracking │     │  reports/         │
                                              │  (experiments)   │     │  metrics.json     │
                                              └──────────────────┘     └──────────────────┘
                                                       │
                                                       ▼
                                              ┌──────────────────┐     ┌──────────────────┐
                                              │  model.pkl       │────▶│  FastAPI Server   │
                                              │  (best model)    │     │  (Docker / EC2)   │
                                              └──────────────────┘     └──────────────────┘
```

**DVC Pipeline Stages** (`dvc.yaml`):

| Stage          | Script                         | Input                          | Output                    |
| -------------- | ------------------------------ | ------------------------------ | ------------------------- |
| `preprocess`   | `src/data/preprocess.py`       | `data/raw/*.csv`               | `data/processed/*.csv`    |
| `train`        | `src/models/train.py`          | Cleaned CSV                    | `model.pkl` + MLflow logs |
| `train_all`    | `src/models/train_all_models.py` | Cleaned CSV                  | `model.pkl` + `reports/metrics.json` |
| `evaluate`     | `src/models/evaluate.py`       | `model.pkl` + cleaned CSV      | `reports/metrics.json`    |

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.9+**
- **Git**
- **Docker** (for containerized deployment)

### Installation

```bash
# Clone the repository
git clone https://github.com/dusvanth/customer-churn-mlops.git
cd customer-churn-mlops

# Create and activate a virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Data Setup

The raw dataset is tracked by DVC. Either pull via DVC or download manually:

```bash
# Option 1: Pull with DVC (if remote is configured)
dvc pull

# Option 2: Download manually from Kaggle
# Place telco_customer_churn.csv in data/raw/
```

> **Dataset**: [Telco Customer Churn on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (~978 KB, 7,043 rows)

---

## 📖 Usage

### Run the Full DVC Pipeline

```bash
# Run all stages (preprocess → train → evaluate)
dvc repro
```

DVC automatically skips stages whose dependencies haven't changed.

### Run Individual Stages

```bash
# Preprocess raw data
dvc repro preprocess
# or directly:
python src/data/preprocess.py

# Train single model (Logistic Regression)
python src/models/train.py

# Train & compare all models (LR, RF, XGBoost, LightGBM)
python src/models/train_all_models.py

# Evaluate the saved model
python src/models/evaluate.py
```

---

## 📊 MLflow — Experiment Tracking

MLflow tracks every training run with hyperparameters, metrics, and serialized models.

### Launch the MLflow UI

```bash
mlflow ui
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

### What's Tracked

| Data Point          | Example                                    |
| ------------------- | ------------------------------------------ |
| **Experiment**      | `telco-churn`                              |
| **Parameters**      | `n_estimators=200`, `max_depth=6`, etc.    |
| **Metrics**         | accuracy, precision, recall, F1, ROC-AUC   |
| **Artifacts**       | Serialized model files                     |
| **Run Names**       | `LogisticRegression`, `RandomForest`, etc. |

### Query Runs Programmatically

```python
import mlflow
runs = mlflow.search_runs(experiment_names=["telco-churn"])
print(runs[["params.model_name", "metrics.accuracy", "metrics.f1_score", "metrics.roc_auc"]])
```

---

## 📦 DVC — Data Versioning

DVC versions both data files and model artifacts so experiments are fully reproducible.

### Common DVC Commands

```bash
# Check pipeline status (which stages are outdated)
dvc status

# Reproduce the pipeline
dvc repro

# Track a new data file
dvc add data/raw/new_dataset.csv

# Push data to remote storage
dvc push

# Pull data from remote storage
dvc pull

# View pipeline DAG
dvc dag
```

### Pipeline DAG

```
  +------------------------------+
  | data/raw/telco_customer_churn |
  +------------------------------+
                 |
                 v
          +------------+
          | preprocess |
          +------------+
                 |
                 v
      +-------------------+
      |      train        |
      | train_all_models  |
      +-------------------+
                 |
                 v
          +------------+
          |  evaluate  |
          +------------+
```

---

## 🤖 Models Tested

The pipeline trains and compares the following models:

| Model                   | Library       | Key Hyperparameters                        |
| ----------------------- | ------------- | ------------------------------------------ |
| **Logistic Regression** | scikit-learn  | `max_iter=1000`                            |
| **Random Forest**       | scikit-learn  | `n_estimators=200`, `max_depth=10`         |
| **XGBoost**             | xgboost       | `n_estimators=200`, `max_depth=6`, `lr=0.1`|
| **LightGBM**            | lightgbm      | `n_estimators=200`, `max_depth=6`, `lr=0.1`|

### Metrics Evaluated

| Metric        | Description                                    |
| ------------- | ---------------------------------------------- |
| **Accuracy**  | Overall correctness of predictions             |
| **Precision** | Ratio of true positives to predicted positives |
| **Recall**    | Ratio of true positives to actual positives    |
| **F1 Score**  | Harmonic mean of precision and recall          |
| **ROC-AUC**   | Area under the receiver operating curve        |

The best model (by accuracy) is automatically saved as `model.pkl`.

---

## 🌐 FastAPI — REST API

The trained model is served via a FastAPI application with interactive Swagger documentation.

### Start the API Server

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

Open the interactive docs at [http://localhost:8000/docs](http://localhost:8000/docs).

### API Endpoints

| Method | Endpoint      | Description                         |
| ------ | ------------- | ----------------------------------- |
| `GET`  | `/health`     | Health check (is model loaded?)     |
| `GET`  | `/model-info` | Model type, feature count, status   |
| `POST` | `/predict`    | Predict churn for a customer        |

### Example: Predict Churn

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": 1,
    "SeniorCitizen": 0,
    "Partner": 1,
    "Dependents": 0,
    "tenure": 12,
    "PhoneService": 1,
    "PaperlessBilling": 1,
    "MonthlyCharges": 70.35,
    "TotalCharges": 844.20,
    "InternetService_Fiber_optic": 1,
    "Contract_One_year": 1,
    "PaymentMethod_Electronic_check": 1
  }'
```

**Response:**
```json
{
  "churn_prediction": 0,
  "churn_probability": 0.2341,
  "confidence": "High"
}
```

---

## 🐳 Docker Deployment

### Build & Run Locally

```bash
# Build the Docker image
docker build -t churn-prediction-api .

# Run the container
docker run -p 8000:8000 churn-prediction-api

# Or use Docker Compose
docker-compose up -d
```

Verify it's running:

```bash
curl http://localhost:8000/health
# {"status": "healthy", "model_loaded": true}
```

### Deploy to AWS EC2

```bash
# 1. SSH into your EC2 instance
ssh -i your-key.pem ec2-user@<ec2-public-ip>

# 2. Install Docker on EC2
sudo yum update -y
sudo yum install -y docker
sudo service docker start
sudo usermod -a -G docker ec2-user

# 3. Clone the repository
git clone https://github.com/dusvanth/customer-churn-mlops.git
cd customer-churn-mlops

# 4. Build and run
docker build -t churn-prediction-api .
docker run -d -p 8000:8000 --name churn-api churn-prediction-api

# 5. Open port 8000 in your EC2 Security Group (AWS Console)
# The API is now available at: http://<ec2-public-ip>:8000/docs
```

### Docker Image Details

| Property        | Value                                |
| --------------- | ------------------------------------ |
| Base image      | `python:3.11-slim`                   |
| Build type      | Multi-stage (builder + production)   |
| User            | Non-root (`appuser`)                 |
| Port            | 8000                                 |
| Health check    | `GET /health` every 30s              |
| Prod deps only  | `requirements-prod.txt`              |

---

## ⚙️ CI/CD — GitHub Actions

The CI/CD pipeline (`.github/workflows/ci-cd.yml`) automates the entire workflow:

```
Push/PR to main
       │
       ├── 🔍 Lint ──────── flake8 code quality checks
       │
       ├── 🧪 Test ──────── pytest + import verification
       │
       ├── 🚂 Train ─────── DVC repro (full pipeline)
       │                     ↳ Upload model & metrics artifacts
       │
       └── 🐳 Docker ────── Build image → Push to GHCR
```

| Job      | Trigger             | What it does                                  |
| -------- | ------------------- | --------------------------------------------- |
| `lint`   | Push / PR to `main` | Runs flake8 on `src/`                         |
| `test`   | Push / PR to `main` | Runs pytest, verifies all imports work        |
| `train`  | Push to `main` only | Runs `dvc repro`, uploads model artifact      |
| `docker` | Push to `main` only | Builds Docker image, pushes to GitHub Container Registry |

---

## 📋 Quick Reference

```bash
# ─── Setup ────────────────────────────────
git clone https://github.com/dusvanth/customer-churn-mlops.git
cd customer-churn-mlops
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt

# ─── Data ─────────────────────────────────
dvc pull                              # Pull data from DVC remote
dvc status                            # Check which stages are outdated
dvc dag                               # View pipeline DAG

# ─── Training ─────────────────────────────
dvc repro                             # Run full pipeline
python src/models/train_all_models.py # Train all models with comparison
python src/models/evaluate.py         # Evaluate the best model

# ─── Experiment Tracking ──────────────────
mlflow ui                             # Launch MLflow UI on :5000

# ─── API ──────────────────────────────────
uvicorn src.api.app:app --reload      # Start FastAPI server on :8000

# ─── Docker ───────────────────────────────
docker build -t churn-api .           # Build image
docker run -p 8000:8000 churn-api     # Run container
docker-compose up -d                  # Or use Compose
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes (`git commit -m 'Add my feature'`)
4. Push to the branch (`git push origin feature/my-feature`)
5. Open a Pull Request

---

## 📄 License

This project is provided for educational and demonstration purposes.
