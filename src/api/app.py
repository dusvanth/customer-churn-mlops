# src/api/app.py

"""
FastAPI application for serving churn predictions via REST API.

Endpoints:
  GET  /health      — Health check
  GET  /model-info  — Model metadata
  POST /predict     — Predict churn for a customer
"""

import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from src.api.schemas import (
    CustomerFeatures, PredictionResponse,
    HealthResponse, ModelInfoResponse
)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Customer Churn Prediction API",
    description=(
        "REST API for real-time telecom customer churn prediction. "
        "Trained on the Telco Customer Churn dataset with multiple ML models."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")
model = None


def load_model():
    """Load the trained model from disk."""
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"✅ Model loaded from {MODEL_PATH}: {type(model).__name__}")
    else:
        print(f"⚠️  Model file not found at {MODEL_PATH}")


@app.on_event("startup")
async def startup_event():
    """Load model on application startup."""
    load_model()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check if the API and model are operational."""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["System"])
async def model_info():
    """Return metadata about the loaded model."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    n_features = (
        model.n_features_in_ if hasattr(model, "n_features_in_") else -1
    )

    return ModelInfoResponse(
        model_type=type(model).__name__,
        model_file=MODEL_PATH,
        feature_count=n_features,
        status="ready",
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(customer: CustomerFeatures):
    """
    Predict whether a customer will churn.

    Accepts customer features as JSON and returns:
    - `churn_prediction`: 0 (No Churn) or 1 (Churn)
    - `churn_probability`: probability between 0.0 and 1.0
    - `confidence`: Low / Medium / High
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert input to DataFrame
        input_data = pd.DataFrame([customer.model_dump()])

        # Predict
        prediction = int(model.predict(input_data)[0])
        probability = float(model.predict_proba(input_data)[0][1])

        # Confidence label
        if probability < 0.3 or probability > 0.7:
            confidence = "High"
        elif probability < 0.4 or probability > 0.6:
            confidence = "Medium"
        else:
            confidence = "Low"

        return PredictionResponse(
            churn_prediction=prediction,
            churn_probability=round(probability, 4),
            confidence=confidence,
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")
