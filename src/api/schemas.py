# src/api/schemas.py

"""
Pydantic request/response schemas for the FastAPI churn prediction API.
"""

from pydantic import BaseModel, Field
from typing import Optional


class CustomerFeatures(BaseModel):
    """Input features for a single customer prediction."""

    gender: int = Field(..., ge=0, le=1, description="0=Male, 1=Female")
    SeniorCitizen: int = Field(..., ge=0, le=1, description="0=No, 1=Yes")
    Partner: int = Field(..., ge=0, le=1, description="0=No, 1=Yes")
    Dependents: int = Field(..., ge=0, le=1, description="0=No, 1=Yes")
    tenure: int = Field(..., ge=0, description="Months with company")
    PhoneService: int = Field(..., ge=0, le=1, description="0=No, 1=Yes")
    PaperlessBilling: int = Field(..., ge=0, le=1, description="0=No, 1=Yes")
    MonthlyCharges: float = Field(..., ge=0, description="Monthly charge amount")
    TotalCharges: float = Field(..., ge=0, description="Total charges to date")

    # One-hot encoded features (set to 0 or 1)
    MultipleLines_No_phone_service: Optional[int] = Field(0, ge=0, le=1)
    MultipleLines_Yes: Optional[int] = Field(0, ge=0, le=1)
    InternetService_Fiber_optic: Optional[int] = Field(0, ge=0, le=1)
    InternetService_No: Optional[int] = Field(0, ge=0, le=1)
    OnlineSecurity_No_internet_service: Optional[int] = Field(0, ge=0, le=1)
    OnlineSecurity_Yes: Optional[int] = Field(0, ge=0, le=1)
    OnlineBackup_No_internet_service: Optional[int] = Field(0, ge=0, le=1)
    OnlineBackup_Yes: Optional[int] = Field(0, ge=0, le=1)
    DeviceProtection_No_internet_service: Optional[int] = Field(0, ge=0, le=1)
    DeviceProtection_Yes: Optional[int] = Field(0, ge=0, le=1)
    TechSupport_No_internet_service: Optional[int] = Field(0, ge=0, le=1)
    TechSupport_Yes: Optional[int] = Field(0, ge=0, le=1)
    StreamingTV_No_internet_service: Optional[int] = Field(0, ge=0, le=1)
    StreamingTV_Yes: Optional[int] = Field(0, ge=0, le=1)
    StreamingMovies_No_internet_service: Optional[int] = Field(0, ge=0, le=1)
    StreamingMovies_Yes: Optional[int] = Field(0, ge=0, le=1)
    Contract_One_year: Optional[int] = Field(0, ge=0, le=1)
    Contract_Two_year: Optional[int] = Field(0, ge=0, le=1)
    PaymentMethod_Credit_card_automatic: Optional[int] = Field(0, ge=0, le=1)
    PaymentMethod_Electronic_check: Optional[int] = Field(0, ge=0, le=1)
    PaymentMethod_Mailed_check: Optional[int] = Field(0, ge=0, le=1)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
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
                    "PaymentMethod_Electronic_check": 1,
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Response from a churn prediction."""

    churn_prediction: int = Field(..., description="0=No Churn, 1=Churn")
    churn_probability: float = Field(..., description="Probability of churn (0.0 to 1.0)")
    confidence: str = Field(..., description="Low / Medium / High confidence label")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    """Model metadata response."""

    model_type: str
    model_file: str
    feature_count: int
    status: str
