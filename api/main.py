"""
Credit Scoring API
Loads the best model from MLflow Model Registry and serves predictions.
"""

import logging
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import src.features.transformers  # noqa: F401 — registers classes for joblib

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
REGISTERED_MODEL_NAME = "credit-scoring-model"
PIPELINE_PATH = Path("data/processed/preprocessing_pipeline.joblib")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Credit Scoring API",
    description="Predicts probability of financial distress in the next 2 years.",
    version="0.1.0",
)

# Global model and pipeline — loaded once at startup
model = None
pipeline = None


@app.on_event("startup")
async def load_model() -> None:
    global model, pipeline

    logger.info("Loading preprocessing pipeline...")
    pipeline = joblib.load(PIPELINE_PATH)
    logger.info("Pipeline loaded.")

    logger.info(f"Loading model from MLflow registry: {REGISTERED_MODEL_NAME}")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{REGISTERED_MODEL_NAME}/1"
    model = mlflow.sklearn.load_model(model_uri)
    logger.info("Model loaded.")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class CustomerFeatures(BaseModel):
    revolving_utilization: float = Field(
        ..., ge=0, description="Revolving utilization of unsecured lines (0-1)"
    )
    age: int = Field(..., gt=0, lt=120, description="Age of borrower in years")
    times_30_59_days_late: int = Field(
        ..., ge=0, description="Number of times 30-59 days past due"
    )
    debt_ratio: float = Field(..., ge=0, description="Monthly debt / monthly income")
    monthly_income: float = Field(..., ge=0, description="Monthly income in USD")
    open_credit_lines: int = Field(
        ..., ge=0, description="Number of open credit lines"
    )
    times_90_days_late: int = Field(
        ..., ge=0, description="Number of times 90+ days past due"
    )
    real_estate_loans: int = Field(
        ..., ge=0, description="Number of real estate loans"
    )
    times_60_89_days_late: int = Field(
        ..., ge=0, description="Number of times 60-89 days past due"
    )
    number_of_dependents: float = Field(
        ..., ge=0, description="Number of dependents"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "revolving_utilization": 0.75,
                    "age": 45,
                    "times_30_59_days_late": 0,
                    "debt_ratio": 0.35,
                    "monthly_income": 5000.0,
                    "open_credit_lines": 8,
                    "times_90_days_late": 0,
                    "real_estate_loans": 1,
                    "times_60_89_days_late": 0,
                    "number_of_dependents": 2.0,
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    default_probability: float = Field(
        ..., description="Probability of financial distress (0-1)"
    )
    risk_tier: str = Field(
        ..., description="Risk classification: LOW / MEDIUM / HIGH"
    )
    model_version: str = Field(..., description="Model version used for prediction")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def classify_risk(prob: float) -> str:
    if prob < 0.10:
        return "LOW"
    elif prob < 0.25:
        return "MEDIUM"
    return "HIGH"


def features_to_array(customer: CustomerFeatures) -> np.ndarray:
    """Convert Pydantic model to numpy array in FEATURE_COLS order."""
    return np.array([[
        customer.revolving_utilization,
        customer.age,
        customer.times_30_59_days_late,
        customer.debt_ratio,
        customer.monthly_income,
        customer.open_credit_lines,
        customer.times_90_days_late,
        customer.real_estate_loans,
        customer.times_60_89_days_late,
        customer.number_of_dependents,
    ]])


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "pipeline_loaded": pipeline is not None,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(customer: CustomerFeatures) -> PredictionResponse:
    if model is None or pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    try:
        X_raw = features_to_array(customer)
        X_processed = pipeline.transform(X_raw)
        prob = float(model.predict_proba(X_processed)[0][1])
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    return PredictionResponse(
        default_probability=round(prob, 4),
        risk_tier=classify_risk(prob),
        model_version="1",
    )


@app.get("/")
async def root() -> dict:
    return {"message": "Credit Scoring API", "docs": "/docs"}