"""FastAPI service exposing the product-return prediction model.

Endpoints
---------
``GET  /health``  -- service + model status (used by Docker HEALTHCHECK and CI)
``POST /predict`` -- predict whether an order will be returned
``GET  /``        -- service metadata

Run locally:  uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

import json
from datetime import datetime, timezone

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from app.schemas import HealthResponse, OrderFeatures, PredictionResponse
from src.config import MODEL_INFO_PATH, MODEL_PATH, MODEL_VERSION, PREDICTION_THRESHOLD
from src.features import engineer_features

# ---------------------------------------------------------------------------
# Model loading (at import time so the container fails fast on a bad image)
# ---------------------------------------------------------------------------
try:
    model = joblib.load(MODEL_PATH)
    with open(MODEL_INFO_PATH, encoding="utf-8") as fh:
        model_info = json.load(fh)
    MODEL_NAME = model_info.get("best_model", "unknown")
    MODEL_LOADED = True
except Exception as exc:  # pragma: no cover - only hit on a broken build
    model = None
    model_info = {}
    MODEL_NAME = None
    MODEL_LOADED = False
    LOAD_ERROR = str(exc)

app = FastAPI(
    title="E-Commerce Product Return Prediction API",
    description=(
        "Predicts whether a purchased product is likely to be returned, based "
        "on product category, price, discount, delivery time, customer rating "
        "and previous return history."
    ),
    version=MODEL_VERSION,
)


def _risk_level(probability: float) -> str:
    """Map a return probability onto a business-friendly risk level."""
    if probability >= 0.60:
        return "High"
    if probability >= 0.40:
        return "Medium"
    return "Low"


def _frame_from_order(order: OrderFeatures) -> pd.DataFrame:
    """Build a single-row raw-order dataframe shaped like the training data."""
    data = {
        "Product_Category": [order.product_category],
        "Product_Price_INR": [order.product_price_inr],
        "Discount_Percent": [order.discount_percent],
        "Expected_Delivery_Days": [order.expected_delivery_days],
        "Actual_Delivery_Days": [order.actual_delivery_days],
        "Customer_Rating_PreOrder": [order.customer_rating_preorder],
        "Previous_Returns": [order.previous_returns],
        "Previous_Orders": [order.previous_orders],
        "Payment_Method": [order.payment_method],
        "Damaged_Package_Flag": [order.damaged_package_flag],
        "Package_Size": [order.package_size],
        "Promotion_Active": [order.promotion_active],
    }
    return pd.DataFrame(data)


@app.get("/", tags=["meta"])
def root() -> JSONResponse:
    """Service metadata."""
    return JSONResponse(
        {
            "service": "E-Commerce Product Return Prediction API",
            "version": MODEL_VERSION,
            "docs": "/docs",
            "endpoints": ["GET /health", "POST /predict"],
        }
    )


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    """Health/readiness probe used by Docker and the CI smoke test."""
    payload = {
        "status": "healthy" if MODEL_LOADED else "unhealthy",
        "model_loaded": MODEL_LOADED,
        "model": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if not MODEL_LOADED:
        payload["detail"] = LOAD_ERROR
    return HealthResponse(**payload)


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(order: OrderFeatures) -> PredictionResponse:
    """Predict whether the given order will be returned."""
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    raw = _frame_from_order(order)
    features = engineer_features(raw)

    probability = float(model.predict_proba(features)[0, 1])
    predicted_return = probability >= PREDICTION_THRESHOLD
    prediction = "Returned" if predicted_return else "Not Returned"

    return PredictionResponse(
        order_id=order.order_id,
        prediction=prediction,
        return_probability=round(probability, 4),
        risk_level=_risk_level(probability),
        model=MODEL_NAME,
        model_version=MODEL_VERSION,
        threshold=PREDICTION_THRESHOLD,
    )
