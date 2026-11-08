"""Pydantic request/response schemas for the prediction API."""

from typing import Optional

from pydantic import BaseModel, Field


class OrderFeatures(BaseModel):
    """Raw attributes of a single purchased order.

    Fields mirror the training dataset. Optional fields model missing data
    exactly like the training CSV: they are imputed by the model pipeline.
    """

    product_category: str = Field(..., examples=["Electronics"], description="Product category")
    product_price_inr: float = Field(..., gt=0, examples=[22055.97])
    discount_percent: Optional[float] = Field(
        None, ge=0, le=100, examples=[21.6], description="Discount applied (0-100)"
    )
    expected_delivery_days: int = Field(..., ge=0, examples=[9])
    actual_delivery_days: Optional[float] = Field(None, ge=0, examples=[12])
    customer_rating_preorder: Optional[float] = Field(
        None, ge=1, le=5, examples=[4.4]
    )
    previous_returns: int = Field(0, ge=0, examples=[0])
    previous_orders: int = Field(..., gt=0, examples=[33])
    payment_method: Optional[str] = Field(None, examples=["Card"])
    damaged_package_flag: str = Field("No", examples=["Yes"])
    package_size: str = Field("Medium", examples=["Large"])
    promotion_active: str = Field("No", examples=["Yes"])
    order_id: Optional[str] = Field(None, examples=["ECOM90001"])


class PredictionResponse(BaseModel):
    """Prediction returned by ``POST /predict``."""

    order_id: Optional[str] = None
    prediction: str = Field(..., description="'Returned' or 'Not Returned'")
    return_probability: float = Field(..., ge=0, le=1)
    risk_level: str = Field(..., description="'High', 'Medium' or 'Low'")
    model: str
    model_version: str
    threshold: float


class HealthResponse(BaseModel):
    """Liveness/readiness payload returned by ``GET /health``."""

    status: str
    model_loaded: bool
    model: Optional[str] = None
    model_version: Optional[str] = None
    timestamp: str
