"""API contract tests for the FastAPI service (Task 4).

These run against the trained artifact with FastAPI's TestClient, so they
double as the integration tests executed in the CI/CD pipeline (Task 6).
"""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

VALID_ORDER = {
    "product_category": "Electronics",
    "product_price_inr": 22055.97,
    "discount_percent": 21.6,
    "expected_delivery_days": 9,
    "actual_delivery_days": 12,
    "customer_rating_preorder": 4.4,
    "previous_returns": 0,
    "previous_orders": 33,
    "payment_method": "Card",
    "damaged_package_flag": "No",
    "package_size": "Large",
    "promotion_active": "Yes",
    "order_id": "ECOM-TEST-0001",
}


def test_health_endpoint_reports_healthy_model():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "healthy"
    assert body["model_loaded"] is True
    assert body["model_version"]
    assert body["model"]


def test_root_endpoint_lists_service():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Product Return Prediction" in resp.json()["service"]


def test_predict_returns_valid_payload():
    resp = client.post("/predict", json=VALID_ORDER)
    assert resp.status_code == 200
    body = resp.json()
    assert body["prediction"] in {"Returned", "Not Returned"}
    assert 0.0 <= body["return_probability"] <= 1.0
    assert body["risk_level"] in {"High", "Medium", "Low"}
    assert body["order_id"] == "ECOM-TEST-0001"
    assert body["model_version"]


def test_predict_high_risk_order_maps_to_returned():
    """A damaged, heavily-discounted, late parcel from a serial returner."""
    risky = {
        **VALID_ORDER,
        "discount_percent": 70.0,
        "damaged_package_flag": "Yes",
        "previous_returns": 8,
        "previous_orders": 10,
        "customer_rating_preorder": 1.5,
        "actual_delivery_days": 15,
        "expected_delivery_days": 5,
    }
    body = client.post("/predict", json=risky).json()
    assert body["prediction"] == "Returned"
    assert body["risk_level"] == "High"


def test_predict_handles_missing_optional_fields():
    partial = {
        "product_category": "Fashion",
        "product_price_inr": 1500.0,
        "expected_delivery_days": 6,
        "previous_orders": 12,
    }
    resp = client.post("/predict", json=partial)
    assert resp.status_code == 200
    assert resp.json()["prediction"] in {"Returned", "Not Returned"}


def test_predict_rejects_invalid_price():
    bad = {**VALID_ORDER, "product_price_inr": -5}
    assert client.post("/predict", json=bad).status_code == 422


def test_predict_rejects_invalid_discount():
    bad = {**VALID_ORDER, "discount_percent": 150}
    assert client.post("/predict", json=bad).status_code == 422


def test_predict_rejects_missing_required_field():
    bad = {k: v for k, v in VALID_ORDER.items() if k != "product_price_inr"}
    assert client.post("/predict", json=bad).status_code == 422


def test_predict_rejects_invalid_rating():
    bad = {**VALID_ORDER, "customer_rating_preorder": 9}
    assert client.post("/predict", json=bad).status_code == 422
