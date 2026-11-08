"""Central configuration for the Product Return Prediction system."""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "Section7_Q4_Ecommerce_Product_Return_CICD.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
MODEL_INFO_PATH = ARTIFACTS_DIR / "model_info.json"

# ---------------------------------------------------------------------------
# Dataset schema
# ---------------------------------------------------------------------------
TARGET_COLUMN = "Return_Status"
TARGET_MAPPING = {"Not Returned": 0, "Returned": 1}
ID_COLUMN = "Order_ID"

NUMERIC_COLUMNS = [
    "Product_Price_INR",
    "Discount_Percent",
    "Expected_Delivery_Days",
    "Actual_Delivery_Days",
    "Customer_Rating_PreOrder",
    "Previous_Returns",
    "Previous_Orders",
]

CATEGORICAL_COLUMNS = [
    "Product_Category",
    "Payment_Method",
    "Damaged_Package_Flag",
    "Package_Size",
    "Promotion_Active",
]

# ---------------------------------------------------------------------------
# Training settings
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20

# Model / service version (semantic). Bumped whenever the model or the
# application behaviour changes so that /health can prove which build is live.
MODEL_VERSION = "1.1.0"

# Decision threshold used by the API to convert return probability into a
# label. Exposed as an environment variable so it can be tuned per deployment
# without changing code.
PREDICTION_THRESHOLD = float(os.environ.get("PREDICTION_THRESHOLD", "0.5"))
