"""Feature engineering shared by training and inference.

The engineered features required by the assignment are:

* **Discount percentage / effective price** -- ``Discount_Percent`` arrives
  with missing and invalid entries; it is sanitised during cleaning and an
  ``Effective_Price_INR`` feature (what the customer actually pays) is
  derived from it.
* **Delivery delay** -- ``Delivery_Delay_Days`` = actual - expected delivery
  time, plus a binary ``Is_Late_Delivery`` flag.
* **Customer return frequency** -- ``Customer_Return_Rate`` =
  previous returns / previous orders (0 when the customer has no history).
"""

import numpy as np
import pandas as pd

# Engineered (derived) columns produced by :func:`engineer_features`.
ENGINEERED_COLUMNS = [
    "Effective_Price_INR",
    "Delivery_Delay_Days",
    "Is_Late_Delivery",
    "Customer_Return_Rate",
]

# Complete list of model inputs after feature engineering.
NUMERIC_FEATURES = [
    "Product_Price_INR",
    "Discount_Percent",
    "Expected_Delivery_Days",
    "Actual_Delivery_Days",
    "Customer_Rating_PreOrder",
    "Previous_Returns",
    "Previous_Orders",
    "Effective_Price_INR",
    "Delivery_Delay_Days",
    "Is_Late_Delivery",
    "Customer_Return_Rate",
]

CATEGORICAL_FEATURES = [
    "Product_Category",
    "Payment_Method",
    "Damaged_Package_Flag",
    "Package_Size",
    "Promotion_Active",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features to a raw (already cleaned) order dataframe.

    Works for both the full training dataframe and the single-row dataframe
    built from a FastAPI request, so training and inference always compute
    exactly the same features.
    """
    out = df.copy()

    # 1. Effective price after discount (uses the sanitised discount %).
    out["Effective_Price_INR"] = out["Product_Price_INR"] * (
        1 - out["Discount_Percent"] / 100
    )

    # 2. Delivery delay: positive => the parcel arrived late.
    out["Delivery_Delay_Days"] = out["Actual_Delivery_Days"] - out["Expected_Delivery_Days"]
    # NaN comparisons evaluate to False, so missing data => not late.
    out["Is_Late_Delivery"] = (out["Delivery_Delay_Days"] > 0).astype("int64")

    # 3. Customer return frequency: share of past orders that were returned.
    orders = out["Previous_Orders"].astype("float64")
    returns_ = out["Previous_Returns"].astype("float64")
    out["Customer_Return_Rate"] = np.where(orders > 0, returns_ / orders, 0.0)

    return out


def select_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return only the columns the model expects, in canonical order."""
    return df[ALL_FEATURES]


__all__ = [
    "ENGINEERED_COLUMNS",
    "NUMERIC_FEATURES",
    "CATEGORICAL_FEATURES",
    "ALL_FEATURES",
    "engineer_features",
    "select_model_features",
]
