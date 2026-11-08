"""Tests for feature engineering (Task 2)."""

import pandas as pd

from src.features import (
    ALL_FEATURES,
    ENGINEERED_COLUMNS,
    engineer_features,
    select_model_features,
)


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Product_Category": ["Electronics", "Fashion"],
            "Product_Price_INR": [1000.0, 2000.0],
            "Discount_Percent": [20.0, 50.0],
            "Expected_Delivery_Days": [5, 10],
            "Actual_Delivery_Days": [7.0, 8.0],
            "Customer_Rating_PreOrder": [4.0, 2.0],
            "Previous_Returns": [2, 9],
            "Previous_Orders": [10, 10],
            "Payment_Method": ["Card", "COD"],
            "Damaged_Package_Flag": ["No", "Yes"],
            "Package_Size": ["Medium", "Large"],
            "Promotion_Active": ["Yes", "No"],
        }
    )


def test_engineered_columns_are_added():
    out = engineer_features(_sample_frame())
    for col in ENGINEERED_COLUMNS:
        assert col in out.columns


def test_effective_price_uses_discount_percentage():
    out = engineer_features(_sample_frame())
    assert out["Effective_Price_INR"].iloc[0] == 800.0  # 1000 * (1 - 0.20)
    assert out["Effective_Price_INR"].iloc[1] == 1000.0  # 2000 * (1 - 0.50)


def test_delivery_delay_computed_correctly():
    out = engineer_features(_sample_frame())
    assert out["Delivery_Delay_Days"].iloc[0] == 2.0  # 7 - 5 => late
    assert out["Delivery_Delay_Days"].iloc[1] == -2.0  # 8 - 10 => early
    assert out["Is_Late_Delivery"].iloc[0] == 1
    assert out["Is_Late_Delivery"].iloc[1] == 0


def test_customer_return_frequency():
    out = engineer_features(_sample_frame())
    assert abs(out["Customer_Return_Rate"].iloc[0] - 0.2) < 1e-9
    assert abs(out["Customer_Return_Rate"].iloc[1] - 0.9) < 1e-9


def test_return_frequency_zero_when_no_order_history():
    df = _sample_frame()
    df.loc[0, "Previous_Orders"] = 0
    df.loc[0, "Previous_Returns"] = 0
    out = engineer_features(df)
    assert out["Customer_Return_Rate"].iloc[0] == 0.0


def test_missing_values_do_not_crash_engineering():
    df = _sample_frame()
    df.loc[0, ["Discount_Percent", "Actual_Delivery_Days"]] = float("nan")
    out = engineer_features(df)  # must not raise
    assert pd.isna(out["Effective_Price_INR"].iloc[0])
    assert pd.isna(out["Delivery_Delay_Days"].iloc[0])
    assert out["Is_Late_Delivery"].iloc[0] == 0


def test_select_model_features_returns_canonical_columns():
    out = select_model_features(engineer_features(_sample_frame()))
    assert list(out.columns) == ALL_FEATURES
