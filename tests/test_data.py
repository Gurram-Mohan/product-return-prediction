"""Tests for data loading and cleaning (Task 1: preprocessing)."""

import pandas as pd

from src.config import DATA_PATH, TARGET_COLUMN
from src.data_loader import clean_data, load_and_clean_data, load_raw_data


def test_raw_data_loads_with_expected_shape():
    df = load_raw_data()
    assert df.shape[0] >= 1000
    assert "Return_Status" in df.columns


def test_clean_data_removes_invalid_prices():
    df = pd.DataFrame({"Product_Price_INR": [-10, 0, 500.0, 100.0]})
    out = clean_data(df)
    assert out["Product_Price_INR"].isna().sum() == 2  # -10 and 0 invalidated
    assert out["Product_Price_INR"].iloc[2] == 500.0
    assert out["Product_Price_INR"].iloc[3] == 100.0


def test_clean_data_removes_invalid_discounts():
    df = pd.DataFrame({"Discount_Percent": [5.0, 125.0, -3.0, 50.0]})
    out = clean_data(df)
    assert pd.isna(out.loc[1, "Discount_Percent"])  # 125% is impossible
    assert pd.isna(out.loc[2, "Discount_Percent"])  # negative discount
    assert out.loc[0, "Discount_Percent"] == 5.0
    assert out.loc[3, "Discount_Percent"] == 50.0


def test_clean_data_removes_invalid_ratings():
    df = pd.DataFrame({"Customer_Rating_PreOrder": [4.0, 5.5, 0.5, 3.0]})
    out = clean_data(df)
    assert pd.isna(out["Customer_Rating_PreOrder"].iloc[1])
    assert pd.isna(out["Customer_Rating_PreOrder"].iloc[2])


def test_full_clean_keeps_valid_rows_and_target():
    df = load_and_clean_data()
    assert set(df[TARGET_COLUMN].unique()) <= {"Returned", "Not Returned"}
    # Invalid values must never survive cleaning.
    assert (df["Product_Price_INR"].dropna() > 0).all()
    assert df["Discount_Percent"].dropna().between(0, 100).all()
    assert df["Customer_Rating_PreOrder"].dropna().between(1, 5).all()


def test_missing_values_are_still_present_for_imputation():
    """Cleaning masks invalid values as NaN; imputation happens in the pipeline."""
    df = load_and_clean_data()
    raw = load_raw_data()
    # The raw file really contains missing data ...
    assert raw.isna().sum().sum() > 0
    # ... and clean_data keeps NaN (does not crash or fill with junk).
    assert df.isna().sum().sum() >= raw.isna().sum().sum() - 1


def test_dataset_path_exists():
    assert DATA_PATH.exists()
