"""Data loading and cleaning for the e-commerce returns dataset.

Cleaning rules
--------------
* The CSV is read with ``utf-8-sig`` to strip the byte-order mark.
* Whitespace is stripped from every text column.
* Invalid numeric values are converted to ``NaN`` so that they can be repaired
  by the median/mode imputers inside the scikit-learn pipeline:
    - ``Product_Price_INR`` <= 0
    - ``Discount_Percent`` < 0 or > 100
    - ``Customer_Rating_PreOrder`` outside [1, 5]
    - ``Actual_Delivery_Days`` / ``Expected_Delivery_Days`` < 0
    - ``Previous_Returns`` > ``Previous_Orders`` (impossible history)
* The target is validated to contain only {"Returned", "Not Returned"}.
"""

from pathlib import Path

import pandas as pd

from src.config import CATEGORICAL_COLUMNS, DATA_PATH, TARGET_COLUMN


def load_raw_data(path: str | Path | None = None) -> pd.DataFrame:
    """Load the raw CSV exactly as stored (no cleaning)."""
    path = Path(path) if path is not None else DATA_PATH
    return pd.read_csv(path, encoding="utf-8-sig")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *df* with invalid values masked as ``NaN``.

    Missing values are intentionally left as ``NaN`` here; they are imputed
    inside the model pipeline so that training and inference behave the same
    way (the FastAPI ``/predict`` endpoint may also receive missing fields).
    """
    out = df.copy()

    # Normalise text columns (strip stray whitespace / case of booleans).
    # NOTE: object dtype is kept (not pandas "string") because scikit-learn's
    # SimpleImputer cannot handle pd.NA values.
    for col in out.select_dtypes(include="object").columns:
        out[col] = out[col].str.strip()
    for col in ("Damaged_Package_Flag", "Promotion_Active"):
        if col in out.columns:
            out[col] = out[col].str.title()  # 'yes' -> 'Yes'
    if "Package_Size" in out.columns:
        out["Package_Size"] = out["Package_Size"].str.title()

    # --- invalid numeric values -> NaN (repaired later by imputers) --------
    if "Product_Price_INR" in out.columns:
        out.loc[out["Product_Price_INR"] <= 0, "Product_Price_INR"] = float("nan")

    if "Discount_Percent" in out.columns:
        bad = (out["Discount_Percent"] < 0) | (out["Discount_Percent"] > 100)
        out.loc[bad, "Discount_Percent"] = float("nan")

    if "Customer_Rating_PreOrder" in out.columns:
        bad = (out["Customer_Rating_PreOrder"] < 1) | (out["Customer_Rating_PreOrder"] > 5)
        out.loc[bad, "Customer_Rating_PreOrder"] = float("nan")

    for col in ("Actual_Delivery_Days", "Expected_Delivery_Days"):
        if col in out.columns:
            out.loc[out[col] < 0, col] = float("nan")

    if {"Previous_Returns", "Previous_Orders"}.issubset(out.columns):
        impossible = out["Previous_Returns"] > out["Previous_Orders"]
        out.loc[impossible, "Previous_Returns"] = float("nan")

    # --- validate the target -----------------------------------------------
    if TARGET_COLUMN in out.columns:
        valid = out[TARGET_COLUMN].isin(["Returned", "Not Returned"])
        out = out[valid].reset_index(drop=True)

    return out


def load_and_clean_data(path: str | Path | None = None) -> pd.DataFrame:
    """Convenience helper: load the CSV and apply :func:`clean_data`."""
    return clean_data(load_raw_data(path))


__all__ = ["load_raw_data", "clean_data", "load_and_clean_data", "CATEGORICAL_COLUMNS"]
