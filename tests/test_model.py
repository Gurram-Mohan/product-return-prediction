"""Tests for the trained model artifact (Task 3)."""

import joblib
import pytest

from src.config import MODEL_INFO_PATH, MODEL_PATH
from src.data_loader import load_and_clean_data
from src.features import ALL_FEATURES, engineer_features


@pytest.fixture(scope="module")
def pipeline():
    assert MODEL_PATH.exists(), (
        "artifacts/model.pkl is missing - run `python -m src.train` first"
    )
    return joblib.load(MODEL_PATH)


@pytest.fixture(scope="module")
def info():
    assert MODEL_INFO_PATH.exists(), (
        "artifacts/model_info.json is missing - run `python -m src.train` first"
    )
    import json

    with open(MODEL_INFO_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def test_model_info_contains_version_and_best_model(info):
    assert info["model_version"]
    assert info["best_model"] in {
        "Logistic Regression",
        "Random Forest",
        "Gradient Boosting",
    }


def test_model_info_records_metrics_for_all_candidates(info):
    names = {m["model"] for m in info["metrics"]}
    assert {
        "Logistic Regression",
        "Random Forest",
        "Gradient Boosting",
    }.issubset(names)
    for metric in info["metrics"]:
        assert 0.0 <= metric["accuracy"] <= 1.0
        assert 0.0 <= metric["f1_score"] <= 1.0


def test_pipeline_predicts_valid_labels(pipeline):
    df = engineer_features(load_and_clean_data().head(30))
    X = df[ALL_FEATURES]
    preds = pipeline.predict(X)
    assert set(preds).issubset({0, 1})


def test_pipeline_probabilities_are_bounded(pipeline):
    df = engineer_features(load_and_clean_data().head(30))
    probs = pipeline.predict_proba(df[ALL_FEATURES])
    assert ((probs >= 0) & (probs <= 1)).all()
    assert (probs.sum(axis=1) == pytest.approx(1.0))


def test_pipeline_handles_missing_and_unknown_values(pipeline):
    """One row with missing numerics and an unseen category must not crash."""
    df = engineer_features(load_and_clean_data().head(1))
    X = df[ALL_FEATURES].copy()
    X.loc[X.index[0], "Discount_Percent"] = float("nan")
    X.loc[X.index[0], "Product_Category"] = "Groceries"  # unseen in training
    pred = pipeline.predict(X)
    assert pred[0] in {0, 1}
