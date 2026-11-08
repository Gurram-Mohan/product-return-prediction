"""Train, evaluate and persist the product-return classifiers.

The script:
1. loads + cleans the dataset and engineers features,
2. trains three classifiers (Logistic Regression, Random Forest, Gradient
   Boosting) on a stratified 80/20 split,
3. evaluates each on the held-out test set (accuracy, precision, recall,
   F1, ROC-AUC),
4. saves the best pipeline (preprocessor + estimator) to
   ``artifacts/model.pkl`` and a metadata JSON next to it.

Run with:  python -m src.train
"""

import json
import platform
from datetime import datetime, timezone

import joblib
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    ARTIFACTS_DIR,
    DATA_PATH,
    MODEL_INFO_PATH,
    MODEL_PATH,
    MODEL_VERSION,
    RANDOM_STATE,
    TARGET_COLUMN,
    TARGET_MAPPING,
    TEST_SIZE,
)
from src.data_loader import load_and_clean_data
from src.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, engineer_features


def build_preprocessor() -> ColumnTransformer:
    """Impute + scale numeric columns, impute + one-hot encode categoricals."""
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ]
    )


def get_candidate_models() -> dict:
    """The classification models compared by this training run.

    Hyperparameters were selected with 5-fold cross-validated ROC-AUC/F1 on
    the training data; the settings below favour generalisation because the
    dataset signal is weak (best achievable CV ROC-AUC ~ 0.57).
    """
    return {
        "Logistic Regression": LogisticRegression(
            C=0.2,  # v1.1.0: relaxed from 0.1 after CV comparison
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=2,
            random_state=RANDOM_STATE,
        ),
    }


def evaluate(model_name: str, y_true: pd.Series, y_pred, y_prob) -> dict:
    """Compute the evaluation metrics for one model."""
    return {
        "model": model_name,
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred)), 4),
        "recall": round(float(recall_score(y_true, y_pred)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
    }


def main() -> dict:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- data
    df = load_and_clean_data(DATA_PATH)
    y = df[TARGET_COLUMN].map(TARGET_MAPPING)
    X_raw = df.drop(columns=[TARGET_COLUMN, "Order_ID"])
    X = engineer_features(X_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"Dataset: {len(df)} rows | train={len(X_train)} test={len(X_test)}")
    print(f"Return rate: train={y_train.mean():.3f} test={y_test.mean():.3f}\n")

    # ------------------------------------------------------------- training
    results = []
    fitted_pipelines = {}
    for name, estimator in get_candidate_models().items():
        pipe = Pipeline(
            steps=[("preprocessor", build_preprocessor()), ("classifier", estimator)]
        )
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        metrics = evaluate(name, y_test, y_pred, y_prob)
        results.append(metrics)
        fitted_pipelines[name] = pipe
        print(
            f"{name:<20} acc={metrics['accuracy']:.4f}  precision={metrics['precision']:.4f}"
            f"  recall={metrics['recall']:.4f}  f1={metrics['f1_score']:.4f}"
            f"  roc_auc={metrics['roc_auc']:.4f}"
        )

    # ------------------------------------------------------- model selection
    best = max(results, key=lambda m: (m["f1_score"], m["roc_auc"]))
    best_name = best["model"]
    best_pipe = fitted_pipelines[best_name]
    print(f"\nBest model: {best_name} (F1={best['f1_score']:.4f}, "
          f"ROC-AUC={best['roc_auc']:.4f})")

    # ------------------------------------------------------------ artifacts
    joblib.dump(best_pipe, MODEL_PATH)
    y_pred_best = best_pipe.predict(X_test)
    report = classification_report(
        y_test, y_pred_best, target_names=["Not Returned", "Returned"]
    )
    conf_matrix = confusion_matrix(y_test, y_pred_best).tolist()

    info = {
        "model_version": MODEL_VERSION,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "best_model": best_name,
        "dataset": {
            "path": str(DATA_PATH.name),
            "rows": int(len(df)),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "return_rate": round(float(y.mean()), 4),
        },
        "metrics": results,
        "best_model_classification_report": report,
        "best_model_confusion_matrix": conf_matrix,
        "features": {
            "numeric": NUMERIC_FEATURES,
            "categorical": CATEGORICAL_FEATURES,
        },
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
        },
    }
    with open(MODEL_INFO_PATH, "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2)

    print(f"\nSaved model pipeline -> {MODEL_PATH}")
    print(f"Saved model metadata -> {MODEL_INFO_PATH}")
    print("\nClassification report (best model):\n" + report)
    return info


if __name__ == "__main__":
    main()
