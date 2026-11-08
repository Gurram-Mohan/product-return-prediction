# E-Commerce Product Return Prediction System (with CI/CD)

End-to-end machine learning system that predicts whether a purchased product
is likely to be **returned**, served through a **FastAPI** microservice and
shipped through an automated **GitHub Actions CI/CD pipeline** that builds a
**Docker image** and publishes it to **GitHub Container Registry (ghcr.io)**.

![Pipeline](https://img.shields.io/badge/CI%2FCD-GitHub_Actions-2088FF?logo=githubactions&logoColor=white)
![API](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)
![Registry](https://img.shields.io/badge/Registry-GHCR-2496ED?logo=docker&logoColor=white)

---

## 1. Problem

Given an e-commerce order (product category, price, discount, expected vs.
actual delivery time, pre-order customer rating, previous return history,
payment method, package attributes), predict:

> **Returned** or **Not Returned**

## 2. Architecture

```
                       ┌──────────────────────────────────────────────┐
                       │                GitHub Repository             │
                       │                                              │
  git push ──────────► │  src/          training + feature engineering│
                       │  app/          FastAPI service               │
                       │  tests/        pytest suite (28 tests)       │
                       │  artifacts/    trained model + metadata      │
                       │  Dockerfile    production image recipe       │
                       └───────────────────┬──────────────────────────┘
                                           │ on: push to main
                                           ▼
                       ┌──────────────────────────────────────────────┐
                       │        GitHub Actions CI/CD pipeline         │
                       │                                              │
                       │  Job 1: Validate & Test                      │
                       │    • pip install dependencies               │
                       │    • ruff lint (code validation)             │
                       │    • python -m src.train (train + evaluate)  │
                       │    • pytest (automated testing)              │
                       │                                              │
                       │  Job 2: Build & Publish  (needs: quality)    │
                       │    • docker build                            │
                       │    • smoke test (/health + /predict)         │
                       │    • docker push → ghcr.io/<owner>/<repo>    │
                       └───────────────────┬──────────────────────────┘
                                           │
                                           ▼
                       ┌──────────────────────────────────────────────┐
                       │   ghcr.io/<owner>/<product-return-prediction>│
                       │   tags: latest · <version> · sha-<commit>    │
                       └───────────────────┬──────────────────────────┘
                                           │ docker pull / docker run
                                           ▼
                       ┌──────────────────────────────────────────────┐
                       │  Docker Desktop / any Docker host            │
                       │  uvicorn FastAPI on :8000                    │
                       │  GET /health   POST /predict                 │
                       └──────────────────────────────────────────────┘
```

## 3. Repository layout

```
.
├── .github/workflows/ci-cd.yml   # CI/CD pipeline (lint → test → build → GHCR)
├── app/                          # FastAPI service
│   ├── main.py                   #   /health, /predict endpoints
│   └── schemas.py                #   Pydantic request/response models
├── src/                          # ML source package
│   ├── config.py                 #   paths, constants, MODEL_VERSION
│   ├── data_loader.py            #   loading + cleaning (Task 1)
│   ├── features.py               #   feature engineering (Task 2)
│   └── train.py                  #   training + evaluation (Task 3)
├── tests/                        # pytest suite (Task 6)
│   ├── test_data.py              #   cleaning rules
│   ├── test_features.py          #   engineered-feature math
│   ├── test_model.py             #   artifact contract
│   └── test_api.py               #   API contract incl. error paths
├── artifacts/                    # committed trained model (Task 5)
│   ├── model.pkl                 #   best pipeline (preprocessor + model)
│   └── model_info.json           #   metrics, versions, feature lists
├── data/                         # raw dataset (CSV)
├── Dockerfile                    # production image (python:3.13-slim)
├── requirements.txt              # runtime dependencies
├── requirements-dev.txt          # lint + test dependencies
└── pyproject.toml                # ruff + pytest configuration
```

## 4. Data preprocessing (Task 1)

The dataset (`data/Section7_Q4_Ecommerce_Product_Return_CICD.csv`, 1200
orders, target rate 22.6 % returned) is cleaned by `src/data_loader.py`:

| Issue found in data | Treatment |
|---|---|
| UTF-8 BOM in CSV header | read with `encoding="utf-8-sig"` |
| 36 missing values in `Product_Price_INR`, `Discount_Percent`, `Actual_Delivery_Days`, `Customer_Rating_PreOrder`, `Payment_Method` | left as `NaN` → **median imputer** (numeric) / `"Missing"` constant imputer (categorical) inside the model pipeline |
| 3 rows with `Product_Price_INR <= 0` | invalidated → `NaN` → imputed |
| 2 rows with `Discount_Percent > 100` | invalidated → `NaN` → imputed |
| Ratings outside [1, 5], negative delivery days, returns > orders | invalidated → `NaN` → imputed |
| Text columns | whitespace stripped, case normalised (`yes`→`Yes`) |
| Categorical encoding | `OneHotEncoder(handle_unknown="ignore")` in the pipeline |
| Numeric scaling | `StandardScaler` after imputation |
| Class imbalance (77/23) | `class_weight="balanced"` |

## 5. Feature engineering (Task 2)

`src/features.py` — applied identically at training and inference time:

| Feature | Formula | Rationale |
|---|---|---|
| `Effective_Price_INR` | `price × (1 − discount% / 100)` | what the customer actually pays |
| `Delivery_Delay_Days` | `actual − expected delivery days` | > 0 means the parcel was late |
| `Is_Late_Delivery` | `1 if delay > 0 else 0` | simple late-delivery signal |
| `Customer_Return_Rate` | `previous_returns / previous_orders` | customer return frequency |

## 6. Models & evaluation (Task 3)

Three classifiers trained on a stratified 80/20 split (`random_state=42`),
evaluated on the held-out test set (v1.1.0 results):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** (C=0.2, balanced) | 0.6042 | 0.2796 | 0.4815 | **0.3537** | **0.5746** |
| Random Forest (300 trees, d=6) | 0.7000 | 0.3393 | 0.3519 | 0.3455 | 0.5593 |
| Gradient Boosting (lr=0.05, d=2) | 0.7625 | 0.2857 | 0.0370 | 0.0656 | 0.5708 |

**Best model: Logistic Regression** (highest F1 and ROC-AUC) → saved to
`artifacts/model.pkl` with its preprocessor; full metrics in
`artifacts/model_info.json` — also exposed live by `GET /model-info`.

> **Honest note on performance:** the dataset carries a weak, noisy signal —
> the strongest predictors are `Damaged_Package_Flag` (40.6 % return rate vs
> 21.5 %), `Product_Category` (Fashion 31 %) and delivery duration. 5-fold
> cross-validation puts the achievable ceiling near ROC-AUC ≈ 0.57, which the
> selected model reaches. The pipeline is deliberately balanced
> (`class_weight="balanced"`) so recall of true returns is prioritised over
> raw accuracy, which majority-class prediction would inflate.

## 7. FastAPI service (Task 4)

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000   # local
# interactive docs: http://localhost:8000/docs
```

### `GET /health`
```json
{"status":"healthy","model_loaded":true,"model":"Logistic Regression",
 "model_version":"1.0.0","timestamp":"2026-11-08T12:29:48+00:00"}
```

### `POST /predict`
```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "product_category": "Electronics", "product_price_inr": 22055.97,
    "discount_percent": 21.6, "expected_delivery_days": 9,
    "actual_delivery_days": 12, "customer_rating_preorder": 4.4,
    "previous_returns": 0, "previous_orders": 33,
    "payment_method": "Card", "damaged_package_flag": "No",
    "package_size": "Large", "promotion_active": "Yes",
    "order_id": "ECOM-0001"
  }'
```
```json
{"order_id":"ECOM-0001","prediction":"Not Returned",
 "return_probability":0.3923,"risk_level":"Low",
 "model":"Logistic Regression","model_version":"1.0.0","threshold":0.5}
```

Optional fields may be omitted or `null` — the same imputers used in
training repair them at inference time. Invalid payloads (negative price,
discount > 100, rating outside 1–5, missing required fields) return HTTP
`422` with a detailed error body.

The decision threshold is configurable per deployment:
`PREDICTION_THRESHOLD=0.4 docker run ...`

## 8. CI/CD pipeline (Tasks 6–7)

`.github/workflows/ci-cd.yml` runs on every **push to `main`**, every **pull
request** and via manual **workflow_dispatch**:

1. **Validate & Test** job — install dependencies (`pip install -r
   requirements-dev.txt`), lint with **ruff**, retrain + evaluate the models
   (`python -m src.train`), run the **pytest** suite (28 tests).
2. **Build & Publish** job (only after a green quality gate on `main`) —
   `docker build` the image, **smoke test** it by starting the container and
   hitting `/health` + `/predict`, then `docker push` three tags to **GHCR**:
   `latest`, the semantic model version, and `sha-<commit>`.

Authentication to GHCR uses the built-in `GITHUB_TOKEN` with
`packages: write` permission — no manual secrets required.

## 9. Run the Docker image (Tasks 7 & 9)

```bash
# public repo: no login needed
docker pull ghcr.io/<github-username>/product-return-prediction:latest

docker run -d --name return-prediction-api -p 8000:8000 \
  ghcr.io/<github-username>/product-return-prediction:latest

curl http://localhost:8000/health
```

The image includes a `HEALTHCHECK` and runs as a non-root user. To watch the
container in **Docker Desktop**, open the *Containers* tab — the image shows
a green *healthy* badge once `/health` responds.

## 10. Local development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

python -m src.train    # retrain + evaluate models
pytest -v              # run the test suite
ruff check .           # lint
uvicorn app.main:app --reload   # dev server
```

## 11. Release history

See [CHANGELOG.md](CHANGELOG.md). Every change to `main` automatically
produces a new immutable image tagged `sha-<commit>` on GHCR.
