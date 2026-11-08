# Changelog

All notable changes to the Product Return Prediction system are documented
here. The format follows [Keep a Changelog](https://keepachangelog.com/) and
versioning is semantic.

## [1.0.0] - 2026-11-08

### Added
- Data loading and cleaning pipeline (missing values, invalid prices /
  discounts / ratings, whitespace and case normalisation).
- Feature engineering: `Effective_Price_INR`, `Delivery_Delay_Days`,
  `Is_Late_Delivery`, `Customer_Return_Rate`.
- Training script comparing Logistic Regression, Random Forest and
  Gradient Boosting; best model (Logistic Regression, test F1 0.3467,
  ROC-AUC 0.5750) persisted to `artifacts/`.
- FastAPI service with `GET /health` and `POST /predict` endpoints,
  Pydantic validation, configurable decision threshold.
- pytest suite (28 tests) covering cleaning, features, model artifact and
  API contract.
- Dockerfile (python:3.13-slim, non-root user, HEALTHCHECK on /health).
- GitHub Actions CI/CD pipeline: lint → retrain → test → build → smoke test
  → publish to GitHub Container Registry.
