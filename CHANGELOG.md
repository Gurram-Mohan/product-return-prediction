# Changelog

All notable changes to the Product Return Prediction system are documented
here. The format follows [Keep a Changelog](https://keepachangelog.com/) and
versioning is semantic.

## [1.1.0] - 2026-11-08

### Changed
- Model configuration: Logistic Regression regularization relaxed
  (`C` 0.1 -> 0.2) after cross-validated comparison (CV ROC-AUC 0.5701 vs
  0.5712 - statistically equivalent, slightly smoother decision boundary).
- `MODEL_VERSION` bumped to 1.1.0; `/health` now reports the new version so
  a running container can be tied to its source commit and image tag.

### Added
- `GET /model-info` endpoint exposing training metadata (metrics of all
  candidate models, feature lists, training environment) from the running
  service.
- Test coverage for the new endpoint (29 tests total).

### Fixed
- Exact dependency pinning so the Docker image can load the committed
  scikit-learn model artifact (caught by the CI smoke test in run #1).

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
