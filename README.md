# Credit Risk Scoring: from training to a REST API

An end-to-end MLOps project: train and compare credit-risk models, track every run in **MLflow**, promote the best
one through the **model registry**, and serve it with a validated **FastAPI** REST API. Tests run in **GitHub Actions**,
and everything runs in **Docker Compose**.

```
train.py ──► 3 models, 5-fold CV ──► MLflow tracking (params, metrics, model artifacts)
                                         │
                                         ▼  best CV AUC
                              MLflow registry: credit-risk@production
                                         │
api.py (FastAPI) ◄──── loads at startup ─┘  ──►  POST /predict  →  default probability + decision
```

## Data

[German Credit](https://www.openml.org/d/31) (OpenML `credit-g`): 1,000 loan applicants with 20 attributes, labelled
good or bad risk. It comes with a cost matrix: **approving a bad applicant costs 5x more than rejecting a good one**.

Two protected attributes are excluded from the features: `personal_status` (mixes sex and marital status) and
`foreign_worker`. Credit scoring is classed as high-risk under the EU AI Act, so this is a deliberate choice, not
data cleaning. `age` is still used and is the obvious next attribute to audit.

## Results

Selection is done on cross-validated AUC from the training split; the 20% test split is only used for reporting.

| Model | CV ROC AUC | Test ROC AUC | Test recall (bad) | Cost per applicant |
|---|---|---|---|---|
| **Random forest** (promoted) | **0.789** | 0.799 | 0.87 | **0.535** |
| Hist. gradient boosting | 0.785 | 0.794 | 0.82 | 0.550 |
| Logistic regression | 0.766 | 0.805 | 0.87 | 0.595 |
| Approve everyone (baseline) | | | 0.00 | 1.500 |

The decision threshold is not 0.5: it is chosen on out-of-fold predictions to minimise the business cost, and lands at
**0.25** for the promoted model. That cuts the expected cost per applicant by about 64% versus approving everyone.

Logistic regression has the best test AUC but the worst CV AUC; with only 200 test rows, that gap is noise, and
picking it because of the test score would be leakage.

## Run it

```bash
docker compose up -d mlflow
docker compose run --rm train      # trains, logs to MLflow, registers the best model
docker compose up -d api
```

- MLflow UI: http://localhost:5000 (runs, metrics, registered model)
- API docs (Swagger): http://localhost:8000/docs

```bash
curl localhost:8000/model-info     # threshold, model run id, valid values for each categorical field
```

## API

| Endpoint | Description |
|---|---|
| `GET /health` | liveness + model loaded |
| `GET /model-info` | model run id, threshold, allowed category values |
| `POST /predict` | batch of 1 to 1,000 applicants → default probability + `approve` / `manual_review` |

Input is validated at the boundary: numeric ranges, no extra fields, and categorical values checked against the
categories the model was trained on (the one-hot encoder would otherwise silently ignore an unknown value).
High-risk applicants are routed to **manual review**, never auto-rejected, since GDPR art. 22 restricts fully
automated decisions with legal effect.

## Tests

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest -q
```

9 tests: health, batch prediction, API output equal to the model's own `predict_proba`, rejection of invalid input
(unknown category, out-of-range values, extra field, empty batch), and the cost-based threshold logic.
They run on every push via [GitHub Actions](.github/workflows/ci.yml).
