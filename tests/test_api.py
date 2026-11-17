import os
import tempfile

import mlflow
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

import train

X, y = train.load_data()
MODEL_DIR = os.path.join(tempfile.mkdtemp(), "model")
mlflow.sklearn.save_model(
    train.build_pipeline(LogisticRegression(max_iter=2000)).fit(X, y), MODEL_DIR, metadata={"threshold": 0.3}
)
os.environ["MODEL_URI"] = MODEL_DIR

import api  # noqa: E402  (reads MODEL_URI at import)


@pytest.fixture(scope="module")
def client():
    with TestClient(api.app) as c:
        yield c


def rows(n=3):
    return X.head(n).to_dict("records")


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}


def test_predict_batch(client):
    body = client.post("/predict", json={"applicants": rows()}).json()
    assert body["threshold"] == 0.3
    assert len(body["predictions"]) == 3
    for p in body["predictions"]:
        assert 0 <= p["default_probability"] <= 1
        assert p["decision"] == ("manual_review" if p["default_probability"] >= 0.3 else "approve")


def test_predictions_match_model(client):
    body = client.post("/predict", json={"applicants": rows(5)}).json()
    expected = api.state["model"].predict_proba(X.head(5))[:, 1]
    np.testing.assert_allclose([p["default_probability"] for p in body["predictions"]], expected, atol=1e-4)


@pytest.mark.parametrize("change", [
    {"purpose": "spaceship"},          # unknown category
    {"credit_amount": -100},           # out of range
    {"age": 12},                       # out of range
    {"unexpected_field": 1},           # extra field
])
def test_invalid_input_rejected(client, change):
    applicant = {**rows(1)[0], **change}
    assert client.post("/predict", json={"applicants": [applicant]}).status_code == 422


def test_empty_batch_rejected(client):
    assert client.post("/predict", json={"applicants": []}).status_code == 422


def test_best_threshold_prefers_catching_defaults():
    y_true = np.array([0, 0, 0, 0, 1, 1])
    proba = np.array([0.1, 0.2, 0.3, 0.45, 0.4, 0.9])
    # missing the 0.4 default costs 5, flagging the 0.45 good applicant costs 1 -> threshold <= 0.4
    assert train.best_threshold(y_true, proba) <= 0.4
    assert train.total_cost(y_true, proba, 0.4) == 1
