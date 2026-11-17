"""REST API serving the production credit-risk model from the MLflow registry."""
import os
from contextlib import asynccontextmanager
from typing import Literal

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

MODEL_URI = os.getenv("MODEL_URI", "models:/credit-risk@production")
state = {}


@asynccontextmanager
async def lifespan(_app):
    path = mlflow.artifacts.download_artifacts(MODEL_URI)  # resolves registry aliases once
    model = mlflow.sklearn.load_model(path)
    info = mlflow.models.Model.load(path)
    encoder = model[0].named_transformers_["cat"]
    state.update(
        model=model,
        threshold=float(info.metadata["threshold"]),
        run_id=info.run_id,
        # valid values per categorical field, taken from the fitted encoder
        categories={col: sorted(cats) for col, cats in zip(encoder.feature_names_in_, encoder.categories_)},
    )
    yield


app = FastAPI(title="Credit Risk API", version="1.0.0", lifespan=lifespan)


class Applicant(BaseModel):
    model_config = ConfigDict(extra="forbid")

    checking_status: str
    duration: int = Field(gt=0, le=120, description="Loan duration in months")
    credit_history: str
    purpose: str
    credit_amount: float = Field(gt=0, le=1_000_000)
    savings_status: str
    employment: str
    installment_commitment: int = Field(ge=1, le=4, description="Instalment as % of income, bucketed 1-4")
    other_parties: str
    residence_since: int = Field(ge=1, le=4)
    property_magnitude: str
    age: int = Field(ge=18, le=100)
    other_payment_plans: str
    housing: str
    existing_credits: int = Field(ge=1, le=10)
    job: str
    num_dependents: int = Field(ge=0, le=10)
    own_telephone: str


class Batch(BaseModel):
    applicants: list[Applicant] = Field(min_length=1, max_length=1000)


class Prediction(BaseModel):
    default_probability: float
    decision: Literal["approve", "manual_review"]


class PredictionResponse(BaseModel):
    predictions: list[Prediction]
    threshold: float
    model_run_id: str | None


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state}


@app.get("/model-info")
def model_info():
    return {"model_uri": MODEL_URI, "run_id": state["run_id"], "threshold": state["threshold"],
            "categories": state["categories"]}


@app.post("/predict", response_model=PredictionResponse)
def predict(batch: Batch):
    df = pd.DataFrame([a.model_dump() for a in batch.applicants])
    errors = [
        f"{col}: {sorted(set(df[col]) - set(allowed))} not in {allowed}"
        for col, allowed in state["categories"].items()
        if set(df[col]) - set(allowed)
    ]
    if errors:  # the encoder would silently ignore unknown values: reject them instead
        raise HTTPException(status_code=422, detail=errors)

    model = state["model"]
    proba = model.predict_proba(df[list(model.feature_names_in_)])[:, 1]
    # High risk goes to a human, never to an automatic rejection (GDPR art. 22).
    return {
        "predictions": [
            {"default_probability": round(float(p), 4), "decision": "manual_review" if p >= state["threshold"] else "approve"}
            for p in proba
        ],
        "threshold": state["threshold"],
        "model_run_id": state["run_id"],
    }
