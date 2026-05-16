"""Typed API contracts."""
from __future__ import annotations

from pydantic import BaseModel


class ModelCard(BaseModel):
    id: str
    name: str
    kind: str
    features: list[str]
    task: str
    version: int | None = None
    metrics: dict | None = None


class PredictRequest(BaseModel):
    features: dict[str, float]


class PredictResponse(BaseModel):
    model_id: str
    prediction: float
    confidence: float | None = None


class RetrainResponse(BaseModel):
    job_id: str
    status: str


class JobStatus(BaseModel):
    job_id: str
    model_id: str
    status: str
    log: list[dict]
    result: dict | None = None
