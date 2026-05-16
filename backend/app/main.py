"""ModelMesh FastAPI service — serve any sklearn model, watch it drift, retrain.

GET  /api/models                       list registered models
POST /api/models                       register (model.joblib + train.csv)
GET  /api/models/{id}                  model card + live drift summary
POST /api/models/{id}/predict          inference (logged for drift analysis)
GET  /api/models/{id}/drift            full per-feature drift report
POST /api/models/{id}/retrain          submit labelled CSV → threaded retrain
GET  /api/jobs/{job_id}                retrain job status
"""
from __future__ import annotations

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from . import __version__, registry, store
from .monitor import drift_report
from .retrain import get_job, start_retrain
from .schemas import (
    JobStatus,
    ModelCard,
    PredictRequest,
    PredictResponse,
    RetrainResponse,
)

app = FastAPI(title="ModelMesh", version=__version__)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


@app.on_event("startup")
def _startup() -> None:
    store.init_db()


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "version": __version__, "models": len(store.list_models())}


@app.get("/api/models", response_model=list[ModelCard])
def models() -> list[ModelCard]:
    return [ModelCard(**m) for m in store.list_models()]


@app.post("/api/models", response_model=ModelCard)
async def register_model(
    name: str = Form(...),
    target: str = Form(""),
    model: UploadFile = File(...),
    train_csv: UploadFile = File(...),
) -> ModelCard:
    try:
        meta = registry.register(
            name, await model.read(), await train_csv.read(), target or None
        )
    except registry.RegistryError as e:
        raise HTTPException(400, str(e))
    full = store.get_model(meta["id"])
    return ModelCard(**full)


@app.get("/api/models/{model_id}", response_model=ModelCard)
def model_card(model_id: str) -> ModelCard:
    m = store.get_model(model_id)
    if not m:
        raise HTTPException(404, "unknown model id")
    return ModelCard(**m)


@app.post("/api/models/{model_id}/predict", response_model=PredictResponse)
def predict(model_id: str, req: PredictRequest) -> PredictResponse:
    try:
        y, conf = registry.predict(model_id, req.features)
    except registry.RegistryError as e:
        raise HTTPException(400, str(e))
    return PredictResponse(model_id=model_id, prediction=y, confidence=conf)


@app.get("/api/models/{model_id}/drift")
def drift(model_id: str) -> dict:
    try:
        return drift_report(model_id)
    except KeyError:
        raise HTTPException(404, "unknown model id")


@app.post("/api/models/{model_id}/retrain", response_model=RetrainResponse)
async def retrain(
    model_id: str,
    target: str = Form(...),
    labelled_csv: UploadFile = File(...),
) -> RetrainResponse:
    if not store.get_model(model_id):
        raise HTTPException(404, "unknown model id")
    job_id = start_retrain(model_id, await labelled_csv.read(), target)
    return RetrainResponse(job_id=job_id, status="queued")


@app.get("/api/jobs/{job_id}", response_model=JobStatus)
def job_status(job_id: str) -> JobStatus:
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "unknown job id")
    return JobStatus(**job)
