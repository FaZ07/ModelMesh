"""Threaded auto-retrain.

When drift is significant the operator (or an automated trigger) submits a
fresh labelled CSV. We refit a *clone* of the deployed estimator, evaluate it
on a held-out split, and only promote the challenger if it beats the champion.
This is the "shadow → promote" pattern, done synchronously in a worker thread
(a production system would swap the ThreadPoolExecutor for Celery/Redis — the
job interface here is identical).
"""
from __future__ import annotations

import io
import threading
import time
import uuid

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split

from . import registry, store
from .config import MODEL_DIR
from .drift import FeatureReference

_jobs: dict[str, dict] = {}
_lock = threading.Lock()


def get_job(job_id: str) -> dict | None:
    return _jobs.get(job_id)


def start_retrain(model_id: str, labelled_csv: bytes, target: str) -> str:
    job_id = uuid.uuid4().hex[:10]
    _jobs[job_id] = {
        "job_id": job_id, "model_id": model_id, "status": "queued",
        "started": time.time(), "log": [], "result": None,
    }
    threading.Thread(
        target=_run, args=(job_id, model_id, labelled_csv, target), daemon=True
    ).start()
    return job_id


def _say(job: dict, msg: str) -> None:
    job["log"].append({"t": round(time.time() - job["started"], 2), "msg": msg})


def _run(job_id: str, model_id: str, csv_bytes: bytes, target: str) -> None:
    job = _jobs[job_id]
    job["status"] = "running"
    try:
        meta = store.get_model(model_id)
        if not meta:
            raise ValueError("unknown model id")
        est = registry._estimator(model_id)

        _say(job, "Loading fresh labelled data")
        df = pd.read_csv(io.BytesIO(csv_bytes))
        if target not in df.columns:
            raise ValueError(f"target column '{target}' not in CSV")
        feats = meta["features"]
        X = df[feats].to_numpy(dtype="float64")
        y = df[target].to_numpy()

        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)

        _say(job, f"Scoring champion (v{meta['version']}) on holdout")
        champ_score = _score(est, Xte, yte, meta["task"])

        _say(job, f"Fitting challenger ({meta['kind']})")
        challenger = clone(est)
        challenger.fit(Xtr, ytr)
        chall_score = _score(challenger, Xte, yte, meta["task"])

        promote = chall_score > champ_score
        _say(job, f"champion={champ_score:.4f}  challenger={chall_score:.4f}")

        if promote:
            with _lock:
                joblib.dump(challenger, MODEL_DIR / f"{model_id}.joblib")
                registry.invalidate(model_id)
                new_ref = {
                    f: FeatureReference.fit(df[f].to_numpy(dtype="float64")).__dict__
                    for f in feats
                }
                store.update_model(
                    model_id,
                    version=meta["version"] + 1,
                    reference=new_ref,
                    metrics={"holdout_score": round(chall_score, 4),
                             "previous": round(champ_score, 4)},
                )
            _say(job, f"✅ Promoted → v{meta['version'] + 1}")
        else:
            _say(job, "✋ Challenger did not beat champion — kept champion")

        job["result"] = {
            "promoted": promote,
            "champion_score": round(champ_score, 4),
            "challenger_score": round(chall_score, 4),
            "new_version": meta["version"] + (1 if promote else 0),
        }
        job["status"] = "done"
    except Exception as e:  # noqa: BLE001
        _say(job, f"ERROR: {e}")
        job["status"] = "failed"
        job["result"] = {"error": str(e)}


def _score(est, X: np.ndarray, y: np.ndarray, task: str) -> float:
    pred = est.predict(X)
    if task == "classification":
        return float((pred == y).mean())
    # regression → R² (higher is better, like accuracy)
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) + 1e-12
    return 1.0 - ss_res / ss_tot
