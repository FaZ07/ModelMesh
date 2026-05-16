"""Model registry — register any scikit-learn estimator, auto-derive its
input schema and per-feature reference distribution, and serve predictions.

Accepts a pickled/joblib estimator + the training feature CSV. The CSV is
used only to (a) learn feature names and (b) fit drift reference histograms —
the model itself is treated as an opaque `predict` black box, so anything with
a scikit-learn-style API works.
"""
from __future__ import annotations

import io
import uuid

import joblib
import numpy as np
import pandas as pd

from . import store
from .config import MODEL_DIR
from .drift import FeatureReference


class RegistryError(RuntimeError):
    pass


def _load_estimator(blob: bytes):
    try:
        return joblib.load(io.BytesIO(blob))
    except Exception as e:  # noqa: BLE001
        raise RegistryError(f"could not unpickle estimator: {e}") from e


def register(name: str, model_blob: bytes, train_csv: bytes, target: str | None) -> dict:
    est = _load_estimator(model_blob)
    if not hasattr(est, "predict"):
        raise RegistryError("estimator has no .predict() method")

    df = pd.read_csv(io.BytesIO(train_csv))
    if target and target in df.columns:
        df = df.drop(columns=[target])
    features = [c for c in df.columns]
    if not features:
        raise RegistryError("training CSV has no feature columns")

    # Validate the model actually runs on this schema.
    sample = df.iloc[:5].to_numpy(dtype="float64")
    try:
        est.predict(sample)
    except Exception as e:  # noqa: BLE001
        raise RegistryError(f"model.predict failed on training schema: {e}") from e

    task = "classification" if hasattr(est, "predict_proba") else "regression"
    reference = {
        f: FeatureReference.fit(df[f].to_numpy(dtype="float64")).__dict__
        for f in features
    }

    model_id = uuid.uuid4().hex[:12]
    joblib.dump(est, MODEL_DIR / f"{model_id}.joblib")

    meta = {
        "id": model_id,
        "name": name,
        "kind": type(est).__name__,
        "features": features,
        "task": task,
        "reference": reference,
        "metrics": None,
        "version": 1,
    }
    store.insert_model(meta)
    return {k: meta[k] for k in ("id", "name", "kind", "features", "task")}


# ── inference ────────────────────────────────────────────────────────────
_cache: dict[str, object] = {}


def _estimator(model_id: str):
    if model_id not in _cache:
        path = MODEL_DIR / f"{model_id}.joblib"
        if not path.exists():
            raise RegistryError("model artifact missing on disk")
        _cache[model_id] = joblib.load(path)
    return _cache[model_id]


def invalidate(model_id: str) -> None:
    _cache.pop(model_id, None)


def predict(model_id: str, feature_map: dict) -> tuple[float, float | None]:
    meta = store.get_model(model_id)
    if not meta:
        raise RegistryError("unknown model id")
    missing = [f for f in meta["features"] if f not in feature_map]
    if missing:
        raise RegistryError(f"missing features: {missing}")

    est = _estimator(model_id)
    x = np.array([[float(feature_map[f]) for f in meta["features"]]], dtype="float64")
    y = est.predict(x)[0]

    conf: float | None = None
    if hasattr(est, "predict_proba"):
        conf = float(np.max(est.predict_proba(x)[0]))

    store.log_prediction(model_id, feature_map, float(y), conf)
    return float(y), conf
