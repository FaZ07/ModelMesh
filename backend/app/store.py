"""SQLite persistence — model registry metadata + every prediction logged.

Logging every prediction with its feature vector is what makes post-deployment
drift analysis possible at all. WAL mode keeps it safe under concurrent
FastAPI + retrain-thread access.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager

from .config import DB_PATH

_lock = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS models (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    kind        TEXT NOT NULL,
    features    TEXT NOT NULL,        -- json list of feature names
    task        TEXT NOT NULL,        -- classification | regression
    reference   TEXT NOT NULL,        -- json {feat: FeatureReference}
    metrics     TEXT,                 -- json holdout metrics
    version     INTEGER DEFAULT 1,
    created_at  REAL NOT NULL,
    active      INTEGER DEFAULT 1
);
CREATE TABLE IF NOT EXISTS predictions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id    TEXT NOT NULL,
    features    TEXT NOT NULL,        -- json {feat: value}
    output      REAL NOT NULL,
    confidence  REAL,
    ts          REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_pred_model ON predictions(model_id, id);
"""


@contextmanager
def conn():
    c = sqlite3.connect(DB_PATH, timeout=30)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    try:
        yield c
        c.commit()
    finally:
        c.close()


def init_db() -> None:
    """Idempotent — safe to call at import and on every startup/worker boot."""
    with conn() as c:
        c.executescript(SCHEMA)


# ── models ───────────────────────────────────────────────────────────────
def insert_model(meta: dict) -> None:
    with _lock, conn() as c:
        c.execute(
            """INSERT INTO models
               (id,name,kind,features,task,reference,metrics,version,created_at,active)
               VALUES (?,?,?,?,?,?,?,?,?,1)""",
            (
                meta["id"], meta["name"], meta["kind"],
                json.dumps(meta["features"]), meta["task"],
                json.dumps(meta["reference"]), json.dumps(meta.get("metrics")),
                meta.get("version", 1), time.time(),
            ),
        )


def get_model(model_id: str) -> dict | None:
    with conn() as c:
        r = c.execute("SELECT * FROM models WHERE id=?", (model_id,)).fetchone()
    if not r:
        return None
    d = dict(r)
    d["features"] = json.loads(d["features"])
    d["reference"] = json.loads(d["reference"])
    d["metrics"] = json.loads(d["metrics"]) if d["metrics"] else None
    return d


def list_models() -> list[dict]:
    with conn() as c:
        rows = c.execute("SELECT * FROM models ORDER BY created_at DESC").fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["features"] = json.loads(d["features"])
        d.pop("reference")
        d["metrics"] = json.loads(d["metrics"]) if d["metrics"] else None
        out.append(d)
    return out


def update_model(model_id: str, **fields) -> None:
    if not fields:
        return
    sets = ", ".join(f"{k}=?" for k in fields)
    vals = [json.dumps(v) if isinstance(v, (dict, list)) else v for v in fields.values()]
    with _lock, conn() as c:
        c.execute(f"UPDATE models SET {sets} WHERE id=?", (*vals, model_id))


# ── predictions ──────────────────────────────────────────────────────────
def log_prediction(model_id: str, features: dict, output: float, conf: float | None) -> None:
    with _lock, conn() as c:
        c.execute(
            "INSERT INTO predictions (model_id,features,output,confidence,ts) VALUES (?,?,?,?,?)",
            (model_id, json.dumps(features), float(output), conf, time.time()),
        )


def recent_predictions(model_id: str, limit: int) -> list[dict]:
    with conn() as c:
        rows = c.execute(
            "SELECT * FROM predictions WHERE model_id=? ORDER BY id DESC LIMIT ?",
            (model_id, limit),
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["features"] = json.loads(d["features"])
        out.append(d)
    return list(reversed(out))


def prediction_count(model_id: str) -> int:
    with conn() as c:
        return c.execute(
            "SELECT COUNT(*) FROM predictions WHERE model_id=?", (model_id,)
        ).fetchone()[0]


# Ensure the schema exists no matter how the app is launched (uvicorn,
# gunicorn worker, TestClient, a bare `import app.store`). CREATE TABLE
# IF NOT EXISTS makes this safe to run every import.
init_db()
