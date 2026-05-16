"""End-to-end backend proof: register -> predict (with injected drift) ->
drift report -> retrain. Run from backend/ with the venv python.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.main import app  # noqa: E402

SAMPLE = Path(__file__).resolve().parents[1] / "sample"
c = TestClient(app)


def main() -> None:
    assert c.get("/api/health").json()["status"] == "ok"

    # 1 · register
    with open(SAMPLE / "model.joblib", "rb") as m, open(SAMPLE / "train.csv", "rb") as t:
        r = c.post(
            "/api/models",
            data={"name": "breast-cancer-rf", "target": "target"},
            files={"model": ("model.joblib", m), "train_csv": ("train.csv", t)},
        )
    assert r.status_code == 200, r.text
    mid = r.json()["id"]
    feats = r.json()["features"]
    print(f"[OK] registered model {mid}  ({len(feats)} features, {r.json()['kind']})")

    df = pd.read_csv(SAMPLE / "train.csv").drop(columns=["target"])
    rng = np.random.default_rng(0)

    # 2 · in-distribution traffic
    for i in range(140):
        row = df.iloc[rng.integers(0, len(df))].to_dict()
        c.post(f"/api/models/{mid}/predict",
               json={"features": {k: float(v) for k, v in row.items()}})
    base = c.get(f"/api/models/{mid}/drift").json()
    print(f"[OK] in-dist  max_psi={base['overall']['max_psi']:.4f} "
          f"status={base['overall']['status']}")

    # 3 · inject covariate shift on feature 0
    f0 = feats[0]
    for i in range(160):
        row = df.iloc[rng.integers(0, len(df))].to_dict()
        row[f0] = row[f0] * 1.8 + 6.0
        c.post(f"/api/models/{mid}/predict",
               json={"features": {k: float(v) for k, v in row.items()}})
    drift = c.get(f"/api/models/{mid}/drift").json()
    top = drift["features"][0]
    print(f"[OK] drifted  max_psi={drift['overall']['max_psi']:.4f} "
          f"status={drift['overall']['status']}  "
          f"top={top['feature']} psi={top['psi']:.3f} js={top['js']:.3f}")
    assert drift["overall"]["status"] == "significant", "drift not detected!"
    assert top["feature"] == f0, "wrong feature flagged"

    # 4 · shadow-promote retrain
    with open(SAMPLE / "fresh_labelled.csv", "rb") as f:
        r = c.post(f"/api/models/{mid}/retrain",
                   data={"target": "target"},
                   files={"labelled_csv": ("fresh.csv", f)})
    job = r.json()["job_id"]
    import time
    for _ in range(60):
        js = c.get(f"/api/jobs/{job}").json()
        if js["status"] in ("done", "failed"):
            break
        time.sleep(0.3)
    assert js["status"] == "done", js
    res = js["result"]
    print(f"[OK] retrain  champion={res['champion_score']} "
          f"challenger={res['challenger_score']} promoted={res['promoted']}")

    print("\n[PASS] ModelMesh backend pipeline verified end-to-end.")


if __name__ == "__main__":
    main()
