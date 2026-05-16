# ModelMesh

![CI](https://github.com/FaZ07/ModelMesh/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

### Local ML model serving + real-time drift detection + auto-retrain — no cloud, no API keys

Models don't fail loudly. They decay *silently* as the world drifts away from
their training data. ModelMesh is the missing layer: register any scikit-learn
model, it auto-serves it behind a typed API, **logs every prediction**, and
continuously measures distribution drift with three industry-standard
detectors. When the model decays, it runs a **shadow-and-promote** retrain.

> The full post-deployment ML lifecycle — *serve → monitor → detect → recover* —
> in one self-contained, offline platform.

---

## Why this is hard (and why it matters)

Training a model is the easy 20%. The expensive 80% is everything after
deployment:

| Real failure | What teams usually do | ModelMesh |
|---|---|---|
| Covariate shift creeps in over weeks | Nobody notices until KPIs drop | PSI + JS divergence per feature, live |
| Sudden concept drift (a pipeline changes upstream) | Found in a post-mortem | ADWIN online change-point detector |
| "Should we retrain?" is a gut call | Retrain blindly, hope it helps | Challenger must beat champion on holdout to promote |
| Drift analysis needs historical inputs | Inputs were never logged | Every prediction persisted with its feature vector |

Every one of these is a line item in real ML-engineer job descriptions.

---

## The three detectors (the core IP)

| Detector | Type | What it catches | Rule of thumb |
|---|---|---|---|
| **PSI** (Population Stability Index) | Batch, per-feature | Gradual covariate shift | `<0.1` stable · `0.1–0.25` moderate · `>0.25` significant |
| **Jensen–Shannon divergence** | Batch, per-feature | Distribution shape change, bounded [0,1] | complements PSI when bins empty |
| **ADWIN** (adaptive windowing) | **Streaming**, online | Abrupt concept drift on model confidence | Hoeffding-bound change point |

PSI bins by *reference quantiles* (robust to skew). ADWIN is a faithful
simplified Bifet–Gavaldà detector with a Hoeffding bound on the difference of
sub-window means. All pure NumPy — no black boxes.

---

## Architecture

```
  register ─▶ joblib estimator + train.csv
                      │
                      ├─▶ infer schema (feature names, task)
                      ├─▶ fit FeatureReference (quantile hist) per feature
                      └─▶ SQLite registry  (WAL, concurrent-safe)
                      
  POST /predict ─▶ model.predict ─▶ ✎ log(features, output, confidence) ─▶ SQLite
                                              │
                                              ▼
  GET /drift ─▶ recent N preds  ─▶ PSI / JS per feature  ┐
                                 └▶ ADWIN on confidence   ├─▶ drift report + charts
                                                          ┘
  POST /retrain ─▶ worker thread:
        clone(estimator).fit(fresh)  ─▶  score challenger vs champion on holdout
        challenger wins?  ──yes──▶  swap artifact, bump version, refresh reference
                          ──no───▶  keep champion (logged, auditable)
```

---

## Quick start

### 1 · Backend

```bash
cd backend
python -m venv .venv && .venv\Scripts\activate     # Windows
pip install -r requirements.txt

# build a demo model + data (sklearn breast-cancer, no download)
python scripts/make_sample_model.py

uvicorn app.main:app --reload --port 8000
```

### 2 · Frontend

```bash
cd frontend
npm install
npm run dev          # http://localhost:5181
```

### 3 · See drift happen live

1. **Register** → upload `backend/sample/model.joblib` + `backend/sample/train.csv` (target = `target`)
2. In another terminal, stream a controlled covariate shift:
   ```bash
   cd backend
   python scripts/simulate_drift.py <MODEL_ID>
   ```
3. Watch the dashboard: PSI gauge climbs, the shifting feature turns red,
   ADWIN flips to **CHANGE**, the reference-vs-live histogram visibly separates.
4. **Trigger retrain** → upload `backend/sample/fresh_labelled.csv` (target = `target`)
   → watch the shadow-promote job log and the version bump.

---

## API

| Method | Route | Purpose |
|---|---|---|
| `GET`  | `/api/models` | list registered models |
| `POST` | `/api/models` | register estimator + training CSV |
| `POST` | `/api/models/{id}/predict` | inference (logged for drift) |
| `GET`  | `/api/models/{id}/drift` | full per-feature drift report |
| `POST` | `/api/models/{id}/retrain` | submit labelled CSV → threaded retrain |
| `GET`  | `/api/jobs/{job_id}` | retrain job status + log |

Interactive docs at `http://localhost:8000/docs`.

---

## Tech stack

**ML**: scikit-learn · NumPy · pandas · joblib · custom PSI / JS / ADWIN
**Backend**: FastAPI · Pydantic v2 · SQLite (WAL) · threaded job runner
**Frontend**: React 18 · Vite · Tailwind · dependency-free inline-SVG charts
**Zero external services. Zero API keys. Runs air-gapped.**

> Production note: the retrain job interface is intentionally identical to a
> Celery task — swapping the worker thread for Celery + Redis is a one-file
> change. Kept threaded here so the whole platform runs with a single command.

---

## Screenshots

| Model Registry | Drift Dashboard | Retrain Log |
|---|---|---|
| Register any sklearn estimator + CSV | PSI gauge turns red, shifting feature highlighted | Shadow-promote job log with champion vs challenger |

> Stream a covariate shift with `simulate_drift.py` to see the dashboard react live.

---

## License

MIT

---

*Built by Mohamed Fazil — AI/ML & Full-Stack Engineer.*
