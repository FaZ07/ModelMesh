"""Drift monitoring service — joins the reference distribution with the live
prediction log and produces a per-feature drift report + an online ADWIN
verdict on model confidence.
"""
from __future__ import annotations

import numpy as np

from . import store
from .config import DRIFT_WINDOW
from .drift import ADWIN, FeatureReference, js_divergence, psi, severity

# One ADWIN detector per model, kept warm across requests.
_adwin: dict[str, ADWIN] = {}


def drift_report(model_id: str, window: int = DRIFT_WINDOW) -> dict:
    meta = store.get_model(model_id)
    if not meta:
        raise KeyError("unknown model id")

    rows = store.recent_predictions(model_id, window)
    total = store.prediction_count(model_id)
    if len(rows) < 20:
        return {
            "model_id": model_id,
            "samples": len(rows),
            "total_predictions": total,
            "ready": False,
            "message": "Need ≥20 logged predictions to assess drift.",
            "features": [],
            "overall": {"max_psi": 0.0, "status": "stable", "drifting_features": 0},
            "stream": {"adwin_drift": False, "drift_points": []},
        }

    feats = meta["features"]
    cur = {f: np.array([r["features"][f] for r in rows], dtype="float64") for f in feats}

    report = []
    for f in feats:
        ref = FeatureReference(**meta["reference"][f])
        p = psi(ref, cur[f])
        report.append({
            "feature": f,
            "psi": round(p, 4),
            "js": round(js_divergence(ref, cur[f]), 4),
            "status": severity(p),
            "ref_mean": round(ref.mean, 4),
            "cur_mean": round(float(cur[f].mean()), 4),
            "shift": round(float(cur[f].mean()) - ref.mean, 4),
            "ref_hist": _hist(ref, cur[f]),
        })
    report.sort(key=lambda d: d["psi"], reverse=True)

    max_psi = max(r["psi"] for r in report)
    drifting = sum(1 for r in report if r["status"] != "stable")

    # Streaming concept-drift on model confidence (classification only).
    adw = _adwin.setdefault(model_id, ADWIN())
    stream_drift = False
    signal = [r["confidence"] for r in rows if r["confidence"] is not None]
    for v in signal[-50:]:  # feed the freshest slice
        stream_drift = adw.update(v) or stream_drift

    return {
        "model_id": model_id,
        "samples": len(rows),
        "total_predictions": total,
        "ready": True,
        "features": report,
        "overall": {
            "max_psi": round(max_psi, 4),
            "status": severity(max_psi),
            "drifting_features": drifting,
        },
        "stream": {
            "adwin_drift": stream_drift,
            "drift_points": adw.drift_points[-10:],
        },
    }


def _hist(ref: FeatureReference, cur: np.ndarray, bins: int = 12) -> dict:
    edges = np.asarray(ref.edges)
    lo, hi = float(edges[0]), float(edges[-1])
    common = np.linspace(lo, hi, bins + 1)
    rc = np.array(ref.expected)  # already proportions over ref edges
    # Re-bin current onto the shared axis for an overlay chart.
    ch, _ = np.histogram(cur, bins=common)
    ch = ch / max(ch.sum(), 1)
    # Approximate reference density on the shared axis from its quantile edges.
    centers = 0.5 * (edges[:-1] + edges[1:])
    rh, _ = np.histogram(centers, bins=common, weights=rc)
    rh = rh / max(rh.sum(), 1e-9)
    return {
        "bins": [round(float(x), 3) for x in common.tolist()],
        "reference": [round(float(x), 4) for x in rh.tolist()],
        "current": [round(float(x), 4) for x in ch.tolist()],
    }
