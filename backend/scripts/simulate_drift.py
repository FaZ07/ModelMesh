"""Stream predictions into a registered model with a controlled covariate
shift so the drift dashboard visibly lights up.

Usage:
  python scripts/simulate_drift.py <MODEL_ID> [--host http://localhost:8000]

Phase 1 (0–120 reqs): in-distribution traffic   → PSI stays "stable"
Phase 2 (120+ reqs):  progressive shift on the   → PSI climbs, ADWIN fires
                       first feature
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

SAMPLE = Path(__file__).resolve().parents[1] / "sample" / "train.csv"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_id")
    ap.add_argument("--host", default="http://localhost:8000")
    ap.add_argument("--n", type=int, default=300)
    args = ap.parse_args()

    if not SAMPLE.exists():
        sys.exit("Run scripts/make_sample_model.py first.")

    df = pd.read_csv(SAMPLE).drop(columns=["target"], errors="ignore")
    feats = list(df.columns)
    rng = np.random.default_rng(7)
    url = f"{args.host}/api/models/{args.model_id}/predict"

    for i in range(args.n):
        row = df.iloc[rng.integers(0, len(df))].to_dict()
        if i >= 120:  # inject growing shift on feature 0
            mag = (i - 120) / 80.0
            row[feats[0]] = row[feats[0]] * (1.0 + 0.9 * mag) + 4.0 * mag

        r = requests.post(url, json={"features": {k: float(v) for k, v in row.items()}})
        if r.status_code != 200:
            sys.exit(f"[{r.status_code}] {r.text}")

        if i % 30 == 0:
            phase = "IN-DIST" if i < 120 else "DRIFTING"
            print(f"  req {i:3d}  [{phase}]  pred={r.json()['prediction']}")
        time.sleep(0.01)

    print(f"\n✅ Sent {args.n} predictions. Open the dashboard → drift should be 'significant'.")


if __name__ == "__main__":
    main()
