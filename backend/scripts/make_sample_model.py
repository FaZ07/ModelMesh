"""Train a demo model + emit reference and drifted CSVs — fully offline.

Uses sklearn's built-in breast-cancer dataset (no download). Produces:
  sample/model.joblib        a RandomForest classifier
  sample/train.csv           reference feature distribution (register with this)
  sample/fresh_labelled.csv  new labelled data for the retrain demo
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

OUT = Path(__file__).resolve().parents[1] / "sample"
OUT.mkdir(exist_ok=True)


def main() -> None:
    data = load_breast_cancer()
    # Keep 8 features for a readable demo UI.
    cols = list(data.feature_names[:8])
    X = pd.DataFrame(data.data[:, :8], columns=cols)
    y = pd.Series(data.target, name="target")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    clf = RandomForestClassifier(n_estimators=120, random_state=0)
    clf.fit(Xtr, ytr)
    acc = (clf.predict(Xte) == yte).mean()

    joblib.dump(clf, OUT / "model.joblib")
    Xtr.assign(target=ytr.values).to_csv(OUT / "train.csv", index=False)

    # Fresh labelled batch (slightly shifted) for the retrain workflow.
    rng = np.random.default_rng(1)
    fresh = Xte.copy()
    fresh[cols[0]] *= 1.05 + rng.normal(0, 0.02, len(fresh))
    fresh.assign(target=yte.values).to_csv(OUT / "fresh_labelled.csv", index=False)

    print(f"[OK] model.joblib  (RandomForest, holdout acc={acc:.3f})")
    print(f"[OK] train.csv     ({len(Xtr)} rows, {len(cols)} features)")
    print(f"[OK] fresh_labelled.csv ({len(fresh)} rows)")
    print(f"     -> {OUT}")


if __name__ == "__main__":
    main()
