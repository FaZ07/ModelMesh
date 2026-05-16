"""Configuration & filesystem layout."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STORE_DIR = ROOT / "store"
MODEL_DIR = STORE_DIR / "models"
DB_PATH = STORE_DIR / "modelmesh.db"

for d in (STORE_DIR, MODEL_DIR):
    d.mkdir(parents=True, exist_ok=True)

# --- drift thresholds (industry rules of thumb) -------------------------
PSI_BINS = 10
PSI_NO_SHIFT = 0.10        # < 0.10  → stable
PSI_MODERATE = 0.25        # 0.10–0.25 moderate · > 0.25 significant
DRIFT_WINDOW = 200         # most recent N predictions used as "current"
ADWIN_DELTA = 0.002        # confidence for Hoeffding change bound
