"""Drift detection maths — the core IP of ModelMesh.

Three complementary detectors:

1. **PSI** (Population Stability Index) — the bank/credit-risk industry
   standard for covariate shift. Binned by *reference* quantiles so it is
   robust to skew.

2. **Jensen–Shannon divergence** — symmetric, bounded [0,1] distribution
   distance; complements PSI and is well-defined when a bin empties.

3. **ADWIN-style detector** — Hoeffding-bound adaptive windowing for
   *streaming* concept-drift detection on a 1-D signal (e.g. model
   confidence). Flags the change point online without a fixed window.

All pure NumPy — no scikit-learn needed for the detectors themselves.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .config import ADWIN_DELTA, PSI_BINS, PSI_MODERATE, PSI_NO_SHIFT


# ── reference summary ─────────────────────────────────────────────────────
@dataclass
class FeatureReference:
    """Quantile bin edges + expected proportions from training data."""

    edges: list[float]
    expected: list[float]
    mean: float
    std: float

    @classmethod
    def fit(cls, x: np.ndarray, bins: int = PSI_BINS) -> "FeatureReference":
        x = np.asarray(x, dtype="float64")
        x = x[np.isfinite(x)]
        # Quantile edges → roughly equal mass per bin in the reference.
        qs = np.linspace(0, 1, bins + 1)
        edges = np.unique(np.quantile(x, qs))
        if edges.size < 3:  # near-constant feature → widen artificially
            edges = np.array([x.min() - 1e-6, x.mean(), x.max() + 1e-6])
        counts, _ = np.histogram(x, bins=edges)
        expected = counts / max(counts.sum(), 1)
        return cls(
            edges=edges.tolist(),
            expected=_smooth(expected).tolist(),
            mean=float(x.mean()),
            std=float(x.std() + 1e-12),
        )


def _smooth(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = p + eps
    return p / p.sum()


# ── PSI & JS divergence ───────────────────────────────────────────────────
def psi(ref: FeatureReference, current: np.ndarray) -> float:
    cur = np.asarray(current, dtype="float64")
    cur = cur[np.isfinite(cur)]
    if cur.size == 0:
        return 0.0
    counts, _ = np.histogram(cur, bins=np.asarray(ref.edges))
    actual = _smooth(counts / max(counts.sum(), 1))
    expected = np.asarray(ref.expected)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def js_divergence(ref: FeatureReference, current: np.ndarray) -> float:
    cur = np.asarray(current, dtype="float64")
    cur = cur[np.isfinite(cur)]
    if cur.size == 0:
        return 0.0
    counts, _ = np.histogram(cur, bins=np.asarray(ref.edges))
    q = _smooth(counts / max(counts.sum(), 1))
    p = np.asarray(ref.expected)
    m = 0.5 * (p + q)
    kl = lambda a, b: np.sum(a * np.log(a / b))  # noqa: E731
    return float(0.5 * kl(p, m) + 0.5 * kl(q, m))


def severity(psi_value: float) -> str:
    if psi_value < PSI_NO_SHIFT:
        return "stable"
    if psi_value < PSI_MODERATE:
        return "moderate"
    return "significant"


# ── ADWIN-style streaming change detector ─────────────────────────────────
@dataclass
class ADWIN:
    """Adaptive windowing change detector (Bifet & Gavaldà, simplified).

    Maintains a window of recent values. After each insert it searches for a
    split such that the two sub-windows' means differ by more than a Hoeffding
    bound; if found, the older sub-window is dropped and a change is reported.
    """

    delta: float = ADWIN_DELTA
    max_window: int = 1000
    window: deque[float] = field(default_factory=deque)
    drift_points: list[int] = field(default_factory=list)
    _t: int = 0

    def update(self, value: float) -> bool:
        self._t += 1
        self.window.append(float(value))
        if len(self.window) > self.max_window:
            self.window.popleft()
        changed = self._detect()
        if changed:
            self.drift_points.append(self._t)
        return changed

    def _detect(self) -> bool:
        w = np.asarray(self.window, dtype="float64")
        n = w.size
        if n < 20:
            return False
        total = w.sum()
        left_sum = 0.0
        for i in range(5, n - 5):  # require min sub-window size
            left_sum += w[i - 1]
            n0, n1 = i, n - i
            m0 = left_sum / n0
            m1 = (total - left_sum) / n1
            # Hoeffding bound for difference of means (variance-free form).
            m = 1.0 / (1.0 / n0 + 1.0 / n1)
            eps = np.sqrt((1.0 / (2 * m)) * np.log(4.0 / self.delta))
            if abs(m0 - m1) > eps:
                # Drop the stale older half.
                for _ in range(n0):
                    self.window.popleft()
                return True
        return False
