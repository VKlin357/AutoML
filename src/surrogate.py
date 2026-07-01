"""
Cheap surrogate model that learns: (config) -> validation primary metric.

We use it to FILTER the LLM's proposals before paying the cost of even a
cheap-rung neural training. Inspired by SeqNAS (Bayesian-optimized NAS
with a graph-kernel surrogate) but using gradient-boosted trees on a
flattened feature representation, which works fine for our small
(50–200 trial) histories and avoids any extra deps.

Workflow used by the orchestrator:

    surr = ConfigSurrogate()
    for each completed trial:
        surr.add(cfg, primary)
    surr.fit()                     # cheap: ~1s on 100 rows

    # When the LLM gives us K candidates, we score them all and only
    # train the top ``keep`` ones at cheap rung.
    scores = [surr.score(c) for c in candidates]

The surrogate is also exposed to the LLM in the reflection prompt, as
"the surrogate currently believes that the most predictive features of
val score are: ...", which sometimes helps the agent reason.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .search_space import featurize_config


# ---------------------------------------------------------------------------
# Lazy import: lightgbm is optional. If absent, fall back to sklearn GBR.
# ---------------------------------------------------------------------------

def _make_regressor():
    try:
        import lightgbm as lgb
        return lgb.LGBMRegressor(
            n_estimators=200, learning_rate=0.05,
            num_leaves=15, min_data_in_leaf=2,
            verbose=-1, random_state=0,
        )
    except Exception:
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=3, random_state=0,
        )


# ---------------------------------------------------------------------------
# Featurization helper: align dicts with shared column space
# ---------------------------------------------------------------------------

def _to_matrix(feat_dicts: List[Dict[str, float]]) -> Tuple[np.ndarray, List[str]]:
    keys = sorted({k for d in feat_dicts for k in d.keys()})
    X = np.zeros((len(feat_dicts), len(keys)), dtype=np.float32)
    for i, d in enumerate(feat_dicts):
        for j, k in enumerate(keys):
            X[i, j] = d.get(k, 0.0)
    return X, keys


# ---------------------------------------------------------------------------
# Surrogate class
# ---------------------------------------------------------------------------

@dataclass
class ConfigSurrogate:
    histories: List[Tuple[Dict[str, float], float]] = field(default_factory=list)
    model: Any = None
    keys: List[str] = field(default_factory=list)
    fitted: bool = False
    n_train: int = 0

    def add(self, cfg: Dict[str, Any], primary: float):
        self.histories.append((featurize_config(cfg), float(primary)))
        self.fitted = False

    def fit(self) -> bool:
        if len(self.histories) < 5:
            self.fitted = False
            return False
        feats = [h[0] for h in self.histories]
        y = np.array([h[1] for h in self.histories], dtype=np.float32)
        # If all-equal targets, surrogate is meaningless
        if np.std(y) < 1e-9:
            self.fitted = False
            return False
        X, keys = _to_matrix(feats)
        self.model = _make_regressor()
        self.model.fit(X, y)
        self.keys = keys
        self.n_train = len(y)
        self.fitted = True
        return True

    def score(self, cfg: Dict[str, Any]) -> float:
        if not self.fitted:
            return 0.0
        f = featurize_config(cfg)
        x = np.array([[f.get(k, 0.0) for k in self.keys]], dtype=np.float32)
        return float(self.model.predict(x)[0])

    def rank(self, cfgs: List[Dict[str, Any]]) -> List[int]:
        """Return indices of cfgs sorted by predicted score descending."""
        if not self.fitted:
            return list(range(len(cfgs)))
        s = [self.score(c) for c in cfgs]
        return list(np.argsort(s)[::-1])

    def feature_importances(self, top_k: int = 10) -> List[Tuple[str, float]]:
        """For the reflection prompt — which features the surrogate currently
        thinks matter most."""
        if not self.fitted:
            return []
        try:
            imp = np.asarray(self.model.feature_importances_, dtype=np.float64)
        except Exception:
            return []
        if imp.sum() <= 0:
            return []
        imp = imp / imp.sum()
        order = np.argsort(imp)[::-1][:top_k]
        return [(self.keys[i], float(imp[i])) for i in order]
