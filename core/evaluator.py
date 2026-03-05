from __future__ import annotations

import time
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
import joblib

from .metrics import score_metric
from .pipeline_builder import build_pipeline


def evaluate_cv(
    df: pd.DataFrame,
    target: str,
    task: str,
    metric: str,
    cand: Dict[str, Any],
    seed: int,
    cv_folds: int,
    max_onehot_categories: int
) -> Dict[str, Any]:
    X = df.drop(columns=[target])
    y = df[target].to_numpy()

    pipe, used_cols = build_pipeline(df, target, task, cand, seed, max_onehot_categories)

    splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed) if task in ("binary", "multiclass") \
        else KFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    scores = []
    t0 = time.time()

    for tr_idx, va_idx in splitter.split(X, y):
        Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
        ytr, yva = y[tr_idx], y[va_idx]

        # clone
        pipe_fold = joblib.loads(joblib.dumps(pipe))
        pipe_fold.fit(Xtr, ytr)

        ypred = pipe_fold.predict(Xva)
        yproba = None
        if task in ("binary", "multiclass") and metric in ("roc_auc", "logloss"):
            yproba = pipe_fold.predict_proba(Xva)

        s = score_metric(task, metric, yva, ypred, yproba)
        scores.append(s)

    return {
        "score_mean": float(np.mean(scores)),
        "score_std": float(np.std(scores)),
        "fit_time_s": float(time.time() - t0),
        "used_cols": used_cols,
        "eval_mode": "cv",
    }


def evaluate_holdout(
    df: pd.DataFrame,
    target: str,
    task: str,
    metric: str,
    cand: Dict[str, Any],
    seed: int,
    holdout_size: float,
    max_onehot_categories: int
) -> Dict[str, Any]:
    X = df.drop(columns=[target])
    y = df[target].to_numpy()

    strat = (task in ("binary", "multiclass"))
    Xtr, Xva, ytr, yva = train_test_split(
        X, y, test_size=holdout_size, random_state=seed, stratify=y if strat else None
    )

    dftr = pd.concat([Xtr, pd.Series(ytr, name=target, index=Xtr.index)], axis=1)
    pipe, used_cols = build_pipeline(dftr, target, task, cand, seed, max_onehot_categories)

    t0 = time.time()
    pipe.fit(Xtr, ytr)

    ypred = pipe.predict(Xva)
    yproba = None
    if task in ("binary", "multiclass") and metric in ("roc_auc", "logloss"):
        yproba = pipe.predict_proba(Xva)

    s = score_metric(task, metric, yva, ypred, yproba)

    return {
        "score_mean": float(s),
        "score_std": 0.0,
        "fit_time_s": float(time.time() - t0),
        "used_cols": used_cols,
        "eval_mode": "holdout",
    }