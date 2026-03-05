from __future__ import annotations

import math
from typing import Optional
import numpy as np

from sklearn.metrics import (
    roc_auc_score, accuracy_score, log_loss,
    mean_squared_error, mean_absolute_error, r2_score
)


def score_metric(task: str, metric: str, y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None) -> float:
    """
    Returns score where HIGHER is better.
    Loss metrics are returned as negative values.
    """
    if task in ("binary", "multiclass"):
        if metric == "accuracy":
            return float(accuracy_score(y_true, y_pred))
        if metric == "logloss":
            if y_proba is None:
                raise ValueError("logloss requires predict_proba outputs.")
            return -float(log_loss(y_true, y_proba))
        if metric == "roc_auc":
            if task != "binary":
                raise ValueError("roc_auc supported only for binary in this agent.")
            if y_proba is None:
                raise ValueError("roc_auc requires predict_proba outputs.")
            if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                return float(roc_auc_score(y_true, y_proba[:, 1]))
            return float(roc_auc_score(y_true, y_proba))
        raise ValueError(f"Unknown metric: {metric}")

    if metric == "rmse":
        return -float(math.sqrt(mean_squared_error(y_true, y_pred)))
    if metric == "mae":
        return -float(mean_absolute_error(y_true, y_pred))
    if metric == "r2":
        return float(r2_score(y_true, y_pred))
    raise ValueError(f"Unknown metric: {metric}")