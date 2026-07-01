from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_squared_error,
    recall_score,
    roc_auc_score,
)

TaskType = Literal["binary", "multiclass", "regression"]


@dataclass
class MetricResult:
    primary: float          # accuracy / auc / -rmse  — used in final reports & comparisons
    search_score: float     # dense proxy for NAS selection (includes logloss + bal_acc)
    metrics: Dict[str, float]
    higher_is_better: bool


def infer_task_type(y: np.ndarray) -> TaskType:
    y = np.asarray(y)
    if np.issubdtype(y.dtype, np.floating):
        return "regression"
    uniq = np.unique(y)
    if len(uniq) <= 2:
        return "binary"
    return "multiclass"


def compute_metrics(
    task,
    y_true: np.ndarray,
    y_pred_proba: Optional[np.ndarray] = None,
    y_pred: Optional[np.ndarray] = None,
) -> MetricResult:
    y_true = np.asarray(y_true)

    if task == "regression":
        assert y_pred is not None
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        primary = -rmse
        return MetricResult(
            primary=primary,
            search_score=primary,
            metrics={"rmse": rmse},
            higher_is_better=True,
        )

    if task == "binary":
        assert y_pred_proba is not None
        proba = np.nan_to_num(
            y_pred_proba.reshape(-1),
            nan=0.5,
            posinf=1.0,
            neginf=0.0,
        )
        proba = np.clip(proba, 1e-6, 1 - 1e-6)
        pred = (proba >= 0.5).astype(int)
        acc = float(accuracy_score(y_true, pred))
        present_labels = np.unique(y_true)
        bal_acc = float(recall_score(
            y_true, pred, labels=present_labels, average="macro", zero_division=0
        ))
        ll = float(log_loss(y_true, proba, labels=[0, 1]))
        unique = np.unique(y_true)
        auc_defined = len(unique) == 2
        if auc_defined:
            auc = float(roc_auc_score(y_true, proba))
            primary = auc
        else:
            # Temporal validation blocks may occasionally contain one class.
            # AUC is undefined there, so use accuracy as a stable selection
            # fallback instead of poisoning the whole trial with -inf.
            auc = None
            primary = acc
        # Dense search proxy: auc + balanced_acc bonus - logloss penalty
        search_score = primary + 0.05 * (bal_acc - acc) - 0.01 * ll
        return MetricResult(
            primary=primary,
            search_score=search_score,
            metrics={"auc": auc, "auc_defined": bool(auc_defined),
                     "logloss": ll, "acc": acc, "balanced_acc": bal_acc,
                     "search_score": search_score},
            higher_is_better=True,
        )

    # multiclass
    assert y_pred_proba is not None
    proba = np.clip(y_pred_proba, 1e-9, 1.0)
    pred = np.argmax(proba, axis=1)
    labels = np.arange(proba.shape[1])
    present_labels = np.unique(y_true)
    acc = float(accuracy_score(y_true, pred))
    # Validation may legitimately contain only a subset of classes (e.g.
    # official ECG5000 TRAIN has very few samples of class 5). Balanced
    # accuracy is defined over classes observable in this evaluation split,
    # while macro-F1/logloss retain the model's full output label space.
    bal_acc = float(recall_score(
        y_true, pred, labels=present_labels, average="macro", zero_division=0
    ))
    macro_f1 = float(f1_score(
        y_true, pred, labels=labels, average="macro", zero_division=0
    ))
    ll = float(log_loss(y_true, proba, labels=labels))

    # Dense search proxy for NAS:
    #   - balanced_acc catches class-imbalance that raw acc ignores
    #   - macro_f1 adds soft signal (model improving minority classes)
    #   - logloss penalty: model that improves calibration/separation also improves NAS feedback
    # Final report still uses raw accuracy for honest CatBoost comparison.
    search_score = (
        acc
        + 0.10 * (bal_acc - acc)        # bonus for balanced improvement
        + 0.05 * (macro_f1 - acc)       # bonus for minority-class improvement
        - 0.015 * ll                     # soft logloss penalty
    )

    return MetricResult(
        primary=acc,
        search_score=search_score,
        metrics={
            "acc": acc,
            "balanced_acc": bal_acc,
            "macro_f1": macro_f1,
            "logloss": ll,
            "search_score": search_score,
        },
        higher_is_better=True,
    )
