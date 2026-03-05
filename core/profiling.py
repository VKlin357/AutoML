from __future__ import annotations

from typing import Any, Dict, List
import pandas as pd

from .utils import sha_fingerprint


def infer_task(df: pd.DataFrame, target: str) -> str:
    y = df[target]
    nun = y.nunique(dropna=True)

    if pd.api.types.is_float_dtype(y) and nun > 20:
        return "regression"
    if nun <= 2:
        return "binary"
    return "multiclass"


def default_metric(task: str) -> str:
    if task == "binary":
        return "roc_auc"
    if task == "multiclass":
        return "accuracy"
    return "rmse"


def profile_tabular(df: pd.DataFrame, target: str) -> Dict[str, Any]:
    X = df.drop(columns=[target])
    y = df[target]

    n_rows, n_cols = X.shape

    num_cols: List[str] = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols: List[str] = [c for c in X.columns if c not in num_cols]

    missing_top20 = (X.isna().mean().sort_values(ascending=False).head(20)).to_dict()
    nunique_top20 = X.nunique(dropna=True).sort_values(ascending=False).head(20).to_dict()

    id_like_cols = []
    for c in X.columns:
        if X[c].nunique(dropna=True) > 0.95 * n_rows:
            id_like_cols.append(c)

    y_info = {
        "dtype": str(y.dtype),
        "n_unique": int(y.nunique(dropna=True)),
        "missing_rate": float(y.isna().mean()),
    }

    balance = None
    if y_info["n_unique"] <= 50 and not pd.api.types.is_float_dtype(y):
        vc = y.value_counts(dropna=False)
        balance = {
            "top_counts": vc.head(10).to_dict(),
            "imbalance_ratio": float(vc.max() / max(1, vc.min()))
        }

    # small stable fingerprint (shape + head csv)
    head_csv = df.head(200).to_csv(index=False)
    fingerprint = sha_fingerprint(f"{df.shape}|{head_csv}")

    return {
        "n_rows": int(n_rows),
        "n_features": int(n_cols),
        "numeric_cols": num_cols,
        "categorical_cols": cat_cols,
        "missing_top20": missing_top20,
        "nunique_top20": {k: int(v) for k, v in nunique_top20.items()},
        "id_like_cols": id_like_cols[:20],
        "target": y_info,
        "class_balance": balance,
        "df_fingerprint": fingerprint,
    }