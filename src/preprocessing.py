"""
Pluggable preprocessing for tabular NAS.

The dataset loader (``data.py``) now returns the *raw* train/val/test
DataFrames and column metadata. Each trial then applies its own
``Preprocessor`` (chosen by the LLM agent / random sampler).

Why:
- Lets the agent treat preprocessing as part of the search space.
- Avoids leakage: every preprocessor is fit on TRAIN only.
- Makes ``data.load_*`` a pure dataset I/O function — easier to test
  and to reuse for new dataset sources.

Two sub-choices are exposed today:
- ``num_encoder``  in {standard, quantile, none}
- ``cat_encoder``  in {embedding, onehot}

The output is a ``PreparedSplit`` with the (X_num, X_cat, y) ndarrays
plus the categorical cardinalities (for embedding layers).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer

from .metrics import TaskType

# ---------------------------------------------------------------------------
# Returned arrays
# ---------------------------------------------------------------------------

@dataclass
class PreparedSplit:
    X_train_num: np.ndarray
    X_val_num:   np.ndarray
    X_test_num:  np.ndarray

    X_train_cat: np.ndarray
    X_val_cat:   np.ndarray
    X_test_cat:  np.ndarray

    y_train: np.ndarray
    y_val:   np.ndarray
    y_test:  np.ndarray

    cat_cardinalities: List[int]
    n_num_after: int
    n_cat_after: int

    task: TaskType
    n_classes: int

# ---------------------------------------------------------------------------
# Numeric encoders
# ---------------------------------------------------------------------------

def _impute_num(train: pd.DataFrame, *others: pd.DataFrame, num_cols: List[str]):
    if not num_cols:
        empties = [np.zeros((len(train), 0), np.float32)] + [
            np.zeros((len(o), 0), np.float32) for o in others
        ]
        return tuple(empties)
    tr = train[num_cols].copy()
    med = tr.median(numeric_only=True)
    # median can itself be NaN if column is mostly NaN — fall back to 0
    med = med.fillna(0.0)
    tr_filled = tr.fillna(med).to_numpy(np.float32)
    # Replace any remaining inf/nan (e.g. float64→float32 overflow in physics datasets)
    np.nan_to_num(tr_filled, copy=False, nan=0.0, posinf=1e6, neginf=-1e6)
    out = [tr_filled]
    for o in others:
        o_filled = o[num_cols].copy().fillna(med).to_numpy(np.float32)
        np.nan_to_num(o_filled, copy=False, nan=0.0, posinf=1e6, neginf=-1e6)
        out.append(o_filled)
    return tuple(out)

def _sanitize(arr: np.ndarray) -> np.ndarray:
    """Replace NaN / ±inf with 0 / ±1e6. Works on any float array."""
    return np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=1e6, neginf=-1e6)

def _encode_num_standard(tr: np.ndarray, *others) -> Tuple[np.ndarray, ...]:
    if tr.shape[1] == 0:
        return (tr, *others)
    tr = _sanitize(tr)
    mean = tr.mean(axis=0, keepdims=True)
    std = tr.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    out = [(tr - mean) / std]
    for o in others:
        out.append((_sanitize(o) - mean) / std)
    return tuple(out)

def _encode_num_quantile(tr: np.ndarray, *others) -> Tuple[np.ndarray, ...]:
    if tr.shape[1] == 0:
        return (tr, *others)
    n_quantiles = max(10, min(1000, tr.shape[0]))
    qt = QuantileTransformer(
        n_quantiles=n_quantiles, output_distribution="normal", subsample=int(1e6),
        random_state=0,
    )
    # Clip to [-10, 10]: QuantileTransformer with output_distribution="normal" can produce
    # ±inf (via scipy.stats.norm.ppf) for val/test values outside the training range,
    # which silently become NaN when cast to float32. Clipping prevents NaN propagation.
    tr_t = qt.fit_transform(tr).astype(np.float32)
    np.clip(tr_t, -10.0, 10.0, out=tr_t)
    out = [tr_t]
    for o in others:
        o_t = qt.transform(o).astype(np.float32)
        np.clip(o_t, -10.0, 10.0, out=o_t)
        out.append(o_t)
    return tuple(out)

def _encode_num_none(tr: np.ndarray, *others):
    return (tr, *others)

# ---------------------------------------------------------------------------
# Categorical encoders
# ---------------------------------------------------------------------------

def _encode_cat_embedding(
    train: pd.DataFrame, *others: pd.DataFrame, cat_cols: List[str]
) -> Tuple[List[np.ndarray], List[int]]:
    """Per-column int encoding [0, card-1] with 0 reserved for OOV / missing."""
    if not cat_cols:
        empties = [np.zeros((len(train), 0), dtype=np.int64)]
        for o in others:
            empties.append(np.zeros((len(o), 0), dtype=np.int64))
        return empties, []

    tr_arrs, cardinalities = [], []
    other_arrs: List[List[np.ndarray]] = [[] for _ in others]

    for c in cat_cols:
        tr_col = train[c].astype("object").fillna("__MISSING__")
        uniq = pd.unique(tr_col)
        vocab = {val: i + 1 for i, val in enumerate(uniq)}  # 0 = OOV
        tr_arrs.append(tr_col.map(lambda v: vocab.get(v, 0)).astype("int64").to_numpy().reshape(-1, 1))
        for j, o in enumerate(others):
            o_col = o[c].astype("object").fillna("__MISSING__")
            other_arrs[j].append(o_col.map(lambda v: vocab.get(v, 0)).astype("int64").to_numpy().reshape(-1, 1))
        cardinalities.append(len(vocab) + 1)  # +OOV slot

    out = [np.concatenate(tr_arrs, axis=1)]
    for cols in other_arrs:
        out.append(np.concatenate(cols, axis=1))
    return out, cardinalities

def _encode_cat_onehot(
    train: pd.DataFrame, *others: pd.DataFrame, cat_cols: List[str], num_train: np.ndarray,
    num_others: list,
) -> Tuple[np.ndarray, list, list, List[int]]:
    """One-hot encoding folded into the *numeric* tensor (no embedding tables).

    To prevent the categorical signal from disappearing for embedding-only
    models we still return a 0-column int matrix for the cat output and
    instead append the dummies to the numeric tensor.
    """
    if not cat_cols:
        empties = [np.zeros((len(train), 0), dtype=np.int64)]
        for o in others:
            empties.append(np.zeros((len(o), 0), dtype=np.int64))
        return num_train, list(num_others), empties, []  # type: ignore

    # determine vocab from train
    vocabs = []
    for c in cat_cols:
        tr_col = train[c].astype("object").fillna("__MISSING__")
        uniq = list(pd.unique(tr_col))
        vocabs.append(uniq)

    def _to_dummies(df: pd.DataFrame) -> np.ndarray:
        cols = []
        for c, uniq in zip(cat_cols, vocabs):
            col = df[c].astype("object").fillna("__MISSING__")
            mat = np.zeros((len(col), len(uniq)), dtype=np.float32)
            idx = col.map({v: i for i, v in enumerate(uniq)})
            valid = idx.notna()
            mat[np.where(valid)[0], idx[valid].astype(int).to_numpy()] = 1.0
            cols.append(mat)
        return np.concatenate(cols, axis=1)

    tr_dum = _to_dummies(train)
    other_dums = [_to_dummies(o) for o in others]
    new_num_train = np.concatenate([num_train, tr_dum], axis=1)
    new_num_others = [np.concatenate([no, od], axis=1) for no, od in zip(num_others, other_dums)]
    cat_empty_train = np.zeros((len(train), 0), dtype=np.int64)
    cat_empty_others = [np.zeros((len(o), 0), dtype=np.int64) for o in others]
    return new_num_train, new_num_others, [cat_empty_train, *cat_empty_others], []

# ---------------------------------------------------------------------------
# Top-level Preprocessor
# ---------------------------------------------------------------------------

@dataclass
class Preprocessor:
    num_encoder: str = "standard"      # standard | quantile | none
    cat_encoder: str = "embedding"     # embedding | onehot

    def __post_init__(self):
        assert self.num_encoder in ("standard", "quantile", "none"), self.num_encoder
        assert self.cat_encoder in ("embedding", "onehot"), self.cat_encoder

    def fit_transform(
        self,
        X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame,
        y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray,
        num_cols: List[str], cat_cols: List[str],
        task: TaskType, n_classes: int,
    ) -> PreparedSplit:
        # 1) Numeric: impute (train median) then encode
        tr_n, va_n, te_n = _impute_num(X_train, X_val, X_test, num_cols=num_cols)

        if self.num_encoder == "standard":
            tr_n, va_n, te_n = _encode_num_standard(tr_n, va_n, te_n)
        elif self.num_encoder == "quantile":
            tr_n, va_n, te_n = _encode_num_quantile(tr_n, va_n, te_n)
        # "none" -> pass-through

        # 2) Categorical: embedding (int matrix + cardinalities) or one-hot folded into numeric
        if self.cat_encoder == "embedding":
            cats, cardinalities = _encode_cat_embedding(X_train, X_val, X_test, cat_cols=cat_cols)
            tr_c, va_c, te_c = cats[0], cats[1], cats[2]
            X_train_num, X_val_num, X_test_num = tr_n, va_n, te_n
        else:
            (X_train_num, others, cats_empty, cardinalities) = _encode_cat_onehot(
                X_train, X_val, X_test, cat_cols=cat_cols,
                num_train=tr_n, num_others=[va_n, te_n],  # type: ignore
            )
            X_val_num, X_test_num = others[0], others[1]
            tr_c, va_c, te_c = cats_empty[0], cats_empty[1], cats_empty[2]

        return PreparedSplit(
            X_train_num=X_train_num.astype(np.float32),
            X_val_num=X_val_num.astype(np.float32),
            X_test_num=X_test_num.astype(np.float32),
            X_train_cat=tr_c.astype(np.int64),
            X_val_cat=va_c.astype(np.int64),
            X_test_cat=te_c.astype(np.int64),
            y_train=y_train, y_val=y_val, y_test=y_test,
            cat_cardinalities=cardinalities,
            n_num_after=int(X_train_num.shape[1]),
            n_cat_after=int(tr_c.shape[1]),
            task=task, n_classes=n_classes,
        )

def make_preprocessor(cfg: Dict) -> Preprocessor:
    return Preprocessor(
        num_encoder=cfg.get("num_encoder", "standard"),
        cat_encoder=cfg.get("cat_encoder", "embedding"),
    )
