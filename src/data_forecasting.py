"""
Forecasting datasets loader.

Supported datasets:
  - ETTh1, ETTh2  : Electricity Transformer Temperature (hourly)
  - ETTm1, ETTm2  : Electricity Transformer Temperature (15-min)
  - Weather       : 21 meteorological indicators
  - Exchange      : exchange rates for 8 currencies

All datasets are reformulated as tabular regression:
  Input:  last L timesteps (lookback window) → flat feature vector
  Output: next H timesteps (prediction horizon, default H=1)

For CatBoost: use make_forecasting_features() to add lag/rolling features.
For LLM-NAS:  use the raw flat lookback vector directly.

Usage:
    from src.data_forecasting import load_forecasting_raw
    raw, summary = load_forecasting_raw("etth1", lookback=96, horizon=1)
"""
from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

FORECASTING_DATASETS = {
    "etth1": "ETT-small/ETTh1.csv",
    "etth2": "ETT-small/ETTh2.csv",
    "ettm1": "ETT-small/ETTm1.csv",
    "ettm2": "ETT-small/ETTm2.csv",
    "weather": "weather/weather.csv",
    "exchange": "exchange_rate/exchange_rate.csv",
}

ETT_URLS = [
    "https://raw.githubusercontent.com/thuml/Time-Series-Library/main/dataset/ETT-small/ETTh1.csv",
    "https://raw.githubusercontent.com/thuml/Time-Series-Library/main/dataset/ETT-small/ETTh2.csv",
    "https://raw.githubusercontent.com/thuml/Time-Series-Library/main/dataset/ETT-small/ETTm1.csv",
    "https://raw.githubusercontent.com/thuml/Time-Series-Library/main/dataset/ETT-small/ETTm2.csv",
]

DATASET_URLS = {
    "etth1":    "https://raw.githubusercontent.com/thuml/Time-Series-Library/main/dataset/ETT-small/ETTh1.csv",
    "etth2":    "https://raw.githubusercontent.com/thuml/Time-Series-Library/main/dataset/ETT-small/ETTh2.csv",
    "ettm1":    "https://raw.githubusercontent.com/thuml/Time-Series-Library/main/dataset/ETT-small/ETTm1.csv",
    "ettm2":    "https://raw.githubusercontent.com/thuml/Time-Series-Library/main/dataset/ETT-small/ETTm2.csv",
    "weather":  "https://raw.githubusercontent.com/thuml/Time-Series-Library/main/dataset/weather/weather.csv",
    "exchange": "https://raw.githubusercontent.com/thuml/Time-Series-Library/main/dataset/exchange_rate/exchange_rate.csv",
}


@dataclass
class ForecastingRaw:
    """Preprocessed forecasting dataset ready for tabular models."""
    X_train: np.ndarray   # (n_train, n_features)
    y_train: np.ndarray   # (n_train,) or (n_train, horizon)
    X_val:   np.ndarray
    y_val:   np.ndarray
    X_test:  np.ndarray
    y_test:  np.ndarray

    target_col: str
    feature_cols: List[str]
    lookback: int
    horizon: int
    freq: str             # 'h' for hourly, '15min', 'd' for daily
    task: str             # always 'regression'
    n_classes: int        # always 1


def _download_csv(name: str, cache_dir: Path) -> pd.DataFrame:
    """Download dataset CSV to cache and return as DataFrame."""
    cache_file = cache_dir / f"{name}.csv"
    if cache_file.exists():
        print(f"[Forecasting] Loading {name} from cache: {cache_file}")
        return pd.read_csv(cache_file)

    url = DATASET_URLS.get(name)
    if url is None:
        raise ValueError(f"Unknown forecasting dataset: {name!r}. "
                         f"Available: {list(DATASET_URLS.keys())}")

    print(f"[Forecasting] Downloading {name} from {url} ...")
    try:
        import urllib.request
        with urllib.request.urlopen(url, timeout=60) as r:
            content = r.read()
        cache_file.write_bytes(content)
        print(f"[Forecasting] Saved to {cache_file}")
        return pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise RuntimeError(
            f"[Forecasting] Failed to download {name}: {e}\n"
            f"Manually download from:\n  {url}\n"
            f"and place at:\n  {cache_file}"
        )


def _make_windows(
    values: np.ndarray,
    lookback: int,
    horizon: int,
    target_idx: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding windows from multivariate time series.

    Parameters
    ----------
    values : (T, C) array — time × channels
    lookback : number of past steps to use as features
    horizon : number of future steps to predict (1 = next step)
    target_idx : channel index to predict

    Returns
    -------
    X : (N, lookback * C)  — flat feature vectors
    y : (N,) or (N, horizon) — targets
    """
    T, C = values.shape
    n_windows = T - lookback - horizon + 1
    if n_windows <= 0:
        raise ValueError(f"Not enough data: T={T}, lookback={lookback}, horizon={horizon}")

    X = np.zeros((n_windows, lookback * C), dtype=np.float32)
    if horizon == 1:
        y = np.zeros(n_windows, dtype=np.float32)
    else:
        y = np.zeros((n_windows, horizon), dtype=np.float32)

    for i in range(n_windows):
        X[i] = values[i:i + lookback].flatten()
        if horizon == 1:
            y[i] = values[i + lookback, target_idx]
        else:
            y[i] = values[i + lookback:i + lookback + horizon, target_idx]

    return X, y


def load_forecasting_raw(
    name: str,
    lookback: int = 96,
    horizon: int = 1,
    target_col: Optional[str] = None,
    train_frac: float = 0.7,
    val_frac: float = 0.1,
    seed: int = 42,
    cache_dir: Optional[str] = None,
) -> Tuple[ForecastingRaw, dict]:
    """
    Load a forecasting dataset as tabular regression.

    Parameters
    ----------
    name       : one of 'etth1', 'etth2', 'ettm1', 'ettm2', 'weather', 'exchange'
    lookback   : number of past steps used as features (default 96)
    horizon    : number of future steps to predict (default 1 = next step)
    target_col : which column to predict (default: first numeric column)
    train_frac : fraction for training (default 0.7)
    val_frac   : fraction for validation (default 0.1)

    Returns
    -------
    (ForecastingRaw, summary_dict)
    """
    name = name.lower()
    cache_dir = Path(cache_dir or Path.home() / ".cache" / "nas_datasets" / "forecasting")
    cache_dir.mkdir(parents=True, exist_ok=True)

    df = _download_csv(name, cache_dir)

    # Drop date/time column if present
    date_cols = [c for c in df.columns if c.lower() in ("date", "datetime", "timestamp", "time")]
    if date_cols:
        df = df.drop(columns=date_cols)

    # All remaining columns are numeric features
    df = df.select_dtypes(include=[np.number]).dropna()
    all_cols = list(df.columns)

    if not all_cols:
        raise ValueError(f"No numeric columns found in {name}")

    # Default target: first column (OT for ETT, last col for others)
    if target_col is None:
        if "OT" in all_cols:
            target_col = "OT"
        else:
            target_col = all_cols[-1]

    target_idx = all_cols.index(target_col)
    values = df.values.astype(np.float32)   # (T, C)
    T = len(values)

    # Normalize per channel (fit on train only to avoid leakage)
    train_end = int(T * train_frac)
    val_end   = int(T * (train_frac + val_frac))

    mean = values[:train_end].mean(axis=0)
    std  = values[:train_end].std(axis=0) + 1e-8
    values = (values - mean) / std

    # Chronological split BEFORE windowing (no leakage)
    train_vals = values[:train_end]
    val_vals   = values[train_end - lookback:val_end]   # overlap for context
    test_vals  = values[val_end - lookback:]

    X_train, y_train = _make_windows(train_vals, lookback, horizon, target_idx)
    X_val,   y_val   = _make_windows(val_vals,   lookback, horizon, target_idx)
    X_test,  y_test  = _make_windows(test_vals,  lookback, horizon, target_idx)

    # Determine frequency
    freq_map = {"etth1": "h", "etth2": "h", "ettm1": "15min", "ettm2": "15min",
                "weather": "h", "exchange": "d", "electricity": "h", "traffic": "h"}
    freq = freq_map.get(name, "h")

    n_features = lookback * len(all_cols)
    feature_cols = [f"t{t}_ch{c}" for t in range(lookback) for c in range(len(all_cols))]

    raw = ForecastingRaw(
        X_train=X_train, y_train=y_train,
        X_val=X_val,     y_val=y_val,
        X_test=X_test,   y_test=y_test,
        target_col=target_col,
        feature_cols=feature_cols,
        lookback=lookback,
        horizon=horizon,
        freq=freq,
        task="regression",
        n_classes=1,
    )

    summary = {
        "dataset_name": name,
        "task": "regression",
        "target_col": target_col,
        "target_idx": target_idx,
        "n_channels": len(all_cols),
        "all_channels": all_cols,
        "lookback": lookback,
        "horizon": horizon,
        "freq": freq,
        "T_total": T,
        "n_train": len(X_train),
        "n_val":   len(X_val),
        "n_test":  len(X_test),
        "n_features": n_features,
        "split_strategy": "chronological_no_leakage",
        "normalization": "standard_fit_on_train",
    }

    print(f"[Forecasting] {name.upper()}: {T} timesteps → "
          f"train={len(X_train)}, val={len(X_val)}, test={len(X_test)} windows")
    print(f"  features: {lookback} steps × {len(all_cols)} channels = {n_features}")
    print(f"  target: '{target_col}' (idx={target_idx}), horizon={horizon}")

    return raw, summary
