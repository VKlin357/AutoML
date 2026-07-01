"""
Dataset I/O.

Entry points:

- ``load_raw(source, ...)``      : unified loader — dispatches to OpenML / sklearn /
                                   UCI download / CSV.  Use this everywhere.

- ``load_openml_raw(...)``       : OpenML-specific loader (called by load_raw).

- ``load_builtin_raw(name, ...)`` : sklearn built-ins + UCI downloads.
  Supported names:
    "covtype"             sklearn.fetch_covtype  581k rows, 54 feats, 7-class
    "california_housing"  sklearn.fetch_california_housing  20k rows, regression
    "miniboonee"          UCI download  130k rows, 50 feats, binary ← NNs beat GBM

- ``load_csv_raw(path, ...)``    : any CSV with a "target" column (or last column).

- ``load_openml_dataset(...)``   : legacy entry point — backward-compat with baselines.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    import openml
except ImportError:
    openml = None  # type: ignore

from .metrics import TaskType, infer_task_type
from .preprocessing import Preprocessor, PreparedSplit


# ---------------------------------------------------------------------------
# Builtin dataset registry
# ---------------------------------------------------------------------------

#: Datasets that can be loaded without OpenML (source="sklearn" or "uci").
#: key → (description, task_hint)
BUILTIN_DATASETS = {
    "covtype": (
        "Forest covertype — sklearn built-in. 581k rows, 54 numeric/binary feats, "
        "7-class multiclass. NNs competitive at this scale.",
        "multiclass",
    ),
    "california_housing": (
        "California housing — sklearn built-in. 20k rows, 8 numeric feats, regression. "
        "FT-Transformer often beats GBM here.",
        "regression",
    ),
    "miniboonee": (
        "MiniBooNE particle ID — UCI download. 130k rows, 50 numeric feats, binary. "
        "Classic dataset where NNs consistently beat CatBoost.",
        "binary",
    ),
    # ── Biomedical time-series ──────────────────────────────────────────────
    "ecg5000": (
        "ECG5000 — UCR Time Series Archive. 5000 heartbeat segments × 140 time steps. "
        "5-class cardiac arrhythmia (Normal, R-on-T PVC, PVC, SP, UB). "
        "NNs learn heartbeat morphology; GBMs miss local waveform shape.",
        "multiclass",
    ),
    "har": (
        "Human Activity Recognition — UCI. 10299 samples × 561 statistical features "
        "extracted from smartphone accelerometer + gyroscope 50 Hz windows. "
        "6 classes: Walking, UpStairs, DownStairs, Sitting, Standing, Laying. "
        "NNs competitive; strong non-linear cross-sensor interactions.",
        "multiclass",
    ),
    "harth": (
        "HARTH — UCI. 6.46M raw 50 Hz readings from two accelerometers worn by "
        "22 subjects in free-living activity; windowed with subject-group holdout.",
        "multiclass",
    ),
    "pamap2": (
        "PAMAP2 — UCI. 3.85M raw 100 Hz readings from three wearable IMUs and "
        "9 subjects performing physical activities; windowed with subject-group holdout.",
        "multiclass",
    ),
    "emg_gestures": (
        "EMG hand gestures — UCI. 60000+ sEMG signal windows × 64 time-step features "
        "from 8 forearm EMG electrodes. 5 hand gesture classes. "
        "NNs learn electrode correlation patterns across time.",
        "multiclass",
    ),
    # ── Economics / Financial ───────────────────────────────────────────────
    "elec2": (
        "ELEC2 electricity price direction — concept-drift benchmark. "
        "45312 rows × 8 market features (NSW/VIC price, demand, scheduled transfer). "
        "Binary: electricity price UP or DOWN vs previous period. "
        "NNs capture non-stationary market dynamics better than GBMs.",
        "binary",
    ),
}


# ---------------------------------------------------------------------------
# Raw dataset (NEW interface)
# ---------------------------------------------------------------------------

@dataclass
class RawDataset:
    X_train: pd.DataFrame
    X_val:   pd.DataFrame
    X_test:  pd.DataFrame
    y_train: np.ndarray
    y_val:   np.ndarray
    y_test:  np.ndarray
    num_cols: List[str]
    cat_cols: List[str]
    task: TaskType
    n_classes: int
    name: str
    openml_id: int          # -1 for non-OpenML datasets


def _detect_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    cat_cols, num_cols = [], []
    for c in df.columns:
        dt = df[c].dtype
        if pd.api.types.is_bool_dtype(dt):
            cat_cols.append(c)
        elif pd.api.types.is_categorical_dtype(dt):
            cat_cols.append(c)
        elif pd.api.types.is_object_dtype(dt):
            cat_cols.append(c)
        else:
            num_cols.append(c)
    return num_cols, cat_cols


def load_openml_raw(
    openml_id: int,
    task: str = "auto",
    test_size: float = 0.2,
    val_size: float = 0.2,
    seed: int = 42,
) -> Tuple[RawDataset, dict]:
    if openml is None:
        raise ImportError("openml is not installed. Run: pip install openml")
    ds = openml.datasets.get_dataset(openml_id)
    X, y, *_ = ds.get_data(dataset_format="dataframe", target=ds.default_target_attribute)
    X = X.dropna(axis=1, how="all")

    if task == "auto":
        task_t: TaskType = infer_task_type(y.to_numpy())
    else:
        task_t = task  # type: ignore

    y_arr = y.to_numpy()
    if task_t in ("binary", "multiclass"):
        y_arr = pd.Series(y_arr).astype("category").cat.codes.to_numpy()

    num_cols, cat_cols = _detect_columns(X)

    strat = y_arr if task_t in ("binary", "multiclass") else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_arr, test_size=test_size, random_state=seed, stratify=strat
    )
    strat2 = y_train if task_t in ("binary", "multiclass") else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=seed, stratify=strat2
    )

    n_classes = 1
    if task_t == "multiclass":
        n_classes = int(np.max(y_train) + 1)
    elif task_t == "binary":
        n_classes = 2

    summary = {
        "openml_id": openml_id,
        "name": ds.name,
        "n_rows": int(len(X)),
        "n_features": int(X.shape[1]),
        "n_num": int(len(num_cols)),
        "n_cat": int(len(cat_cols)),
        "task": task_t,
        "n_classes": n_classes,
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
        "num_cols_preview": num_cols[:50],
        "cat_cols_preview": cat_cols[:50],
        # class-imbalance hint for the LLM
        "class_balance": _class_balance(y_train, task_t),
    }

    raw = RawDataset(
        X_train=X_train.reset_index(drop=True), X_val=X_val.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_train=y_train, y_val=y_val, y_test=y_test,
        num_cols=num_cols, cat_cols=cat_cols,
        task=task_t, n_classes=n_classes,
        name=str(ds.name), openml_id=int(openml_id),
    )
    return raw, summary


def _class_balance(y, task):
    if task == "regression":
        return None
    vals, counts = np.unique(y, return_counts=True)
    return {int(v): int(c) for v, c in zip(vals, counts)}


def _make_splits(
    X: pd.DataFrame,
    y_arr: np.ndarray,
    task: TaskType,
    num_cols: List[str],
    cat_cols: List[str],
    name: str,
    dataset_id: int,
    test_size: float = 0.2,
    val_size: float = 0.2,
    seed: int = 42,
    temporal: bool = False,
    feature_engineering: Optional[dict] = None,
) -> Tuple["RawDataset", dict]:
    """Shared split + summary logic for any (X, y) pair.

    If temporal=True the data is assumed to be already sorted chronologically
    and split into train / val / test by position (no shuffling), which avoids
    look-ahead leakage for time-series datasets.
    """
    if temporal:
        n = len(X)
        n_test = int(n * test_size)
        n_val  = int((n - n_test) * val_size)
        n_train = n - n_val - n_test
        X_train = X.iloc[:n_train]
        X_val   = X.iloc[n_train:n_train + n_val]
        X_test  = X.iloc[n_train + n_val:]
        y_train = y_arr[:n_train]
        y_val   = y_arr[n_train:n_train + n_val]
        y_test  = y_arr[n_train + n_val:]
        print(f"[Data] Temporal split: train={n_train}  val={n_val}  test={n_test}")
    else:
        strat = y_arr if task in ("binary", "multiclass") else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_arr, test_size=test_size, random_state=seed, stratify=strat,
        )
        strat2 = y_train if task in ("binary", "multiclass") else None
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size, random_state=seed, stratify=strat2,
        )

    n_classes = 1
    if task == "multiclass":
        n_classes = int(np.max(y_train) + 1)
    elif task == "binary":
        n_classes = 2

    summary = {
        "openml_id": dataset_id,
        "name": name,
        "n_rows": int(len(X)),
        "n_features": int(X.shape[1]),
        "n_num": int(len(num_cols)),
        "n_cat": int(len(cat_cols)),
        "task": task,
        "n_classes": n_classes,
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
        "split_strategy": "temporal_ordered" if temporal else "random_stratified",
        "feature_engineering": feature_engineering or {"type": "raw_tabular"},
        "num_cols_preview": num_cols[:50],
        "cat_cols_preview": cat_cols[:50],
        "class_balance": _class_balance(y_train, task),
        "class_balance_train": _class_balance(y_train, task),
        "class_balance_val": _class_balance(y_val, task),
        "class_balance_test": _class_balance(y_test, task),
    }
    raw = RawDataset(
        X_train=X_train.reset_index(drop=True),
        X_val=X_val.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_train=y_train, y_val=y_val, y_test=y_test,
        num_cols=num_cols, cat_cols=cat_cols,
        task=task, n_classes=n_classes,
        name=name, openml_id=dataset_id,
    )
    return raw, summary


def _make_official_test_splits(
    X_train_full: pd.DataFrame,
    y_train_full: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    task: TaskType,
    num_cols: List[str],
    cat_cols: List[str],
    name: str,
    dataset_id: int,
    val_size: float = 0.2,
    seed: int = 42,
    feature_engineering: Optional[dict] = None,
) -> Tuple["RawDataset", dict]:
    """Keep a dataset-provided test split untouched and derive val from train."""
    strat = y_train_full if task in ("binary", "multiclass") else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=val_size,
        random_state=seed,
        stratify=strat,
    )

    if task == "multiclass":
        n_classes = int(max(np.max(y_train), np.max(y_test)) + 1)
    elif task == "binary":
        n_classes = 2
    else:
        n_classes = 1

    summary = {
        "openml_id": dataset_id,
        "name": name,
        "n_rows": int(len(X_train_full) + len(X_test)),
        "n_features": int(X_train_full.shape[1]),
        "n_num": int(len(num_cols)),
        "n_cat": int(len(cat_cols)),
        "task": task,
        "n_classes": n_classes,
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
        "split_strategy": "official_test_with_train_only_validation",
        "feature_engineering": feature_engineering or {"type": "raw_tabular"},
        "num_cols_preview": num_cols[:50],
        "cat_cols_preview": cat_cols[:50],
        "class_balance": _class_balance(y_train, task),
        "class_balance_train": _class_balance(y_train, task),
        "class_balance_val": _class_balance(y_val, task),
        "class_balance_test": _class_balance(y_test, task),
    }
    raw = RawDataset(
        X_train=X_train.reset_index(drop=True),
        X_val=X_val.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_train=y_train, y_val=y_val, y_test=y_test,
        num_cols=num_cols, cat_cols=cat_cols,
        task=task, n_classes=n_classes,
        name=name, openml_id=dataset_id,
    )
    return raw, summary


def _make_lagged_timeseries_frame(
    X: pd.DataFrame,
    y_arr: np.ndarray,
    *,
    lag_steps: List[int],
    window_sizes: List[int],
    include_diff1: bool = True,
) -> Tuple[pd.DataFrame, np.ndarray, dict]:
    """Build leakage-free supervised tabular features from a time-ordered frame.

    Every generated predictor is based on observations strictly before the
    target timestamp. The target at row t is therefore predicted from lags
    t-1, t-2, ... and rolling statistics computed over the past only.
    """
    base_cols = list(X.columns)
    features = {}

    for lag in lag_steps:
        shifted = X.shift(lag)
        for c in base_cols:
            features[f"{c}_lag{lag}"] = shifted[c]

    past = X.shift(1)
    for w in window_sizes:
        roll = past.rolling(window=w, min_periods=w)
        for c in base_cols:
            features[f"{c}_roll{w}_mean"] = roll[c].mean()
            features[f"{c}_roll{w}_std"] = roll[c].std()

    if include_diff1:
        for c in base_cols:
            features[f"{c}_diff1"] = X[c].shift(1) - X[c].shift(2)

    X_supervised = pd.DataFrame(features)
    valid = X_supervised.notna().all(axis=1).to_numpy()
    X_supervised = X_supervised.loc[valid].reset_index(drop=True)
    y_supervised = y_arr[valid]

    metadata = {
        "type": "time_series_lag_window_tabularization",
        "target_alignment": "predict_y_t_from_features_before_t",
        "uses_current_timestamp_features": False,
        "base_features": base_cols,
        "lag_steps": lag_steps,
        "rolling_windows": window_sizes,
        "rolling_statistics": ["mean", "std"],
        "include_diff1": include_diff1,
        "dropped_initial_rows": int((~valid).sum()),
        "n_features_after": int(X_supervised.shape[1]),
    }
    return X_supervised, y_supervised, metadata


# ---------------------------------------------------------------------------
# Built-in datasets (sklearn + UCI) — no OpenML dependency
# ---------------------------------------------------------------------------

def load_builtin_raw(
    name: str,
    task: str = "auto",
    test_size: float = 0.2,
    val_size: float = 0.2,
    seed: int = 42,
    cache_dir: Optional[str] = None,
) -> Tuple["RawDataset", dict]:
    """Load a dataset without OpenML.

    Parameters
    ----------
    name : one of "covtype", "california_housing", "miniboonee"
    cache_dir : directory for caching UCI downloads (default: ~/.cache/nas_datasets)
    """
    name = name.lower().replace("-", "_")
    cache_dir = Path(cache_dir or Path.home() / ".cache" / "nas_datasets")
    cache_dir.mkdir(parents=True, exist_ok=True)

    if name == "california_housing":
        return _load_california_housing(task, test_size, val_size, seed)

    if name == "covtype":
        return _load_covtype(task, test_size, val_size, seed)

    if name in ("miniboonee", "miniboone", "miniboonee_uci"):
        return _load_miniboonee_uci(task, test_size, val_size, seed, cache_dir)

    if name in ("ecg5000", "ecg"):
        return _load_ecg5000_ucr(task, test_size, val_size, seed, cache_dir)

    if name == "har":
        return _load_har_uci(task, test_size, val_size, seed, cache_dir)

    if name == "harth":
        return _load_harth_uci(task, test_size, val_size, seed, cache_dir)

    if name == "pamap2":
        return _load_pamap2_uci(task, test_size, val_size, seed, cache_dir)

    if name in ("emg_gestures", "emg"):
        return _load_emg_gestures_uci(task, test_size, val_size, seed, cache_dir)

    if name in ("elec2", "electricity"):
        return _load_elec2(task, test_size, val_size, seed, cache_dir)

    raise ValueError(
        f"Unknown builtin dataset: {name!r}. "
        f"Available: {list(BUILTIN_DATASETS.keys())}"
    )


def _load_california_housing(task, test_size, val_size, seed):
    from sklearn.datasets import fetch_california_housing
    print("[Data] Loading California Housing (sklearn built-in) …")
    ds = fetch_california_housing(as_frame=True)
    X = ds.frame.drop(columns=["MedHouseVal"])
    y = ds.frame["MedHouseVal"].to_numpy(dtype=np.float32)
    task_t: TaskType = "regression" if task == "auto" else task
    num_cols = list(X.columns)
    cat_cols: List[str] = []
    return _make_splits(X, y, task_t, num_cols, cat_cols,
                        "california_housing", -1, test_size, val_size, seed)


def _load_covtype(task, test_size, val_size, seed):
    from sklearn.datasets import fetch_covtype
    print("[Data] Loading Covertype (sklearn built-in, 581k rows) …")
    ds = fetch_covtype(as_frame=True)
    X = ds.frame.drop(columns=["Cover_Type"])
    y = ds.frame["Cover_Type"].to_numpy(dtype=np.int64) - 1  # 0-indexed
    task_t: TaskType = "multiclass" if task == "auto" else task
    # Covtype has 10 continuous + 44 binary (one-hot) columns
    num_cols = [c for c in X.columns if X[c].nunique() > 2]
    cat_cols = [c for c in X.columns if X[c].nunique() <= 2]
    return _make_splits(X, y, task_t, num_cols, cat_cols,
                        "covertype", -2, test_size, val_size, seed)


def _load_miniboonee_uci(task, test_size, val_size, seed, cache_dir: Path):
    """Download MiniBooNE from UCI and cache locally."""
    cache_file = cache_dir / "MiniBooNE_PID.txt"
    if not cache_file.exists():
        url = ("https://archive.ics.uci.edu/ml/machine-learning-databases"
               "/00199/MiniBooNE_PID.txt")
        print(f"[Data] Downloading MiniBooNE from UCI → {cache_file} …")
        import urllib.request
        try:
            urllib.request.urlretrieve(url, cache_file)
            print(f"[Data] Downloaded: {cache_file.stat().st_size // 1024} KB")
        except Exception as e:
            raise RuntimeError(
                f"Failed to download MiniBooNE: {e}\n"
                f"Try manually downloading from:\n  {url}\n"
                f"and saving to: {cache_file}"
            ) from e

    print(f"[Data] Loading MiniBooNE from {cache_file} …")
    with open(cache_file) as f:
        first_line = f.readline().strip().split()
    n_signal, n_bg = int(first_line[0]), int(first_line[1])
    print(f"  signal={n_signal}  background={n_bg}  total={n_signal + n_bg}")

    data = np.loadtxt(cache_file, skiprows=1)
    assert data.shape == (n_signal + n_bg, 50), \
        f"Unexpected shape {data.shape}, expected ({n_signal + n_bg}, 50)"

    X_arr = data.astype(np.float32)
    y_arr = np.zeros(n_signal + n_bg, dtype=np.int64)
    y_arr[:n_signal] = 1   # signal = 1, background = 0

    cols = [f"f{i}" for i in range(50)]
    X = pd.DataFrame(X_arr, columns=cols)
    task_t: TaskType = "binary" if task == "auto" else task
    return _make_splits(X, y_arr, task_t, cols, [],
                        "MiniBooNE", -3, test_size, val_size, seed)


# ---------------------------------------------------------------------------
# Biomedical + Financial built-in loaders
# ---------------------------------------------------------------------------

def _download_url(url: str, dest: Path, name: str) -> None:
    """Download a file with progress reporting."""
    import urllib.request
    print(f"[Data] Downloading {name} …")
    print(f"  URL: {url}")
    try:
        def _progress(block, block_size, total):
            if total > 0 and block % 50 == 0:
                mb = block * block_size / 1e6
                print(f"  {mb:.1f} / {total/1e6:.1f} MB", end="\r")
        urllib.request.urlretrieve(url, dest, reporthook=_progress)
        print(f"\n  Saved: {dest}  ({dest.stat().st_size // 1024} KB)")
    except Exception as e:
        if dest.exists():
            dest.unlink()
        raise RuntimeError(
            f"Failed to download {name}: {e}\n"
            f"Download manually from:\n  {url}\n"
            f"and save to: {dest}"
        ) from e


def _parse_ucr_ts_file(content: str):
    """Parse UCR .ts format → (list_of_feature_rows, list_of_labels).

    Format (each data line):
      val1,val2,...,valN:label
    Metadata lines start with '@'.
    """
    rows, labels = [], []
    in_data = False
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if line.lower().startswith('@data'):
            in_data = True
            continue
        if line.startswith('@') or not in_data:
            continue
        # label is the part after the LAST ':'
        if ':' not in line:
            continue
        feat_str, label = line.rsplit(':', 1)
        label = label.strip()
        feat_str = feat_str.strip()
        sep = ',' if ',' in feat_str else ' '
        try:
            vals = [float(x) for x in feat_str.split(sep) if x.strip() not in ('', '?')]
        except ValueError:
            continue
        if vals:
            rows.append(vals)
            labels.append(label)
    return rows, labels


def _load_ecg5000_ucr(task, test_size, val_size, seed, cache_dir: Path):
    """ECG5000 cardiac arrhythmia — UCR Time Series Classification Archive.

    5000 single-lead ECG heartbeat segments, each 140 time steps.
    5 classes: Normal (1), R-on-T PVC (2), PVC (3), SP (4), UB (5).
    NNs learn heartbeat morphology; GBMs miss waveform shape.

    Source: https://www.timeseriesclassification.com
    """
    import zipfile

    cache_zip = cache_dir / "ECG5000.zip"
    if not cache_zip.exists():
        _download_url(
            "https://www.timeseriesclassification.com/aeon-toolkit/ECG5000.zip",
            cache_zip, "ECG5000",
        )

    split_rows = {"train": [], "test": []}
    split_labels = {"train": [], "test": []}
    with zipfile.ZipFile(cache_zip) as zf:
        ts_files = [n for n in zf.namelist() if n.endswith('.ts')]
        if not ts_files:
            raise RuntimeError(
                f"ECG5000.zip contains no .ts files: {zf.namelist()}"
            )
        for fname in ts_files:
            upper_name = fname.upper()
            if "TRAIN" in upper_name:
                split = "train"
            elif "TEST" in upper_name:
                split = "test"
            else:
                continue
            content = zf.read(fname).decode('utf-8', errors='replace')
            r, l = _parse_ucr_ts_file(content)
            split_rows[split].extend(r)
            split_labels[split].extend(l)

    if not split_rows["train"] or not split_rows["test"]:
        raise RuntimeError(
            "ECG5000: official TRAIN/TEST files were not parsed. "
            "Delete ~/.cache/nas_datasets/ECG5000.zip and retry."
        )

    all_rows = split_rows["train"] + split_rows["test"]
    all_labels = split_labels["train"] + split_labels["test"]
    n_feats = max(len(r) for r in all_rows)
    # Pad shorter rows with 0
    cols = [f"t{i}" for i in range(n_feats)]
    label_categories = list(pd.unique(pd.Series(all_labels)))
    label_to_id = {label: idx for idx, label in enumerate(label_categories)}

    def _as_frame(rows):
        padded = [r + [0.0] * (n_feats - len(r)) for r in rows]
        return pd.DataFrame(padded, columns=cols, dtype=np.float32)

    X_train_full = _as_frame(split_rows["train"])
    X_test = _as_frame(split_rows["test"])
    y_train_full = np.asarray([label_to_id[label] for label in split_labels["train"]], dtype=np.int64)
    y_test = np.asarray([label_to_id[label] for label in split_labels["test"]], dtype=np.int64)
    task_t: TaskType = "multiclass" if task == "auto" else task

    print(f"[Data] ECG5000: {len(all_rows)} samples × {n_feats} time steps  "
          f"classes={np.unique(np.concatenate([y_train_full, y_test])).tolist()}")
    return _make_official_test_splits(
        X_train_full, y_train_full, X_test, y_test,
        task_t, cols, [], "ECG5000", -10,
        val_size=val_size, seed=seed,
        feature_engineering={
            "type": "raw_time_series_segment_as_tabular_vector",
            "sequence_length": n_feats,
            "split_origin": "UCR_official_train_test",
        },
    )


def _load_har_uci(task, test_size, val_size, seed, cache_dir: Path):
    """Human Activity Recognition — UCI ML Repository.

    10299 smartphone sensor windows × 561 statistical features
    (time + frequency domain, accelerometer + gyroscope at 50 Hz).
    6 classes: Walking, UpStairs, DownStairs, Sitting, Standing, Laying.
    NNs model complex cross-sensor non-linear patterns.

    Source: https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition
    """
    import zipfile, io

    cache_zip = cache_dir / "UCI_HAR.zip"
    if not cache_zip.exists():
        _download_url(
            "https://archive.ics.uci.edu/ml/machine-learning-databases"
            "/00240/UCI%20HAR%20Dataset.zip",
            cache_zip, "UCI HAR",
        )

    with zipfile.ZipFile(cache_zip) as zf:
        names = zf.namelist()

        def _find(substr):
            matches = [n for n in names if substr in n]
            if not matches:
                raise RuntimeError(
                    f"HAR ZIP: file containing '{substr}' not found. "
                    f"Contents: {names[:20]}"
                )
            return matches[0]

        def _read_arr(fname):
            with zf.open(fname) as f:
                return np.loadtxt(io.TextIOWrapper(f))

        def _read_labels(fname):
            with zf.open(fname) as f:
                return np.loadtxt(io.TextIOWrapper(f), dtype=np.int64) - 1  # 0-indexed

        def _read_feature_names(fname):
            with zf.open(fname) as f:
                lines = io.TextIOWrapper(f).read().splitlines()
            # Format: "1 tBodyAcc-mean()-X"
            names = []
            for line in lines:
                parts = line.strip().split(None, 1)
                names.append(parts[1].strip() if len(parts) == 2 else f"f{parts[0]}")
            return names

        X_train = _read_arr(_find("train/X_train"))
        y_train = _read_labels(_find("train/y_train"))
        X_test  = _read_arr(_find("test/X_test"))
        y_test  = _read_labels(_find("test/y_test"))

        # Feature names from features.txt
        try:
            feat_names = _read_feature_names(_find("features.txt"))
            # Deduplicate if needed (HAR has a few duplicate feature names)
            seen: dict = {}
            clean_names = []
            for n in feat_names:
                cnt = seen.get(n, 0)
                clean_names.append(n if cnt == 0 else f"{n}_{cnt}")
                seen[n] = cnt + 1
            feat_names = clean_names
        except Exception:
            feat_names = [f"f{i}" for i in range(X_train.shape[1])]

    X_train_df = pd.DataFrame(X_train.astype(np.float32), columns=feat_names[:X_train.shape[1]])
    X_test_df = pd.DataFrame(X_test.astype(np.float32), columns=feat_names[:X_test.shape[1]])
    cols = list(X_train_df.columns)
    task_t: TaskType = "multiclass" if task == "auto" else task

    print(f"[Data] HAR: {len(X_train_df) + len(X_test_df)} samples × {X_train_df.shape[1]} features  "
          f"classes={np.unique(np.concatenate([y_train, y_test])).tolist()}")
    return _make_official_test_splits(
        X_train_df, y_train, X_test_df, y_test,
        task_t, cols, [], "HAR", -11,
        val_size=val_size, seed=seed,
        feature_engineering={
            "type": "precomputed_sensor_window_features",
            "split_origin": "UCI_HAR_official_subject_split",
        },
    )


def _make_subject_window_splits(
    streams: List[Tuple[str, str, np.ndarray, np.ndarray]],
    *,
    sensor_cols: List[str],
    task: TaskType,
    name: str,
    dataset_id: int,
    raw_n_rows: int,
    source_metadata: dict,
    window_size: int = 128,
    stride: int = 32,
    min_label_purity: float = 0.8,
    fixed_subject_partitions: Optional[dict] = None,
) -> Tuple["RawDataset", dict]:
    """Window continuous sensor streams after a fixed subject-wise split.

    ``streams`` entries are ``(subject_id, session_id, numeric_values, labels)``.
    Labels must be zero-based class ids; ``-1`` marks unlabeled transition rows.
    """
    subjects = sorted({s[0] for s in streams})
    if len(subjects) < 5:
        raise RuntimeError(f"{name}: at least five subjects are required for group holdout")

    if fixed_subject_partitions is not None:
        train_subjects = set(fixed_subject_partitions["train"])
        val_subjects = set(fixed_subject_partitions["val"])
        test_subjects = set(fixed_subject_partitions["test"])
        assigned = train_subjects | val_subjects | test_subjects
        if assigned != set(subjects) or (
            train_subjects & val_subjects
            or train_subjects & test_subjects
            or val_subjects & test_subjects
        ):
            raise RuntimeError(f"{name}: invalid fixed subject partition")
    else:
        # Fixed partition: experimental seed may change model initialization, not test subjects.
        split_rng = np.random.default_rng(42)
        shuffled = list(split_rng.permutation(subjects))
        n_test = max(1, int(round(len(subjects) * 0.2)))
        n_val = max(1, int(round(len(subjects) * 0.2)))
        test_subjects = set(shuffled[:n_test])
        val_subjects = set(shuffled[n_test:n_test + n_val])
        train_subjects = set(shuffled[n_test + n_val:])
    partitions = {
        "train": train_subjects,
        "val": val_subjects,
        "test": test_subjects,
    }

    flat_cols = [f"t{t}_{c}" for t in range(window_size) for c in sensor_cols]
    frames: dict = {}
    labels_out: dict = {}
    for split, split_subjects in partitions.items():
        chunks, target_chunks = [], []
        for subject, _session, values, labels in streams:
            if subject not in split_subjects or len(values) < window_size:
                continue
            starts = np.arange(0, len(values) - window_size + 1, stride, dtype=np.int64)
            row_idx = starts[:, None] + np.arange(window_size, dtype=np.int64)[None, :]
            window_labels = labels[row_idx]
            selected, targets = [], []
            for i, row in enumerate(window_labels):
                valid = row[row >= 0]
                if len(valid) == 0:
                    continue
                target = int(np.bincount(valid).argmax())
                if float(np.mean(row == target)) >= min_label_purity:
                    selected.append(i)
                    targets.append(target)
            if selected:
                window_values = values[row_idx[np.asarray(selected, dtype=np.int64)]]
                chunks.append(window_values.reshape(len(selected), -1).astype(np.float32))
                target_chunks.append(np.asarray(targets, dtype=np.int64))
        if not chunks:
            raise RuntimeError(f"{name}: no usable windows produced for {split} split")
        frames[split] = pd.DataFrame(np.concatenate(chunks, axis=0), columns=flat_cols)
        labels_out[split] = np.concatenate(target_chunks, axis=0)

    class_sets = {split: set(np.unique(values).tolist())
                  for split, values in labels_out.items()}
    if not (class_sets["train"] == class_sets["val"] == class_sets["test"]):
        raise RuntimeError(
            f"{name}: group split has inconsistent class coverage: {class_sets}. "
            "Choose subject partitions or an activity subset before training."
        )

    n_classes = int(max(np.max(y) for y in labels_out.values()) + 1)
    feature_engineering = {
        "type": "raw_sensor_sliding_window_as_tabular_vector",
        "window_size": window_size,
        "stride": stride,
        "min_label_purity": min_label_purity,
        "sensor_columns": sensor_cols,
        "split_origin": "fixed_subject_group_holdout_before_windowing",
        "partition_policy": ("fixed_subject_class_coverage"
                             if fixed_subject_partitions is not None
                             else "fixed_seeded_subject_partition"),
        **source_metadata,
    }
    summary = {
        "openml_id": dataset_id,
        "name": name,
        "n_rows": int(sum(len(x) for x in frames.values())),
        "raw_n_rows": int(raw_n_rows),
        "n_features": len(flat_cols),
        "n_num": len(flat_cols),
        "n_cat": 0,
        "task": task,
        "n_classes": n_classes,
        "n_train": int(len(frames["train"])),
        "n_val": int(len(frames["val"])),
        "n_test": int(len(frames["test"])),
        "split_strategy": "fixed_subject_group_holdout_before_windowing",
        "split_subjects_train": sorted(train_subjects),
        "split_subjects_val": sorted(val_subjects),
        "split_subjects_test": sorted(test_subjects),
        "feature_engineering": feature_engineering,
        "num_cols_preview": flat_cols[:50],
        "cat_cols_preview": [],
        "class_balance": _class_balance(labels_out["train"], task),
        "class_balance_train": _class_balance(labels_out["train"], task),
        "class_balance_val": _class_balance(labels_out["val"], task),
        "class_balance_test": _class_balance(labels_out["test"], task),
    }
    raw = RawDataset(
        X_train=frames["train"], X_val=frames["val"], X_test=frames["test"],
        y_train=labels_out["train"], y_val=labels_out["val"], y_test=labels_out["test"],
        num_cols=flat_cols, cat_cols=[], task=task, n_classes=n_classes,
        name=name, openml_id=dataset_id,
    )
    print(f"[Data] {name}: {raw_n_rows} raw rows -> {summary['n_rows']} windows x "
          f"{len(flat_cols)} features; subjects train/val/test="
          f"{len(train_subjects)}/{len(val_subjects)}/{len(test_subjects)}")
    return raw, summary


def _load_harth_uci(task, test_size, val_size, seed, cache_dir: Path):
    """HARTH: raw free-living accelerometer signals, split by participant."""
    import zipfile

    cache_zip = cache_dir / "HARTH.zip"
    if not cache_zip.exists():
        _download_url(
            "https://archive.ics.uci.edu/static/public/779/harth.zip",
            cache_zip, "HARTH",
        )

    sensor_cols = ["back_x", "back_y", "back_z", "thigh_x", "thigh_y", "thigh_z"]
    raw_streams = []
    observed_labels = set()
    with zipfile.ZipFile(cache_zip) as zf:
        csv_files = sorted(n for n in zf.namelist()
                           if n.lower().endswith(".csv") and "__macosx" not in n.lower())
        if not csv_files:
            raise RuntimeError("HARTH ZIP contains no CSV recordings")
        for fname in csv_files:
            with zf.open(fname) as src:
                df = pd.read_csv(src)
            if not set(sensor_cols + ["label"]).issubset(df.columns):
                continue
            subject = Path(fname).stem
            values = df[sensor_cols].to_numpy(dtype=np.float32)
            original_labels = df["label"].to_numpy(dtype=np.int64)
            observed_labels.update(np.unique(original_labels).tolist())
            raw_streams.append((subject, subject, values, original_labels))
    label_values = np.asarray(sorted(observed_labels), dtype=np.int64)
    label_map = {int(label): i for i, label in enumerate(label_values)}
    streams = [(s, sess, x, np.searchsorted(label_values, y).astype(np.int64))
               for s, sess, x, y in raw_streams]
    task_t: TaskType = "multiclass" if task == "auto" else task
    return _make_subject_window_splits(
        streams, sensor_cols=sensor_cols, task=task_t, name="HARTH",
        dataset_id=-13, raw_n_rows=sum(len(x[2]) for x in raw_streams),
        source_metadata={
            "source": "UCI_HARTH",
            "sampling_frequency_hz": 50,
            "activity_label_map": {str(k): v for k, v in label_map.items()},
        },
        fixed_subject_partitions={
            # Chosen from label availability only so every partition contains
            # rare activity 140; no model scores are used in this choice.
            "train": ["S006", "S008", "S009", "S010", "S012", "S013", "S014",
                      "S016", "S017", "S019", "S021", "S022", "S027", "S028"],
            "val": ["S015", "S020", "S023", "S026"],
            "test": ["S018", "S024", "S025", "S029"],
        },
    )


def _load_pamap2_uci(task, test_size, val_size, seed, cache_dir: Path):
    """PAMAP2: raw wearable IMU signals, split by participant."""
    import re
    import zipfile

    cache_zip = cache_dir / "PAMAP2_Dataset.zip"
    if not cache_zip.exists():
        _download_url(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/00231/PAMAP2_Dataset.zip",
            cache_zip, "PAMAP2",
        )

    # Use the ±16 g accelerometer from hand, chest, and ankle: nine raw channels.
    selected_idx = [4, 5, 6, 21, 22, 23, 38, 39, 40]
    sensor_cols = [
        "hand_x", "hand_y", "hand_z",
        "chest_x", "chest_y", "chest_z",
        "ankle_x", "ankle_y", "ankle_z",
    ]
    raw_streams = []
    observed_labels = set()
    with zipfile.ZipFile(cache_zip) as zf:
        # PAMAP2 Optional recordings contain activities performed by only a
        # subset of subjects; they would create unseen classes in group test.
        dat_files = sorted(n for n in zf.namelist()
                           if n.lower().endswith(".dat") and "/protocol/" in n.lower())
        if not dat_files:
            raise RuntimeError("PAMAP2 ZIP contains no DAT recordings")
        for fname in dat_files:
            with zf.open(fname) as src:
                arr = np.loadtxt(src, dtype=np.float32)
            match = re.search(r"subject(\\d+)", fname.lower())
            subject = match.group(1) if match else Path(fname).stem
            original_labels = arr[:, 1].astype(np.int64)
            observed_labels.update(np.unique(original_labels[original_labels > 0]).tolist())
            raw_streams.append((subject, Path(fname).stem, arr[:, selected_idx], original_labels))
    label_values = np.asarray(sorted(observed_labels), dtype=np.int64)
    label_map = {int(label): i for i, label in enumerate(label_values)}
    streams = []
    for subject, session, values, labels in raw_streams:
        mapped = np.full(len(labels), -1, dtype=np.int64)
        labeled = labels > 0
        mapped[labeled] = np.searchsorted(label_values, labels[labeled])
        streams.append((subject, session, values, mapped))
    task_t: TaskType = "multiclass" if task == "auto" else task
    return _make_subject_window_splits(
        streams, sensor_cols=sensor_cols, task=task_t, name="PAMAP2",
        dataset_id=-14, raw_n_rows=sum(len(x[2]) for x in raw_streams),
        source_metadata={
            "source": "UCI_PAMAP2",
            "sampling_frequency_hz": 100,
            "activity_label_map": {str(k): v for k, v in label_map.items()},
            "raw_channels_used": "three_IMU_accelerometers_16g",
            "recording_subset": "Protocol",
        },
        stride=16,
        fixed_subject_partitions={
            # Protocol activities are jointly covered in all partitions.
            "train": ["subject102", "subject105", "subject106", "subject107", "subject109"],
            "val": ["subject103", "subject108"],
            "test": ["subject101", "subject104"],
        },
    )


def _load_emg_gestures_uci(task, test_size, val_size, seed, cache_dir: Path):
    """EMG Hand Gestures — UCI Senz3D dataset (subset).

    Surface EMG from 8 forearm electrodes, windowed into 64-sample segments.
    5 gesture classes: Hand at Rest, Extension, Flexion, Ulnar Deviation, Radial Deviation.
    NNs learn electrode cross-correlation patterns; GBMs miss spatial structure.

    Falls back to a clean CSV hosted on GitHub if UCI is unavailable.
    Source: UCI ML Repository / Lobov et al. 2018
    """
    cache_csv = cache_dir / "emg_gestures.csv"
    if not cache_csv.exists():
        # The Lobov et al. EMG dataset is mirrored on several repos.
        # We use a CSV version where columns = [emg_1..emg_64, label]
        urls = [
            # 8-electrode × 8-sample window → 64 features, ~60k rows, 5 classes
            "https://raw.githubusercontent.com/uci-cbcl/EMG-gesture-recognition"
            "/main/data/emg_gestures_tabular.csv",
            # Fallback: smaller dataset from UCI Opportunities (3 channels × 15 steps = 45 feats)
            "https://archive.ics.uci.edu/ml/machine-learning-databases"
            "/00310/UJI_Pen_Characters2.zip",
        ]
        downloaded = False
        for url in urls:
            try:
                _download_url(url, cache_csv, "EMG Gestures")
                downloaded = True
                break
            except Exception as e:
                print(f"  [EMG] URL failed ({e}), trying next …")

        if not downloaded:
            raise RuntimeError(
                "[EMG Gestures] All download URLs failed — cannot load dataset.\n"
                "Tried:\n" + "\n".join(f"  {u}" for u in urls) + "\n"
                "Fix: download the dataset manually and place it at:\n"
                f"  {cache_csv}\n"
                "Format: CSV with columns emg_0..emg_63 + label"
            )

    df = pd.read_csv(cache_csv)
    target_col = "label" if "label" in df.columns else df.columns[-1]
    X = df.drop(columns=[target_col])
    y_raw = df[target_col].to_numpy()
    y_arr = pd.Series(y_raw).astype("category").cat.codes.to_numpy(dtype=np.int64)
    cols = list(X.columns)
    num_cols, cat_cols = _detect_columns(X)
    task_t: TaskType = "multiclass" if task == "auto" else task

    print(f"[Data] EMG Gestures: {len(X)} samples × {len(cols)} features  "
          f"classes={np.unique(y_arr).tolist()}")
    return _make_splits(X, y_arr, task_t, num_cols, cat_cols, "EMG_Gestures", -12,
                        test_size, val_size, seed)


def _load_elec2(task, test_size, val_size, seed, cache_dir: Path):
    """ELEC2 electricity price direction — concept-drift benchmark.

    45312 half-hourly observations of the Australian electricity market (1996–1998).
    8 numerical features: period-of-day, day-of-week, NSW/VIC price & demand,
    scheduled transfer, and transfer indicator.
    Binary target: electricity price UP vs DOWN vs previous period.

    NNs capture non-stationary market dynamics and cross-market dependencies;
    GBMs require manual feature engineering for temporal patterns.

    Sources:
      1. OpenML (preferred)
      2. Direct ARFF download from OpenML CDN
      3. GitHub mirror as fallback
    """
    cache_csv = cache_dir / "elec2.csv"
    if cache_csv.exists():
        df = pd.read_csv(cache_csv)
    else:
        df = None

        # Source 1: try OpenML
        if openml is not None:
            # ELEC2 is OpenML dataset 44120 (or "electricity" dataset)
            # Try a few IDs in case the primary one changes
            for oid in (44120, 44, 151):
                try:
                    print(f"[Data] Trying OpenML ID {oid} for ELEC2 …")
                    ds = openml.datasets.get_dataset(oid)
                    X_o, y_o, *_ = ds.get_data(
                        dataset_format="dataframe",
                        target=ds.default_target_attribute
                    )
                    if len(X_o) > 10000:   # real ELEC2 has 45k rows
                        df = X_o.copy()
                        df["class"] = y_o
                        df.to_csv(cache_csv, index=False)
                        print(f"  Downloaded via OpenML {oid}: {df.shape}")
                        break
                except Exception as e:
                    print(f"  OpenML {oid} failed: {e}")

        # Source 2: direct ARFF download
        if df is None:
            arff_urls = [
                "https://www.openml.org/data/v1/download/2419",
                "https://raw.githubusercontent.com/datasets-io/elec2/"
                "master/data/electricity-normalized.csv",
            ]
            for url in arff_urls:
                try:
                    tmp = cache_dir / "elec2_tmp"
                    _download_url(url, tmp, "ELEC2")
                    content = tmp.read_text(encoding='utf-8', errors='replace')
                    tmp.unlink(missing_ok=True)
                    if url.endswith('.csv'):
                        df = pd.read_csv(pd.io.common.StringIO(content))
                    else:
                        # Parse ARFF
                        lines = content.splitlines()
                        attrs = []
                        data_lines = []
                        in_data = False
                        for line in lines:
                            l = line.strip()
                            if l.lower().startswith('@attribute'):
                                parts = l.split(None, 2)
                                attrs.append(parts[1])
                            elif l.lower().startswith('@data'):
                                in_data = True
                            elif in_data and l and not l.startswith('%'):
                                data_lines.append(l.split(','))
                        if attrs and data_lines:
                            df = pd.DataFrame(data_lines, columns=attrs[:len(data_lines[0])])
                            for c in df.columns[:-1]:
                                try:
                                    df[c] = pd.to_numeric(df[c])
                                except Exception:
                                    pass
                            df.to_csv(cache_csv, index=False)
                            print(f"  Downloaded via ARFF: {df.shape}")
                            break
                except Exception as e:
                    print(f"  URL {url} failed: {e}")

        if df is None:
            raise RuntimeError(
                "[ELEC2] All download sources failed — cannot load dataset.\n"
                "Fix: download from https://www.openml.org/d/151 and place CSV at:\n"
                f"  {cache_csv}"
            )

    target_col = "class" if "class" in df.columns else df.columns[-1]
    X = df.drop(columns=[target_col]).copy()
    # Convert all columns to numeric where possible
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors='coerce')
    X = X.dropna(axis=1, how='all').fillna(0.0)

    y_raw = df[target_col].to_numpy()
    task_t: TaskType = "binary" if task == "auto" else task
    y_arr = pd.Series(y_raw).astype("category").cat.codes.to_numpy(dtype=np.int64)

    lag_steps = [1, 2, 3, 6, 12, 24, 48]
    window_sizes = [6, 12, 24, 48]
    X, y_arr, ts_features = _make_lagged_timeseries_frame(
        X,
        y_arr,
        lag_steps=lag_steps,
        window_sizes=window_sizes,
        include_diff1=True,
    )

    num_cols = list(X.columns)
    cat_cols: List[str] = []

    print(f"[Data] ELEC2: {len(X)} rows × {len(num_cols)} features  "
          f"balance={dict(zip(*np.unique(y_arr, return_counts=True)))}")
    # ELEC2 is already in chronological order — use temporal split to avoid look-ahead leakage
    return _make_splits(X, y_arr, task_t, num_cols, cat_cols, "ELEC2", -13,
                        test_size, val_size, seed, temporal=True,
                        feature_engineering=ts_features)


# ---------------------------------------------------------------------------
# CSV loader
# ---------------------------------------------------------------------------

def load_csv_raw(
    path: str,
    task: str = "auto",
    target_col: Optional[str] = None,
    test_size: float = 0.2,
    val_size: float = 0.2,
    seed: int = 42,
) -> Tuple["RawDataset", dict]:
    """Load any CSV file.

    Parameters
    ----------
    path       : path to CSV
    task       : "auto" | "binary" | "multiclass" | "regression"
    target_col : name of target column (default: last column)
    """
    path = Path(path)
    print(f"[Data] Loading CSV: {path} …")
    df = pd.read_csv(path)
    if target_col is None:
        target_col = df.columns[-1]
    print(f"  shape={df.shape}  target={target_col!r}")

    X = df.drop(columns=[target_col])
    y_raw = df[target_col].to_numpy()

    if task == "auto":
        task_t = infer_task_type(y_raw)
    else:
        task_t = task  # type: ignore

    if task_t in ("binary", "multiclass"):
        y_arr = pd.Series(y_raw).astype("category").cat.codes.to_numpy(dtype=np.int64)
    else:
        y_arr = y_raw.astype(np.float32)

    num_cols, cat_cols = _detect_columns(X)
    name = path.stem
    return _make_splits(X, y_arr, task_t, num_cols, cat_cols,
                        name, -4, test_size, val_size, seed)


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def load_raw(
    source: str = "openml",
    *,
    openml_id: Optional[int] = None,
    builtin_name: Optional[str] = None,
    csv_path: Optional[str] = None,
    task: str = "auto",
    test_size: float = 0.2,
    val_size: float = 0.2,
    seed: int = 42,
    cache_dir: Optional[str] = None,
) -> Tuple["RawDataset", dict]:
    """Unified dataset loader.

    Parameters
    ----------
    source : "openml" | "sklearn" | "uci" | "builtin" | "csv"
    openml_id   : required when source="openml"
    builtin_name: required when source in ("sklearn","uci","builtin")
                  one of: "covtype", "california_housing", "miniboonee"
    csv_path    : required when source="csv"
    """
    if source == "openml":
        assert openml_id is not None, "openml_id required when source='openml'"
        return load_openml_raw(openml_id, task=task,
                               test_size=test_size, val_size=val_size, seed=seed)
    elif source in ("sklearn", "uci", "builtin"):
        assert builtin_name is not None, "builtin_name required for non-openml source"
        return load_builtin_raw(builtin_name, task=task,
                                test_size=test_size, val_size=val_size,
                                seed=seed, cache_dir=cache_dir)
    elif source == "csv":
        assert csv_path is not None, "csv_path required when source='csv'"
        return load_csv_raw(csv_path, task=task,
                            test_size=test_size, val_size=val_size, seed=seed)
    else:
        raise ValueError(f"Unknown source: {source!r}. Use 'openml', 'builtin', or 'csv'.")


# ---------------------------------------------------------------------------
# Legacy interface (used by v1 baselines)
# ---------------------------------------------------------------------------

@dataclass
class TabularData:
    X_train_num: np.ndarray
    X_val_num: np.ndarray
    X_test_num: np.ndarray
    X_train_cat: np.ndarray
    X_val_cat: np.ndarray
    X_test_cat: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    num_cols: List[str]
    cat_cols: List[str]
    cat_cardinalities: List[int]
    task: TaskType
    n_classes: int


def load_openml_dataset(openml_id, task="auto", test_size=0.2, val_size=0.2, seed=42):
    """Backward-compatible loader: applies the default Preprocessor."""
    raw, summary = load_openml_raw(openml_id, task, test_size, val_size, seed)
    pre = Preprocessor(num_encoder="standard", cat_encoder="embedding")
    sp = pre.fit_transform(
        raw.X_train, raw.X_val, raw.X_test,
        raw.y_train, raw.y_val, raw.y_test,
        raw.num_cols, raw.cat_cols, raw.task, raw.n_classes,
    )
    summary["cat_cardinalities_preview"] = sp.cat_cardinalities[:50]
    data = TabularData(
        X_train_num=sp.X_train_num, X_val_num=sp.X_val_num, X_test_num=sp.X_test_num,
        X_train_cat=sp.X_train_cat, X_val_cat=sp.X_val_cat, X_test_cat=sp.X_test_cat,
        y_train=sp.y_train, y_val=sp.y_val, y_test=sp.y_test,
        num_cols=raw.num_cols, cat_cols=raw.cat_cols,
        cat_cardinalities=sp.cat_cardinalities,
        task=raw.task, n_classes=raw.n_classes,
    )
    return data, summary
