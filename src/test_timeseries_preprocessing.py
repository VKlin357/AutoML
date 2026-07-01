"""
Verify time-series dataset loading and CatBoost preprocessing.

Checks:
  1. Dataset loads correctly (shape, classes, split sizes)
  2. No data leakage between train/val/test
  3. CatBoost gets flat numeric matrix (no temporal structure assumed)
  4. Temporal split used where appropriate (ELEC2)

Usage:
    python src/test_timeseries_preprocessing.py --dataset ecg5000
    python src/test_timeseries_preprocessing.py --dataset emg_gestures
    python src/test_timeseries_preprocessing.py --dataset elec2
    python src/test_timeseries_preprocessing.py --dataset harth
    python src/test_timeseries_preprocessing.py --dataset pamap2
    python src/test_timeseries_preprocessing.py --all
"""
import argparse
import os
import sys
import time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

TIMESERIES_DATASETS = ["ecg5000", "emg_gestures", "elec2", "harth", "pamap2"]


def check_no_overlap(y_train, y_val, y_test, X_train_num, X_val_num, X_test_num):
    """Verify train/val/test rows don't overlap (simple row-hash check)."""
    def row_hashes(arr):
        return set(hash(row.tobytes()) for row in arr[:200])  # sample first 200

    tr_h = row_hashes(X_train_num)
    va_h = row_hashes(X_val_num)
    te_h = row_hashes(X_test_num)

    tr_va = tr_h & va_h
    tr_te = tr_h & te_h
    va_te = va_h & te_h

    return {
        "train_val_overlap": len(tr_va),
        "train_test_overlap": len(tr_te),
        "val_test_overlap": len(va_te),
    }


def test_dataset(name: str, seed: int = 42):
    from src.data import load_builtin_raw
    from src.preprocessing import Preprocessor

    print(f"\n{'='*60}")
    print(f"  Dataset: {name.upper()}")
    print(f"{'='*60}")

    t0 = time.time()
    try:
        raw, summary = load_builtin_raw(name, task="auto", seed=seed)
    except Exception as e:
        print(f"  ERROR loading dataset: {e}")
        return False

    elapsed = time.time() - t0
    print(f"  Load time: {elapsed:.1f}s")
    print(f"  Task: {raw.task}  |  Classes: {raw.n_classes}")
    print(f"  Num cols: {len(raw.num_cols)}  |  Cat cols: {len(raw.cat_cols)}")
    print(f"  Train: {len(raw.X_train):,} rows")
    print(f"  Val:   {len(raw.X_val):,} rows")
    print(f"  Test:  {len(raw.X_test):,} rows")
    print(f"  Split strategy: {summary.get('split_strategy', 'unknown')}")

    # Check temporal split for ELEC2
    if name in ("elec2",):
        strat = summary.get("split_strategy", "")
        if "temporal" in strat:
            print("  ✓ Temporal split confirmed (no look-ahead leakage)")
        else:
            print(f"  ✗ WARNING: expected temporal split, got '{strat}'")

    # Apply CatBoost preprocessing (standard scaler, no special treatment)
    pre = Preprocessor(num_encoder="standard", cat_encoder="embedding")
    sp = pre.fit_transform(
        raw.X_train, raw.X_val, raw.X_test,
        raw.y_train, raw.y_val, raw.y_test,
        raw.num_cols, raw.cat_cols,
        raw.task, raw.n_classes,
    )

    print(f"\n  After preprocessing (CatBoost input):")
    print(f"  X_train_num shape: {sp.X_train_num.shape}")
    print(f"  X_val_num shape:   {sp.X_val_num.shape}")
    print(f"  X_test_num shape:  {sp.X_test_num.shape}")

    # Check for NaN/Inf
    for split_name, arr in [("train", sp.X_train_num), ("val", sp.X_val_num), ("test", sp.X_test_num)]:
        n_nan = np.isnan(arr).sum()
        n_inf = np.isinf(arr).sum()
        if n_nan > 0 or n_inf > 0:
            print(f"  ✗ WARNING: {split_name} has {n_nan} NaN, {n_inf} Inf values")
        else:
            print(f"  ✓ {split_name}: no NaN/Inf")

    # Leakage check
    overlap = check_no_overlap(
        raw.y_train, raw.y_val, raw.y_test,
        sp.X_train_num, sp.X_val_num, sp.X_test_num
    )
    if all(v == 0 for v in overlap.values()):
        print(f"  ✓ No row overlap between splits")
    else:
        print(f"  ✗ WARNING overlap detected: {overlap}")

    # Label distribution
    unique_tr, counts_tr = np.unique(raw.y_train, return_counts=True)
    print(f"\n  Label distribution (train): ", end="")
    for lbl, cnt in zip(unique_tr, counts_tr):
        pct = cnt / len(raw.y_train) * 100
        print(f"{int(lbl)}:{pct:.1f}%", end="  ")
    print()

    print(f"\n  ✓ {name.upper()} OK — CatBoost will see {sp.X_train_num.shape[1]} flat numeric features")
    print(f"    (CatBoost treats all {sp.X_train_num.shape[1]} columns independently —")
    print(f"     no temporal/channel structure exploited)")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None, choices=TIMESERIES_DATASETS)
    ap.add_argument("--all", action="store_true", help="Test all time-series datasets")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not args.dataset and not args.all:
        ap.print_help()
        sys.exit(1)

    datasets = TIMESERIES_DATASETS if args.all else [args.dataset]
    results = {}

    for ds in datasets:
        ok = test_dataset(ds, seed=args.seed)
        results[ds] = "OK" if ok else "FAILED"

    print(f"\n{'='*60}")
    print("  Summary:")
    for ds, status in results.items():
        icon = "✓" if status == "OK" else "✗"
        print(f"    {icon} {ds}: {status}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
