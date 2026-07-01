"""
Forecasting experiment runner: CatBoost+FE vs LLM-NAS.

Runs on ETTh1, ETTh2, ETTm1, ETTm2, Weather, Exchange Rate datasets.

Usage:
    # CatBoost + LightGBM with feature engineering
    python src/run_forecasting.py --dataset etth1 --out_dir experiments/forecasting/etth1

    # LLM-NAS
    python src/run_forecasting.py --dataset etth1 --out_dir experiments/forecasting/etth1 --llm_nas

    # Both together
    python src/run_forecasting.py --dataset etth1 --out_dir experiments/forecasting/etth1 --all
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def save_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, default=str))


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    mse  = float(np.mean((y_true - y_pred) ** 2))
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(mse))
    # R^2
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = float(1 - ss_res / (ss_tot + 1e-8))
    return {"mse": mse, "mae": mae, "rmse": rmse, "r2": r2}


def run_catboost(raw, out_dir: Path, seed: int = 42, use_fe: bool = True):
    """CatBoost baseline with optional feature engineering."""
    from catboost import CatBoostRegressor
    from src.forecasting_features import make_forecasting_features

    print("\n" + "="*60)
    print("  CatBoost" + (" + Feature Engineering" if use_fe else " (raw)"))
    print("="*60)

    if use_fe:
        X_tr = make_forecasting_features(raw.X_train, raw.n_channels, raw.lookback)
        X_va = make_forecasting_features(raw.X_val,   raw.n_channels, raw.lookback)
        X_te = make_forecasting_features(raw.X_test,  raw.n_channels, raw.lookback)
        print(f"  Feature shape: {X_tr.shape[1]} (raw={raw.X_train.shape[1]} + engineered)")
    else:
        X_tr, X_va, X_te = raw.X_train, raw.X_val, raw.X_test
        print(f"  Feature shape: {X_tr.shape[1]} (raw only)")

    t0 = time.time()
    m = CatBoostRegressor(
        iterations=2000, learning_rate=0.05, depth=8,
        random_seed=seed, verbose=100,
        early_stopping_rounds=100,
    )
    m.fit(X_tr, raw.y_train, eval_set=(X_va, raw.y_val), use_best_model=True)
    elapsed = time.time() - t0

    val_metrics  = compute_regression_metrics(raw.y_val,  m.predict(X_va))
    test_metrics = compute_regression_metrics(raw.y_test, m.predict(X_te))

    print(f"\n  Val  MSE={val_metrics['mse']:.6f}  MAE={val_metrics['mae']:.6f}")
    print(f"  Test MSE={test_metrics['mse']:.6f}  MAE={test_metrics['mae']:.6f}  R²={test_metrics['r2']:.4f}")
    print(f"  Time: {elapsed:.1f}s  |  Best iter: {m.best_iteration_}")

    result = {
        "model": "catboost" + ("_fe" if use_fe else "_raw"),
        "use_fe": use_fe,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "test_mse": test_metrics["mse"],
        "test_mae": test_metrics["mae"],
        "test_rmse": test_metrics["rmse"],
        "test_r2": test_metrics["r2"],
        "best_iteration": int(m.best_iteration_),
        "seconds": elapsed,
        "n_features": int(X_tr.shape[1]),
    }
    fname = "baseline_catboost_fe.json" if use_fe else "baseline_catboost_raw.json"
    save_json(out_dir / fname, result)
    return result


def run_lightgbm(raw, out_dir: Path, seed: int = 42, use_fe: bool = True):
    """LightGBM baseline with optional feature engineering."""
    import lightgbm as lgb
    from src.forecasting_features import make_forecasting_features

    print("\n" + "="*60)
    print("  LightGBM" + (" + Feature Engineering" if use_fe else " (raw)"))
    print("="*60)

    if use_fe:
        X_tr = make_forecasting_features(raw.X_train, raw.n_channels, raw.lookback)
        X_va = make_forecasting_features(raw.X_val,   raw.n_channels, raw.lookback)
        X_te = make_forecasting_features(raw.X_test,  raw.n_channels, raw.lookback)
    else:
        X_tr, X_va, X_te = raw.X_train, raw.X_val, raw.X_test

    t0 = time.time()
    dtrain = lgb.Dataset(X_tr, label=raw.y_train)
    dval   = lgb.Dataset(X_va, label=raw.y_val, reference=dtrain)

    params = dict(
        objective="regression", metric="mse",
        learning_rate=0.05, num_leaves=127, max_depth=8,
        n_estimators=2000, seed=seed, verbose=-1,
    )
    callbacks = [lgb.early_stopping(100, verbose=False), lgb.log_evaluation(100)]
    m = lgb.train(params, dtrain, valid_sets=[dval], callbacks=callbacks)
    elapsed = time.time() - t0

    val_metrics  = compute_regression_metrics(raw.y_val,  m.predict(X_va))
    test_metrics = compute_regression_metrics(raw.y_test, m.predict(X_te))

    print(f"\n  Val  MSE={val_metrics['mse']:.6f}  MAE={val_metrics['mae']:.6f}")
    print(f"  Test MSE={test_metrics['mse']:.6f}  MAE={test_metrics['mae']:.6f}  R²={test_metrics['r2']:.4f}")
    print(f"  Time: {elapsed:.1f}s")

    result = {
        "model": "lightgbm" + ("_fe" if use_fe else "_raw"),
        "use_fe": use_fe,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "test_mse": test_metrics["mse"],
        "test_mae": test_metrics["mae"],
        "test_rmse": test_metrics["rmse"],
        "test_r2": test_metrics["r2"],
        "seconds": elapsed,
        "n_features": int(X_tr.shape[1]),
    }
    fname = "baseline_lightgbm_fe.json" if use_fe else "baseline_lightgbm_raw.json"
    save_json(out_dir / fname, result)
    return result


def run_llm_nas(raw, dataset: str, out_dir: Path, seed: int = 42,
                budget: int = 30, api_key: str = "", model: str = "gpt-4o"):
    """LLM-NAS on forecasting dataset (tabular regression)."""
    import subprocess

    print("\n" + "="*60)
    print(f"  LLM-NAS  budget={budget}  model={model}")
    print("="*60)

    # Save dataset as CSV for run_nas_v2.py --csv mode
    import pandas as pd
    tmp_dir = out_dir / "tmp_csv"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    def save_split(X, y, path):
        cols = [f"f{i}" for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=cols)
        df["target"] = y
        df.to_csv(path, index=False)

    train_csv = tmp_dir / "train.csv"
    val_csv   = tmp_dir / "val.csv"
    test_csv  = tmp_dir / "test.csv"

    save_split(raw.X_train, raw.y_train, train_csv)
    save_split(raw.X_val,   raw.y_val,   val_csv)
    save_split(raw.X_test,  raw.y_test,  test_csv)

    cmd = [
        sys.executable, "src/run_nas_v2.py",
        "--csv", str(train_csv),
        "--task", "regression",
        "--out_dir", str(out_dir / "nas_trials"),
        "--budget", str(budget),
        "--random_warmup", "10",
        "--mode", "batch",
        "--batch_n", "20",
        "--batch_k", "4",
        "--seed", str(seed),
        "--llm_model", model,
    ]

    env = os.environ.copy()
    env["OPENAI_API_KEY"] = api_key

    print(f"  Running: {' '.join(cmd[:6])} ...")
    t0 = time.time()
    result = subprocess.run(cmd, env=env, cwd=ROOT)
    elapsed = time.time() - t0

    print(f"\n  LLM-NAS finished in {elapsed:.1f}s (exit={result.returncode})")
    return result.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    choices=["etth1", "etth2", "ettm1", "ettm2", "weather", "exchange"])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--lookback", type=int, default=96)
    ap.add_argument("--horizon",  type=int, default=1)
    ap.add_argument("--seed",     type=int, default=42)
    ap.add_argument("--budget",   type=int, default=30)
    ap.add_argument("--llm_model", default="gpt-4o")
    ap.add_argument("--api_key",  default=None)

    ap.add_argument("--baselines", action="store_true", help="Run CatBoost + LightGBM")
    ap.add_argument("--llm_nas",   action="store_true", help="Run LLM-NAS")
    ap.add_argument("--all",       action="store_true", help="Run everything")
    ap.add_argument("--raw_only",  action="store_true",
                    help="Run CatBoost without feature engineering (ablation)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    from src.data_forecasting import load_forecasting_raw
    print(f"\nLoading {args.dataset.upper()} ...")
    raw, summary = load_forecasting_raw(
        args.dataset, lookback=args.lookback, horizon=args.horizon, seed=args.seed
    )
    # Attach n_channels for feature engineering
    raw.n_channels = summary["n_channels"]
    save_json(out_dir / "dataset_summary.json", summary)

    results = {}

    run_bl = args.baselines or args.all
    run_nas = args.llm_nas or args.all

    if run_bl:
        # CatBoost with feature engineering (strong baseline)
        results["catboost_fe"] = run_catboost(raw, out_dir, args.seed, use_fe=True)

        # CatBoost raw (ablation — shows FE helps)
        if args.raw_only or args.all:
            results["catboost_raw"] = run_catboost(raw, out_dir, args.seed, use_fe=False)

        # LightGBM with FE
        results["lightgbm_fe"] = run_lightgbm(raw, out_dir, args.seed, use_fe=True)

    if run_nas:
        api_key = args.api_key or os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            print("ERROR: set OPENAI_API_KEY or pass --api_key")
        else:
            run_llm_nas(raw, args.dataset, out_dir, args.seed,
                        args.budget, api_key, args.llm_model)

    # Final summary
    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    for name, r in results.items():
        print(f"  {name:25s}  MSE={r['test_mse']:.6f}  MAE={r['test_mae']:.6f}  R²={r['test_r2']:.4f}")
    save_json(out_dir / "results_summary.json", results)


if __name__ == "__main__":
    main()
