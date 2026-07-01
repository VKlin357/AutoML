"""
Main entry point for Forecasting NAS experiments.

Runs:
  1. CatBoost (raw flat features)       — weak baseline
  2. CatBoost + Feature Engineering     — strong baseline
  3. LightGBM + Feature Engineering     — strong baseline
  4. LLM-NAS (forecasting-aware)        — our method

Usage:
    # Full experiment
    python src/run_forecasting_nas.py \
        --dataset etth1 \
        --out_dir experiments/forecasting/etth1 \
        --budget 30 --seed 42 --llm_model gpt-4o

    # Only baselines (no API key needed)
    python src/run_forecasting_nas.py \
        --dataset etth1 --out_dir experiments/forecasting/etth1 \
        --skip_nas

    # Only NAS (no CatBoost)
    python src/run_forecasting_nas.py \
        --dataset etth1 --out_dir experiments/forecasting/etth1 \
        --skip_baselines
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


def compute_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    mse  = float(np.mean((y_true - y_pred) ** 2))
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(mse))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = float(1 - ss_res / (ss_tot + 1e-8))
    return {"mse": mse, "mae": mae, "rmse": rmse, "r2": r2}


# ─────────────────────────────────────────────────────────────────────────────
# Baselines
# ─────────────────────────────────────────────────────────────────────────────

def run_catboost_baseline(raw, out_dir: Path, seed: int, use_fe: bool) -> dict:
    from catboost import CatBoostRegressor

    label = "CatBoost+FE" if use_fe else "CatBoost raw"
    print(f"\n{'='*60}\n  {label}\n{'='*60}")

    if use_fe:
        from src.forecasting_features import make_forecasting_features
        freq = getattr(raw, 'freq', 'h')
        X_tr = make_forecasting_features(raw.X_train, raw.n_channels, raw.lookback, freq=freq)
        X_va = make_forecasting_features(raw.X_val,   raw.n_channels, raw.lookback, freq=freq)
        X_te = make_forecasting_features(raw.X_test,  raw.n_channels, raw.lookback, freq=freq)
    else:
        X_tr, X_va, X_te = raw.X_train, raw.X_val, raw.X_test

    print(f"  Feature dim: {X_tr.shape[1]}")
    t0 = time.time()

    m = CatBoostRegressor(
        iterations=2000, learning_rate=0.05, depth=8,
        random_seed=seed, verbose=100, early_stopping_rounds=100,
    )
    m.fit(X_tr, raw.y_train, eval_set=(X_va, raw.y_val), use_best_model=True)
    elapsed = time.time() - t0

    val_met  = compute_metrics(raw.y_val,  m.predict(X_va))
    test_met = compute_metrics(raw.y_test, m.predict(X_te))

    print(f"\n  Val  MSE={val_met['mse']:.6f}  MAE={val_met['mae']:.6f}")
    print(f"  Test MSE={test_met['mse']:.6f}  MAE={test_met['mae']:.6f}  R²={test_met['r2']:.4f}")
    print(f"  Time: {elapsed:.1f}s  Best iter: {m.best_iteration_}")

    result = {
        "model": "catboost_fe" if use_fe else "catboost_raw",
        "use_fe": use_fe, "n_features": X_tr.shape[1],
        "val_metrics": val_met, "test_metrics": test_met,
        "test_mse": test_met["mse"], "test_mae": test_met["mae"],
        "test_r2": test_met["r2"], "seconds": elapsed,
    }
    fname = "baseline_catboost_fe.json" if use_fe else "baseline_catboost_raw.json"
    save_json(out_dir / fname, result)
    return result


def run_lightgbm_baseline(raw, out_dir: Path, seed: int) -> dict:
    import lightgbm as lgb
    from src.forecasting_features import make_forecasting_features

    print(f"\n{'='*60}\n  LightGBM + FE\n{'='*60}")

    freq = getattr(raw, 'freq', 'h')
    X_tr = make_forecasting_features(raw.X_train, raw.n_channels, raw.lookback, freq=freq)
    X_va = make_forecasting_features(raw.X_val,   raw.n_channels, raw.lookback, freq=freq)
    X_te = make_forecasting_features(raw.X_test,  raw.n_channels, raw.lookback, freq=freq)

    print(f"  Feature dim: {X_tr.shape[1]}")
    t0 = time.time()

    dtrain = lgb.Dataset(X_tr, label=raw.y_train)
    dval   = lgb.Dataset(X_va, label=raw.y_val, reference=dtrain)
    params = dict(objective="regression", metric="mse", learning_rate=0.05,
                  num_leaves=127, max_depth=8, seed=seed, verbose=-1)
    cbs = [lgb.early_stopping(100, verbose=False), lgb.log_evaluation(100)]
    m = lgb.train(params, dtrain, num_boost_round=2000, valid_sets=[dval], callbacks=cbs)
    elapsed = time.time() - t0

    val_met  = compute_metrics(raw.y_val,  m.predict(X_va))
    test_met = compute_metrics(raw.y_test, m.predict(X_te))

    print(f"\n  Val  MSE={val_met['mse']:.6f}  MAE={val_met['mae']:.6f}")
    print(f"  Test MSE={test_met['mse']:.6f}  MAE={test_met['mae']:.6f}  R²={test_met['r2']:.4f}")
    print(f"  Time: {elapsed:.1f}s")

    result = {
        "model": "lightgbm_fe", "use_fe": True, "n_features": X_tr.shape[1],
        "val_metrics": val_met, "test_metrics": test_met,
        "test_mse": test_met["mse"], "test_mae": test_met["mae"],
        "test_r2": test_met["r2"], "seconds": elapsed,
    }
    save_json(out_dir / "baseline_lightgbm_fe.json", result)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    choices=["etth1", "etth2", "ettm1", "ettm2", "weather", "exchange",
                             "electricity", "traffic"])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--lookback",  type=int, default=96)
    ap.add_argument("--horizon",   type=int, default=1)
    ap.add_argument("--seed",      type=int, default=42)
    ap.add_argument("--budget",    type=int, default=30)
    ap.add_argument("--warmup",    type=int, default=10)
    ap.add_argument("--batch_n",   type=int, default=10,  help="LLM proposes N configs")
    ap.add_argument("--batch_k",   type=int, default=3,   help="Train top-K from N")
    ap.add_argument("--llm_model", default="gpt-4o")
    ap.add_argument("--api_key",   default=None)
    ap.add_argument("--device",    default=None)
    ap.add_argument("--skip_baselines", action="store_true")
    ap.add_argument("--skip_nas",       action="store_true")
    ap.add_argument("--no_fe_ablation", action="store_true",
                    help="Skip CatBoost raw (only run CatBoost+FE)")
    args = ap.parse_args()

    api_key = args.api_key or os.getenv("OPENAI_API_KEY", "")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    from src.data_forecasting import load_forecasting_raw
    print(f"\nLoading {args.dataset.upper()} (lookback={args.lookback}, horizon={args.horizon})")
    raw, summary = load_forecasting_raw(
        args.dataset, lookback=args.lookback, horizon=args.horizon, seed=args.seed
    )
    raw.n_channels = summary["n_channels"]
    save_json(out_dir / "dataset_summary.json", summary)

    results = {}

    # ── Baselines ──
    if not args.skip_baselines:
        if not args.no_fe_ablation:
            results["catboost_raw"] = run_catboost_baseline(raw, out_dir, args.seed, use_fe=False)
        results["catboost_fe"]  = run_catboost_baseline(raw, out_dir, args.seed, use_fe=True)
        try:
            results["lightgbm_fe"] = run_lightgbm_baseline(raw, out_dir, args.seed)
        except ImportError:
            print("  LightGBM not installed — skipping")

    # ── LLM-NAS ──
    if not args.skip_nas:
        if not api_key:
            print("\nERROR: set OPENAI_API_KEY or pass --api_key")
        else:
            from src.forecasting.orchestrator import run_forecasting_nas
            nas_result = run_forecasting_nas(
                dataset_name=args.dataset,
                X_train=raw.X_train, y_train=raw.y_train,
                X_val=raw.X_val,     y_val=raw.y_val,
                X_test=raw.X_test,   y_test=raw.y_test,
                lookback=raw.lookback,
                horizon=raw.horizon,
                n_channels=raw.n_channels,
                target_idx=summary.get("target_idx", -1),
                dataset_info=summary,
                out_dir=str(out_dir / "nas"),
                budget=args.budget,
                random_warmup=args.warmup,
                batch_n=args.batch_n,
                batch_k=args.batch_k,
                seed=args.seed,
                device=args.device,
                api_key=api_key,
                llm_model=args.llm_model,
            )
            best = nas_result.get("best", {})
            if best:
                results["llm_nas"] = {
                    "model": "llm_nas",
                    "best_family": best.get("config", {}).get("arch", {}).get("family"),
                    "val_mse": best.get("val_mse"),
                    "val_mae": best.get("val_mae"),
                    "test_mse": best.get("test_mse"),
                    "test_mae": best.get("test_mae"),
                }

    # ── Final summary ──
    print(f"\n{'='*60}")
    print(f"  FINAL SUMMARY — {args.dataset.upper()}")
    print(f"{'='*60}")
    header = f"  {'Model':25s}  {'Test MSE':>12}  {'Test MAE':>12}  {'R²':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name, r in results.items():
        mse = r.get("test_mse") or r.get("test_metrics", {}).get("mse", float("nan"))
        mae = r.get("test_mae") or r.get("test_metrics", {}).get("mae", float("nan"))
        r2  = r.get("test_r2")  or r.get("test_metrics", {}).get("r2",  float("nan"))
        print(f"  {name:25s}  {mse:12.6f}  {mae:12.6f}  {r2:8.4f}")

    save_json(out_dir / "results_summary.json", results)
    print(f"\n  Results saved to {out_dir}/results_summary.json")


if __name__ == "__main__":
    main()
