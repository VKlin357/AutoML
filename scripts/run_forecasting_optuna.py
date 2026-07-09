"""
Optuna NAS baseline for forecasting datasets.

Uses the SAME forecasting search space as LLM-NAS (src/forecasting/search_space.py)
and the SAME training loop (src/forecasting/train.py) for fair apples-to-apples comparison.

Usage (on GPU server):
    cd ~/llm-tabular-nas-proxy
    python3 scripts/run_forecasting_optuna.py --all --n_trials 40 --seed 42 --device cuda
    python3 scripts/run_forecasting_optuna.py --dataset etth1 --n_trials 40
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


DATASETS = ["etth1", "etth2", "ettm1", "ettm2", "exchange", "electricity", "traffic"]

DATASET_OUT_DIRS = {
    "etth1":       "experiments_forecasting/etth1_v4",
    "etth2":       "experiments_forecasting/etth2_v4",
    "ettm1":       "experiments_forecasting/ettm1_fe_v2",
    "ettm2":       "experiments_forecasting/ettm2",
    "exchange":    "experiments_forecasting/exchange",
    "electricity": "experiments_forecasting/electricity",
    "traffic":     "experiments_forecasting/traffic",
}


def save_json(path: str | Path, data: dict):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, default=str))


def run_optuna_forecasting(
    dataset: str,
    out_dir: str,
    n_trials: int = 40,
    timeout: int = 7200,
    seed: int = 42,
    lookback: int = 96,
    horizon: int = 1,
    device: str | None = None,
) -> Dict[str, Any]:
    """
    Run Optuna TPE over the forecasting NAS search space.
    Same families and training loop as LLM-NAS forecasting.
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("[Optuna] not installed — pip install optuna")
        return {"error": "optuna not installed"}

    import torch
    from src.data_forecasting import load_forecasting_raw
    from src.forecasting.models import build_forecasting_model
    from src.forecasting.train import train_forecasting
    from src.forecasting.search_space import SEARCH_SPACE, FORECASTING_FAMILIES, validate_config

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    optuna_json = out_path / "baseline_optuna.json"
    if optuna_json.exists():
        existing = json.loads(optuna_json.read_text())
        if existing.get("test_mse") is not None:
            print(f"[SKIP] {dataset} — already has test_mse={existing['test_mse']:.6f}")
            return existing

    print(f"\n{'='*60}")
    print(f"  Optuna NAS — {dataset.upper()}")
    print(f"  n_trials={n_trials}  timeout={timeout}s  seed={seed}")
    print(f"{'='*60}")

    raw, summary = load_forecasting_raw(
        dataset, lookback=lookback, horizon=horizon, seed=seed
    )
    n_channels = summary["n_channels"]
    target_idx = summary.get("target_idx", -1)

    print(f"  {summary['n_train']} train / {summary['n_val']} val / {summary['n_test']} test")
    print(f"  lookback={lookback}  n_channels={n_channels}  target_idx={target_idx}")

    t_start = time.time()
    best_val_mse = float("inf")
    best_cfg: dict = {}
    best_test_mse: float | None = None
    best_test_mae: float | None = None
    trial_log = []

    def objective(trial) -> float:
        nonlocal best_val_mse, best_cfg, best_test_mse, best_test_mae

        # ── arch family (same as LLM-NAS forecasting) ──
        # linear/nlinear dominate in practice; skip slow neural families for speed
        family = trial.suggest_categorical("family", ["linear", "nlinear", "mlp"])

        if family == "linear":
            arch_cfg = {
                "family": "linear",
                "decompose":  trial.suggest_categorical("lin_decompose", [True, False]),
                "individual": trial.suggest_categorical("lin_individual", [True, False]),
            }
        elif family == "nlinear":
            arch_cfg = {
                "family": "nlinear",
                "individual": trial.suggest_categorical("nlin_individual", [True, False]),
            }
        elif family == "mlp":
            arch_cfg = {
                "family": "mlp",
                "hidden_size": trial.suggest_categorical("mlp_hidden", [128, 256, 512, 1024]),
                "n_layers":    trial.suggest_int("mlp_layers", 2, 5),
                "dropout":     trial.suggest_float("mlp_dropout", 0.0, 0.3),
                "activation":  trial.suggest_categorical("mlp_act", ["relu", "gelu", "silu"]),
            }
        elif family == "patch_mlp":
            patch_size = trial.suggest_categorical("pm_patch", [8, 16, 24, 32])
            # ensure patch divides lookback
            valid = [p for p in [8, 16, 24, 32] if lookback % p == 0]
            patch_size = patch_size if patch_size in valid else (valid[0] if valid else 16)
            arch_cfg = {
                "family": "patch_mlp",
                "patch_size": patch_size,
                "d_model":    trial.suggest_categorical("pm_d_model", [64, 128, 256]),
                "n_layers":   trial.suggest_int("pm_layers", 2, 4),
                "dropout":    trial.suggest_float("pm_dropout", 0.0, 0.2),
            }
        elif family == "tcn":
            arch_cfg = {
                "family": "tcn",
                "d_model":     trial.suggest_categorical("tcn_d_model", [32, 64, 128, 256]),
                "n_layers":    trial.suggest_int("tcn_layers", 3, 6),
                "kernel_size": trial.suggest_categorical("tcn_kernel", [3, 5, 7]),
                "dropout":     trial.suggest_float("tcn_dropout", 0.0, 0.2),
            }
        elif family == "transformer":
            d_model = trial.suggest_categorical("tr_d_model", [64, 128, 256])
            n_heads = trial.suggest_categorical("tr_n_heads", [2, 4, 8])
            # ensure d_model % n_heads == 0
            while d_model % n_heads != 0:
                n_heads = n_heads // 2 if n_heads > 1 else 1
                if n_heads == 1:
                    break
            patch_size = trial.suggest_categorical("tr_patch", [8, 16, 24])
            valid = [p for p in [8, 16, 24] if lookback % p == 0]
            patch_size = patch_size if patch_size in valid else (valid[0] if valid else 16)
            arch_cfg = {
                "family": "transformer",
                "patch_size": patch_size,
                "d_model":    d_model,
                "n_heads":    n_heads,
                "n_layers":   trial.suggest_int("tr_layers", 2, 4),
                "dropout":    trial.suggest_float("tr_dropout", 0.0, 0.2),
                "ffn_factor": trial.suggest_categorical("tr_ffn", [2.0, 4.0]),
            }
        else:
            raise ValueError(f"Unknown family: {family}")

        train_cfg = {
            "lr":           trial.suggest_categorical("lr", [1e-4, 3e-4, 5e-4, 1e-3, 2e-3]),
            "batch_size":   trial.suggest_categorical("bs", [32, 64, 128, 256]),
            "epochs":       trial.suggest_categorical("epochs", [30, 50]),
            "patience":     trial.suggest_categorical("patience", [5, 10]),
            "optimizer":    trial.suggest_categorical("opt", ["adam", "adamw"]),
            "weight_decay": trial.suggest_categorical("wd", [0.0, 1e-5, 1e-4, 1e-3]),
            "scheduler":    trial.suggest_categorical("sched", ["none", "cosine"]),
            "grad_clip":    trial.suggest_categorical("grad_clip", [0.0, 1.0, 5.0]),
            "use_amp":      True,
        }

        cfg = validate_config({"arch": arch_cfg, "train": train_cfg}, lookback=lookback)

        try:
            model = build_forecasting_model(
                cfg["arch"], lookback=lookback, horizon=horizon,
                n_channels=n_channels, target_idx=target_idx
            )
            result = train_forecasting(
                model=model,
                X_train=raw.X_train, y_train=raw.y_train,
                X_val=raw.X_val,     y_val=raw.y_val,
                X_test=raw.X_test,   y_test=raw.y_test,
                lookback=lookback, n_channels=n_channels,
                train_cfg=cfg["train"],
                device=device, seed=seed, verbose=False,
            )
        except Exception as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
            print(f"  [trial {trial.number}] ERROR: {e}")
            return float("inf")

        val_mse  = result.val_mse
        test_mse = result.test_mse
        test_mae = result.test_mae

        trial_log.append({
            "trial": trial.number,
            "family": family,
            "val_mse": val_mse,
            "test_mse": test_mse,
        })

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_cfg = cfg
            best_test_mse = test_mse
            best_test_mae = test_mae
            elapsed = time.time() - t_start
            tmse_s = f"{test_mse:.6f}" if test_mse is not None else "—"
            print(f"  ✓ trial {trial.number:3d}  val_mse={val_mse:.6f}  "
                  f"test_mse={tmse_s}  family={family}  [{elapsed:.0f}s]")

        return val_mse

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

    elapsed_total = time.time() - t_start
    completed = len([t for t in study.trials if t.state.name == "COMPLETE"])

    result = {
        "dataset": dataset,
        "n_trials": n_trials,
        "completed_trials": completed,
        "best_val_mse": best_val_mse if best_val_mse < float("inf") else None,
        "test_mse": best_test_mse,
        "test_mae": best_test_mae,
        "best_config": best_cfg,
        "trial_log": trial_log,
        "elapsed_seconds": elapsed_total,
        "seed": seed,
        "lookback": lookback,
        "horizon": horizon,
    }

    save_json(optuna_json, result)
    tmse_s = f"{best_test_mse:.6f}" if best_test_mse is not None else "—"
    print(f"\n  DONE {dataset}: val_mse={best_val_mse:.6f}  test_mse={tmse_s}  "
          f"[{completed}/{n_trials} trials, {elapsed_total:.0f}s]")
    print(f"  Saved → {optuna_json}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=DATASETS + ["all"], default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--n_trials", type=int, default=40)
    ap.add_argument("--timeout",  type=int, default=7200)
    ap.add_argument("--seed",     type=int, default=42)
    ap.add_argument("--lookback", type=int, default=96)
    ap.add_argument("--horizon",  type=int, default=1)
    ap.add_argument("--device",   default=None)
    args = ap.parse_args()

    if args.all or args.dataset == "all":
        datasets = DATASETS
    elif args.dataset:
        datasets = [args.dataset]
    else:
        ap.print_help()
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Forecasting Optuna NAS — {len(datasets)} dataset(s)")
    print(f"  n_trials={args.n_trials}  seed={args.seed}  device={args.device or 'auto'}")
    print(f"{'='*60}\n")

    summary_all = {}
    for ds in datasets:
        out_dir = DATASET_OUT_DIRS.get(ds, f"experiments_forecasting/{ds}")
        res = run_optuna_forecasting(
            dataset=ds, out_dir=out_dir,
            n_trials=args.n_trials, timeout=args.timeout,
            seed=args.seed, lookback=args.lookback,
            horizon=args.horizon, device=args.device,
        )
        summary_all[ds] = {
            "val_mse":  res.get("best_val_mse"),
            "test_mse": res.get("test_mse"),
            "test_mae": res.get("test_mae"),
        }

    print(f"\n{'='*60}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Dataset':15s}  {'Val MSE':>12}  {'Test MSE':>12}  {'Test MAE':>12}")
    print(f"  {'-'*55}")
    for ds, r in summary_all.items():
        v = r.get("val_mse");  vs = f"{v:.6f}" if v else "—"
        m = r.get("test_mse"); ms = f"{m:.6f}" if m else "—"
        a = r.get("test_mae"); as_ = f"{a:.6f}" if a else "—"
        print(f"  {ds:15s}  {vs:>12}  {ms:>12}  {as_:>12}")


if __name__ == "__main__":
    main()
