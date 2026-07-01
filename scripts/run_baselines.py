"""
Run baseline models on a single dataset.

Baselines:
  CatBoost      — default settings (de-facto SOTA for tabular data)
  LightGBM      — default settings (fast GBDT, usually on par with CatBoost)
  Random NAS    — random search over the same NN search space (ablation)
  Optuna TPE    — Bayesian HP optimisation over the same NN space (key comparison)

Usage examples:
  # Just CatBoost + LightGBM
  python scripts/run_baselines.py --openml_id 40981 --out_dir runs/helena

  # Full comparison including Optuna (match --optuna_trials to your LLM-NAS budget)
  python scripts/run_baselines.py --openml_id 40981 --out_dir runs/helena \\
      --optuna --optuna_trials 25 --random_nas

  # Built-in dataset (no OpenML download needed)
  python scripts/run_baselines.py --builtin miniboonee --out_dir runs/miniboonee \\
      --optuna --optuna_trials 25
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.run_baselines_core import run_baselines


def main():
    ap = argparse.ArgumentParser(description="Baseline runner: CatBoost + LightGBM + Optuna + Random NAS")

    # --- Data source ---
    ap.add_argument("--openml_id", type=int, default=None,
                    help="OpenML dataset ID (e.g. 40981=helena, 41150=MiniBooNE)")
    ap.add_argument("--builtin", default=None,
                    choices=["covtype", "california_housing", "miniboonee", "har", "harth", "pamap2", "ecg5000"],
                    help="Built-in dataset (no OpenML download needed)")
    ap.add_argument("--csv", default=None,
                    help="Path to CSV file (last column = target)")
    ap.add_argument("--task", default="auto",
                    choices=["auto", "binary", "multiclass", "regression"])
    ap.add_argument("--out_dir", required=True,
                    help="Output directory for all baseline results")
    ap.add_argument("--seed", type=int, default=42)

    # --- Which baselines to run ---
    ap.add_argument("--no_lgbm", action="store_true", help="Skip LightGBM")
    ap.add_argument("--random_nas", action="store_true",
                    help="Run random NAS ablation (same budget, random arch selection)")
    ap.add_argument("--random_nas_budget", type=int, default=25,
                    help="Trial budget for random NAS (should match LLM-NAS budget)")

    # --- Optuna ---
    ap.add_argument("--optuna", action="store_true",
                    help="Run Optuna TPE baseline (pip install optuna first)")
    ap.add_argument("--optuna_trials", type=int, default=25,
                    help="Number of Optuna trials (should match LLM-NAS --budget for fair comparison)")
    ap.add_argument("--optuna_timeout", type=int, default=3600,
                    help="Hard wall-clock cap for Optuna in seconds (default: 1 hour)")

    ap.add_argument("--device", default=None, help="cuda / cpu (auto-detected if omitted)")
    args = ap.parse_args()

    # Resolve data source
    source = "openml"
    builtin_name = None
    csv_path = None
    if args.builtin:
        source = "builtin"
        builtin_name = args.builtin
    elif args.csv:
        source = "csv"
        csv_path = args.csv
    elif args.openml_id is None:
        ap.error("One of --openml_id, --builtin, or --csv is required")

    run_baselines(
        openml_id=args.openml_id,
        task=args.task,
        out_dir=args.out_dir,
        seed=args.seed,
        run_lgbm=not args.no_lgbm,
        run_random_nas=args.random_nas,
        random_nas_budget=args.random_nas_budget,
        device=args.device,
        source=source,
        builtin_name=builtin_name,
        csv_path=csv_path,
    )

    # Optuna — called separately from baselines_automl to keep run_baselines_core clean
    if args.optuna:
        print("\n--- Optuna TPE baseline ---")
        from src.baselines_automl import run_optuna_baseline
        run_optuna_baseline(
            openml_id=args.openml_id,
            task=args.task,
            out_dir=args.out_dir,
            seed=args.seed,
            n_trials=args.optuna_trials,
            timeout=args.optuna_timeout,
            source=source,
            builtin_name=builtin_name,
            csv_path=csv_path,
            device=args.device,
        )


if __name__ == "__main__":
    main()
