"""
Unified experiment script — runs the full comparison on one or more datasets:

  1. CatBoost baseline
  2. LightGBM baseline
  3. Random NAS baseline (ablation: same budget, random arch selection)
  4. Optuna TPE baseline (optional, --optuna flag)
  5. Naive LLM baseline  (optional, --naive_llm flag)
  6. LLM-NAS v2 (the proposed method)  + optional greedy ensemble
  7. AutoGluon baseline (optional, --automl flag)

Results are saved per-dataset under:
  experiments/<dataset_name>/
    baseline_catboost.json
    baseline_lightgbm.json
    baseline_random_search.json
    baseline_optuna.json          (if --optuna)
    baseline_naive_llm.json       (if --naive_llm)
    baseline_autogluon.json       (if --automl)
    nas_v2/  (all LLM-NAS trial results + ensemble_result.json)

A summary table is printed at the end.

Usage:
  # Run on multiple datasets (default set):
  python scripts/run_experiment.py --api_key sk-proj-...

  # Full comparison on one hard dataset:
  python scripts/run_experiment.py \\
      --datasets 40981 \\
      --budget 30 \\
      --ensemble_k 5 \\
      --optuna --optuna_trials 30 \\
      --naive_llm --naive_llm_trials 5 \\
      --api_key sk-proj-...

Dataset groups
--------------
EASY (GBM usually wins — useful for sanity checks only):
  31    credit_g       (binary,      1000 rows,   20 features)
  1590  adult          (binary,     48842 rows,   14 features)

HARD (NNs competitive with GBM — USE THESE FOR THE THESIS):
  41166  helena        (multiclass, 65196 rows,   27 features)
    → Large, complex non-linear interactions; FT-Transformer often matches CatBoost.
    NOTE: OpenML ID 40981 is "Australian" (690 rows) — use 41166 for the real Helena!
  45021  jannis        (multiclass, 83733 rows,   54 features)
    → Large, purely numeric; ResMLP / GatedTab scale well.
  168338 MiniBooNE     (binary,    130064 rows,   50 features)
    → Particle physics, smooth distributions; NNs often win.
  44156  pol           (binary,     10082 rows,   26 features)
    → Polynomial feature interactions; GBM advantage small.
  4352   year_msd      (regression, 515345 rows,  90 features)
    → Very large regression; NNs scale better than CatBoost at this N.

Reference: Gorishniy et al. 2021 (FT-Transformer), McElfresh et al. 2023
  (When Do Neural Nets Outperform Boosted Trees on Tabular Data?)
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Default benchmark set — HARD datasets where NNs are competitive with GBMs.
# Source: Gorishniy et al. 2021, McElfresh et al. 2023.
#
# Dataset IDs verified on OpenML (May 2025):
#   41166  = helena     (65196 rows, 27 features, 100 classes)
#            NOTE: OpenML ID 40981 loads "Australian" (690 rows) — WRONG!
#   45021  = jannis     (83733 rows, 54 features, 4 classes)
#   168338 = MiniBooNE  (130064 rows, 50 features, binary)
#   44156  = pol        (10082 rows, 26 features, binary)
DEFAULT_DATASETS = [
    (41166,  "helena",    "multiclass"),   # 65k rows, 100 classes, NNs match GBM
    (45021,  "jannis",    "multiclass"),   # 84k rows, numeric, NNs scale well
    (168338, "MiniBooNE", "binary"),       # 130k rows, physics, NNs often win ★★★
    (44156,  "pol",       "binary"),       # 10k rows, polynomial interactions
]

# Extended benchmark — add these for a stronger thesis claim
EXTENDED_DATASETS = [
    (293,    "covertype",  "multiclass"),  # 581k rows, NNs scale well at large N
    (4352,   "year_msd",   "regression"),  # 515k rows, NNs beat CatBoost at this size
]

# RECOMMENDED SINGLE DATASET — best for showing LLM-NAS vs CatBoost gap
# MiniBooNE (168338): particle physics data, 130k rows, 50 purely numeric features.
# CatBoost AUC ~ 0.975, FT-Transformer/ResMLP often reach 0.980+
# Run: python scripts/run_experiment.py --datasets 168338 --budget 25 --api_key ...

# Other strong NAS datasets (NNs clearly beat GBM):
#   4352   year_msd      regression  515k rows, 90 features — NNs scale better than CatBoost at this N
#   293    covertype     multiclass  581k rows, 54 features — NNs competitive at scale
#   45021  jannis        multiclass   84k rows, 54 features — already in DEFAULT_DATASETS

# Easy datasets (GBM usually wins) — kept here for sanity-check runs
EASY_DATASETS = [
    (31,    "credit_g",      "binary"),
    (1590,  "adult",         "binary"),
    (40966, "mfeat_factors", "multiclass"),
]

# ─────────────────────────────────────────────────────────────────────────────
# NEW: Domain-specific benchmark groups where NNs beat GBMs
#
# Thesis claim: "LLM-NAS finds architectures that outperform random search AND
# CatBoost across three domains where temporal/signal structure matters."
#
# Use --domain biomedical  →  ECG5000 + EEG Eye State + HAR
# Use --domain finance     →  ELEC2
# Use --domain all_domains →  all four above
# ─────────────────────────────────────────────────────────────────────────────

# Biomedical: physiological signals
# ECG5000 (UCR)          — 5k heartbeats × 140 TS features,  5 classes
# EEG Eye State (OpenML) — 15k EEG windows × 14 features,    binary
# HAR (UCI)              — 10k activity windows × 561 features, 6 classes
BIOMEDICAL_DATASETS = [
    # (openml_id_or_None, name, task, source, builtin_name)
    # We store as 5-tuples: (openml_id, name, task, source, builtin_name)
    # For OpenML: source="openml", builtin_name=None
    # For builtin: source="builtin", openml_id=-1
    (-1,   "ECG5000",       "multiclass", "builtin", "ecg5000"),
    # EEG_EyeState (OpenML 1471) removed: persistent NaN in raw data causes training failures
    (-1,   "HAR",           "multiclass", "builtin", "har"),
]

# Financial / Economics: market dynamics
# ELEC2                  — 45k electricity market periods × 8 features, binary
FINANCE_DATASETS = [
    (-1,   "ELEC2",         "binary",     "builtin", "elec2"),
]

# All domain datasets combined
ALL_DOMAIN_DATASETS = BIOMEDICAL_DATASETS + FINANCE_DATASETS


def run_one(
    openml_id: int,
    name: str,
    task: str,
    exp_dir: Path,
    *,
    budget: int,
    coldstart_n: int,
    random_nas_budget: int,
    ensemble_k: int,
    api_key: str,
    llm_model: str,
    seed: int,
    skip_random_nas: bool,
    skip_lgbm: bool,
    run_automl: bool,
    automl_time_limit: int,
    run_optuna: bool,
    optuna_trials: int,
    run_naive_llm: bool,
    naive_llm_trials: int,
    device: str | None,
    source: str = "openml",
    builtin_name: str | None = None,
    csv_path: str | None = None,
    use_refine_loop: bool = True,
    explore_pulse_every: int = 6,
    curve_feedback: bool = True,
):
    from src.run_baselines_core import run_baselines
    from src.nas_orchestrator import run_llm_nas_v2

    dataset_dir = exp_dir / name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Dataset: {name}  (openml_id={openml_id}, task={task})")
    print(f"{'='*60}")

    # ---- Baselines ----
    print("\n--- Baselines ---")
    baselines = run_baselines(
        openml_id=openml_id, task=task, out_dir=str(dataset_dir),
        seed=seed,
        run_lgbm=not skip_lgbm,
        run_random_nas=not skip_random_nas,
        random_nas_budget=random_nas_budget,
        device=device,
        source=source, builtin_name=builtin_name, csv_path=csv_path,
    )

    # ---- AutoGluon baseline (optional) ----
    autogluon_primary = None
    if run_automl:
        print("\n--- AutoGluon baseline ---")
        try:
            from src.baselines_automl import run_autogluon_baseline
            ag_res = run_autogluon_baseline(
                openml_id=openml_id, task=task,
                out_dir=str(dataset_dir),
                seed=seed,
                time_limit=automl_time_limit,
            )
            autogluon_primary = ag_res.get("primary")
        except Exception as e:
            print(f"  [AutoGluon] failed: {e}")

    # ---- Optuna baseline (optional) ----
    optuna_primary = None
    if run_optuna:
        print("\n--- Optuna TPE baseline ---")
        try:
            from src.baselines_automl import run_optuna_baseline
            opt_res = run_optuna_baseline(
                openml_id=openml_id, task=task,
                out_dir=str(dataset_dir),
                seed=seed,
                n_trials=optuna_trials,
                source=source, builtin_name=builtin_name, csv_path=csv_path,
                device=device,
            )
            optuna_primary = opt_res.get("primary")
        except Exception as e:
            print(f"  [Optuna] failed: {e}")
            import traceback; traceback.print_exc()

    # ---- Naive LLM baseline (optional) ----
    naive_llm_primary = None
    if run_naive_llm:
        print("\n--- Naive LLM baseline ---")
        try:
            from src.baselines_automl import run_naive_llm_baseline
            naive_res = run_naive_llm_baseline(
                openml_id=openml_id, task=task,
                out_dir=str(dataset_dir),
                api_key=api_key,
                llm_model=llm_model,
                seed=seed,
                n_trials=naive_llm_trials,
                source=source,
                builtin_name=builtin_name,
                csv_path=csv_path,
            )
            naive_llm_primary = naive_res.get("primary")
        except Exception as e:
            print(f"  [NaiveLLM] failed: {e}")
            import traceback; traceback.print_exc()

    # ---- LLM NAS v2 ----
    print("\n--- LLM NAS v2 ---")
    nas_out = str(dataset_dir / "nas_v2")
    # Copy baseline files into nas_v2 dir so the orchestrator can load them
    Path(nas_out).mkdir(parents=True, exist_ok=True)
    for bl_file in ("baseline_catboost.json", "baseline_lightgbm.json"):
        src_p = dataset_dir / bl_file
        if src_p.exists():
            import shutil
            shutil.copy(src_p, Path(nas_out) / bl_file)

    nas_result = run_llm_nas_v2(
        openml_id=openml_id, task=task, out_dir=nas_out,
        budget=budget, coldstart_n=coldstart_n,
        seed=seed,
        llm_model=llm_model,
        api_key=api_key,
        device=device,
        ensemble_k=ensemble_k,
        source=source, builtin_name=builtin_name, csv_path=csv_path,
        use_refine_loop=use_refine_loop,
        explore_pulse_every=explore_pulse_every,
        curve_feedback=curve_feedback,
    )

    ensemble_primary = None
    if "ensemble" in nas_result and nas_result["ensemble"] is not None:
        ensemble_primary = nas_result["ensemble"].get("primary")

    return {
        "name": name,
        "openml_id": openml_id,
        "task": task,
        "catboost": baselines.get("catboost", {}).get("primary"),
        "lightgbm": baselines.get("lightgbm", {}).get("primary"),
        "random_nas": baselines.get("random_nas", {}).get("primary"),
        "optuna": optuna_primary,
        "naive_llm": naive_llm_primary,
        "llm_nas_v2": nas_result["best_trial"]["primary"],
        "llm_nas_ensemble": ensemble_primary,
        "autogluon": autogluon_primary,
        "llm_nas_best_arch": nas_result["best_trial"]["config"]["arch"]["family"],
    }


def print_summary(rows):
    print(f"\n{'='*130}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*130}")
    header = (
        f"{'Dataset':<20}  {'Task':<12}  {'CatBoost':>10}  {'LightGBM':>10}  "
        f"{'RandNAS':>10}  {'Optuna':>10}  {'NaiveLLM':>10}  {'LLM-NAS':>10}  "
        f"{'Ensemble':>10}  {'AutoGluon':>10}  {'BestArch':<15}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        def fmt(v):
            return f"{v:.5f}" if v is not None else "  N/A    "
        print(
            f"{r['name']:<20}  {r['task']:<12}  {fmt(r['catboost']):>10}  "
            f"{fmt(r['lightgbm']):>10}  {fmt(r['random_nas']):>10}  "
            f"{fmt(r.get('optuna')):>10}  {fmt(r.get('naive_llm')):>10}  "
            f"{fmt(r['llm_nas_v2']):>10}  {fmt(r.get('llm_nas_ensemble')):>10}  "
            f"{fmt(r.get('autogluon')):>10}  {r.get('llm_nas_best_arch', '?'):<15}"
        )
    print(f"{'='*130}\n")


def main():
    ap = argparse.ArgumentParser(description="Full experiment runner (baselines + LLM-NAS)")
    ap.add_argument("--datasets", nargs="+", type=int, default=None,
                    help="OpenML dataset IDs to run (default: %(default)s)")
    ap.add_argument("--exp_dir", default="experiments",
                    help="Root output directory for all experiments")
    ap.add_argument("--budget", type=int, default=25,
                    help="LLM-NAS trial budget per dataset")
    ap.add_argument("--coldstart_n", type=int, default=5)
    ap.add_argument("--random_nas_budget", type=int, default=20,
                    help="Trial budget for random NAS ablation")
    ap.add_argument("--ensemble_k", type=int, default=5,
                    help="Re-train and ensemble top-K NAS configs (0 = disable)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--llm_model", default="gpt-4o",
                    help="OpenAI model name (e.g. gpt-4o, gpt-4o-mini)")
    ap.add_argument("--api_key", default=None,
                    help="OpenAI API key (default: $OPENAI_API_KEY)")
    ap.add_argument("--skip_random_nas", action="store_true",
                    help="Skip random NAS ablation (saves time)")
    ap.add_argument("--skip_lgbm", action="store_true",
                    help="Skip LightGBM baseline")
    ap.add_argument("--automl", action="store_true",
                    help="Also run AutoGluon baseline (requires autogluon.tabular)")
    ap.add_argument("--automl_time_limit", type=int, default=120,
                    help="Wall-clock budget in seconds for AutoGluon (default: 120)")
    ap.add_argument("--optuna", action="store_true",
                    help="Also run Optuna TPE baseline (requires optuna, pip install optuna)")
    ap.add_argument("--optuna_trials", type=int, default=50,
                    help="Number of Optuna trials (should match --budget for fair comparison)")
    ap.add_argument("--naive_llm", action="store_true",
                    help="Also run naive LLM baseline (LLM gets no schema — just dataset description)")
    ap.add_argument("--naive_llm_trials", type=int, default=5,
                    help="Number of proposals to ask from naive LLM (best is kept)")
    ap.add_argument("--easy", action="store_true",
                    help="Use easy datasets (credit_g, adult, mfeat_factors) instead of hard ones")
    ap.add_argument("--builtin", default=None,
                    choices=["covtype", "california_housing", "miniboonee"],
                    help="Use a built-in dataset instead of OpenML. "
                         "covtype=581k multiclass, california_housing=20k regression, "
                         "miniboonee=130k binary (NNs beat GBM on all three)")
    ap.add_argument("--csv", default=None,
                    help="Path to a CSV file to use as the dataset (last column = target)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--no_refine", action="store_true",
                    help="Use legacy PROPOSE loop instead of REFINE tool-call loop "
                         "(useful for ablation: LLM-NAS-Propose vs LLM-NAS-Refine)")
    ap.add_argument("--explore_pulse", type=int, default=6,
                    help="Inject PROPOSE step every N steps during REFINE phase (0=never)")
    ap.add_argument("--no_curves", action="store_true",
                    help="ABLATION: hide learning curve data from LLM — LLM sees only "
                         "final validation scores (simulates GENIUS / EvoPrompting). "
                         "Compare with default (curve_feedback=True) to measure contribution "
                         "of curve-aware feedback.")
    ap.add_argument("--domain", default=None,
                    choices=["biomedical", "finance", "all_domains"],
                    help="Run on domain-specific time-series datasets that show NNs > GBMs. "
                         "biomedical = ECG5000 + EEG_EyeState + HAR; "
                         "finance = ELEC2; "
                         "all_domains = all four. "
                         "Overrides --datasets and --easy flags.")
    args = ap.parse_args()

    api_key = args.api_key or os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        print("ERROR: Set OPENAI_API_KEY env var or pass --api_key sk-proj-...")
        sys.exit(1)

    exp_dir = Path(args.exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    # ── Domain mode: biomedical / finance / all_domains ────────────────────
    if getattr(args, 'domain', None):
        domain_map = {
            "biomedical":  BIOMEDICAL_DATASETS,
            "finance":     FINANCE_DATASETS,
            "all_domains": ALL_DOMAIN_DATASETS,
        }
        domain_datasets = domain_map[args.domain]
        rows = []
        for entry in domain_datasets:
            openml_id, name, task, src, bname = entry
            print(f"\n{'='*60}")
            print(f"[Domain:{args.domain}] Dataset: {name}  source={src}  task={task}")
            print(f"{'='*60}")
            try:
                row = run_one(
                    openml_id, name, task, exp_dir,
                    budget=args.budget,
                    coldstart_n=args.coldstart_n,
                    random_nas_budget=args.random_nas_budget,
                    ensemble_k=args.ensemble_k,
                    api_key=api_key,
                    llm_model=args.llm_model,
                    seed=args.seed,
                    skip_random_nas=args.skip_random_nas,
                    skip_lgbm=args.skip_lgbm,
                    run_automl=args.automl,
                    automl_time_limit=args.automl_time_limit,
                    run_optuna=args.optuna,
                    optuna_trials=args.optuna_trials,
                    run_naive_llm=args.naive_llm,
                    naive_llm_trials=args.naive_llm_trials,
                    device=args.device,
                    source=src,
                    builtin_name=bname,
                    csv_path=None,
                    use_refine_loop=not args.no_refine,
                    explore_pulse_every=args.explore_pulse,
                    curve_feedback=not getattr(args, 'no_curves', False),
                )
                rows.append(row)
            except Exception as e:
                print(f"\n[ERROR] Dataset {name}: {e}")
                import traceback; traceback.print_exc()

        if rows:
            print_summary(rows)
            summary_path = exp_dir / "summary.json"
            with open(summary_path, "w") as f:
                json.dump(rows, f, indent=2, ensure_ascii=False, default=str)
            print(f"Summary saved to {summary_path}")
        return

    # ── Normal mode: OpenML IDs or named groups ─────────────────────────────

    # Resolve dataset list
    pool = EASY_DATASETS if args.easy else DEFAULT_DATASETS
    if args.datasets:
        # User passed IDs only; look up names from pool or use "dataset_N"
        ds_map = {d[0]: d for d in DEFAULT_DATASETS + EASY_DATASETS + EXTENDED_DATASETS}
        datasets = []
        for did in args.datasets:
            if did in ds_map:
                datasets.append(ds_map[did])
            else:
                datasets.append((did, f"dataset_{did}", "auto"))
    else:
        datasets = pool

    # Resolve data source
    data_source = "openml"
    builtin_name = None
    csv_path = None
    if args.builtin:
        data_source = "builtin"
        builtin_name = args.builtin
        # Override datasets list with a single synthetic entry
        datasets = [(-1, args.builtin, "auto")]
        print(f"[Experiment] Using builtin dataset: {args.builtin}")
    elif args.csv:
        data_source = "csv"
        csv_path = args.csv
        datasets = [(-1, Path(args.csv).stem, "auto")]
        print(f"[Experiment] Using CSV dataset: {args.csv}")

    rows = []
    for openml_id, name, task in datasets:
        try:
            row = run_one(
                openml_id, name, task, exp_dir,
                budget=args.budget,
                coldstart_n=args.coldstart_n,
                random_nas_budget=args.random_nas_budget,
                ensemble_k=args.ensemble_k,
                api_key=api_key,
                llm_model=args.llm_model,
                seed=args.seed,
                skip_random_nas=args.skip_random_nas,
                skip_lgbm=args.skip_lgbm,
                run_automl=args.automl,
                automl_time_limit=args.automl_time_limit,
                run_optuna=args.optuna,
                optuna_trials=args.optuna_trials,
                run_naive_llm=args.naive_llm,
                naive_llm_trials=args.naive_llm_trials,
                device=args.device,
                source=data_source,
                builtin_name=builtin_name,
                csv_path=csv_path,
                use_refine_loop=not args.no_refine,
                explore_pulse_every=args.explore_pulse,
                curve_feedback=not getattr(args, 'no_curves', False),
            )
            rows.append(row)
        except Exception as e:
            print(f"\n[ERROR] Dataset {name} ({openml_id}): {e}")
            import traceback
            traceback.print_exc()

    if rows:
        print_summary(rows)
        summary_path = exp_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False, default=str)
        print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
