"""
Run the v2 LLM-guided NAS pipeline on a single OpenML dataset.

Usage examples:

  # API key from env:
  export OPENAI_API_KEY=sk-proj-...
  python scripts/run_nas_v2.py --openml_id 31 --out_dir runs/credit_v2 --budget 25

  # API key as flag:
  python scripts/run_nas_v2.py \\
      --openml_id 1590 \\
      --task binary \\
      --out_dir runs/adult_v2 \\
      --budget 30 \\
      --coldstart_n 6 \\
      --llm_model gpt-4o-mini \\
      --api_key sk-proj-...

The script writes results to out_dir/:
  dataset_summary.json
  trials/trial_NNN/  (config.json, metrics.json, model.pt, history.json)
  trials_index.json
  best_trial.json
  final_report.json
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.nas_orchestrator import run_llm_nas_v2


def main():
    ap = argparse.ArgumentParser(description="LLM-NAS v2 runner")
    ap.add_argument("--openml_id", type=int, required=False, default=None,
                    help="OpenML dataset ID (e.g. 31=credit-g, 1590=adult)")
    ap.add_argument("--task", default="auto",
                    choices=["auto", "binary", "multiclass", "regression"])
    ap.add_argument("--out_dir", required=True,
                    help="Output directory for trial results")
    ap.add_argument("--budget", type=int, default=25,
                    help="Total number of trials (cold-start included)")
    ap.add_argument("--coldstart_n", type=int, default=5,
                    help="Number of LLM cold-start (diverse init) trials. "
                         "Used when --random_warmup=0 (default). "
                         "Otherwise replaced by random_warmup random trials.")
    ap.add_argument("--random_warmup", type=int, default=0,
                    help="Number of PURE RANDOM trials before LLM refine-loop. "
                         "If 0 (default) — use --coldstart_n LLM cold-start trials. "
                         "If >0 — skip LLM cold-start, use N random trials instead "
                         "(advisor's idea: gives LLM rich diverse history to reason from).")
    ap.add_argument("--reflect_every", type=int, default=8,
                    help="LLM reflection every N evolution steps")
    ap.add_argument("--crossover_p", type=float, default=0.25,
                    help="Probability of crossover vs. mutation at each step")
    ap.add_argument("--population_size", type=int, default=20)
    ap.add_argument("--tournament_size", type=int, default=5)
    ap.add_argument("--surr_candidates", type=int, default=3,
                    help="Number of LLM proposals per step (surrogate picks best)")
    ap.add_argument("--no_multifidelity", action="store_true",
                    help="Disable multi-fidelity (always use full evaluation)")
    ap.add_argument("--no_refine", action="store_true",
                    help="Use legacy PROPOSE loop instead of REFINE tool-call loop")
    ap.add_argument("--explore_pulse", type=int, default=6,
                    help="Inject a PROPOSE step every N steps during REFINE phase (0=never)")
    ap.add_argument("--mode", default="refine",
                    choices=["refine", "freeform", "multi_refine", "critic_corrector", "batch"],
                    help="LLM-NAS evolutionary mode: "
                         "'refine' = 22-action vocab (default); "
                         "'multi_refine' = up to 3 actions per turn + LR throttle; "
                         "'critic_corrector' = 2-stage Critic→Corrector pipeline; "
                         "'freeform' = LLM proposes config diff directly; "
                         "'batch' = LLM proposes N candidates, surrogate filters, train top-K.")
    ap.add_argument("--batch_n", type=int, default=15,
                    help="[batch mode] how many configs LLM proposes per round (default: 15)")
    ap.add_argument("--batch_k", type=int, default=3,
                    help="[batch mode] how many configs to train per round (default: 3)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--llm_model", default="gpt-4o",
                    help="LLM model id (default: gpt-4o full — structured-output is "
                         "~2-3x stronger than gpt-4o-mini on tabular NAS reasoning). "
                         "Other strong options: 'gpt-4-turbo', 'o1-mini', "
                         "'o1-preview' (reasoning), 'anthropic/claude-sonnet-4.5' "
                         "(via OpenRouter). For cheap/fast: 'gpt-4o-mini'.")
    ap.add_argument("--llm_temperature", type=float, default=0.7)
    ap.add_argument("--api_key", default=None,
                    help="API key (OpenAI or OpenRouter, default: $OPENAI_API_KEY)")
    ap.add_argument("--base_url", default="",
                    help="Custom API base URL (e.g. https://openrouter.ai/api/v1/chat/completions)")
    ap.add_argument("--require_llm", action="store_true",
                    help="Abort if an LLM proposal fails or is invalid; do not silently use random fallback.")
    ap.add_argument("--ensemble_k", type=int, default=0,
                    help="Re-train and ensemble top-K NAS configs at full fidelity (0 = disable)")
    ap.add_argument("--ensemble_only", action="store_true",
                    help="Skip NAS search, only run ensemble on existing trials_index.json")
    ap.add_argument("--device", default=None, help="cuda / cpu (auto-detected if omitted)")
    ap.add_argument("--builtin", default=None,
                    choices=["covtype", "california_housing", "miniboonee",
                             "ecg5000", "har", "harth", "pamap2", "elec2", "emg_gestures",
                             "etth1", "etth2", "ettm1", "ettm2", "weather", "exchange"],
                    help="Built-in dataset instead of OpenML "
                         "(covtype / california_housing / miniboonee / ecg5000 / har / harth / pamap2 / elec2 / emg_gestures / "
                         "etth1 / etth2 / ettm1 / ettm2 / weather / exchange)")
    ap.add_argument("--lookback", type=int, default=96,
                    help="Lookback window for forecasting datasets (default 96)")
    ap.add_argument("--horizon", type=int, default=1,
                    help="Prediction horizon for forecasting datasets (default 1)")
    ap.add_argument("--csv", default=None,
                    help="Path to CSV file (last column = target)")
    ap.add_argument("--resume", action="store_true",
                    help="Skip warmup phase — load existing warmup trials from "
                         "out_dir/trials_index.json and continue with LLM phase. "
                         "Useful when warmup completed but LLM phase was broken.")
    ap.add_argument("--extend", action="store_true",
                    help="Load ALL existing trials from out_dir and run `budget` MORE trials. "
                         "Use after a completed run to add extra refine trials on top of batch. "
                         "Example: ran 40 batch trials, now extend with 15 refine: "
                         "--extend --mode refine --budget 15")
    args = ap.parse_args()

    api_key = args.api_key or os.getenv("OPENAI_API_KEY", "")
    if not api_key and not args.ensemble_only:
        print("ERROR: Set OPENAI_API_KEY env var or pass --api_key sk-proj-...")
        sys.exit(1)

    # Forecasting datasets → convert to CSV and run as regression
    FORECASTING_DATASETS = {"etth1", "etth2", "ettm1", "ettm2", "weather", "exchange"}

    if args.builtin and args.builtin in FORECASTING_DATASETS:
        from src.data_forecasting import load_forecasting_raw
        import pandas as pd, tempfile
        print(f"[Forecasting] Loading {args.builtin} (lookback={args.lookback}, horizon={args.horizon}) ...")
        raw, summary = load_forecasting_raw(
            args.builtin, lookback=args.lookback, horizon=args.horizon, seed=args.seed
        )
        # Save as CSV for NAS pipeline
        tmp_dir = Path(args.out_dir) / "tmp_forecasting_csv"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        n_feat = raw.X_train.shape[1]
        cols = [f"f{i}" for i in range(n_feat)]
        for split_name, X, y in [("train", raw.X_train, raw.y_train),
                                   ("val",   raw.X_val,   raw.y_val),
                                   ("test",  raw.X_test,  raw.y_test)]:
            df = pd.DataFrame(X, columns=cols)
            df["target"] = y
            df.to_csv(tmp_dir / f"{split_name}.csv", index=False)
        csv_path = str(tmp_dir / "train.csv")
        source = "csv"
        builtin_name = None
        if args.task == "auto":
            args.task = "regression"
        print(f"[Forecasting] Converted to CSV: {n_feat} features, "
              f"train={len(raw.X_train)}, val={len(raw.X_val)}, test={len(raw.X_test)}")
    else:
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

    if args.ensemble_only:
        # Only run ensemble on existing trials — no new LLM calls
        from src.nas_orchestrator import run_ensemble_only
        run_ensemble_only(
            out_dir=args.out_dir,
            ensemble_k=args.ensemble_k,
            seed=args.seed,
            device=args.device,
            openml_id=args.openml_id,
            task=args.task,
            source=source,
            builtin_name=builtin_name,
            csv_path=csv_path,
        )
    else:
        run_llm_nas_v2(
            openml_id=args.openml_id,
            task=args.task,
            out_dir=args.out_dir,
            budget=args.budget,
            coldstart_n=args.coldstart_n,
            reflect_every=args.reflect_every,
            surr_candidates_k=args.surr_candidates,
            use_refine_loop=not args.no_refine,
            explore_pulse_every=args.explore_pulse,
            mode=args.mode,
            batch_n=args.batch_n,
            batch_k=args.batch_k,
            random_warmup=args.random_warmup,
            crossover_p=args.crossover_p,
            population_size=args.population_size,
            tournament_size=args.tournament_size,
            use_multi_fidelity=not args.no_multifidelity,
            seed=args.seed,
            llm_model=args.llm_model,
            llm_temperature=args.llm_temperature,
            api_key=api_key,
            base_url=args.base_url,
            device=args.device,
            ensemble_k=args.ensemble_k,
            source=source,
            builtin_name=builtin_name,
            csv_path=csv_path,
            resume=args.resume,
            extend=args.extend,
            require_llm=args.require_llm,
        )


if __name__ == "__main__":
    main()
