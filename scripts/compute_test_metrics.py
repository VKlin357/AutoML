"""
Compute test-set metrics for all existing seed-42 experiments.

The original runs saved only validation metrics in ensemble_result.json.
This script re-runs ensemble_only (re-trains top-K configs on the same data)
and saves a new ensemble_result.json that includes test_primary.

Usage (run from repo root on GPU machine):
    python scripts/compute_test_metrics.py --exp_dir experiments_v9 --ensemble_k 7

The script writes:
  <exp_dir>/<name>/ensemble_result.json   (updated, now includes test_primary)
  <exp_dir>/test_metrics_summary.json     (all results in one file)

Requirements: GPU machine with the same environment as the original runs.
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Dataset registry: name → (source, openml_id, builtin_name, task)
# ---------------------------------------------------------------------------
DATASETS = {
    "volkert":    ("openml", 41166,  None,          "multiclass"),
    "jannis":     ("openml", 45021,  None,          "multiclass"),
    "miniboonee": ("builtin", -3,    "miniboonee",  "binary"),
    "helena":     ("openml", 41169,  None,          "multiclass"),
    "adult":      ("openml", 1590,   None,          "binary"),
    # sensor / extra datasets
    "har":        ("builtin", -1,    "har",         "multiclass"),
    "harth":      ("openml", 43921,  None,          "multiclass"),
    "pamap2":     ("openml", 44994,  None,          "multiclass"),
    "ecg5000":    ("builtin", -2,    "ecg5000",     "multiclass"),
    # EEG Eye State (openml 1471) — used in eeg_llm_s42 / eeg_rnd40 / eeg_baselines
    "eeg":        ("openml", 1471,   None,          "binary"),
    # Other extra datasets encountered in experiments_v9
    "airlines":   ("openml", 1169,   None,          "binary"),
    "gesture":    ("openml", 4538,   None,          "multiclass"),
    "pol":        ("openml", 722,    None,          "multiclass"),
    "sensorless": ("openml", 9981,   None,          "multiclass"),
    "elec2":      ("openml", 151,    None,          "binary"),
    "traffic":    ("openml", 1568,   None,          "multiclass"),
}

# Map experiment-directory name patterns → dataset key
# We match the start of the directory name.
DIR_TO_DATASET = {
    "volkert":    "volkert",
    "jannis":     "jannis",
    "miniboonee": "miniboonee",
    "helena":     "helena",
    "adult":      "adult",
    "har_":       "har",
    "harth":      "harth",
    "pamap2":     "pamap2",
    "ecg5000":    "ecg5000",
    "eeg_":       "eeg",       # eeg_llm_s42, eeg_rnd40 — EEG Eye State openml:1471
    "airlines":   "airlines",
    "gesture":    "gesture",
    "pol_":       "pol",
    "sensorless": "sensorless",
    "elec2":      "elec2",
    "traffic":    "traffic",
}


def _resolve_dataset(exp_name: str, exp_dir: Path = None):
    """Return (source, openml_id, builtin_name, task) for an experiment dir name.

    Priority:
    1. Read dataset_summary.json from the experiment dir (most accurate — uses
       the exact same source that the original training used).
    2. Fall back to the hardcoded DATASETS registry.
    """
    # --- Try reading dataset_summary.json first ---
    if exp_dir is not None:
        summary_path = exp_dir / "dataset_summary.json"
        if summary_path.exists():
            try:
                import json as _json
                summary = _json.loads(summary_path.read_text())
                openml_id = summary.get("openml_id")
                task = summary.get("task")

                # Negative openml_id means it was loaded as a builtin
                if openml_id is not None and openml_id < 0:
                    # Determine builtin name from experiment dir prefix
                    for prefix, key in DIR_TO_DATASET.items():
                        if exp_name.startswith(prefix):
                            # Use the builtin_name from registry
                            _, _, bname, _ = DATASETS.get(key, (None, None, None, None))
                            if bname:
                                return "builtin", None, bname, task
                    # Fallback: guess builtin name from dir prefix
                    name_lower = exp_name.split("_")[0]
                    return "builtin", None, name_lower, task

                # Positive openml_id: use OpenML source
                if openml_id and openml_id > 0 and task:
                    return "openml", openml_id, None, task
            except Exception:
                pass  # Fall through to registry lookup

    # --- Fall back to hardcoded registry ---
    for prefix, key in DIR_TO_DATASET.items():
        if exp_name.startswith(prefix):
            src, oid, bname, task = DATASETS[key]
            return src, oid, bname, task
    return None


def _load_dataset(source, openml_id, builtin_name, task, seed):
    from src.data import load_raw
    return load_raw(
        source=source,
        openml_id=openml_id if openml_id and openml_id > 0 else None,
        task=task,
        builtin_name=builtin_name,
        seed=seed,
    )


def _run_ensemble(exp_dir: Path, source, openml_id, builtin_name, task,
                  seed: int, ensemble_k: int, device):
    """Re-run ensemble_only on an existing experiment directory."""
    trials_path = exp_dir / "trials_index.json"
    if not trials_path.exists():
        print(f"  SKIP — no trials_index.json in {exp_dir}")
        return None

    raw_data = json.loads(trials_path.read_text())
    all_trials = raw_data["trials"] if isinstance(raw_data, dict) and "trials" in raw_data else raw_data
    # Filter out trials without config (cheap-screening artifacts)
    valid_trials = [t for t in all_trials if "config" in t]
    if len(valid_trials) < len(all_trials):
        print(f"  Filtered {len(all_trials) - len(valid_trials)} trials without config")
    all_trials = valid_trials
    print(f"  Loaded {len(all_trials)} trials")

    raw, _ = _load_dataset(source, openml_id, builtin_name, task, seed)

    from src.ensemble import ensemble_top_k
    result = ensemble_top_k(
        all_trials, raw,
        k=ensemble_k,
        method="greedy",
        seed=seed,
        device=device,
        out_dir=str(exp_dir),
    )
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_dir", default="experiments_v9",
                    help="Root experiments directory")
    ap.add_argument("--names", nargs="*", default=None,
                    help="Only process these experiment dirs (default: all with ensemble_result.json missing test)")
    ap.add_argument("--ensemble_k", type=int, default=7)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None, help="cuda / cpu")
    ap.add_argument("--force", action="store_true",
                    help="Re-compute even if test_primary already exists")
    args = ap.parse_args()

    exp_root = Path(args.exp_dir)
    if not exp_root.exists():
        print(f"ERROR: {exp_root} does not exist")
        sys.exit(1)

    # Collect experiment dirs to process
    if args.names:
        dirs = [exp_root / n for n in args.names]
    else:
        dirs = sorted([d for d in exp_root.iterdir() if d.is_dir()])

    summary = {}
    skipped = []

    for exp_dir in dirs:
        exp_name = exp_dir.name

        # Skip baselines / optuna / rnd dirs — they don't have LLM-NAS trials
        skip_keywords = ["baselines", "optuna", "rnd", "random", "lgb_only", "rerun"]
        if any(k in exp_name for k in skip_keywords):
            continue

        ens_path = exp_dir / "ensemble_result.json"
        # Allow computing even if ensemble_result.json doesn't exist yet
        # (e.g. pamap2_llm_s42 — trials ran but ensemble was never called)
        if not ens_path.exists():
            if not (exp_dir / "trials_index.json").exists():
                skipped.append(exp_name)
                continue
            existing = {}
        else:
            existing = json.loads(ens_path.read_text())

        # Check if test already computed
        if not args.force and existing.get("test_primary") is not None:
            print(f"[SKIP] {exp_name} — test_primary already present: {existing['test_primary']:.5f}")
            summary[exp_name] = {
                "val_primary":  existing.get("val_primary", existing.get("primary")),
                "test_primary": existing["test_primary"],
                "status": "already_done",
            }
            continue

        dataset_info = _resolve_dataset(exp_name, exp_dir)
        if dataset_info is None:
            print(f"[SKIP] {exp_name} — unknown dataset, add to DIR_TO_DATASET")
            continue

        source, openml_id, builtin_name, task = dataset_info
        print(f"\n{'='*60}")
        print(f"Processing: {exp_name}")
        print(f"  dataset: source={source}, openml_id={openml_id}, task={task}")
        print(f"  seed={args.seed}, ensemble_k={args.ensemble_k}")

        try:
            result = _run_ensemble(
                exp_dir, source, openml_id, builtin_name, task,
                seed=args.seed, ensemble_k=args.ensemble_k, device=args.device,
            )
            if result:
                val = result.get("val_primary")
                test = result.get("test_primary")
                print(f"  ✓ val={val:.5f}  test={test:.5f}")
                summary[exp_name] = {
                    "val_primary": val,
                    "test_primary": test,
                    "status": "computed",
                }
            else:
                summary[exp_name] = {"status": "failed"}
        except Exception as e:
            print(f"  ERROR: {e}")
            summary[exp_name] = {"status": "error", "error": str(e)}

    # Print final table
    print(f"\n{'='*60}")
    print(f"{'Experiment':<35} {'Val':>8} {'Test':>8}  Status")
    print("-" * 60)
    for name, info in sorted(summary.items()):
        val  = f"{info['val_primary']:.5f}"  if info.get('val_primary')  else "  N/A  "
        test = f"{info['test_primary']:.5f}" if info.get('test_primary') else "  N/A  "
        print(f"{name:<35} {val:>8} {test:>8}  {info['status']}")

    # Save summary
    out = exp_root / "test_metrics_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved summary → {out}")

    if skipped:
        print(f"\nDirs with no ensemble_result.json (skipped): {skipped}")


if __name__ == "__main__":
    main()
