"""
Подробный эксперимент на одном датасете: CatBoost baseline + LLM-NAS.
Логи максимально детальные — каждый триал, каждая итерация, все метрики.

Usage:
    python src/run_verbose_experiment.py --dataset pamap2 --out_dir experiments/pamap2_verbose
    python src/run_verbose_experiment.py --dataset ecg5000 --out_dir experiments/ecg5000_verbose
"""
import argparse
import json
import os
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

SEPARATOR = "=" * 70


def log(msg: str, f=None):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    if f:
        f.write(line + "\n")
        f.flush()


def print_dataset_info(raw, summary, f=None):
    import numpy as np
    log(SEPARATOR, f)
    log("  DATASET INFO", f)
    log(SEPARATOR, f)
    log(f"  Name:           {summary.get('dataset_name', '?')}", f)
    log(f"  Task:           {raw.task}  |  Classes: {raw.n_classes}", f)
    log(f"  Num features:   {len(raw.num_cols)}", f)
    log(f"  Cat features:   {len(raw.cat_cols)}", f)
    log(f"  Train rows:     {len(raw.X_train):,}", f)
    log(f"  Val rows:       {len(raw.X_val):,}", f)
    log(f"  Test rows:      {len(raw.X_test):,}", f)
    log(f"  Split strategy: {summary.get('split_strategy', 'unknown')}", f)
    log("", f)

    # Label distribution
    unique, counts = zip(*sorted(
        zip(*__import__('numpy').unique(raw.y_train, return_counts=True))
    ))
    log("  Class distribution (train):", f)
    for lbl, cnt in zip(unique, counts):
        pct = cnt / len(raw.y_train) * 100
        bar = "█" * int(pct / 2)
        log(f"    class {int(lbl):2d}: {cnt:6,}  ({pct:5.1f}%)  {bar}", f)
    log(SEPARATOR, f)


def run_catboost_verbose(raw, out_dir: Path, seed: int, f=None):
    import numpy as np
    from src.preprocessing import Preprocessor
    from src.metrics import compute_metrics

    log("", f)
    log(SEPARATOR, f)
    log("  CATBOOST BASELINE", f)
    log(SEPARATOR, f)

    pre = Preprocessor(num_encoder="standard", cat_encoder="embedding")
    sp = pre.fit_transform(
        raw.X_train, raw.X_val, raw.X_test,
        raw.y_train, raw.y_val, raw.y_test,
        raw.num_cols, raw.cat_cols, raw.task, raw.n_classes,
    )

    # Concatenate num + cat for CatBoost
    def concat(num, cat):
        if cat.shape[1] > 0:
            return np.concatenate([num, cat.astype(np.float32)], axis=1)
        return num

    Xtr = concat(sp.X_train_num, sp.X_train_cat)
    Xva = concat(sp.X_val_num, sp.X_val_cat)
    Xte = concat(sp.X_test_num, sp.X_test_cat)

    log(f"  Input shape: {Xtr.shape}  (flat numeric, no temporal structure)", f)
    log(f"  CatBoost sees {Xtr.shape[1]} independent columns", f)
    log(f"  → cannot exploit channel correlations or time ordering", f)
    log("", f)

    try:
        from catboost import CatBoostClassifier, CatBoostRegressor
    except ImportError:
        log("ERROR: catboost not installed", f)
        return None

    cb_params = dict(
        iterations=2000,
        learning_rate=0.05,
        depth=8,
        random_seed=seed,
        verbose=50,  # print every 50 iterations
        use_best_model=True,
        early_stopping_rounds=100,
    )
    log(f"  Params: {cb_params}", f)
    log("", f)

    t0 = time.time()
    if raw.task == "binary":
        m = CatBoostClassifier(loss_function="Logloss", **cb_params)
        m.fit(Xtr, sp.y_train, eval_set=(Xva, sp.y_val))
        val_met = compute_metrics("binary", sp.y_val,   y_pred_proba=m.predict_proba(Xva)[:, 1])
        test_met = compute_metrics("binary", sp.y_test, y_pred_proba=m.predict_proba(Xte)[:, 1])
    elif raw.task == "multiclass":
        m = CatBoostClassifier(loss_function="MultiClass", **cb_params)
        m.fit(Xtr, sp.y_train, eval_set=(Xva, sp.y_val))
        val_met = compute_metrics("multiclass", sp.y_val,   y_pred_proba=m.predict_proba(Xva))
        test_met = compute_metrics("multiclass", sp.y_test, y_pred_proba=m.predict_proba(Xte))
    else:
        m = CatBoostRegressor(loss_function="RMSE", **cb_params)
        m.fit(Xtr, sp.y_train, eval_set=(Xva, sp.y_val))
        val_met = compute_metrics("regression", sp.y_val,   y_pred=m.predict(Xva))
        test_met = compute_metrics("regression", sp.y_test, y_pred=m.predict(Xte))

    elapsed = time.time() - t0
    log("", f)
    log(f"  Best iteration: {m.best_iteration_}", f)
    log(f"  Time: {elapsed:.1f}s", f)
    log("", f)
    log("  RESULTS:", f)
    for k, v in test_met.metrics.items():
        log(f"    test_{k}: {v:.6f}", f)
    log(f"  >>> CatBoost test_primary = {test_met.primary:.6f}", f)
    log(SEPARATOR, f)

    result = {
        "model": "catboost",
        "test_primary": test_met.primary,
        "val_primary": val_met.primary,
        "test_metrics": test_met.metrics,
        "val_metrics": val_met.metrics,
        "best_iteration": int(m.best_iteration_),
        "seconds": elapsed,
        "n_features": int(Xtr.shape[1]),
        "note": "flat numeric features — no temporal structure exploited",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "catboost_verbose.json").write_text(json.dumps(result, indent=2))
    return result


def run_llm_nas_verbose(dataset: str, out_dir: Path, seed: int,
                        budget: int, api_key: str, model: str, f=None):
    log("", f)
    log(SEPARATOR, f)
    log("  LLM-NAS", f)
    log(SEPARATOR, f)
    log(f"  Budget:  {budget} trials", f)
    log(f"  Warmup:  10 random trials", f)
    log(f"  Mode:    batch (N=20 proposals → top-4 trained)", f)
    log(f"  Model:   {model}", f)
    log("", f)

    cmd = [
        sys.executable, "src/run_nas_v2.py",
        "--builtin", dataset,
        "--task", "auto",
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

    log(f"  Command: {' '.join(cmd)}", f)
    log("", f)

    t0 = time.time()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        cwd=ROOT,
    )

    # Stream output line by line with timestamps
    trial_results = []
    for line in proc.stdout:
        line = line.rstrip()
        # Always print raw line
        ts = datetime.now().strftime("%H:%M:%S")
        out_line = f"[{ts}] {line}"
        print(out_line)
        if f:
            f.write(out_line + "\n")
            f.flush()

        # Parse trial results for summary
        if "primary=" in line and ("TRIAL" in line or "trial" in line):
            trial_results.append(line)

    proc.wait()
    elapsed = time.time() - t0

    log("", f)
    log(f"  LLM-NAS finished in {elapsed:.1f}s  (exit={proc.returncode})", f)
    log(SEPARATOR, f)
    return proc.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    choices=["ecg5000", "harth", "pamap2", "elec2", "emg_gestures", "har"])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--budget", type=int, default=30)
    ap.add_argument("--llm_model", default="gpt-4o-mini")
    ap.add_argument("--api_key", default=None)
    ap.add_argument("--skip_baseline", action="store_true")
    ap.add_argument("--skip_nas", action="store_true")
    args = ap.parse_args()

    api_key = args.api_key or os.getenv("OPENAI_API_KEY", "")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "experiment_verbose.log"
    f = open(log_path, "w")

    log(SEPARATOR, f)
    log(f"  VERBOSE EXPERIMENT: {args.dataset.upper()}", f)
    log(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", f)
    log(f"  Out dir: {out_dir}", f)
    log(SEPARATOR, f)

    # Load dataset
    log("Loading dataset...", f)
    from src.data import load_builtin_raw
    t0 = time.time()
    raw, summary = load_builtin_raw(args.dataset, task="auto", seed=args.seed)
    log(f"Loaded in {time.time()-t0:.1f}s", f)
    print_dataset_info(raw, summary, f)

    cb_result = None
    if not args.skip_baseline:
        cb_result = run_catboost_verbose(raw, out_dir, args.seed, f)

    nas_rc = None
    if not args.skip_nas:
        if not api_key:
            log("ERROR: set OPENAI_API_KEY or pass --api_key", f)
        else:
            nas_rc = run_llm_nas_verbose(
                args.dataset, out_dir, args.seed, args.budget,
                api_key, args.llm_model, f
            )

    # Final summary
    log("", f)
    log(SEPARATOR, f)
    log("  FINAL SUMMARY", f)
    log(SEPARATOR, f)
    if cb_result:
        log(f"  CatBoost  test_primary = {cb_result['test_primary']:.6f}", f)
        log(f"            ({cb_result['n_features']} flat features, no temporal structure)", f)

    # Read NAS best result
    nas_idx = out_dir / "nas_trials" / "trials_index.json"
    if nas_idx.exists():
        import json as _json
        data = _json.loads(nas_idx.read_text())
        trials = data.get("trials", data) if isinstance(data, dict) else data
        if trials:
            best = max(trials, key=lambda t: t.get("primary") or 0)
            arch = best.get("config", {}).get("arch", {}).get("family", "?")
            score = best.get("primary", 0)
            log(f"  LLM-NAS   best_val    = {score:.6f}  (arch={arch})", f)
            if cb_result:
                delta = score - cb_result["test_primary"]
                sign = "✓ LLM-NAS wins" if delta > 0 else "✗ CatBoost wins"
                log(f"  Delta     = {delta:+.6f}  → {sign}", f)

    log(SEPARATOR, f)
    log(f"  Full log saved: {log_path}", f)
    f.close()


if __name__ == "__main__":
    main()
