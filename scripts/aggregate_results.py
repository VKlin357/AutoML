"""
Aggregate multi-seed results into a mean±std table.

Usage:
    # After running compute_test_metrics.py and run_multiseed.sh:
    python scripts/aggregate_results.py --exp_dir experiments_multiseed

    # Or specify seeds explicitly:
    python scripts/aggregate_results.py --exp_dir experiments_multiseed --seeds 42 0 1337

Output:
    - Terminal: formatted table (val and test, mean±std)
    - experiments_multiseed/results_table.json
    - experiments_multiseed/results_table.csv  (paste into LaTeX)
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path


# Datasets to report (display name → dir prefix)
DATASETS = {
    "Volkert":    "volkert",
    "Jannis":     "jannis",
    "MiniBooNE":  "miniboonee",
    "Helena":     "helena",
    "Adult":      "adult",
}


def _load_ens(exp_dir: Path, dataset_prefix: str, seed: int):
    """Load ensemble_result.json for one (dataset, seed)."""
    candidate_dirs = [
        exp_dir / f"{dataset_prefix}_batch_s{seed}",
        exp_dir / f"{dataset_prefix}_llm_s{seed}",
        exp_dir / f"{dataset_prefix}_llm_nas",   # fallback: seed-agnostic name
    ]
    for d in candidate_dirs:
        p = d / "ensemble_result.json"
        if p.exists():
            data = json.loads(p.read_text())
            return data
    return None


def _stats(values):
    """Return (mean, std) or (value, None) if only one point."""
    values = [v for v in values if v is not None]
    if not values:
        return None, None
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, None
    var = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return mean, math.sqrt(var)


def fmt(mean, std, decimals=4):
    """Format as '0.8031 ± 0.0012' or '0.8031' if std is None."""
    if mean is None:
        return "  N/A   "
    s = f"{mean:.{decimals}f}"
    if std is not None:
        s += f" ±{std:.{decimals}f}"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_dir", default="experiments_multiseed",
                    help="Directory containing all seed experiments")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 0, 1337])
    ap.add_argument("--decimals", type=int, default=4)
    args = ap.parse_args()

    exp_dir = Path(args.exp_dir)
    if not exp_dir.exists():
        print(f"ERROR: {exp_dir} does not exist. Run run_multiseed.sh first.")
        sys.exit(1)

    seeds = args.seeds
    D = args.decimals

    # ── Collect data ──────────────────────────────────────────────────────────
    table = {}   # dataset_name → {seed → {val, test}}

    for display_name, prefix in DATASETS.items():
        table[display_name] = {}
        for seed in seeds:
            rec = _load_ens(exp_dir, prefix, seed)
            if rec is None:
                print(f"  [missing] {display_name} seed={seed}")
                table[display_name][seed] = None
            else:
                val  = rec.get("val_primary",  rec.get("primary"))
                test = rec.get("test_primary")
                table[display_name][seed] = {"val": val, "test": test}

    # ── Print table ───────────────────────────────────────────────────────────
    col_w = max(len(n) for n in DATASETS) + 2

    # Header
    header_seeds = "  ".join(f"seed={s}" for s in seeds)
    print(f"\n{'Dataset':<{col_w}}  {'Val (mean±std)':<22}  {'Test (mean±std)':<22}  {header_seeds}")
    print("-" * (col_w + 70 + len(seeds) * 12))

    summary = {}
    for display_name in DATASETS:
        rows = table[display_name]
        vals  = [rows[s]["val"]  if rows.get(s) else None for s in seeds]
        tests = [rows[s]["test"] if rows.get(s) else None for s in seeds]

        val_mean,  val_std  = _stats(vals)
        test_mean, test_std = _stats(tests)

        per_seed_str = "  ".join(
            f"{rows[s]['test']:.{D}f}" if rows.get(s) and rows[s]["test"] else "  N/A  "
            for s in seeds
        )

        print(f"{display_name:<{col_w}}  {fmt(val_mean, val_std, D):<22}  "
              f"{fmt(test_mean, test_std, D):<22}  {per_seed_str}")

        summary[display_name] = {
            "val_mean":  val_mean,
            "val_std":   val_std,
            "test_mean": test_mean,
            "test_std":  test_std,
            "per_seed":  {str(s): rows.get(s) for s in seeds},
        }

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out_json = exp_dir / "results_table.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved → {out_json}")

    # ── Save CSV (LaTeX-friendly) ─────────────────────────────────────────────
    out_csv = exp_dir / "results_table.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Dataset", "Val mean", "Val std", "Test mean", "Test std"]
                        + [f"Test seed={s}" for s in seeds])
        for display_name, info in summary.items():
            per_seed_tests = [
                info["per_seed"].get(str(s), {}).get("test") if info["per_seed"].get(str(s)) else None
                for s in seeds
            ]
            writer.writerow([
                display_name,
                f"{info['val_mean']:.{D}f}"  if info["val_mean"]  is not None else "",
                f"{info['val_std']:.{D}f}"   if info["val_std"]   is not None else "",
                f"{info['test_mean']:.{D}f}" if info["test_mean"] is not None else "",
                f"{info['test_std']:.{D}f}"  if info["test_std"]  is not None else "",
            ] + [f"{v:.{D}f}" if v is not None else "" for v in per_seed_tests])
    print(f"Saved → {out_csv}")

    # ── LaTeX snippet ─────────────────────────────────────────────────────────
    print("\n── LaTeX table snippet ──────────────────────────────────────────")
    print("\\begin{tabular}{lcc}")
    print("\\toprule")
    print("Dataset & Val (mean$\\pm$std) & Test (mean$\\pm$std) \\\\")
    print("\\midrule")
    for display_name, info in summary.items():
        vm, vs = info["val_mean"], info["val_std"]
        tm, ts = info["test_mean"], info["test_std"]
        val_s  = f"${vm:.{D}f} \\pm {vs:.{D}f}$" if vs is not None and vm is not None else (f"${vm:.{D}f}$" if vm else "--")
        test_s = f"${tm:.{D}f} \\pm {ts:.{D}f}$" if ts is not None and tm is not None else (f"${tm:.{D}f}$" if tm else "--")
        print(f"{display_name} & {val_s} & {test_s} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")


if __name__ == "__main__":
    main()
