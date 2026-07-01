"""
Aggregate v3 multi-seed results into a thesis-ready summary table.

Reads experiments_v3/seed{42,1,2026}/<dataset>/best_trial.json and
ensemble_result.json, computes mean ± std across seeds, compares against
random_nas baseline, and writes a markdown table to
experiments_v3/SUMMARY.md.
"""
import json
import os
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
V3 = ROOT / "experiments_v3"
SEEDS = [42, 1, 2026]
DATASETS = ["jannis", "helena", "pol", "miniboone", "HAR", "ECG5000", "ELEC2"]


def collect():
    rows = []
    for ds in DATASETS:
        seed_results = []
        for seed in SEEDS:
            run_dir = V3 / f"seed{seed}" / ds
            best_path = run_dir / "best_trial.json"
            ens_path = run_dir / "ensemble_result.json"
            idx_path = run_dir / "trials_index.json"
            if not best_path.exists():
                continue
            best = json.load(open(best_path))
            ens = json.load(open(ens_path)) if ens_path.exists() else {}
            ops = []
            if idx_path.exists():
                idx = json.load(open(idx_path))
                trials = idx if isinstance(idx, list) else idx.get("trials", [])
                ops = [t.get("op", "") for t in trials]

            seed_results.append({
                "seed": seed,
                "primary": best.get("primary"),
                "ensemble": ens.get("primary") if ens else None,
                "best_arch": best.get("config", {}).get("arch", {}).get("family"),
                "n_forced": sum(1 for o in ops if o == "forced:switch_family"),
                "n_fallback": sum(1 for o in ops if o == "random_refine_fallback" or o == "refine:None"),
                "n_real_refine": sum(1 for o in ops if o.startswith("refine:") and o != "refine:None"),
            })

        if not seed_results:
            continue

        primaries = [r["primary"] for r in seed_results]
        ensembles = [r["ensemble"] for r in seed_results if r["ensemble"] is not None]

        # Try to load Random NAS baseline if present
        rand_path = V3 / f"seed{SEEDS[0]}" / ds / "baselines.json"
        random_nas = None
        catboost = None
        if rand_path.exists():
            bl = json.load(open(rand_path))
            random_nas = bl.get("random_nas", {}).get("primary")
            catboost = bl.get("catboost", {}).get("primary")

        rows.append({
            "dataset": ds,
            "n_seeds": len(seed_results),
            "primary_mean": statistics.mean(primaries),
            "primary_std": statistics.stdev(primaries) if len(primaries) > 1 else 0.0,
            "ensemble_mean": statistics.mean(ensembles) if ensembles else None,
            "random_nas": random_nas,
            "catboost": catboost,
            "forced_total": sum(r["n_forced"] for r in seed_results),
            "fallback_total": sum(r["n_fallback"] for r in seed_results),
            "real_refine_total": sum(r["n_real_refine"] for r in seed_results),
            "per_seed": seed_results,
        })
    return rows


def write_summary(rows):
    out = V3 / "SUMMARY.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write("# v3 multi-seed results\n\n")
        f.write("## Headline table\n\n")
        f.write("| Dataset | seeds | LLM-NAS mean ± std | Ensemble | Random | CatBoost | Δ vs Random |\n")
        f.write("|---|---:|---:|---:|---:|---:|---|\n")
        for r in rows:
            llm = f"{r['primary_mean']:.4f} ± {r['primary_std']:.4f}"
            ens = f"{r['ensemble_mean']:.4f}" if r["ensemble_mean"] is not None else "-"
            rnd = f"{r['random_nas']:.4f}" if r["random_nas"] is not None else "-"
            cb = f"{r['catboost']:.4f}" if r["catboost"] is not None else "-"
            if r["random_nas"] is not None:
                delta = r["primary_mean"] - r["random_nas"]
                tag = "**BEATS**" if delta > 0.005 else ("matches" if delta > -0.005 else "loses")
                vs = f"{tag} ({delta:+.4f})"
            else:
                vs = "-"
            f.write(f"| {r['dataset']} | {r['n_seeds']} | {llm} | {ens} | {rnd} | {cb} | {vs} |\n")

        f.write("\n## Bug-fix verification (per dataset, summed across seeds)\n\n")
        f.write("| Dataset | Real REFINE actions | Fallbacks | forced:switch_family |\n")
        f.write("|---|---:|---:|---:|\n")
        for r in rows:
            f.write(f"| {r['dataset']} | {r['real_refine_total']} | "
                    f"{r['fallback_total']} | {r['forced_total']} |\n")

        # Aggregate stats
        wins = sum(1 for r in rows if r["random_nas"] is not None and r["primary_mean"] > r["random_nas"] + 0.005)
        ties = sum(1 for r in rows if r["random_nas"] is not None and abs(r["primary_mean"] - r["random_nas"]) <= 0.005)
        losses = sum(1 for r in rows if r["random_nas"] is not None and r["primary_mean"] < r["random_nas"] - 0.005)
        f.write(f"\n## Aggregate\n\n")
        f.write(f"- LLM-NAS BEATS Random: **{wins} / {len(rows)}** datasets\n")
        f.write(f"- LLM-NAS matches Random: {ties} / {len(rows)} datasets\n")
        f.write(f"- LLM-NAS loses to Random: {losses} / {len(rows)} datasets\n")
        f.write(f"- Total forced:switch_family across all runs: {sum(r['forced_total'] for r in rows)}\n")
        f.write(f"- Total fallbacks across all runs: {sum(r['fallback_total'] for r in rows)}\n")

    print(f"Wrote {out}")


if __name__ == "__main__":
    rows = collect()
    if not rows:
        print("No v3 results found yet.")
    else:
        write_summary(rows)
        print(f"Aggregated {len(rows)} datasets.")
