"""
Random NAS v3 baseline — честное сравнение с LLM-NAS v3.

Использует ТОЧНО ТЕ ЖЕ механизмы что и LLM-NAS v3:
  - _evaluate_config с cheap→medium→full
  - search_score для продвижения (не raw accuracy)
  - promote_frac_cheap=0.75, promote_frac_medium=0.50
  - Per-family late-bloomer protection (tabm/ft_transformer)
  - Stratified warmup (family quotas)
  - ensemble_top_k для финального результата

Единственное отличие: нет LLM. Конфиги сэмплируются случайно.

Запуск:
    python scripts/run_random_nas_v3.py \\
        --openml_id 41166 --task multiclass \\
        --out_dir experiments_v8/volkert_random_v3_s42 \\
        --budget 120 --ensemble_k 7 --seed 42 --device cuda
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import load_raw
from src.preprocessing import Preprocessor
from src.search_space import (
    ARCH_FAMILIES,
    sample_random_config,
    validate_config,
)
from src.surrogate import ConfigSurrogate
from src.multi_fidelity import CHEAP, MEDIUM, FULL
from src.nas_orchestrator import (
    _prep_from_cfg,
    _cfg_hash,
    _evaluate_config,
    _sel_score,
    _primary_score,
    _trial_record,
)
from src.ensemble import ensemble_top_k
from src.utils import ensure_dir, save_json, seed_everything


def run_random_nas_v3(
    *,
    openml_id: Optional[int] = None,
    task: str = "auto",
    out_dir: str,
    budget: int = 120,
    ensemble_k: int = 7,
    use_multi_fidelity: bool = True,
    seed: int = 42,
    device: Optional[str] = None,
    source: str = "openml",
    builtin_name: Optional[str] = None,
    csv_path: Optional[str] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run Random NAS v3 baseline — same evaluation as LLM-NAS v3, no LLM."""
    t_start = time.time()
    seed_everything(seed)
    rng = random.Random(seed)
    out_dir_p = ensure_dir(out_dir)
    trials_dir = ensure_dir(out_dir_p / "trials")

    # Load data
    print(f"[RandomNAS-v3] Loading dataset …")
    raw, summary = load_raw(
        source=source, openml_id=openml_id,
        builtin_name=builtin_name, csv_path=csv_path,
        task=task, seed=seed,
    )
    print(f"[RandomNAS-v3] {summary.get('n_rows')} rows  "
          f"{summary.get('n_num')} num  {summary.get('n_cat')} cat  "
          f"task={summary.get('task')}")

    # Stratified sampling with family quotas (same as LLM-NAS v3 warmup)
    _quota_weights = {"resmlp": 6, "tabm": 6, "mlp": 4, "ft_transformer": 4,
                      "gated_tab": 2, "autoint": 2}
    _total_weight = sum(_quota_weights.values())

    def _stratified_sample(n: int) -> List[str]:
        fam_list: List[str] = []
        for fam, w in _quota_weights.items():
            count = max(1, round(n * w / _total_weight))
            fam_list.extend([fam] * count)
        rng.shuffle(fam_list)
        fam_list = fam_list[:n]
        while len(fam_list) < n:
            fam_list.append(rng.choice(list(_quota_weights.keys())))
        return fam_list

    families = _stratified_sample(budget)
    family_dist: Dict[str, int] = {}
    for f in families:
        family_dist[f] = family_dist.get(f, 0) + 1
    print(f"[RandomNAS-v3] Family distribution: {family_dist}")

    all_trials: List[Dict] = []
    cheap_scores: List[float] = []
    medium_scores: List[float] = []
    cheap_scores_by_family: Dict[str, List[float]] = {}
    medium_scores_by_family: Dict[str, List[float]] = {}
    seen_hashes: set = set()

    for trial_id, fam in enumerate(families):
        cfg = validate_config(sample_random_config(rng, family=fam))
        h = _cfg_hash(cfg)
        # Allow up to 3 retries to avoid near-duplicate configs
        for _ in range(3):
            if h not in seen_hashes:
                break
            cfg = validate_config(sample_random_config(rng, family=fam))
            h = _cfg_hash(cfg)
        seen_hashes.add(h)

        print(f"  [RandomNAS-v3 {trial_id+1}/{budget}] family={fam}", flush=True)
        prepared = _prep_from_cfg(cfg, raw)

        try:
            result = _evaluate_config(
                cfg, prepared, trials_dir, trial_id,
                use_multi_fidelity=use_multi_fidelity,
                cheap_scores=cheap_scores,
                medium_scores=medium_scores,
                cheap_scores_by_family=cheap_scores_by_family,
                medium_scores_by_family=medium_scores_by_family,
                n_train_rows=int(summary.get("n_rows", 100_000)),
                seed=seed, device=device, verbose=verbose,
            )
        except Exception as e:
            print(f"    ERROR: {e}")
            continue

        rec = _trial_record(trial_id, cfg, result, op="random_v3")
        all_trials.append(rec)
        save_json(out_dir_p / "trials_index.json", {"trials": all_trials})

        if result.rung not in ("cheap_pruned", "medium_pruned", "error"):
            print(f"    primary={result.primary:.5f}  search_score={result.search_score:.5f}  "
                  f"rung={result.rung}")

    # Final selection — two separate bests
    full_trials = [t for t in all_trials
                   if t.get("rung") not in ("cheap_pruned", "medium_pruned", "error")]
    rank_pool = full_trials if full_trials else all_trials

    best_by_primary = max(rank_pool, key=_primary_score)
    best_by_search  = max(rank_pool, key=_sel_score)
    save_json(out_dir_p / "best_trial_by_primary.json", best_by_primary)
    save_json(out_dir_p / "best_trial_by_search.json",  best_by_search)
    save_json(out_dir_p / "best_trial.json",             best_by_primary)

    # Ensemble
    ensemble_result: Dict = {}
    if ensemble_k > 1 and len(full_trials) >= 2:
        print(f"\n[RandomNAS-v3] Running ensemble top-{ensemble_k} …")
        ensemble_result = ensemble_top_k(
            full_trials, raw,
            k=ensemble_k,
            method="greedy",
            seed=seed, device=device,
            out_dir=str(out_dir_p),
        )

    total_time = time.time() - t_start
    best_primary_val = best_by_primary.get("primary", 0.0)
    best_ensemble    = ensemble_result.get("primary")

    final = {
        "mode": "random_nas_v3",
        "seed": seed,
        "budget": budget,
        "n_trials": len(all_trials),
        "n_full_trials": len(full_trials),
        "best_primary": best_primary_val,
        "best_primary_trial_id": best_by_primary.get("trial_id"),
        "best_primary_family": best_by_primary.get("config", {}).get("arch", {}).get("family"),
        "best_search_score": best_by_search.get("search_score"),
        "best_ensemble": best_ensemble,
        "ensemble_k": ensemble_result.get("k"),
        "family_distribution": family_dist,
        "total_seconds": total_time,
        "dataset_summary": summary,
        "ensemble": ensemble_result,
    }
    save_json(out_dir_p / "final_report.json", final)

    print(f"\n[RandomNAS-v3] Done!  best_primary={best_primary_val:.6f}"
          + (f"  best_ensemble={best_ensemble:.6f}" if best_ensemble else "")
          + f"  total={total_time:.0f}s")
    return final


def main():
    p = argparse.ArgumentParser(description="Random NAS v3 baseline")
    p.add_argument("--openml_id", type=int, default=None)
    p.add_argument("--task", default="auto")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--budget", type=int, default=120)
    p.add_argument("--ensemble_k", type=int, default=7)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--source", default="openml")
    p.add_argument("--builtin_name", default=None)
    p.add_argument("--csv_path", default=None)
    p.add_argument("--no_multi_fidelity", action="store_true")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    run_random_nas_v3(
        openml_id=args.openml_id,
        task=args.task,
        out_dir=args.out_dir,
        budget=args.budget,
        ensemble_k=args.ensemble_k,
        use_multi_fidelity=not args.no_multi_fidelity,
        seed=args.seed,
        device=args.device,
        source=args.source,
        builtin_name=args.builtin_name,
        csv_path=args.csv_path,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
