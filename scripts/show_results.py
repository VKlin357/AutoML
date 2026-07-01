"""
show_results.py — читает эксперименты и выводит таблицу результатов.

Запуск:
  python scripts/show_results.py
  python scripts/show_results.py --exp_dir experiments
"""
import json
import sys
import argparse
from pathlib import Path


def load_json_safe(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None


def fmt(v, digits=5):
    if v is None:
        return "  ---  "
    return f"{v:.{digits}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_dir", default="experiments")
    args = ap.parse_args()
    exp = Path(args.exp_dir)

    print()
    print("=" * 120)
    print("РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТОВ")
    print("=" * 120)

    # Main results
    main_summary = load_json_safe(exp / "summary_main.json")
    if main_summary:
        print("\n📊 ОСНОВНАЯ ТАБЛИЦА (hard datasets)")
        print("-" * 120)
        header = f"{'Dataset':<16} {'CatBoost':>10} {'LightGBM':>10} {'RandomNAS':>10} "
        header += f"{'Optuna':>10} {'NaiveLLM':>10} {'LLM-NAS':>10} {'Ensemble':>10} {'BestArch':<15}"
        print(header)
        print("-" * 120)
        for r in main_summary:
            row = (
                f"{r['name']:<16} {fmt(r.get('catboost')):>10} {fmt(r.get('lightgbm')):>10} "
                f"{fmt(r.get('random_nas')):>10} {fmt(r.get('optuna')):>10} "
                f"{fmt(r.get('naive_llm')):>10} {fmt(r.get('llm_nas_v2')):>10} "
                f"{fmt(r.get('llm_nas_ensemble')):>10} {r.get('llm_nas_best_arch','?'):<15}"
            )
            print(row)
        print("-" * 120)

    # Ablation: no curves
    abl_no_curves = load_json_safe(exp / "summary_ablation_no_curves.json")
    if abl_no_curves and main_summary:
        print("\n📊 ABLATION: curve-aware (ваша система) vs curve-blind (как GENIUS/EvoPrompting)")
        print("   Датасет: MiniBooNE (168338)")
        print("-" * 80)
        # Find MiniBooNE in main
        main_mbn = next((r for r in main_summary if r.get("openml_id") == 168338
                         or "MiniBooNE" in r.get("name", "")), None)
        abl_mbn = abl_no_curves[0] if abl_no_curves else None
        if main_mbn:
            print(f"  {'Метод':<40} {'LLM-NAS':>10} {'Ensemble':>10}")
            print(f"  {'-'*62}")
            print(f"  {'LLM-NAS с кривыми (REFINE, proposed)':<40} "
                  f"{fmt(main_mbn.get('llm_nas_v2')):>10} {fmt(main_mbn.get('llm_nas_ensemble')):>10}")
        if abl_mbn:
            print(f"  {'LLM-NAS без кривых (curve-blind ablation)':<40} "
                  f"{fmt(abl_mbn.get('llm_nas_v2')):>10} {fmt(abl_mbn.get('llm_nas_ensemble')):>10}")
        print("-" * 80)

    # Ablation: propose vs refine
    abl_propose = load_json_safe(exp / "summary_ablation_propose.json")
    if abl_propose and main_summary:
        print("\n📊 ABLATION: REFINE (фиксированный словарь) vs PROPOSE (свободная генерация)")
        print("-" * 80)
        main_mbn = next((r for r in main_summary if r.get("openml_id") == 168338
                         or "MiniBooNE" in r.get("name", "")), None)
        abl_mbn = abl_propose[0] if abl_propose else None
        if main_mbn:
            print(f"  {'Метод':<40} {'LLM-NAS':>10} {'Ensemble':>10}")
            print(f"  {'-'*62}")
            print(f"  {'LLM-NAS REFINE (словарь действий)':<40} "
                  f"{fmt(main_mbn.get('llm_nas_v2')):>10} {fmt(main_mbn.get('llm_nas_ensemble')):>10}")
        if abl_mbn:
            print(f"  {'LLM-NAS PROPOSE (как EvoPrompting)':<40} "
                  f"{fmt(abl_mbn.get('llm_nas_v2')):>10} {fmt(abl_mbn.get('llm_nas_ensemble')):>10}")
        print("-" * 80)

    # Per-dataset detailed breakdown
    datasets_found = [d for d in exp.iterdir() if d.is_dir()
                      and (d / "baseline_catboost.json").exists()]
    if datasets_found:
        print("\n📁 Подробные результаты по датасетам:")
        for ds_dir in sorted(datasets_found):
            print(f"\n  {ds_dir.name}:")
            catboost = load_json_safe(ds_dir / "baseline_catboost.json")
            lgbm = load_json_safe(ds_dir / "baseline_lightgbm.json")
            random_nas = load_json_safe(ds_dir / "baseline_random_search.json")
            optuna = load_json_safe(ds_dir / "baseline_optuna.json")
            naive = load_json_safe(ds_dir / "baseline_naive_llm.json")
            best_trial = load_json_safe(ds_dir / "nas_v2" / "best_trial.json")
            ensemble = load_json_safe(ds_dir / "nas_v2" / "ensemble_result.json")
            action_stats = load_json_safe(ds_dir / "nas_v2" / "action_stats.json")

            if catboost:
                print(f"    CatBoost:  {catboost.get('primary', '?'):.5f}")
            if lgbm:
                print(f"    LightGBM:  {lgbm.get('primary', '?'):.5f}")
            if random_nas:
                print(f"    RandomNAS: {random_nas.get('primary', '?'):.5f}")
            if optuna:
                print(f"    Optuna:    {optuna.get('primary', '?'):.5f}")
            if naive:
                print(f"    NaiveLLM:  {naive.get('primary', '?'):.5f}")
            if best_trial:
                arch = best_trial.get("config", {}).get("arch", {}).get("family", "?")
                print(f"    LLM-NAS:   {best_trial.get('primary', '?'):.5f}  "
                      f"(arch={arch})")
            if ensemble:
                print(f"    Ensemble:  {ensemble.get('primary', '?'):.5f}")

            # Top REFINE actions
            if action_stats:
                called = [(a, action_stats[a]["count"]) for a in action_stats
                          if action_stats[a].get("count", 0) > 0]
                called.sort(key=lambda x: -x[1])
                if called:
                    top5 = ", ".join(f"{a}({c})" for a, c in called[:5])
                    print(f"    Top REFINE actions: {top5}")

    if not main_summary and not datasets_found:
        print()
        print("Результаты ещё не найдены.")
        print(f"Запусти: bash scripts/run_thesis_experiments.sh sk-proj-...")
        print(f"Или для быстрого теста: python scripts/run_experiment.py \\")
        print(f"    --datasets 168338 --budget 15 --coldstart_n 5 \\")
        print(f"    --random_nas_budget 15 --optuna --optuna_trials 15 \\")
        print(f"    --naive_llm --naive_llm_trials 3 --ensemble_k 3 \\")
        print(f"    --llm_model gpt-4o-mini --api_key sk-proj-...")

    print()


if __name__ == "__main__":
    main()
