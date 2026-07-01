#!/usr/bin/env bash
# ============================================================================
# Сравнение 3 режимов работы LLM-NAS на ОДИН датасет с ОДНИМ seed.
# Этот ablation — main artifact for thesis: показывает, какой режим лучше.
#
# Использование (на сервере):
#   bash scripts/run_compare_modes.sh <DATASET_NAME> [SEED] [BUDGET]
#
# Пример:
#   bash scripts/run_compare_modes.sh helena         # все 3 mode на helena
#   bash scripts/run_compare_modes.sh pol 42 25      # pol, seed=42, budget=25
#
# Прогоняет:
#   experiments_v6/<dataset>_seed<seed>/refine          ← Mode 1: original
#   experiments_v6/<dataset>_seed<seed>/refine_extended ← Mode 2: +new actions+anti-osc
#   experiments_v6/<dataset>_seed<seed>/freeform        ← Mode 3: changes dict
#
# В конце автоматически распечатывает сравнительную таблицу.
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

DATASET=${1:?Usage: $0 <DATASET> [SEED=42] [BUDGET=25]}
SEED=${2:-42}
BUDGET=${3:-25}

case "${DATASET}" in
  helena)        ARGS="--openml_id 41166 --task multiclass" ;;
  pol)           ARGS="--openml_id 44156 --task binary" ;;
  jannis)        ARGS="--openml_id 45021 --task multiclass" ;;
  connect4)      ARGS="--openml_id 40668 --task multiclass" ;;
  eye_movements) ARGS="--openml_id 1044 --task multiclass" ;;
  covertype)     ARGS="--builtin covtype --task multiclass" ;;
  miniboone)     ARGS="--builtin miniboonee --task binary" ;;
  *) echo "Unknown dataset: ${DATASET}"; exit 1 ;;
esac

BASE_OUT="experiments_v6/${DATASET}_seed${SEED}"
mkdir -p "${BASE_OUT}"

for MODE in refine refine_extended freeform; do
  OUT="${BASE_OUT}/${MODE}"
  mkdir -p "${OUT}"
  echo ""
  echo "=================================================================="
  echo "  Dataset=${DATASET}  Seed=${SEED}  Mode=${MODE}  Budget=${BUDGET}"
  echo "  → ${OUT}"
  echo "=================================================================="

  python3 scripts/run_nas_v2.py \
    ${ARGS} \
    --out_dir "${OUT}" \
    --budget ${BUDGET} \
    --coldstart_n 5 \
    --seed ${SEED} \
    --llm_model gpt-4o-mini \
    --reflect_every 8 \
    --explore_pulse 6 \
    --mode "${MODE}" \
    || echo "  [WARN] ${MODE} failed for ${DATASET} seed=${SEED}"
done

# Comparative table
echo ""
echo "=================================================================="
echo "  COMPARISON TABLE  (${DATASET} seed=${SEED})"
echo "=================================================================="
python3 << PYEOF
import json
from pathlib import Path
from collections import Counter

BASE = Path("${BASE_OUT}")

# Try to load Random NAS baseline if available
rand_bl = None
for path in [
    Path("experiments_v5/baselines/${DATASET}/baselines.json"),
    Path("experiments_v4/${DATASET}/baselines.json"),
    BASE.parent / "baselines.json",
]:
    if path.exists():
        try:
            rand_bl = json.load(open(path)).get("random_nas", {}).get("primary")
            cb_bl = json.load(open(path)).get("catboost", {}).get("primary")
            print(f"  Reference baselines:  CatBoost={cb_bl:.5f}  RandomNAS={rand_bl:.5f}")
            break
        except: pass

print()
print(f"  {'Mode':<22} {'best':>10} {'ensemble':>10} {'real_LLM':>10} {'fallback':>10} {'forced':>8} {'best_op':<35} {'family':<15}")
print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*35} {'-'*15}")

results = []
for mode in ["refine", "refine_extended", "freeform"]:
    mdir = BASE / mode
    bt_path = mdir / "best_trial.json"
    if not bt_path.exists():
        print(f"  {mode:<22} (no output)")
        continue
    bt = json.load(open(bt_path))
    ens_p = mdir / "ensemble_result.json"
    ens = json.load(open(ens_p))["primary"] if ens_p.exists() else None

    idx = json.load(open(mdir / "trials_index.json"))
    trials = idx if isinstance(idx, list) else idx.get("trials", [])
    ops = Counter(t.get("op","") for t in trials)
    real_llm = sum(v for k,v in ops.items() if (
        k.startswith("refine:") or k.startswith("freeform:")
    ) and k not in ("refine:None", "freeform_dedup_fallback", "freeform_empty_fallback", "freeform_error_fallback"))
    fb = ops.get("random_refine_fallback",0) + ops.get("refine:None",0) + \
         ops.get("freeform_dedup_fallback",0) + ops.get("freeform_empty_fallback",0) + ops.get("freeform_error_fallback",0)
    forced = sum(v for k,v in ops.items() if k.startswith("forced:"))

    best_p = bt.get("primary", 0)
    ens_str = f"{ens:.5f}" if ens else "-"
    best_op = bt.get("op","?")[:33]
    best_fam = bt['config']['arch']['family']
    print(f"  {mode:<22} {best_p:>10.5f} {ens_str:>10} {real_llm:>10} {fb:>10} {forced:>8} {best_op:<35} {best_fam:<15}")
    results.append((mode, best_p, ens))

# Winner
if results:
    print()
    winner = max(results, key=lambda r: r[1])
    print(f"  🏆 WINNER (best single): {winner[0]}  primary={winner[1]:.5f}")
    if rand_bl:
        diff_to_rand = winner[1] - rand_bl
        print(f"     Δ vs Random NAS: {diff_to_rand:+.5f}  ({'BEATS' if diff_to_rand>0.005 else 'matches/loses'})")
PYEOF
