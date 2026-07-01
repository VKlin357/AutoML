#!/usr/bin/env bash
# ============================================================================
# Прогнать CatBoost + LightGBM + Random NAS baselines на НОВЫХ датасетах:
#   - Connect-4 (OpenML 40668)
#   - Eye Movements (OpenML 1044)
#   - Covertype (builtin covtype)
#
# Эти 3 датасета мы добавили в v5 чтобы получить "красивую картину" — но
# у нас нет baselines.json для них, поэтому LLM-NAS не с чем сравнивать.
# Этот скрипт это исправляет.
#
# Также можно дописать --random_nas_budget=25 для apples-to-apples сравнения.
#
# Использование:
#   bash scripts/run_baselines_new_datasets.sh         # default: random_nas_budget=25
#   bash scripts/run_baselines_new_datasets.sh 30      # budget=30
# ============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."

BUDGET=${1:-25}

# Куда складывать. Если LLM-NAS уже запускалась в experiments_v5/single/<dataset>_seed42,
# baselines идут ТУДА ЖЕ — чтобы они оказались в одном subdir.
declare -a DATASETS=(
  "connect4:--openml_id 40668 --task multiclass"
  "eye_movements:--openml_id 1044 --task multiclass"
  "covertype:--builtin covtype --task multiclass"
)

for entry in "${DATASETS[@]}"; do
  IFS=':' read -r NAME ARGS <<<"${entry}"
  OUT_DIR="experiments_v5/baselines/${NAME}"
  mkdir -p "${OUT_DIR}"

  echo ""
  echo "=================================================================="
  echo "  Baselines for ${NAME}"
  echo "  ${ARGS}"
  echo "  Output: ${OUT_DIR}   random_nas_budget=${BUDGET}"
  echo "=================================================================="

  python3 scripts/run_baselines.py \
    ${ARGS} \
    --out_dir "${OUT_DIR}" \
    --random_nas \
    --random_nas_budget ${BUDGET} \
    --seed 42 \
    || { echo "  [WARN] baselines for ${NAME} failed"; continue; }

  echo ""
  echo "  --- Quick summary for ${NAME} ---"
  python3 << PYEOF
import json
from pathlib import Path
p = Path("${OUT_DIR}/baselines.json")
if p.exists():
    bl = json.load(open(p))
    for k, v in bl.items():
        if isinstance(v, dict):
            pri = v.get("primary")
            if pri is not None:
                print(f"    {k:<12} primary = {pri:.5f}")
PYEOF
done

echo ""
echo "=================================================================="
echo "  ALL DONE. Now compare with LLM-NAS results:"
echo ""
echo "  python3 -c '"
echo "import json"
echo "from pathlib import Path"
echo "for ds in [\"connect4\", \"eye_movements\", \"covertype\"]:"
echo "    bl = json.load(open(f\"experiments_v5/baselines/{ds}/baselines.json\"))"
echo "    bt = json.load(open(f\"experiments_v5/single/{ds}_seed42/best_trial.json\"))"
echo "    print(f\"{ds}: CatBoost={bl[\\\"catboost\\\"][\\\"primary\\\"]:.4f}  Random={bl[\\\"random_nas\\\"][\\\"primary\\\"]:.4f}  LLM={bt[\\\"primary\\\"]:.4f}\")"
echo "  '"
echo "=================================================================="
