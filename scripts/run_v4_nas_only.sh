#!/usr/bin/env bash
# ============================================================================
# v4 LLM-NAS — только поиск, БЕЗ ансамбля (--ensemble_k 0)
# Один seed, все датасеты по очереди.
# После завершения запусти run_v4_ensemble.sh чтобы посчитать ансамбли.
#
# Использование:
#   bash scripts/run_v4_nas_only.sh
# ============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."

if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "ERROR: OPENAI_API_KEY не задан"
  exit 1
fi

SEED=42
BUDGET=25
MODEL="gpt-4o"
OUT_BASE="experiments_v4"

# CSV-датасеты (подставь реальные пути если нужно)
HAR_CSV="${HAR_CSV:-data/har.csv}"
ECG_CSV="${ECG_CSV:-data/ecg5000.csv}"
ELEC_CSV="${ELEC_CSV:-data/elec2.csv}"

run_one() {
  local name=$1
  local out="${OUT_BASE}/${name}"
  mkdir -p "${out}"
  shift 1

  echo ""
  echo "======================================================"
  echo "  ${name}  seed=${SEED}  →  ${out}"
  echo "======================================================"

  python3 scripts/run_nas_v2.py \
    --out_dir "${out}" \
    --budget ${BUDGET} \
    --coldstart_n 5 \
    --seed ${SEED} \
    --llm_model ${MODEL} \
    --reflect_every 8 \
    --explore_pulse 6 \
    --ensemble_k 0 \
    "$@" || echo "  [WARN] ${name} завершился с ошибкой, продолжаем"
}

echo ""
echo "======================================================"
echo "  v4 LLM-NAS без ансамбля | model=${MODEL} | seed=${SEED}"
echo "======================================================"

# --- OpenML датасеты ---
run_one "jannis"  --openml_id 45021 --task multiclass
run_one "helena"  --openml_id 41166 --task multiclass
run_one "pol"     --openml_id 44156 --task binary

# --- Builtin ---
run_one "miniboone" --builtin miniboonee --task binary

# --- CSV датасеты (раскомментируй если CSV есть на сервере) ---
# run_one "HAR"     --csv "${HAR_CSV}"  --task multiclass
# run_one "ECG5000" --csv "${ECG_CSV}"  --task multiclass
# run_one "ELEC2"   --csv "${ELEC_CSV}" --task binary

echo ""
echo "======================================================"
echo "  Поиск завершён. Запусти ансамбль:"
echo "    bash scripts/run_v4_ensemble.sh"
echo "======================================================"
