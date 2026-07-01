#!/usr/bin/env bash
# ============================================================================
# v4 Ансамбль — запускается ПОСЛЕ run_v4_nas_only.sh
# Читает trials_index.json из каждого датасета и строит ансамбль top-5.
#
# Использование:
#   bash scripts/run_v4_ensemble.sh
# ============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."

SEED=42
OUT_BASE="experiments_v4"

HAR_CSV="${HAR_CSV:-data/har.csv}"
ECG_CSV="${ECG_CSV:-data/ecg5000.csv}"
ELEC_CSV="${ELEC_CSV:-data/elec2.csv}"

ensemble_one() {
  local name=$1
  local out="${OUT_BASE}/${name}"
  shift 1

  if [ ! -f "${out}/trials_index.json" ]; then
    echo "  [SKIP] ${name} — trials_index.json не найден в ${out}"
    return
  fi

  echo ""
  echo "======================================================"
  echo "  Ансамбль: ${name}  →  ${out}"
  echo "======================================================"

  python3 scripts/run_nas_v2.py \
    --out_dir "${out}" \
    --budget 0 \
    --ensemble_k 5 \
    --seed ${SEED} \
    --ensemble_only \
    "$@" || echo "  [WARN] ${name} ансамбль завершился с ошибкой"
}

echo ""
echo "======================================================"
echo "  v4 Ансамбль top-5 | seed=${SEED}"
echo "======================================================"

ensemble_one "jannis"    --openml_id 45021 --task multiclass
ensemble_one "helena"    --openml_id 41166 --task multiclass
ensemble_one "pol"       --openml_id 44156 --task binary
ensemble_one "miniboone" --builtin miniboonee --task binary

# ensemble_one "HAR"     --csv "${HAR_CSV}"  --task multiclass
# ensemble_one "ECG5000" --csv "${ECG_CSV}"  --task multiclass
# ensemble_one "ELEC2"   --csv "${ELEC_CSV}" --task binary

echo ""
echo "======================================================"
echo "  Все ансамбли готовы."
echo "======================================================"
