#!/usr/bin/env bash
# ============================================================================
# v5 full multi-seed run with Fix #10 (relaxed LR-guard) + Fix #11 (family-aware
# cold-start). Includes 2 NEW DL-friendly datasets (Covertype, Connect-4) chosen
# to mirror helena's winning conditions: medium-large, multi-class, structured
# features where DL has a real chance against GBM.
#
# Run on server:
#   cd /workspace/llm-tabular-nas-proxy
#   bash scripts/run_v5_full.sh
# ============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."

if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "ERROR: OPENAI_API_KEY not set"
  exit 1
fi

SEEDS=(42 1 2026)
BUDGET=25

# Original 4 OpenML datasets (drop HAR — saturated 99%+, no signal possible)
OPENML_DATASETS=(
  "jannis:45021:multiclass"
  "helena:41166:multiclass"
  "pol:44156:binary"
  # New DL-friendly additions (mirror helena's success conditions):
  "connect4:40668:multiclass"      # 67k × 42, 3-class, categorical-heavy
  # Eye Movements is small but DL-friendly — quick datapoint:
  "eye_movements:1044:multiclass"  # 10k × 27, 3-class
)

run_one() {
  local name=$1
  local seed=$2
  local out="experiments_v5/seed${seed}/${name}"
  mkdir -p "${out}"

  shift 2
  echo ""
  echo "=================================================================="
  echo "  ${name}  seed=${seed}  →  ${out}"
  echo "=================================================================="

  python scripts/run_nas_v2.py \
    --out_dir "${out}" \
    --budget ${BUDGET} \
    --coldstart_n 5 \
    --seed ${seed} \
    --llm_model gpt-4o-mini \
    --reflect_every 8 \
    --explore_pulse 6 \
    "$@" || echo "  [WARN] ${name} seed=${seed} failed; continuing"
}

for SEED in "${SEEDS[@]}"; do
  echo ""
  echo "############################################"
  echo "  SEED ${SEED}"
  echo "############################################"

  for entry in "${OPENML_DATASETS[@]}"; do
    IFS=':' read -r name oid task <<<"${entry}"
    run_one "${name}" "${SEED}" --openml_id "${oid}" --task "${task}"
  done

  # Built-in dataset: Covertype (multiclass 7, 581k rows, NN-friendly)
  run_one "covertype" "${SEED}" --builtin covtype --task multiclass

  # MiniBooNE (already in pipeline)
  run_one "miniboone" "${SEED}" --builtin miniboonee --task binary
done

echo ""
echo "=================================================================="
echo "  v5 run complete."
echo "=================================================================="

python3 scripts/aggregate_v3.py 2>/dev/null || echo "(aggregate script optional)"
