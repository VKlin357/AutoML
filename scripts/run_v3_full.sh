#!/usr/bin/env bash
# ============================================================================
# v3 full run — 3 seeds × 7 datasets = 21 LLM-NAS runs.
# Total est. wall-clock: ~12-24 hours on a single GPU (shorter with parallelism).
#
# Run only AFTER scripts/run_v3_sanity.sh shows ✅ verdict.
# ============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."

if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "ERROR: OPENAI_API_KEY not set"
  exit 1
fi

SEEDS=(42 1 2026)
BUDGET=25

# Datasets via OpenML — these definitely work on the server
OPENML_DATASETS=(
  "jannis:45021:multiclass"
  "helena:41166:multiclass"
  "pol:44156:binary"
)

# Datasets via --builtin or --csv — paths to be set on server
# Edit these paths if your CSVs live elsewhere
HAR_CSV="${HAR_CSV:-data/har.csv}"
ECG_CSV="${ECG_CSV:-data/ecg5000.csv}"
ELEC_CSV="${ELEC_CSV:-data/elec2.csv}"

run_one() {
  local name=$1
  local seed=$2
  local out="experiments_v3/seed${seed}/${name}"
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

  # OpenML datasets
  for entry in "${OPENML_DATASETS[@]}"; do
    IFS=':' read -r name oid task <<<"${entry}"
    run_one "${name}" "${SEED}" --openml_id "${oid}" --task "${task}"
  done

  # Built-in dataset
  run_one "miniboone" "${SEED}" --builtin miniboonee --task binary

  # CSV datasets — uncomment if you have the CSVs:
  # run_one "HAR"     "${SEED}" --csv "${HAR_CSV}"  --task multiclass
  # run_one "ECG5000" "${SEED}" --csv "${ECG_CSV}"  --task multiclass
  # run_one "ELEC2"   "${SEED}" --csv "${ELEC_CSV}" --task binary
done

echo ""
echo "=================================================================="
echo "  Full run complete. Aggregating results..."
echo "=================================================================="

python3 scripts/aggregate_v3.py
