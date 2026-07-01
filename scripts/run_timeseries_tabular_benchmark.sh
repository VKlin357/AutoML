#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "ERROR: set OPENAI_API_KEY before running LLM-NAS experiments"
  exit 1
fi

mkdir -p logs experiments

BUDGET="${BUDGET:-40}"
WARMUP="${WARMUP:-12}"
BATCH_N="${BATCH_N:-20}"
BATCH_K="${BATCH_K:-4}"
OPTUNA_TRIALS="${OPTUNA_TRIALS:-40}"
SEED="${SEED:-42}"
LLM_MODEL="${LLM_MODEL:-gpt-4o-mini}"

run_pair() {
  local name="$1"
  local builtin="$2"
  local task="$3"

  echo "=== ${name}: LLM-NAS (${task}) ==="
  python src/run_nas_v2.py \
    --builtin "${builtin}" \
    --task "${task}" \
    --out_dir "experiments/${name}_llm_s${SEED}" \
    --budget "${BUDGET}" \
    --random_warmup "${WARMUP}" \
    --mode batch \
    --batch_n "${BATCH_N}" \
    --batch_k "${BATCH_K}" \
    --ensemble_k 7 \
    --seed "${SEED}" \
    --llm_model "${LLM_MODEL}" \
    --require_llm \
    2>&1 | tee "logs/${name}_llm.log"

  echo "=== ${name}: baselines (${task}) ==="
  python src/run_baselines.py \
    --builtin "${builtin}" \
    --task "${task}" \
    --out_dir "experiments/${name}_baselines" \
    --random_nas \
    --random_nas_budget "${BUDGET}" \
    --optuna \
    --optuna_trials "${OPTUNA_TRIALS}" \
    --seed "${SEED}" \
    2>&1 | tee "logs/${name}_baselines.log"
}

# Large raw sensor-stream benchmarks. Windows are built after participant holdout.
run_pair "harth_subject_holdout" "harth" "multiclass"
run_pair "pamap2_subject_holdout" "pamap2" "multiclass"

echo "Finished time-series benchmark."
