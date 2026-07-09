#!/usr/bin/env bash
# ============================================================================
# Optuna NAS baseline for all forecasting datasets.
#
# Usage (on GPU server):
#   cd ~/llm-tabular-nas-proxy
#   bash scripts/run_forecasting_optuna.sh 2>&1 | tee logs/forecasting_optuna.log
#
# Optional env vars:
#   DEVICE=cuda   (default: auto-detect)
#   N_TRIALS=40   (default: 40)
#   SEED=42       (default: 42)
# ============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs

DEVICE="${DEVICE:-cuda}"
N_TRIALS="${N_TRIALS:-40}"
SEED="${SEED:-42}"

echo "======================================"
echo "FORECASTING OPTUNA START: $(date)"
echo "N_TRIALS=$N_TRIALS  SEED=$SEED  DEVICE=$DEVICE"
echo "======================================"

PYTHON="${PYTHON:-$(which python3)}"
"$PYTHON" scripts/run_forecasting_optuna.py \
    --all \
    --n_trials "$N_TRIALS" \
    --seed "$SEED" \
    --device "$DEVICE" \
    --timeout 7200

echo ""
echo "======================================"
echo "FORECASTING OPTUNA DONE: $(date)"
echo "======================================"
