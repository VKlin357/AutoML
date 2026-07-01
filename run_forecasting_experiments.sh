#!/usr/bin/env bash
# =============================================================================
#  LLM-NAS vs Baselines — Forecasting Experiments
#
#  Датасеты: ETTh1, ETTh2, Weather
#  Задача: одношаговая регрессия (lookback=96, horizon=1)
#
#  Использование:
#    export OPENAI_API_KEY="sk-proj-..."
#    bash run_forecasting_experiments.sh
# =============================================================================

set -euo pipefail

ROOT="/root/llm-nas"
EXP="$ROOT/experiments/forecasting"
LOGS="$ROOT/logs/forecasting"
MODEL="gpt-4o"
SEED=42
BUDGET=30
WARMUP=10
LOOKBACK=96
HORIZON=1

mkdir -p "$LOGS" "$EXP"
cd "$ROOT"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "ERROR: export OPENAI_API_KEY='sk-proj-...'"
    exit 1
fi

echo "============================================================"
echo "  Forecasting NAS Experiments"
echo "  Model=$MODEL  Budget=$BUDGET  Lookback=$LOOKBACK  Horizon=$HORIZON"
echo "============================================================"

run_dataset() {
    local ds=$1
    local idx=$2
    local total=$3

    echo ""
    echo "[$idx/$total] $ds"

    nohup python src/run_forecasting_nas.py \
        --dataset "$ds" \
        --out_dir "$EXP/$ds" \
        --lookback $LOOKBACK --horizon $HORIZON \
        --budget $BUDGET --warmup $WARMUP \
        --batch_n 10 --batch_k 3 \
        --seed $SEED --llm_model $MODEL \
        > "$LOGS/${ds}.log" 2>&1 &
    echo "  PID=$!  log: $LOGS/${ds}.log"
}

run_dataset etth1   1 3
sleep 2
run_dataset etth2   2 3
sleep 2
run_dataset weather 3 3

echo ""
echo "============================================================"
echo "  Все 3 эксперимента запущены параллельно."
echo ""
echo "  Следить:"
echo "    tail -f $LOGS/etth1.log"
echo "    tail -f $LOGS/etth2.log"
echo "    tail -f $LOGS/weather.log"
echo ""
echo "  Скачать результаты:"
echo "    rsync -avz -e 'ssh -p ПОРТ' \\"
echo "      root@IP:/root/llm-nas/experiments/forecasting/ \\"
echo "      /Users/vadim/Diplom/llm-tabular-nas-proxy/experiments_forecasting/ \\"
echo "      --exclude='*.pt' --exclude='tmp_forecasting_csv'"
echo "============================================================"
