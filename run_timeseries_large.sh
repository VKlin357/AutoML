#!/usr/bin/env bash
# =============================================================================
#  LLM-NAS vs Baselines — Large Time-Series Experiment
#
#  Датасеты:
#    harth        ~202K окон × 768 признаков  (6 каналов × 128 шагов, 50 Гц)
#    pamap2       ~241K окон × 1152 признаков (9 каналов × 128 шагов, 100 Гц)
#    emg_gestures  ~60K окон × 64  признаков  (8 sEMG электродов × 8 шагов)
#
#  Почему CatBoost хуже:
#    — 768/1152 коррелированных признаков → деревья делают независимые сплиты
#    — FT-Transformer / AutoInt учит cross-channel attention
#    — Временна́я структура внутри окна сохранена в порядке столбцов
#
#  Использование:
#    export OPENAI_API_KEY="sk-proj-..."
#    bash run_timeseries_large.sh
# =============================================================================

set -euo pipefail

ROOT="/root/llm-nas"
EXP="$ROOT/experiments"
LOGS="$ROOT/logs"
MODEL="gpt-4o-mini"
SEED=42
BUDGET=40
WARMUP=10

mkdir -p "$LOGS" "$EXP"
cd "$ROOT"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "ERROR: export OPENAI_API_KEY='sk-proj-...'"
  exit 1
fi

echo "============================================================"
echo "  LLM-NAS Large Time-Series Experiment"
echo "  Model=$MODEL  Budget=$BUDGET  Seed=$SEED"
echo "============================================================"

# ─────────────────────────────────────────────────────────────────────────────
# 1. HARTH — 202K окон × 768 признаков, 12 классов активности
#    Источник: UCI, 22 участника, subject-wise holdout (нет утечки)
#    Акселерометр спина+бедро, 50 Гц, окна 128 шагов, stride=32
# ─────────────────────────────────────────────────────────────────────────────
echo "[1/6] HARTH — LLM-NAS"
nohup python src/run_nas_v2.py \
  --builtin harth --task multiclass \
  --out_dir "$EXP/harth_llm_s42" \
  --budget $BUDGET --random_warmup $WARMUP \
  --mode batch --batch_n 20 --batch_k 4 \
  --seed $SEED --llm_model $MODEL \
  > "$LOGS/harth_llm.log" 2>&1 &
echo "  PID=$!  log: $LOGS/harth_llm.log"

echo "[2/6] HARTH — Baselines (CatBoost + LightGBM + Optuna)"
nohup python src/run_baselines.py \
  --builtin harth --task multiclass \
  --out_dir "$EXP/harth_baselines" \
  --optuna --optuna_trials $BUDGET \
  --seed $SEED \
  > "$LOGS/harth_baselines.log" 2>&1 &
echo "  PID=$!  log: $LOGS/harth_baselines.log"

sleep 5

# ─────────────────────────────────────────────────────────────────────────────
# 2. PAMAP2 — 241K окон × 1152 признаков, 12 классов активности
#    Источник: UCI, 9 участников, 3 IMU датчика (рука/грудь/лодыжка)
#    100 Гц, окна 128 шагов, stride=16
# ─────────────────────────────────────────────────────────────────────────────
echo "[3/6] PAMAP2 — LLM-NAS"
nohup python src/run_nas_v2.py \
  --builtin pamap2 --task multiclass \
  --out_dir "$EXP/pamap2_llm_s42" \
  --budget $BUDGET --random_warmup $WARMUP \
  --mode batch --batch_n 20 --batch_k 4 \
  --seed $SEED --llm_model $MODEL \
  > "$LOGS/pamap2_llm.log" 2>&1 &
echo "  PID=$!  log: $LOGS/pamap2_llm.log"

echo "[4/6] PAMAP2 — Baselines"
nohup python src/run_baselines.py \
  --builtin pamap2 --task multiclass \
  --out_dir "$EXP/pamap2_baselines" \
  --optuna --optuna_trials $BUDGET \
  --seed $SEED \
  > "$LOGS/pamap2_baselines.log" 2>&1 &
echo "  PID=$!  log: $LOGS/pamap2_baselines.log"

sleep 5

# ─────────────────────────────────────────────────────────────────────────────
# 3. EMG Gestures — 60K окон × 64 признака, 5 классов жестов
#    Источник: UCI, 8 sEMG электродов × 8 временны́х точек
# ─────────────────────────────────────────────────────────────────────────────
echo "[5/6] EMG Gestures — LLM-NAS"
nohup python src/run_nas_v2.py \
  --builtin emg_gestures --task multiclass \
  --out_dir "$EXP/emg_llm_s42" \
  --budget $BUDGET --random_warmup $WARMUP \
  --mode batch --batch_n 20 --batch_k 4 \
  --seed $SEED --llm_model $MODEL \
  > "$LOGS/emg_llm.log" 2>&1 &
echo "  PID=$!  log: $LOGS/emg_llm.log"

echo "[6/6] EMG Gestures — Baselines"
nohup python src/run_baselines.py \
  --builtin emg_gestures --task multiclass \
  --out_dir "$EXP/emg_baselines" \
  --optuna --optuna_trials $BUDGET \
  --seed $SEED \
  > "$LOGS/emg_baselines.log" 2>&1 &
echo "  PID=$!  log: $LOGS/emg_baselines.log"

echo ""
echo "============================================================"
echo "  Все 6 задач запущены в фоне."
echo ""
echo "  Смотреть прогресс:"
echo "    watch -n 60 'for f in $LOGS/*.log; do"
echo "      echo \"=== \$(basename \$f) ===\"; tail -2 \$f; done'"
echo ""
echo "  Скачать результаты на мак:"
echo "    rsync -avz -e 'ssh -p 13105' \\"
echo "      root@31.13.223.140:/root/llm-nas/experiments/ \\"
echo "      /Users/vadim/Diplom/llm-tabular-nas-proxy/experiments_v9/"
echo "============================================================"
