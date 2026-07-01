#!/usr/bin/env bash
# =============================================================================
#  LLM-NAS vs CatBoost/LightGBM — Time-Series Tabular Experiment
#
#  Датасеты (все — time-series природы):
#    ecg5000      — 5K  × 140 шагов,  5-class,   сегменты ЭКГ
#    emg_gestures — 60K × 64  шагов,  5-class,   sEMG жесты рук
#    elec2        — 45K × 8   признаков, binary,  электричество (temporal split)
#    harth        — ~100K окон × 6 каналов, multiclass, акселерометр
#
#  HAR уже есть: LLM-NAS 0.9909 > CatBoost 0.9897 ✓
#
#  Использование:
#    export OPENAI_API_KEY="sk-proj-..."
#    bash run_timeseries_exp.sh
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
echo "  LLM-NAS Time-Series Experiment"
echo "  Model=$MODEL  Budget=$BUDGET  Seed=$SEED"
echo "  Запускаем 6 фоновых задач..."
echo "============================================================"

# ─────────────────────────────────────────────────────────────────────────────
# 1. ECG5000 — каждая строка = сегмент ЭКГ из 140 временны́х шагов
#    NN учит форму QRS-комплекса; дерево видит просто 140 числовых столбцов
# ─────────────────────────────────────────────────────────────────────────────
echo "[1/6] ECG5000 — LLM-NAS (batch mode)"
nohup python src/run_nas_v2.py \
  --builtin ecg5000 --task multiclass \
  --out_dir "$EXP/ecg5000_llm_s42" \
  --budget $BUDGET --random_warmup $WARMUP \
  --mode batch --batch_n 20 --batch_k 4 \
  --seed $SEED --llm_model $MODEL \
  > "$LOGS/ecg5000_llm.log" 2>&1 &
echo "  PID=$!"

echo "[2/6] ECG5000 — Baselines (CatBoost + LightGBM + Optuna)"
nohup python src/run_baselines.py \
  --builtin ecg5000 --task multiclass \
  --out_dir "$EXP/ecg5000_baselines" \
  --optuna --optuna_trials $BUDGET \
  --seed $SEED \
  > "$LOGS/ecg5000_baselines.log" 2>&1 &
echo "  PID=$!"

sleep 3

# ─────────────────────────────────────────────────────────────────────────────
# 2. EMG Gestures — 8 ЭМГ электрода × 8 временны́х точек = 64 признака/строка
#    Высококоррелированные каналы → attention-архитектуры (FT-Transformer, AutoInt)
# ─────────────────────────────────────────────────────────────────────────────
echo "[3/6] EMG Gestures — LLM-NAS"
nohup python src/run_nas_v2.py \
  --builtin emg_gestures --task multiclass \
  --out_dir "$EXP/emg_llm_s42" \
  --budget $BUDGET --random_warmup $WARMUP \
  --mode batch --batch_n 20 --batch_k 4 \
  --seed $SEED --llm_model $MODEL \
  > "$LOGS/emg_llm.log" 2>&1 &
echo "  PID=$!"

echo "[4/6] EMG Gestures — Baselines"
nohup python src/run_baselines.py \
  --builtin emg_gestures --task multiclass \
  --out_dir "$EXP/emg_baselines" \
  --optuna --optuna_trials $BUDGET \
  --seed $SEED \
  > "$LOGS/emg_baselines.log" 2>&1 &
echo "  PID=$!"

sleep 3

# ─────────────────────────────────────────────────────────────────────────────
# 3. ELEC2 — хронологический temporal split (нет утечки будущего)
#    Concept drift: распределение меняется со временем → GBM деградирует
# ─────────────────────────────────────────────────────────────────────────────
echo "[5/6] ELEC2 — LLM-NAS (temporal split)"
nohup python src/run_nas_v2.py \
  --builtin elec2 --task binary \
  --out_dir "$EXP/elec2_llm_s42" \
  --budget $BUDGET --random_warmup $WARMUP \
  --mode batch --batch_n 20 --batch_k 4 \
  --seed $SEED --llm_model $MODEL \
  > "$LOGS/elec2_llm.log" 2>&1 &
echo "  PID=$!"

echo "[6/6] ELEC2 — Baselines"
nohup python src/run_baselines.py \
  --builtin elec2 --task binary \
  --out_dir "$EXP/elec2_baselines" \
  --optuna --optuna_trials $BUDGET \
  --seed $SEED \
  > "$LOGS/elec2_baselines.log" 2>&1 &
echo "  PID=$!"

echo ""
echo "============================================================"
echo "  Все 6 задач запущены."
echo ""
echo "  Мониторинг:"
echo "    watch -n 30 'for f in $LOGS/*.log; do echo \"=== \$(basename \$f) ===\"; tail -2 \$f; done'"
echo ""
echo "  Скачать результаты на мак:"
echo "    rsync -avz -e 'ssh -p 13105' \\"
echo "      root@31.13.223.140:/root/llm-nas/experiments/ \\"
echo "      /Users/vadim/Diplom/llm-tabular-nas-proxy/experiments_v9/"
echo "============================================================"
