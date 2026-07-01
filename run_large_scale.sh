#!/usr/bin/env bash
# =============================================================================
#  Large-scale LLM-NAS experiment: sensor / time-series tabular datasets
#  Run this script ON THE VAST.AI SERVER after syncing src/ from Mac.
#
#  Datasets:
#    elec2       — 45K × 8,   binary,    TEMPORAL split (no leakage)
#    ecg5000     — 5K  × 140, 5-class,   ECG morphology
#    emg_gestures— 60K × 64,  4-class,   sEMG sensor channels
#    covtype     — 581K× 54,  7-class,   large-scale stress test
#
#  Usage:
#    export OPENAI_API_KEY="sk-proj-..."
#    bash run_large_scale.sh
# =============================================================================

set -euo pipefail

ROOT="/root/llm-nas"
EXP="$ROOT/experiments"
LOGS="$ROOT/logs"
MODEL="gpt-4o-mini"
SEED=42

mkdir -p "$LOGS" "$EXP"

cd "$ROOT"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "ERROR: set OPENAI_API_KEY first"
  exit 1
fi

echo "============================================================"
echo "  LLM-NAS Large-Scale Experiment"
echo "  Model: $MODEL  |  Seed: $SEED"
echo "============================================================"

# ─────────────────────────────────────────────────────────────────────────────
# 1.  ELEC2  (45K rows, binary, temporal split)
#     Temporal structure → GBMs blind to concept drift → NNs win
# ─────────────────────────────────────────────────────────────────────────────
echo "[1/8] ELEC2 — LLM-NAS (temporal split)"
nohup python src/run_nas_v2.py \
  --builtin elec2 --task binary \
  --out_dir "$EXP/elec2_llm_s42" \
  --budget 40 --random_warmup 10 \
  --mode batch --batch_n 20 --batch_k 4 \
  --seed $SEED --llm_model $MODEL \
  > "$LOGS/elec2_llm.log" 2>&1 &
echo "  PID=$! → $LOGS/elec2_llm.log"

echo "[2/8] ELEC2 — Baselines (CatBoost + LightGBM + Optuna)"
nohup python src/run_baselines.py \
  --builtin elec2 --task binary \
  --out_dir "$EXP/elec2_baselines" \
  --optuna --optuna_trials 40 \
  --seed $SEED \
  > "$LOGS/elec2_baselines.log" 2>&1 &
echo "  PID=$! → $LOGS/elec2_baselines.log"

# wait for a bit so GPU isn't slammed at exactly the same time
sleep 5

# ─────────────────────────────────────────────────────────────────────────────
# 2.  ECG5000  (5K × 140 time-steps, 5-class arrhythmia)
#     NNs learn waveform morphology; GBMs see 140 raw features
# ─────────────────────────────────────────────────────────────────────────────
echo "[3/8] ECG5000 — LLM-NAS"
nohup python src/run_nas_v2.py \
  --builtin ecg5000 --task multiclass \
  --out_dir "$EXP/ecg5000_llm_s42" \
  --budget 40 --random_warmup 10 \
  --mode batch --batch_n 20 --batch_k 4 \
  --seed $SEED --llm_model $MODEL \
  > "$LOGS/ecg5000_llm.log" 2>&1 &
echo "  PID=$! → $LOGS/ecg5000_llm.log"

echo "[4/8] ECG5000 — Baselines"
nohup python src/run_baselines.py \
  --builtin ecg5000 --task multiclass \
  --out_dir "$EXP/ecg5000_baselines" \
  --optuna --optuna_trials 40 \
  --seed $SEED \
  > "$LOGS/ecg5000_baselines.log" 2>&1 &
echo "  PID=$! → $LOGS/ecg5000_baselines.log"

sleep 5

# ─────────────────────────────────────────────────────────────────────────────
# 3.  EMG Gestures  (60K × 64 sEMG channels, 5-class)
#     Highly correlated sensor channels → attention-based NNs excel
# ─────────────────────────────────────────────────────────────────────────────
echo "[5/8] EMG Gestures — LLM-NAS"
nohup python src/run_nas_v2.py \
  --builtin emg_gestures --task multiclass \
  --out_dir "$EXP/emg_llm_s42" \
  --budget 40 --random_warmup 10 \
  --mode batch --batch_n 20 --batch_k 4 \
  --seed $SEED --llm_model $MODEL \
  > "$LOGS/emg_llm.log" 2>&1 &
echo "  PID=$! → $LOGS/emg_llm.log"

echo "[6/8] EMG Gestures — Baselines"
nohup python src/run_baselines.py \
  --builtin emg_gestures --task multiclass \
  --out_dir "$EXP/emg_baselines" \
  --optuna --optuna_trials 40 \
  --seed $SEED \
  > "$LOGS/emg_baselines.log" 2>&1 &
echo "  PID=$! → $LOGS/emg_baselines.log"

sleep 5

# ─────────────────────────────────────────────────────────────────────────────
# 4.  Covertype  (581K × 54, 7-class) — large-scale stress test
#     NNs competitive at scale; tests agent's ability to handle big data
# ─────────────────────────────────────────────────────────────────────────────
echo "[7/8] Covertype — LLM-NAS"
nohup python src/run_nas_v2.py \
  --builtin covtype --task multiclass \
  --out_dir "$EXP/covtype_llm_s42" \
  --budget 30 --random_warmup 8 \
  --mode batch --batch_n 15 --batch_k 3 \
  --seed $SEED --llm_model $MODEL \
  > "$LOGS/covtype_llm.log" 2>&1 &
echo "  PID=$! → $LOGS/covtype_llm.log"

echo "[8/8] Covertype — Baselines"
nohup python src/run_baselines.py \
  --builtin covtype --task multiclass \
  --out_dir "$EXP/covtype_baselines" \
  --optuna --optuna_trials 30 \
  --seed $SEED \
  > "$LOGS/covtype_baselines.log" 2>&1 &
echo "  PID=$! → $LOGS/covtype_baselines.log"

echo ""
echo "============================================================"
echo "  All 8 jobs launched in background."
echo ""
echo "  Monitor progress:"
echo "    tail -f $LOGS/elec2_llm.log"
echo "    tail -f $LOGS/ecg5000_llm.log"
echo "    tail -f $LOGS/emg_llm.log"
echo "    tail -f $LOGS/covtype_llm.log"
echo ""
echo "  Check all at once:"
echo "    for f in $LOGS/*.log; do echo \"=== \$f ===\"; tail -3 \$f; done"
echo ""
echo "  Download results to Mac:"
echo "    rsync -avz -e 'ssh -p 13105' root@31.13.223.140:/root/llm-nas/experiments/ \\"
echo "      /Users/vadim/Diplom/llm-tabular-nas-proxy/experiments_v9/"
echo "============================================================"
