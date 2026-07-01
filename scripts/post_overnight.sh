#!/bin/bash
# Post-overnight cleanup on the GPU server.
# Run this AFTER run_overnight.sh finishes.
#
# Does:
#   1. Compute test for eeg_llm_s42 (was skipped by overnight due to missing mapping — now fixed)
#   2. Compute test for harth_llm_s42_v2 (32 trials, better than harth_llm_s42's 22)
#   3. Run missing optuna baselines (helena, adult, miniboonee, har, harth, eeg, pamap2)
#
# Usage:
#   cd ~/llm-tabular-nas-proxy
#   bash scripts/post_overnight.sh 2>&1 | tee logs/post_overnight.log

set -e
mkdir -p logs

DEVICE="${DEVICE:-cuda}"

echo "======================================"
echo "POST-OVERNIGHT START: $(date)"
echo "======================================"

# ── 1. EEG LLM-NAS ensemble (was skipped overnight due to missing dataset mapping) ──
echo ""
echo "[1/3] EEG LLM-NAS ensemble (eeg_llm_s42)..."

python scripts/compute_test_metrics.py \
    --exp_dir experiments_v9 \
    --names eeg_llm_s42 \
    --ensemble_k 7 --device $DEVICE

echo "  EEG done: $(date +%H:%M)"

# ── 2. harth_llm_s42_v2 (32 trials, better than the 22-trial v1) ──────────────
echo ""
echo "[2/3] harth_llm_s42_v2 ensemble (32 trials)..."

python scripts/compute_test_metrics.py \
    --exp_dir experiments_v9 \
    --names harth_llm_s42_v2 \
    --ensemble_k 7 --device $DEVICE

echo "  harth v2 done: $(date +%H:%M)"

# ── 3. Optuna test metrics ─────────────────────────────────────────────────────
echo ""
echo "[3/3] Optuna тест-метрики..."

# adult и har: 40 триалов уже есть, нужно только дообучить лучшую модель (~5 мин каждый)
echo "  Восстанавливаем test для adult и har (best_params уже сохранены)..."
python scripts/recover_optuna_test.py \
    --exp_dir experiments_v9 \
    --names adult_baselines har_baselines \
    --device $DEVICE

# helena и miniboonee: только 10/14 триалов — нечестное сравнение, перепрогоняем до 40
echo "  Optuna 40 триалов для helena..."
python scripts/run_baselines.py \
    --openml_id 41169 --task multiclass \
    --out_dir experiments_v9/helena_baselines \
    --optuna --optuna_trials 40 --no_lgbm --seed 42 --device $DEVICE

echo "  Optuna 40 триалов для miniboonee..."
python scripts/run_baselines.py \
    --builtin miniboonee --task binary \
    --out_dir experiments_v9/miniboonee_baselines \
    --optuna --optuna_trials 40 --no_lgbm --seed 42 --device $DEVICE

# eeg и pamap2: нет optuna вообще — запускаем с нуля
echo "  Optuna 40 триалов для eeg..."
python scripts/run_baselines.py \
    --openml_id 1471 --task binary \
    --out_dir experiments_v9/eeg_baselines \
    --optuna --optuna_trials 40 --no_lgbm --seed 42 --device $DEVICE

echo "  Optuna 40 триалов для pamap2..."
python scripts/run_baselines.py \
    --openml_id 44994 --task multiclass \
    --out_dir experiments_v9/pamap2_baselines \
    --optuna --optuna_trials 40 --no_lgbm --seed 42 --device $DEVICE

echo ""
echo "======================================"
echo "POST-OVERNIGHT DONE: $(date)"
echo "======================================"

echo ""
echo "=== Full LLM-NAS ensemble summary ==="
python3 - <<'PYEOF'
import json
from pathlib import Path

exp_dir = Path("experiments_v9")
datasets = [
    ("volkert",    "volkert_llm_nas"),
    ("jannis",     "jannis_batch_s42"),
    ("helena",     "helena_batch_s42"),
    ("miniboonee", "miniboonee_batch_s42"),
    ("adult",      "adult_batch_s42"),
    ("har",        "har_llm_s42"),
    ("harth",      "harth_llm_s42_v2"),
    ("eeg",        "eeg_llm_s42"),
    ("pamap2",     "pamap2_llm_s42"),
]

print(f"{'Dataset':<12} {'Val':>8} {'Test':>8}")
print("-" * 32)
for name, dirname in datasets:
    p = exp_dir / dirname / "ensemble_result.json"
    if p.exists():
        d = json.loads(p.read_text())
        val  = d.get("val_primary", d.get("primary", float("nan")))
        test = d.get("test_primary", "MISSING")
        print(f"{name:<12} {val:>8.5f} {str(test)[:8]:>8}")
    else:
        print(f"{name:<12} {'N/A':>8} {'N/A':>8}")
PYEOF
