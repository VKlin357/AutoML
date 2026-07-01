#!/bin/bash
# Overnight pipeline — запускай в новом tmux окне
# Порядок:
#   1. Ждём завершения текущего compute_test_metrics
#   2. Ensemble + test для сенсорных датасетов
#   3. Optuna для Jannis (единственный датасет без optuna)
#   4. AutoGluon на всех 5 датасетах (новый сильный baseline)
#
# Usage:
#   tmux new-window   (Ctrl+B, C)
#   cd ~/llm-tabular-nas-proxy
#   bash scripts/run_overnight.sh 2>&1 | tee logs/overnight.log

set -e
mkdir -p logs experiments_v9/jannis_optuna40 experiments_v9/autogluon

echo "======================================"
echo "OVERNIGHT START: $(date)"
echo "======================================"

# ── 1. Ждём текущий compute_test_metrics ──────────────────────────────────
echo ""
echo "[1/4] Ждём завершения текущего compute_test_metrics..."
while pgrep -f "compute_test_metrics" > /dev/null 2>&1; do
    echo "  ... ещё работает, ждём 60с ($(date +%H:%M))"
    sleep 60
done
echo "  OK, compute_test_metrics завершён: $(date)"

# ── 2. Ensemble + test для сенсорных датасетов ────────────────────────────
echo ""
echo "[2/4] Ensemble + test для сенсорных датасетов..."

python scripts/compute_test_metrics.py \
    --exp_dir experiments_v9 \
    --names har_llm_s42 harth_llm_s42 eeg_llm_s42 pamap2_llm_s42 harth_random_s42_v2 \
    --ensemble_k 7 --device cuda

echo "  Sensors done: $(date)"

# ── 3. Optuna для Jannis (40 trials, тот же бюджет что и LLM-NAS) ─────────
echo ""
echo "[3/4] Optuna baseline для Jannis..."

python scripts/run_baselines.py \
    --openml_id 45021 \
    --task multiclass \
    --out_dir experiments_v9/jannis_optuna40 \
    --optuna \
    --optuna_trials 40 \
    --no_lgbm \
    --seed 42 \
    --device cuda

echo "  Jannis Optuna done: $(date)"

# ── 4. AutoGluon на всех 5 датасетах ──────────────────────────────────────
echo ""
echo "[4/4] AutoGluon baseline..."

# Установка (пропускается если уже установлен)
python -c "import autogluon" 2>/dev/null || {
    echo "  Устанавливаем AutoGluon..."
    pip install autogluon.tabular --quiet
}

run_autogluon() {
    local NAME=$1
    local OPENML_ID=$2
    local TASK=$3
    local BUILTIN=$4

    local OUT="experiments_v9/autogluon/${NAME}"
    mkdir -p "$OUT"

    if [ -f "${OUT}/autogluon_result.json" ]; then
        echo "  [SKIP] AutoGluon $NAME — уже готово"
        return
    fi

    echo "  AutoGluon: $NAME ($(date +%H:%M))..."

    if [ -n "$BUILTIN" ]; then
        python - <<PYEOF
import sys, json
sys.path.insert(0, '.')
from src.data import load_raw
from src.metrics import compute_metrics
from src.utils import save_json
from pathlib import Path
import time, numpy as np

try:
    from autogluon.tabular import TabularDataset, TabularPredictor
except ImportError:
    print("AutoGluon not installed"); sys.exit(0)

raw, _ = load_raw(source='builtin', builtin_name='$BUILTIN', task='$TASK', seed=42)

import pandas as pd
train_df = pd.DataFrame(raw.X_train, columns=[f'f{i}' for i in range(raw.X_train.shape[1])])
train_df['__target__'] = raw.y_train
val_df   = pd.DataFrame(raw.X_val, columns=[f'f{i}' for i in range(raw.X_val.shape[1])])
val_df['__target__'] = raw.y_val
test_df  = pd.DataFrame(raw.X_test, columns=[f'f{i}' for i in range(raw.X_test.shape[1])])

eval_metric = 'roc_auc' if raw.task == 'binary' else 'accuracy'
t0 = time.time()
predictor = TabularPredictor(label='__target__', eval_metric=eval_metric,
                              path='$OUT/ag_model', verbosity=0
).fit(pd.concat([train_df, val_df]), time_limit=1800)

preds = predictor.predict(test_df)
proba = predictor.predict_proba(test_df)

if raw.task == 'binary':
    proba_arr = proba.iloc[:, 1].values if hasattr(proba, 'iloc') else proba
    m = compute_metrics('binary', raw.y_test, y_pred_proba=proba_arr)
else:
    m = compute_metrics('multiclass', raw.y_test, y_pred_proba=proba.values)

result = {'primary': float(m.primary), 'test_primary': float(m.primary),
          'metrics': m.metrics, 'seconds': time.time()-t0, 'dataset': '$NAME'}
save_json(Path('$OUT/autogluon_result.json'), result)
print(f"  AutoGluon $NAME: test={m.primary:.5f}  ({time.time()-t0:.0f}s)")
PYEOF
    else
        python - <<PYEOF
import sys, json
sys.path.insert(0, '.')
from src.data import load_raw
from src.metrics import compute_metrics
from src.utils import save_json
from pathlib import Path
import time, numpy as np

try:
    from autogluon.tabular import TabularDataset, TabularPredictor
except ImportError:
    print("AutoGluon not installed"); sys.exit(0)

raw, _ = load_raw(source='openml', openml_id=$OPENML_ID, task='$TASK', seed=42)

import pandas as pd
train_df = pd.DataFrame(raw.X_train, columns=[f'f{i}' for i in range(raw.X_train.shape[1])])
train_df['__target__'] = raw.y_train
val_df   = pd.DataFrame(raw.X_val, columns=[f'f{i}' for i in range(raw.X_val.shape[1])])
val_df['__target__'] = raw.y_val
test_df  = pd.DataFrame(raw.X_test, columns=[f'f{i}' for i in range(raw.X_test.shape[1])])

eval_metric = 'roc_auc' if raw.task == 'binary' else 'accuracy'
t0 = time.time()
predictor = TabularPredictor(label='__target__', eval_metric=eval_metric,
                              path='$OUT/ag_model', verbosity=0
).fit(pd.concat([train_df, val_df]), time_limit=1800)

proba = predictor.predict_proba(test_df)

if raw.task == 'binary':
    proba_arr = proba.iloc[:, 1].values if hasattr(proba, 'iloc') else proba
    m = compute_metrics('binary', raw.y_test, y_pred_proba=proba_arr)
else:
    m = compute_metrics('multiclass', raw.y_test, y_pred_proba=proba.values)

result = {'primary': float(m.primary), 'test_primary': float(m.primary),
          'metrics': m.metrics, 'seconds': time.time()-t0, 'dataset': '$NAME'}
save_json(Path('$OUT/autogluon_result.json'), result)
print(f"  AutoGluon $NAME: test={m.primary:.5f}  ({time.time()-t0:.0f}s)")
PYEOF
    fi
}

run_autogluon "volkert"    41166  "multiclass"  ""
run_autogluon "jannis"     45021  "multiclass"  ""
run_autogluon "miniboonee" 0      "binary"      "miniboonee"
run_autogluon "helena"     41169  "multiclass"  ""
run_autogluon "adult"      1590   "binary"      ""

echo ""
echo "======================================"
echo "OVERNIGHT DONE: $(date)"
echo "======================================"
echo ""
echo "Результаты AutoGluon:"
for f in experiments_v9/autogluon/*/autogluon_result.json; do
    name=$(echo $f | cut -d'/' -f3)
    python3 -c "import json; d=json.load(open('$f')); print(f'  $name: test={d[\"test_primary\"]:.5f}')" 2>/dev/null
done
