#!/bin/bash
# Запускает всё недостающее в правильном порядке.
# Умный skip: пропускает уже готовое.
#
# Порядок (по приоритету):
#   1. LLM-NAS ensemble+test для сенсорных датасетов (har, harth_v2, eeg, pamap2)
#   2. Optuna test recover для adult и har (40 триалов есть, только 1 дообучение)
#   3. Optuna 40 триалов для helena, miniboonee, jannis, eeg, pamap2
#   4. AutoGluon для 5 RTDL датасетов (1800с на датасет)
#
# Запуск (на GPU сервере, в tmux):
#   cd ~/llm-tabular-nas-proxy
#   bash scripts/run_all_missing.sh 2>&1 | tee logs/run_all_missing_$(date +%Y%m%d_%H%M).log

set -euo pipefail
DEVICE="${DEVICE:-cuda}"
mkdir -p logs experiments_v9/autogluon

echo "=============================================="
echo "START: $(date)"
echo "DEVICE=$DEVICE"
echo "=============================================="

# ══════════════════════════════════════════════════════════════════════════════
# Утилиты
# ══════════════════════════════════════════════════════════════════════════════

has_test() {
    # Возвращает 0 (true) если файл существует и содержит валидный test_primary
    local f=$1
    [ -f "$f" ] && python3 -c "
import json, sys
d = json.load(open('$f'))
t = d.get('test_primary')
sys.exit(0 if (t is not None and str(t) != 'None') else 1)
" 2>/dev/null
}

has_optuna_ok() {
    # test_primary есть И триалов >= min_trials
    local f=$1
    local min=${2:-30}
    [ -f "$f" ] && python3 -c "
import json, sys
d = json.load(open('$f'))
t = d.get('test_primary')
n = int(d.get('n_trials_completed', 0))
sys.exit(0 if (t is not None and n >= $min) else 1)
" 2>/dev/null
}

section() { echo ""; echo "══ $1 ══ $(date +%H:%M)"; }

# ══════════════════════════════════════════════════════════════════════════════
# 1. LLM-NAS ensemble + test для сенсорных датасетов
# ══════════════════════════════════════════════════════════════════════════════
section "1/4 LLM-NAS ensemble (все датасеты)"

for NAME in volkert_llm_nas jannis_batch_s42 helena_batch_s42 miniboonee_batch_s42 adult_batch_s42 har_llm_s42 harth_llm_s42_v2 eeg_llm_s42 pamap2_llm_s42; do
    ENS="experiments_v9/$NAME/ensemble_result.json"
    if has_test "$ENS"; then
        echo "  [SKIP] $NAME — test уже есть"
        continue
    fi
    echo "  ▶ $NAME ($(date +%H:%M))..."
    python scripts/compute_test_metrics.py \
        --exp_dir experiments_v9 \
        --names "$NAME" \
        --ensemble_k 7 --device "$DEVICE"
done

# ══════════════════════════════════════════════════════════════════════════════
# 2. Optuna test recover — adult и har (40 триалов есть, просто нет test_primary)
# ══════════════════════════════════════════════════════════════════════════════
section "2/4 Optuna recover (adult, har)"

NEED_RECOVER=""
for NAME in adult_baselines har_baselines; do
    f="experiments_v9/$NAME/baseline_optuna.json"
    if has_optuna_ok "$f" 30; then
        echo "  [SKIP] $NAME — test уже есть"
    else
        NEED_RECOVER="$NEED_RECOVER $NAME"
    fi
done

if [ -n "$NEED_RECOVER" ]; then
    python scripts/recover_optuna_test.py \
        --exp_dir experiments_v9 \
        --names $NEED_RECOVER \
        --device "$DEVICE"
fi

# ══════════════════════════════════════════════════════════════════════════════
# 3. Optuna 40 триалов для датасетов без нормального результата
# ══════════════════════════════════════════════════════════════════════════════
section "3/4 Optuna 40 триалов"

run_optuna_openml() {
    local NAME=$1 ODIR=$2 OID=$3 TASK=$4
    local f="experiments_v9/$ODIR/baseline_optuna.json"
    if has_optuna_ok "$f" 30; then
        echo "  [SKIP] $NAME — ≥30 триалов с test уже есть"
        return
    fi
    echo "  ▶ Optuna $NAME ($(date +%H:%M))..."
    python scripts/run_baselines.py \
        --openml_id "$OID" --task "$TASK" \
        --out_dir "experiments_v9/$ODIR" \
        --optuna --optuna_trials 40 --no_lgbm \
        --seed 42 --device "$DEVICE"
}

run_optuna_builtin() {
    local NAME=$1 ODIR=$2 BUILTIN=$3 TASK=$4
    local f="experiments_v9/$ODIR/baseline_optuna.json"
    if has_optuna_ok "$f" 30; then
        echo "  [SKIP] $NAME — ≥30 триалов с test уже есть"
        return
    fi
    echo "  ▶ Optuna $NAME ($(date +%H:%M))..."
    python scripts/run_baselines.py \
        --builtin "$BUILTIN" --task "$TASK" \
        --out_dir "experiments_v9/$ODIR" \
        --optuna --optuna_trials 40 --no_lgbm \
        --seed 42 --device "$DEVICE"
}

# helena: было только 10 триалов — перепрогоняем
run_optuna_openml  "helena"     "helena_baselines"     41169 "multiclass"

# miniboonee: было только 14 триалов — перепрогоняем
run_optuna_builtin "miniboonee" "miniboonee_baselines" "miniboonee" "binary"

# jannis: нет optuna вообще
run_optuna_openml  "jannis"     "jannis_optuna40"      45021 "multiclass"

# eeg: нет optuna
run_optuna_openml  "eeg"        "eeg_baselines"        1471  "binary"

# pamap2: нет optuna — UCI builtin (не OpenML)
run_optuna_builtin "pamap2"     "pamap2_baselines"     "pamap2" "multiclass"

# ══════════════════════════════════════════════════════════════════════════════
# 4. AutoGluon на 5 RTDL датасетах
# ══════════════════════════════════════════════════════════════════════════════
section "4/4 AutoGluon"

python -c "import autogluon" 2>/dev/null || {
    echo "  Устанавливаем AutoGluon..."
    pip install autogluon.tabular --quiet
}

run_autogluon() {
    local NAME=$1 OID=$2 TASK=$3 BUILTIN=$4
    local OUT="experiments_v9/autogluon/$NAME"
    if has_test "$OUT/autogluon_result.json"; then
        echo "  [SKIP] AutoGluon $NAME — уже готово"
        return
    fi
    mkdir -p "$OUT"
    echo "  ▶ AutoGluon $NAME ($(date +%H:%M))..."
    python3 - <<PYEOF
import sys, json, time, numpy as np, pandas as pd
sys.path.insert(0, '.')
from src.data import load_raw
from src.metrics import compute_metrics
from src.utils import save_json
from pathlib import Path
from autogluon.tabular import TabularPredictor

BUILTIN = '$BUILTIN'
if BUILTIN:
    raw, _ = load_raw(source='builtin', builtin_name=BUILTIN, task='$TASK', seed=42)
else:
    raw, _ = load_raw(source='openml', openml_id=$OID, task='$TASK', seed=42)

cols = [f'f{i}' for i in range(raw.X_train.shape[1])]
train_df = pd.DataFrame(np.vstack([raw.X_train, raw.X_val]), columns=cols)
train_df['__y__'] = np.concatenate([raw.y_train, raw.y_val])
test_df  = pd.DataFrame(raw.X_test, columns=cols)

metric = 'roc_auc' if raw.task == 'binary' else 'accuracy'
t0 = time.time()
import shutil, os
ag_path = f'/dev/shm/ag_tmp_{os.getpid()}'
try:
    pred = TabularPredictor(
        label='__y__', eval_metric=metric,
        path=ag_path, verbosity=0
    ).fit(train_df, time_limit=1800)

    proba = pred.predict_proba(test_df)
    if raw.task == 'binary':
        arr = proba.iloc[:, 1].values
        m = compute_metrics('binary', raw.y_test, y_pred_proba=arr)
    else:
        m = compute_metrics('multiclass', raw.y_test, y_pred_proba=proba.values)

    result = {
        'test_primary': float(m.primary),
        'metrics': m.metrics,
        'seconds': time.time() - t0,
        'dataset': '$NAME',
    }
    save_json(Path('$OUT/autogluon_result.json'), result)
    print(f'  AutoGluon $NAME: test={m.primary:.5f}  ({result["seconds"]:.0f}s)')
finally:
    shutil.rmtree(ag_path, ignore_errors=True)
PYEOF
}

run_autogluon "volkert"    41166 "multiclass" ""
run_autogluon "jannis"     45021 "multiclass" ""
run_autogluon "helena"     41169 "multiclass" ""
run_autogluon "miniboonee" 0     "binary"     "miniboonee"
run_autogluon "adult"      1590  "binary"     ""

# ══════════════════════════════════════════════════════════════════════════════
# Итоговая сводка
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "=============================================="
echo "DONE: $(date)"
echo "=============================================="

python3 - <<'PYEOF'
import json
from pathlib import Path

exp = Path("experiments_v9")

print("\n── LLM-NAS (ensemble test) ──────────────────")
for name, d in [
    ("volkert",    "volkert_llm_nas"),
    ("jannis",     "jannis_batch_s42"),
    ("helena",     "helena_batch_s42"),
    ("miniboonee", "miniboonee_batch_s42"),
    ("adult",      "adult_batch_s42"),
    ("har",        "har_llm_s42"),
    ("harth",      "harth_llm_s42_v2"),
    ("eeg",        "eeg_llm_s42"),
    ("pamap2",     "pamap2_llm_s42"),
]:
    p = exp / d / "ensemble_result.json"
    if p.exists():
        r = json.loads(p.read_text())
        v = r.get("val_primary", r.get("primary", float("nan")))
        t = r.get("test_primary", "MISSING")
        print(f"  {name:<12} val={v:.4f}  test={str(t)[:8]}")
    else:
        print(f"  {name:<12} НЕТ")

print("\n── Optuna (test) ────────────────────────────")
for name, d in [
    ("volkert",    "volkert_optuna40/baseline_optuna.json"),
    ("jannis",     "jannis_optuna40/baseline_optuna.json"),
    ("helena",     "helena_baselines/baseline_optuna.json"),
    ("miniboonee", "miniboonee_baselines/baseline_optuna.json"),
    ("adult",      "adult_baselines/baseline_optuna.json"),
    ("har",        "har_baselines/baseline_optuna.json"),
    ("harth",      "harth_optuna32/baseline_optuna.json"),
    ("eeg",        "eeg_baselines/baseline_optuna.json"),
    ("pamap2",     "pamap2_baselines/baseline_optuna.json"),
]:
    p = exp / d
    if p.exists():
        r = json.loads(p.read_text())
        t = r.get("test_primary", "MISSING")
        n = r.get("n_trials_completed", "?")
        print(f"  {name:<12} test={str(t)[:8]}  trials={n}")
    else:
        print(f"  {name:<12} НЕТ")

print("\n── AutoGluon (test) ─────────────────────────")
for name in ["volkert", "jannis", "helena", "miniboonee", "adult"]:
    p = exp / "autogluon" / name / "autogluon_result.json"
    if p.exists():
        r = json.loads(p.read_text())
        t = r.get("test_primary", "?")
        print(f"  {name:<12} test={str(t)[:8]}")
    else:
        print(f"  {name:<12} НЕТ")
PYEOF
