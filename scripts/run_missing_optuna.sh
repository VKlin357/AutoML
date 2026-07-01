#!/bin/bash
# Run Optuna baselines for datasets that are missing optuna test metrics.
#
# Datasets covered:
#   helena    — has baseline_optuna.json but NO test_primary (old run bug)
#   adult     — completely missing optuna baseline
#   miniboonee — completely missing optuna baseline
#   har       — completely missing optuna baseline
#   harth     — existing harth_optuna32 has val=test=0.7255 (suspicious, re-run)
#   eeg       — no optuna baseline
#   pamap2    — no optuna baseline
#
# Jannis optuna is handled by run_overnight.sh → experiments_v9/jannis_optuna40/
# Volkert optuna already done → experiments_v9/volkert_optuna40/
#
# Usage (on GPU server):
#   cd ~/llm-tabular-nas-proxy
#   bash scripts/run_missing_optuna.sh 2>&1 | tee logs/missing_optuna.log

set -e
mkdir -p logs

DEVICE="${DEVICE:-cuda}"
N_TRIALS=40
SEED=42

echo "======================================"
echo "MISSING OPTUNA START: $(date)"
echo "======================================"

run_optuna() {
    local NAME=$1
    local OUT_DIR=$2
    shift 2
    local EXTRA="$@"

    if [ -f "${OUT_DIR}/baseline_optuna.json" ]; then
        # Check if test_primary is present and not the suspicious val=test case
        TEST=$(python3 -c "
import json, sys
d = json.load(open('${OUT_DIR}/baseline_optuna.json'))
t = d.get('test_primary')
v = d.get('val_primary', d.get('primary'))
# Flag as suspicious if val==test (old bug) or test is missing
if t is None or (v is not None and abs(float(t)-float(v)) < 1e-9):
    sys.exit(1)
print(t)
" 2>/dev/null) && {
            echo "[SKIP] $NAME — optuna test_primary=$TEST already valid"
            return
        }
        echo "[RE-RUN] $NAME — existing result has missing or suspicious test_primary"
    fi

    mkdir -p "$OUT_DIR"
    echo ""
    echo "▶ Optuna: $NAME → $OUT_DIR  ($(date +%H:%M))"

    python - <<PYEOF
import sys
sys.path.insert(0, '.')
from src.baselines_automl import run_optuna_baseline

result = run_optuna_baseline(
    $EXTRA
    out_dir='$OUT_DIR',
    seed=$SEED,
    n_trials=$N_TRIALS,
    timeout=7200,
    device='$DEVICE',
)
print(f"  DONE: val={result.get('selected_val_primary','?'):.5f}  test={result.get('test_primary','?'):.5f}")
PYEOF

    echo "  $NAME done: $(date +%H:%M)"
}

# ── RTDL datasets ──────────────────────────────────────────────────────────────

run_optuna "helena" "experiments_v9/helena_baselines" \
    "openml_id=41169, task='multiclass', source='openml',"

run_optuna "adult" "experiments_v9/adult_baselines" \
    "openml_id=1590, task='binary', source='openml',"

run_optuna "miniboonee" "experiments_v9/miniboonee_baselines" \
    "openml_id=None, task='binary', source='builtin', builtin_name='miniboonee',"

# ── Sensor datasets ────────────────────────────────────────────────────────────

run_optuna "har" "experiments_v9/har_baselines" \
    "openml_id=None, task='multiclass', source='builtin', builtin_name='har',"

run_optuna "harth" "experiments_v9/harth_baselines" \
    "openml_id=43921, task='multiclass', source='openml',"

run_optuna "eeg" "experiments_v9/eeg_baselines" \
    "openml_id=1471, task='binary', source='openml',"

run_optuna "pamap2" "experiments_v9/pamap2_baselines" \
    "openml_id=44994, task='multiclass', source='openml',"

echo ""
echo "======================================"
echo "MISSING OPTUNA DONE: $(date)"
echo "======================================"

echo ""
echo "Results:"
for dataset in helena adult miniboonee har harth eeg pamap2; do
    f="experiments_v9/${dataset}_baselines/baseline_optuna.json"
    if [ -f "$f" ]; then
        python3 -c "
import json
d = json.load(open('$f'))
t = d.get('test_primary', 'MISSING')
v = d.get('selected_val_primary', d.get('val_primary', d.get('primary', '?')))
print(f'  $dataset: val={float(v):.5f}  test={str(t)[:8]}')
" 2>/dev/null
    else
        echo "  $dataset: NO FILE"
    fi
done
