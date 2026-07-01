#!/bin/bash
# Скачивает с СТАРОГО сервера только то чего нет локально.
# Запускать на МАКЕ.
#
# Использование:
#   bash scripts/fetch_from_old_server.sh
#
# Порт может меняться после reboot — проверь актуальный в Vast.ai UI.

OLD_HOST="185.99.66.50"
OLD_PORT="${OLD_PORT:-16126}"   # переопредели если порт изменился: OLD_PORT=12345 bash ...
RSYNC="/opt/homebrew/bin/rsync"
SSH_OPTS="-p $OLD_PORT -o ServerAliveInterval=15 -o ConnectTimeout=20 -o StrictHostKeyChecking=no"
LOCAL="$HOME/Diplom/llm-tabular-nas-proxy/experiments_v9"
REMOTE="root@$OLD_HOST:~/llm-tabular-nas-proxy/experiments_v9"

echo "======================================"
echo "Старый сервер: $OLD_HOST:$OLD_PORT"
echo "======================================"

# Проверяем доступность
if ! ssh $SSH_OPTS root@$OLD_HOST "echo OK" 2>/dev/null; then
    echo "ОШИБКА: старый сервер недоступен. Проверь порт и статус в Vast.ai."
    exit 1
fi

echo "Сервер доступен. Начинаем скачивать недостающее..."
echo ""

sync_file() {
    local remote_path=$1
    local local_path="$LOCAL/$remote_path"
    local remote_full="$REMOTE/$remote_path"

    if [ -f "$local_path" ]; then
        # Проверяем есть ли test_primary если это ensemble/optuna файл
        test_val=$(python3 -c "
import json, sys
try:
    d = json.load(open('$local_path'))
    t = d.get('test_primary')
    sys.exit(0 if t is not None else 1)
except: sys.exit(1)
" 2>/dev/null)
        if [ $? -eq 0 ]; then
            echo "  [SKIP] $remote_path — уже есть с test"
            return
        fi
        echo "  [UPDATE] $remote_path — есть но без test, обновляем..."
    else
        echo "  [FETCH] $remote_path"
        mkdir -p "$(dirname "$local_path")"
    fi

    $RSYNC -az --progress -e "ssh $SSH_OPTS" \
        "$remote_full" "$local_path" 2>/dev/null \
        && echo "    ✓ готово" || echo "    ✗ не найден на сервере"
}

echo "── LLM-NAS ensembles ──────────────────────────"
sync_file "har_llm_s42/ensemble_result.json"
sync_file "harth_llm_s42_v2/ensemble_result.json"
sync_file "eeg_llm_s42/ensemble_result.json"
sync_file "jannis_llm_nas/ensemble_result.json"
sync_file "helena_llm_nas/ensemble_result.json"
sync_file "miniboonee_llm_nas/ensemble_result.json"
sync_file "adult_llm_nas/ensemble_result.json"

echo ""
echo "── Optuna baselines ───────────────────────────"
sync_file "jannis_optuna40/baseline_optuna.json"
sync_file "eeg_baselines/baseline_optuna.json"
sync_file "pamap2_baselines/baseline_optuna.json"
sync_file "helena_baselines/baseline_optuna.json"
sync_file "miniboonee_baselines/baseline_optuna.json"

echo ""
echo "── Random NAS (ablation) ──────────────────────"
sync_file "jannis_rnd40/baseline_random_search.json"
sync_file "jannis_rnd40/trials_random_index.json"

echo ""
echo "── AutoGluon ──────────────────────────────────"
for ds in volkert jannis helena miniboonee adult; do
    sync_file "autogluon/$ds/autogluon_result.json"
done

echo ""
echo "── test_metrics_summary.json ──────────────────"
sync_file "../test_metrics_summary.json" 2>/dev/null || true

echo ""
echo "======================================"
echo "Итог:"
echo "======================================"
python3 << 'PYEOF'
import json
from pathlib import Path

exp = Path.home() / "Diplom/llm-tabular-nas-proxy/experiments_v9"
checks = [
    ("har_llm_s42/ensemble_result.json",         "LLM-NAS har"),
    ("harth_llm_s42_v2/ensemble_result.json",    "LLM-NAS harth"),
    ("eeg_llm_s42/ensemble_result.json",         "LLM-NAS eeg"),
    ("jannis_llm_nas/ensemble_result.json",      "LLM-NAS jannis"),
    ("helena_llm_nas/ensemble_result.json",      "LLM-NAS helena"),
    ("miniboonee_llm_nas/ensemble_result.json",  "LLM-NAS miniboonee"),
    ("adult_llm_nas/ensemble_result.json",       "LLM-NAS adult"),
    ("jannis_optuna40/baseline_optuna.json",     "Optuna jannis"),
    ("eeg_baselines/baseline_optuna.json",       "Optuna eeg"),
    ("pamap2_baselines/baseline_optuna.json",    "Optuna pamap2"),
    ("helena_baselines/baseline_optuna.json",    "Optuna helena"),
    ("miniboonee_baselines/baseline_optuna.json","Optuna miniboonee"),
    ("jannis_rnd40/baseline_random_search.json", "RndNAS jannis"),
    ("autogluon/volkert/autogluon_result.json",  "AutoGluon volkert"),
    ("autogluon/jannis/autogluon_result.json",   "AutoGluon jannis"),
    ("autogluon/helena/autogluon_result.json",   "AutoGluon helena"),
    ("autogluon/miniboonee/autogluon_result.json","AutoGluon miniboonee"),
    ("autogluon/adult/autogluon_result.json",    "AutoGluon adult"),
]
for rel, label in checks:
    p = exp / rel
    if not p.exists():
        print(f"  ✗ MISSING  {label}")
    else:
        d = json.loads(p.read_text())
        t = d.get("test_primary")
        if t:
            print(f"  ✓ OK       {label}  test={t:.4f}")
        else:
            print(f"  ~ NO TEST  {label}")
PYEOF
