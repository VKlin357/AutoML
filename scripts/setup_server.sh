#!/bin/bash
# Настройка окружения на новом GPU сервере и запуск всех экспериментов.
# Запускать НА СЕРВЕРЕ после rsync:
#   bash scripts/setup_server.sh
#
# ВАЖНО: перед запуском установи API ключ:
#   export OPENAI_API_KEY="sk-..."

set -euo pipefail

cd ~/llm-tabular-nas-proxy
mkdir -p logs

echo "======================================"
echo "SETUP: $(date)"
echo "======================================"

# ── 1. Python venv ──────────────────────────────────────────────
echo "[1/4] Создаём venv..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip --quiet

# ── 2. PyTorch (cu126 работает с драйвером CUDA 13.x) ──────────
echo "[2/4] Устанавливаем PyTorch cu126..."
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu126 --quiet

# ── 3. Остальные зависимости ────────────────────────────────────
echo "[3/4] Устанавливаем зависимости..."
pip install \
    numpy==1.26.4 pandas==2.2.3 scikit-learn==1.5.2 scipy==1.14.1 \
    openml==0.15.1 tqdm==4.66.5 \
    catboost==1.2.7 "lightgbm>=4.0.0" \
    "httpx>=0.27.0" optuna \
    --quiet

# ── 4. Проверка GPU ─────────────────────────────────────────────
echo "[4/4] Проверка GPU..."
python3 -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

echo ""
echo "======================================"
echo "SETUP DONE. Запускаем эксперименты..."
echo "======================================"

# Проверяем API ключ
if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "ОШИБКА: OPENAI_API_KEY не установлен!"
    echo "Запусти: export OPENAI_API_KEY='sk-...'"
    echo "Затем:   bash scripts/run_all_missing.sh"
    exit 1
fi

# Запуск всего в tmux
tmux new-session -d -s experiments 2>/dev/null || true
tmux send-keys -t experiments \
    "cd ~/llm-tabular-nas-proxy && source venv/bin/activate && export OPENAI_API_KEY='${OPENAI_API_KEY}' && bash scripts/run_all_missing.sh 2>&1 | tee logs/run_all_missing_\$(date +%Y%m%d_%H%M).log" \
    Enter

echo ""
echo "Эксперименты запущены в tmux сессии 'experiments'."
echo "Смотреть прогресс: tmux attach -t experiments"
echo "Отсоединиться:    Ctrl+B, затем D"
