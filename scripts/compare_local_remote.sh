#!/usr/bin/env bash
# ============================================================================
# Полное сравнение локального проекта с удалённым сервером.
#
# Использование:
#   cd /Users/vadim/Diplom/llm-tabular-nas-proxy
#   bash scripts/compare_local_remote.sh
#
# Выводит:
#   (1) checksum-сравнение всех .py файлов и скриптов — что отличается
#   (2) ключевые marker-строки для каждого Fix/Prong (Fix #1..#11, Prong A..G)
#   (3) построчный diff для каждого отличающегося файла (опционально)
# ============================================================================
set -euo pipefail

SERVER_HOST="root@79.117.120.96"
SERVER_PORT="20177"
REMOTE_ROOT="/workspace/llm-tabular-nas-proxy"

# Цвета (для красоты)
R=$'\033[0;31m'; G=$'\033[0;32m'; Y=$'\033[0;33m'; B=$'\033[0;34m'; N=$'\033[0m'

cd "$(dirname "$0")/.."
LOCAL_ROOT="$(pwd)"

echo "${B}╔══════════════════════════════════════════════════════════════╗${N}"
echo "${B}║  COMPARE LOCAL ↔ REMOTE  (llm-tabular-nas-proxy)             ║${N}"
echo "${B}╚══════════════════════════════════════════════════════════════╝${N}"
echo "  Local:  ${LOCAL_ROOT}"
echo "  Remote: ${SERVER_HOST}:${REMOTE_ROOT}"
echo ""

# ──────────────────────────────────────────────────────────────────────────────
# СТУПЕНЬ 1 — rsync --dry-run --checksum: какие файлы отличаются
# ──────────────────────────────────────────────────────────────────────────────
echo "${Y}━━━ Step 1: rsync dry-run (checksum-based diff) ━━━${N}"
echo "Файлы, которые потребовали бы upload (= локальные новее или отличаются):"
echo ""

# Используем checksum, а не mtime/size — надёжнее
# Сравниваем src/ + scripts/
DIFF_FILES_TMP=$(mktemp)
rsync -avzn --checksum --no-times -e "ssh -p ${SERVER_PORT}" \
  --include='*/' --include='*.py' --include='*.sh' --include='*.md' \
  --exclude='*' \
  src/ scripts/ \
  "${SERVER_HOST}:${REMOTE_ROOT}/" 2>&1 | tee "${DIFF_FILES_TMP}"

# Извлекаем имена реально-отличающихся файлов
DIFFERING=$(grep -E '^[A-Za-z0-9_/\.]+\.(py|sh|md)$' "${DIFF_FILES_TMP}" || true)

echo ""
if [ -z "${DIFFERING}" ]; then
  echo "${G}✅ Все .py/.sh/.md в src/ и scripts/ идентичны.${N}"
else
  echo "${R}❌ Отличающиеся файлы:${N}"
  echo "${DIFFERING}" | sed 's/^/  - /'
fi

# ──────────────────────────────────────────────────────────────────────────────
# СТУПЕНЬ 2 — Marker-проверка для каждого фикса
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "${Y}━━━ Step 2: проверка маркеров каждого fix на REMOTE ━━━${N}"

# Список (description, file, marker) для всех применённых фиксов
declare -a MARKERS=(
  "Fix #1 max_tokens=8192 default|src/llm/client.py|max_tokens: int = 8192"
  "Fix #1 max_tokens в payload|src/llm/client.py|max_tokens.*self.max_tokens"
  "Fix #2 NOOP_INTENTS coercion|src/llm/client.py|NOOP_INTENTS"
  "Fix #2 _coerced_from marker|src/llm/client.py|_coerced_from"
  "Fix #2 prompt forbids 'None'|src/llm/prompts.py|NEVER use"
  "Fix #3 LR-BIAS rule|src/llm/prompts.py|LR-BIAS GUARD"
  "Fix #3 symmetric reduce_lr|src/llm/prompts.py|val plateaus 3+ epochs"
  "Fix #4 MODE-COLLAPSE GUARD|src/nas_orchestrator.py|MODE-COLLAPSE GUARD"
  "Fix #4 forced:switch_family|src/nas_orchestrator.py|forced:switch_family"
  "Fix #5 cold-start CORNER CASES|src/llm/prompts.py|CORNER CASES"
  "Fix #5 cold-start temperature|src/nas_orchestrator.py|llm.temperature = max"
  "Fix #5 SYSTEM_PROMPT no 1pct|src/llm/prompts.py|extreme regions"
  "Fix #8 PROPOSE n=2 cap|src/nas_orchestrator.py|min(2, max(2, surr_candidates_k)"
  "Fix #9 4-of-5 guard|src/nas_orchestrator.py|dominant_count >= 4"
  "Prong C stochastic widen|src/llm/actions.py|rng.uniform(1.3, 1.8)"
  "Prong C stochastic reduce_lr|src/llm/actions.py|rng.uniform(0.3, 0.7)"
  "Prong C dedup retry loop|src/nas_orchestrator.py|for retry in range(4)"
  "Prong E new actions|src/llm/actions.py|explore_lr_extreme_low"
  "Prong E explore_deep|src/llm/actions.py|explore_deep"
  "Prong D parent_candidates helper|src/nas_orchestrator.py|_build_parent_candidates"
  "Prong B action_history|src/nas_orchestrator.py|action_history: List\\[Dict\\]"
  "Prong F chat() method|src/llm/client.py|def chat"
  "Prong F refine_conversation|src/nas_orchestrator.py|refine_conversation: List"
  "Prong G iterative SYSTEM_PROMPT|src/llm/prompts.py|CONVERSATION PROTOCOL"
  "Prong G chain-of-thought|src/llm/prompts.py|observation"
  "Prong G iteration counter|src/nas_orchestrator.py|ITERATION "
  "Prong G curve_narrative|src/nas_orchestrator.py|curve_narrative"
  "Prong G cold-start kickoff|src/nas_orchestrator.py|SEARCH KICK-OFF"
  "Fix #10 LR_BIAS_LIMIT=4|src/nas_orchestrator.py|LR_BIAS_LIMIT = 4"
  "Fix #10 explore_lr_extreme_low_guard|src/nas_orchestrator.py|explore_lr_extreme_low_guard"
  "Fix #11 family_aware_extras|src/nas_orchestrator.py|family_aware_extras"
  "Fix #11 best_family selection|src/nas_orchestrator.py|best_family = max"
)

ok=0
miss=0
for entry in "${MARKERS[@]}"; do
  IFS='|' read -r desc file marker <<<"${entry}"
  # Проверка на REMOTE через ssh
  if ssh -p "${SERVER_PORT}" "${SERVER_HOST}" "grep -q -- \"${marker}\" \"${REMOTE_ROOT}/${file}\"" 2>/dev/null; then
    echo "  ${G}✓${N} ${desc}"
    ok=$((ok+1))
  else
    # Проверим, есть ли он локально, чтобы понять — он наш или нет
    if grep -q -- "${marker}" "${file}" 2>/dev/null; then
      echo "  ${R}✗ MISSING ON REMOTE${N} (есть локально) — ${desc}"
    else
      echo "  ${Y}⚠ MISSING BOTH (не наш)${N} — ${desc}"
    fi
    miss=$((miss+1))
  fi
done

echo ""
echo "${B}Маркеров проверено: $((ok+miss))   ОК: ${G}${ok}${N}   Отсутствует на remote: ${R}${miss}${N}${N}"

# ──────────────────────────────────────────────────────────────────────────────
# СТУПЕНЬ 3 — построчный diff для каждого отличающегося файла (опционально)
# ──────────────────────────────────────────────────────────────────────────────
if [ -n "${DIFFERING}" ]; then
  echo ""
  echo "${Y}━━━ Step 3: построчные diff'ы (только для отличающихся) ━━━${N}"
  echo "${Y}(хочешь увидеть полный diff каждого файла? — нажми Enter; чтобы пропустить, Ctrl+C)${N}"
  read -r _ || true
  for f in ${DIFFERING}; do
    echo ""
    echo "${B}── diff: ${f} ──${N}"
    REMOTE_TMP=$(mktemp)
    scp -P "${SERVER_PORT}" "${SERVER_HOST}:${REMOTE_ROOT}/${f}" "${REMOTE_TMP}" 2>/dev/null \
      || { echo "  ${R}(не удалось скачать с remote — файл может отсутствовать)${N}"; continue; }
    diff -u "${REMOTE_TMP}" "${f}" | head -80 || true
    rm -f "${REMOTE_TMP}"
  done
fi

rm -f "${DIFF_FILES_TMP}"

echo ""
if [ "${miss}" -eq 0 ] && [ -z "${DIFFERING}" ]; then
  echo "${G}╔══════════════════════════════════════════════════════════════╗${N}"
  echo "${G}║  ✅ LOCAL == REMOTE — всё синхронизировано                   ║${N}"
  echo "${G}╚══════════════════════════════════════════════════════════════╝${N}"
elif [ "${miss}" -eq 0 ]; then
  echo "${Y}╔══════════════════════════════════════════════════════════════╗${N}"
  echo "${Y}║  ⚠ Маркеры все на месте, но rsync видит различия            ║${N}"
  echo "${Y}║    (вероятно, мелкие изменения в комментариях или новых     ║${N}"
  echo "${Y}║     строках). Запусти deploy.sh чтобы догнать.              ║${N}"
  echo "${Y}╚══════════════════════════════════════════════════════════════╝${N}"
else
  echo "${R}╔══════════════════════════════════════════════════════════════╗${N}"
  echo "${R}║  ❌ ${miss} маркер(ов) ОТСУТСТВУЕТ на сервере                   ║${N}"
  echo "${R}║                                                              ║${N}"
  echo "${R}║   Запусти:                                                  ║${N}"
  echo "${R}║     bash scripts/deploy.sh                                  ║${N}"
  echo "${R}╚══════════════════════════════════════════════════════════════╝${N}"
fi
