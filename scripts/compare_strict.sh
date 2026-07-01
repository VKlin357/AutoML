#!/usr/bin/env bash
# ============================================================================
# СТРОГАЯ проверка sync (исправленная).
# Использует ОДИН ssh-вызов чтобы избежать interactive auth prompts.
# Сравнение по SHA256 — самый надёжный метод.
#
#   cd /Users/vadim/Diplom/llm-tabular-nas-proxy
#   bash scripts/compare_strict.sh
# ============================================================================
set -euo pipefail

SERVER_HOST="root@79.117.120.96"
SERVER_PORT="20177"
REMOTE_ROOT="/workspace/llm-tabular-nas-proxy"

R=$'\033[0;31m'; G=$'\033[0;32m'; Y=$'\033[0;33m'; B=$'\033[0;34m'; N=$'\033[0m'

cd "$(dirname "$0")/.."

echo "${B}╔══════════════════════════════════════════════════════════════╗${N}"
echo "${B}║  STRICT COMPARE  (SHA256 + line count)                       ║${N}"
echo "${B}╚══════════════════════════════════════════════════════════════╝${N}"
echo "  Local:  $(pwd)"
echo "  Remote: ${SERVER_HOST}:${REMOTE_ROOT}"
echo ""

CRITICAL_FILES=(
  "src/llm/client.py"
  "src/llm/prompts.py"
  "src/llm/actions.py"
  "src/llm/schema.py"
  "src/llm/logger.py"
  "src/nas_orchestrator.py"
  "src/search_space.py"
  "src/multi_fidelity.py"
  "src/surrogate.py"
  "src/train_nn.py"
  "src/data.py"
  "src/evolution.py"
  "src/ensemble.py"
  "src/preprocessing.py"
  "src/metrics.py"
  "src/utils.py"
  "src/models.py"
  "scripts/run_nas_v2.py"
)

# ──────────────────────────────────────────────────────────────────────────────
# ОДНИМ ssh-вызовом получаем sha256+wc для всех файлов сразу
# ──────────────────────────────────────────────────────────────────────────────
echo "${Y}━━━ Fetching remote SHA256 and line counts (single SSH call) ━━━${N}"

# Формируем shell-команду для удалённой стороны
REMOTE_CMD="cd ${REMOTE_ROOT} && for f in ${CRITICAL_FILES[@]}; do
  if [ -f \"\$f\" ]; then
    h=\$(sha256sum \"\$f\" | awk '{print \$1}')
    l=\$(wc -l < \"\$f\")
    echo \"\$f \$h \$l\"
  else
    echo \"\$f MISSING -\"
  fi
done"

REMOTE_OUT=$(mktemp)
if ! ssh -p "${SERVER_PORT}" -o ConnectTimeout=10 "${SERVER_HOST}" "${REMOTE_CMD}" > "${REMOTE_OUT}" 2>&1; then
  echo "${R}❌ SSH-call failed. Output:${N}"
  cat "${REMOTE_OUT}"
  rm -f "${REMOTE_OUT}"
  exit 1
fi

# Локальные хеши
LOCAL_OUT=$(mktemp)
for f in "${CRITICAL_FILES[@]}"; do
  if [ -f "$f" ]; then
    h=$(sha256sum "$f" 2>/dev/null | awk '{print $1}')
    l=$(wc -l < "$f" | tr -d ' ')
    echo "$f $h $l" >> "${LOCAL_OUT}"
  else
    echo "$f MISSING -" >> "${LOCAL_OUT}"
  fi
done

# ──────────────────────────────────────────────────────────────────────────────
# Сравнение
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "${Y}━━━ Per-file comparison ━━━${N}"
printf "  %-35s  %-10s  %-10s  %-8s  %-8s\n" "file" "local_sha" "remote_sha" "L lines" "R lines"
printf "  %-35s  %-10s  %-10s  %-8s  %-8s\n" "----" "---------" "----------" "-------" "-------"

ALL_OK=1
MISSING_COUNT=0
DIFF_COUNT=0
LINE_DIFF=0
for f in "${CRITICAL_FILES[@]}"; do
  loc_line=$(grep "^${f} " "${LOCAL_OUT}" || echo "${f} ABSENT -")
  rem_line=$(grep "^${f} " "${REMOTE_OUT}" || echo "${f} ABSENT -")

  loc_h=$(echo "${loc_line}" | awk '{print $2}')
  rem_h=$(echo "${rem_line}" | awk '{print $2}')
  loc_l=$(echo "${loc_line}" | awk '{print $3}')
  rem_l=$(echo "${rem_line}" | awk '{print $3}')

  short_loc="${loc_h:0:10}"
  short_rem="${rem_h:0:10}"

  if [ "${loc_h}" = "MISSING" ] || [ "${rem_h}" = "MISSING" ] || [ "${loc_h}" = "ABSENT" ] || [ "${rem_h}" = "ABSENT" ]; then
    printf "  %-35s  ${Y}%-10s  %-10s  %-8s  %-8s  MISSING${N}\n" "$f" "${short_loc}" "${short_rem}" "${loc_l}" "${rem_l}"
    MISSING_COUNT=$((MISSING_COUNT+1))
    ALL_OK=0
  elif [ "${loc_h}" = "${rem_h}" ]; then
    printf "  %-35s  ${G}%-10s  %-10s  %-8s  %-8s  ✓ identical${N}\n" "$f" "${short_loc}" "${short_rem}" "${loc_l}" "${rem_l}"
  else
    printf "  %-35s  ${R}%-10s  %-10s  %-8s  %-8s  ✗ DIFFERS${N}\n" "$f" "${short_loc}" "${short_rem}" "${loc_l}" "${rem_l}"
    DIFF_COUNT=$((DIFF_COUNT+1))
    ALL_OK=0
    if [ "${loc_l}" != "${rem_l}" ]; then
      LINE_DIFF=$((LINE_DIFF+1))
    fi
  fi
done

rm -f "${LOCAL_OUT}" "${REMOTE_OUT}"

# ──────────────────────────────────────────────────────────────────────────────
# Вердикт
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "${B}╔══════════════════════════════════════════════════════════════╗${N}"
if [ "${ALL_OK}" -eq 1 ]; then
  echo "${B}║${G}  ✅ LOCAL == REMOTE  (${#CRITICAL_FILES[@]}/${#CRITICAL_FILES[@]} файлов с identical SHA256)         ${B}║${N}"
  echo "${B}║${G}     Можешь запускать experiments.                            ${B}║${N}"
else
  echo "${B}║${R}  ❌ NOT SYNCED: ${DIFF_COUNT} различаются, ${MISSING_COUNT} missing             ${B}║${N}"
  echo "${B}║${Y}                                                              ${B}║${N}"
  echo "${B}║${Y}   FIX: rsync только нужные файлы на сервер:                  ${B}║${N}"
  echo "${B}║${Y}                                                              ${B}║${N}"
  echo "${B}║${N}     rsync -avz -e \"ssh -p 20177\" \\                            ${B}║${N}"
  echo "${B}║${N}       src/llm/*.py \\                                          ${B}║${N}"
  echo "${B}║${N}       root@79.117.120.96:/workspace/llm-tabular-nas-proxy/src/llm/ ${B}║${N}"
  echo "${B}║${N}                                                              ${B}║${N}"
  echo "${B}║${N}     rsync -avz -e \"ssh -p 20177\" \\                            ${B}║${N}"
  echo "${B}║${N}       src/nas_orchestrator.py \\                               ${B}║${N}"
  echo "${B}║${N}       root@79.117.120.96:/workspace/llm-tabular-nas-proxy/src/ ${B}║${N}"
fi
echo "${B}╚══════════════════════════════════════════════════════════════╝${N}"

exit $((1-ALL_OK))
