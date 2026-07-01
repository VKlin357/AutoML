#!/usr/bin/env bash
# ============================================================================
# Deploy v3 fixes to GPU server.
# Run from local: /Users/vadim/Diplom/llm-tabular-nas-proxy/
#   bash scripts/deploy.sh
# ============================================================================
set -euo pipefail

SERVER_HOST="root@79.117.120.96"
SERVER_PORT="20177"
REMOTE_ROOT="/workspace/llm-tabular-nas-proxy"

echo "=== Deploy: rsync modified source files to server ==="
echo "Target: ${SERVER_HOST}:${REMOTE_ROOT}/"
echo ""

# Push src/llm/*.py (Bug fixes #1, #2, #3, #5)
rsync -avz -e "ssh -p ${SERVER_PORT}" \
  src/llm/client.py \
  src/llm/prompts.py \
  "${SERVER_HOST}:${REMOTE_ROOT}/src/llm/"

# Push src/nas_orchestrator.py (Bug fix #4 hard-rule + Bug fix #5 cold-start temp)
rsync -avz -e "ssh -p ${SERVER_PORT}" \
  src/nas_orchestrator.py \
  "${SERVER_HOST}:${REMOTE_ROOT}/src/"

# Also push the verify + run scripts so we can use them remotely
rsync -avz -e "ssh -p ${SERVER_PORT}" \
  scripts/verify_deploy.sh \
  scripts/run_v3_sanity.sh \
  scripts/run_v3_full.sh \
  "${SERVER_HOST}:${REMOTE_ROOT}/scripts/"

echo ""
echo "=== Deploy complete ==="
echo ""
echo "Next steps:"
echo "  1) Verify markers landed on server:"
echo "       ssh -p ${SERVER_PORT} ${SERVER_HOST} 'cd ${REMOTE_ROOT} && bash scripts/verify_deploy.sh'"
echo "  2) If verify is OK — run sanity test on helena (~30 min):"
echo "       ssh -p ${SERVER_PORT} ${SERVER_HOST} 'cd ${REMOTE_ROOT} && bash scripts/run_v3_sanity.sh'"
echo "  3) If sanity passes — full multi-seed run:"
echo "       ssh -p ${SERVER_PORT} ${SERVER_HOST} 'cd ${REMOTE_ROOT} && bash scripts/run_v3_full.sh'"
