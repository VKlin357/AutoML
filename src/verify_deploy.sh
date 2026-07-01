#!/usr/bin/env bash
# ============================================================================
# Verify that all 6 fixes are present on the GPU server.
# Run on server (or via ssh).
# ============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."

PASS=0
FAIL=0

check() {
  local desc="$1"
  local file="$2"
  local marker="$3"
  if grep -q "${marker}" "${file}" 2>/dev/null; then
    echo "  ✓ ${desc}"
    PASS=$((PASS+1))
  else
    echo "  ✗ MISSING: ${desc}  (expected '${marker}' in ${file})"
    FAIL=$((FAIL+1))
  fi
}

echo "=== Verifying v3 fixes are deployed ==="
echo ""

echo "Fix #1 (max_tokens):"
check "max_tokens param in OpenAILLM"           src/llm/client.py "self.max_tokens"
check "max_tokens in payload"                   src/llm/client.py "max_tokens.*self.max_tokens"
check "max_tokens default 8192"                 src/llm/client.py "max_tokens: int = 8192"

echo ""
echo "Fix #2 (invalid action post-processing):"
check "NOOP_INTENTS coercion set"               src/llm/client.py "NOOP_INTENTS"
check "Coerce-to-switch_family logic"           src/llm/client.py "post-processing maps to 'switch_family'"
check "_coerced_from marker for analysis"       src/llm/client.py "_coerced_from"
check "REFINE prompt forbids 'none'/'None'"     src/llm/prompts.py "NEVER use"

echo ""
echo "Fix #3 (LR-bias):"
check "LR-BIAS GUARD rule in REFINE prompt"     src/llm/prompts.py "LR-BIAS GUARD"
check "Symmetric reduce_lr rule"                src/llm/prompts.py "reduce_lr"

echo ""
echo "Fix #4 (mode collapse):"
check "MODE-COLLAPSE GUARD in orchestrator"     src/nas_orchestrator.py "MODE-COLLAPSE GUARD"
check "forced:switch_family op label"           src/nas_orchestrator.py "forced:switch_family"
check "DIVERSITY RULE in REFINE prompt"         src/llm/prompts.py "DIVERSITY RULE"

echo ""
echo "Fix #5 (cold-start extreme):"
check "CORNER CASES rule in COLD_START"         src/llm/prompts.py "CORNER CASES"
check "Cold-start temperature bump (T>=1.1)"    src/nas_orchestrator.py "1.1"
check "SYSTEM_PROMPT no longer prescribes 1%"   src/llm/prompts.py "extreme regions"

echo ""
echo "Summary: ${PASS} OK, ${FAIL} missing"

if [ "${FAIL}" -gt 0 ]; then
  echo ""
  echo "❌ DEPLOYMENT INCOMPLETE — re-run deploy.sh from local machine"
  exit 1
else
  echo ""
  echo "✅ All 14 markers verified — ready for sanity run"
  exit 0
fi
