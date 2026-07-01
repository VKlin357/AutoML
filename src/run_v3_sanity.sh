#!/usr/bin/env bash
# ============================================================================
# v3 sanity run — single dataset (helena, OpenML 41166), single seed.
# Purpose: verify that the deployed fixes actually trigger:
#   - max_tokens=8192 (no parse_fails in PROPOSE)
#   - NOOP_INTENTS coercion (no random_refine_fallback from "None"/"already_good")
#   - MODE-COLLAPSE GUARD (forced:switch_family > 0)
#
# Run on the GPU server:
#   cd /workspace/llm-tabular-nas-proxy
#   bash scripts/run_v3_sanity.sh
# ============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."

if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "ERROR: OPENAI_API_KEY not set"
  exit 1
fi

OUT_DIR="experiments_v3/sanity/helena"
mkdir -p "${OUT_DIR}"

echo "=== v3 sanity run on helena (OpenML 41166), seed=42, budget=25 ==="
echo "Output: ${OUT_DIR}"
echo ""

python3 scripts/run_nas_v2.py \
  --openml_id 41166 \
  --task multiclass \
  --out_dir "${OUT_DIR}" \
  --budget 25 \
  --coldstart_n 5 \
  --seed 42 \
  --llm_model gpt-4o-mini \
  --reflect_every 8 \
  --explore_pulse 6

echo ""
echo "=== Sanity diagnostics ==="

python3 << 'PYEOF'
import json, os
from collections import Counter

OUT = "experiments_v3/sanity/helena"

# Load trials
idx = json.load(open(f"{OUT}/trials_index.json"))
trials = idx if isinstance(idx, list) else idx.get("trials", [])
ops = Counter(t.get("op", "") for t in trials)

# Load llm log
logs = []
log_path = f"{OUT}/nas_v2/llm_log.jsonl"
if not os.path.exists(log_path):
    log_path = f"{OUT}/llm_log.jsonl"
if os.path.exists(log_path):
    with open(log_path) as f:
        logs = [json.loads(l) for l in f if l.strip()]

# Count parse fails (truncation)
parse_fails = 0
coerced = 0
for e in logs:
    raw = e.get("raw_response", "")
    if raw:
        try:
            json.loads(raw)
        except:
            parse_fails += 1
    if e.get("parsed", {}).get("_coerced_from"):
        coerced += 1

forced = ops.get("forced:switch_family", 0)
fallbacks = ops.get("random_refine_fallback", 0) + ops.get("refine:None", 0)
real_refines = sum(v for k,v in ops.items() if k.startswith("refine:") and k != "refine:None")

print(f"  Total trials:           {len(trials)}")
print(f"  Cold start:             {ops.get('cold_start', 0)}")
print(f"  Propose:                {ops.get('propose', 0)}")
print(f"  Real refine actions:    {real_refines}")
print(f"  forced:switch_family:   {forced}    <-- expected > 0 in v3")
print(f"  random_refine_fallback: {fallbacks}  <-- expected lower in v3")
print(f"  Parse fails (PROPOSE):  {parse_fails}  <-- expected 0 in v3 (was 3 in v2)")
print(f"  Coerced (NOOP→switch):  {coerced}  <-- expected > 0 if LLM still says 'None'")

best = json.load(open(f"{OUT}/best_trial.json"))
print(f"  Best primary:           {best.get('primary'):.4f}")
print(f"  Best op:                {best.get('op')}")
print(f"  Best family:            {best.get('config',{}).get('arch',{}).get('family')}")

# Verdict
print()
print("=== VERDICT ===")
verdict_ok = True
if forced == 0:
    print("  ⚠ MODE-COLLAPSE GUARD did not fire — orchestrator.py likely not deployed")
    verdict_ok = False
if parse_fails > 0:
    print(f"  ⚠ {parse_fails} parse fails — max_tokens=8192 may not be deployed, or model still truncates")
    verdict_ok = False
if fallbacks > 5:
    print(f"  ⚠ {fallbacks} fallbacks remain — may need to re-check post-processing in client.py")
if verdict_ok and forced >= 1 and parse_fails == 0:
    print("  ✅ All v3 fixes verified active. Safe to launch full multi-seed run.")
else:
    print("  ❌ Some fixes did not take effect — DO NOT launch full run yet. Re-deploy and retry.")
PYEOF
