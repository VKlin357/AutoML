#!/usr/bin/env bash
# ============================================================================
# Запуск LLM-NAS на ОДИН датасет с ОДНИМ seed.
# Удобно для quick re-run после фикса или для нового датасета.
#
# Использование:
#   bash scripts/run_single_dataset.sh <DATASET_NAME> [SEED] [BUDGET] [MODEL]
#
# Примеры:
#   bash scripts/run_single_dataset.sh helena                         # gpt-4o-mini, seed=42
#   bash scripts/run_single_dataset.sh jannis 42 25 claude-sonnet     # Claude Sonnet
#   bash scripts/run_single_dataset.sh pol 1 25 claude-haiku          # Claude Haiku (быстрый)
#   bash scripts/run_single_dataset.sh covertype 42 30 gpt-4o         # GPT-4o
#
# Поддерживаемые MODEL (4-й аргумент):
#   gpt-4o-mini      → OpenAI GPT-4o-mini       (дефолт, дёшево)
#   gpt-4o           → OpenAI GPT-4o             (~17x дороже mini, умнее)
#   claude-sonnet    → Claude Sonnet via OpenRouter (лучший баланс)
#   claude-haiku     → Claude Haiku via OpenRouter  (быстрый, дешевле)
#   claude-opus      → Claude Opus via OpenRouter   (самый мощный)
#
# Для Claude нужен OPENROUTER_API_KEY вместо OPENAI_API_KEY:
#   export OPENROUTER_API_KEY=sk-or-...
#
# Поддерживаемые DATASET_NAME:
#   --- OpenML ID-based ---
#   helena       → openml_id 41166, multiclass
#   pol          → openml_id 44156, binary
#   jannis       → openml_id 45021, multiclass
#   connect4     → openml_id 40668, multiclass
#   eye_movements→ openml_id 1044,  multiclass
#   --- built-in ---
#   covertype    → builtin covtype, multiclass
#   miniboone    → builtin miniboonee, binary
# ============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."

DATASET=${1:?Usage: $0 <DATASET_NAME> [SEED=42] [BUDGET=25] [MODEL=gpt-4o-mini]}
SEED=${2:-42}
BUDGET=${3:-25}
MODEL=${4:-gpt-4o-mini}

# ── Резолв модели → реальное имя + base_url + api_key ─────────────────────
case "${MODEL}" in
  gpt-4o-mini)
    LLM_MODEL="gpt-4o-mini"
    LLM_BASE_URL=""
    LLM_API_KEY="${OPENAI_API_KEY:?ERROR: OPENAI_API_KEY not set}"
    ;;
  gpt-4o)
    LLM_MODEL="gpt-4o"
    LLM_BASE_URL=""
    LLM_API_KEY="${OPENAI_API_KEY:?ERROR: OPENAI_API_KEY not set}"
    ;;
  claude-sonnet)
    LLM_MODEL="anthropic/claude-sonnet-4-5"
    LLM_BASE_URL="https://openrouter.ai/api/v1/chat/completions"
    LLM_API_KEY="${OPENROUTER_API_KEY:?ERROR: OPENROUTER_API_KEY not set (get it at openrouter.ai)}"
    ;;
  claude-haiku)
    LLM_MODEL="anthropic/claude-haiku-4-5"
    LLM_BASE_URL="https://openrouter.ai/api/v1/chat/completions"
    LLM_API_KEY="${OPENROUTER_API_KEY:?ERROR: OPENROUTER_API_KEY not set (get it at openrouter.ai)}"
    ;;
  claude-opus)
    LLM_MODEL="anthropic/claude-opus-4-5"
    LLM_BASE_URL="https://openrouter.ai/api/v1/chat/completions"
    LLM_API_KEY="${OPENROUTER_API_KEY:?ERROR: OPENROUTER_API_KEY not set (get it at openrouter.ai)}"
    ;;
  *)
    # произвольная строка модели — передаём как есть, base_url из env если есть
    LLM_MODEL="${MODEL}"
    LLM_BASE_URL="${LLM_BASE_URL:-}"
    LLM_API_KEY="${OPENROUTER_API_KEY:-${OPENAI_API_KEY:-}}"
    if [ -z "${LLM_API_KEY}" ]; then
      echo "ERROR: set OPENAI_API_KEY or OPENROUTER_API_KEY"
      exit 1
    fi
    ;;
esac

# out_dir включает имя модели чтобы не перезатирать результаты разных моделей
MODEL_SLUG=$(echo "${MODEL}" | tr '/' '-')
OUT_DIR="experiments_v5/single/${DATASET}_seed${SEED}_${MODEL_SLUG}"

mkdir -p "${OUT_DIR}"

# Resolve dataset name to LLM-NAS arguments
case "${DATASET}" in
  helena)
    ARGS="--openml_id 41166 --task multiclass"
    ;;
  pol)
    ARGS="--openml_id 44156 --task binary"
    ;;
  jannis)
    ARGS="--openml_id 45021 --task multiclass"
    ;;
  connect4)
    ARGS="--openml_id 40668 --task multiclass"
    ;;
  eye_movements)
    ARGS="--openml_id 1044 --task multiclass"
    ;;
  covertype)
    ARGS="--builtin covtype --task multiclass"
    ;;
  miniboone|miniboonee)
    ARGS="--builtin miniboonee --task binary"
    ;;
  california_housing)
    ARGS="--builtin california_housing --task regression"
    ;;
  *)
    echo "Unknown dataset: ${DATASET}"
    echo "Supported: helena, pol, jannis, connect4, eye_movements, covertype, miniboone, california_housing"
    exit 1
    ;;
esac

echo "=================================================================="
echo "  Single-dataset LLM-NAS run"
echo "  Dataset : ${DATASET}   Seed: ${SEED}   Budget: ${BUDGET}"
echo "  Model   : ${MODEL} → ${LLM_MODEL}"
echo "  Output  : ${OUT_DIR}"
echo "=================================================================="

BASE_URL_ARG=""
if [ -n "${LLM_BASE_URL}" ]; then
  BASE_URL_ARG="--base_url ${LLM_BASE_URL}"
fi

python3 scripts/run_nas_v2.py \
  ${ARGS} \
  --out_dir "${OUT_DIR}" \
  --budget ${BUDGET} \
  --coldstart_n 5 \
  --seed ${SEED} \
  --llm_model "${LLM_MODEL}" \
  --api_key "${LLM_API_KEY}" \
  ${BASE_URL_ARG} \
  --reflect_every 8 \
  --explore_pulse 6

echo ""
echo "=================================================================="
echo "  Quick stats:"
echo "=================================================================="
python3 << PYEOF
import json
from collections import Counter
from pathlib import Path

OUT = Path("${OUT_DIR}")
idx_path = OUT / "trials_index.json"
if not idx_path.exists():
    print("(no trials_index.json found)")
    exit()

idx = json.load(open(idx_path))
trials = idx if isinstance(idx, list) else idx.get("trials", [])
ops = Counter(t.get("op","") for t in trials)
fams = Counter(t.get("config",{}).get("arch",{}).get("family","?") for t in trials)

real_llm = sum(v for k,v in ops.items() if k.startswith("refine:") and k != "refine:None")
fb = ops.get("random_refine_fallback",0) + ops.get("refine:None",0)
forced = sum(v for k,v in ops.items() if k.startswith("forced:"))
fam_aware = ops.get("family_aware_extra", 0)

bt = json.load(open(OUT / "best_trial.json"))
ens_p = OUT / "ensemble_result.json"
ens = json.load(open(ens_p))['primary'] if ens_p.exists() else None

print(f"  Best primary:                  {bt.get('primary'):.5f}")
print(f"  Best op:                       {bt.get('op')}")
print(f"  Best family:                   {bt['config']['arch']['family']}")
if ens:
    print(f"  Ensemble primary:              {ens:.5f}")
print()
print(f"  Real LLM-direct actions:       {real_llm} / 25")
print(f"  Fallbacks (random):            {fb}")
print(f"  Forced (guards):               {forced}")
print(f"  Family-aware extras (Fix #11): {fam_aware}")
print(f"  Family distribution:           {dict(fams.most_common(5))}")
PYEOF
