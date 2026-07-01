#!/bin/bash
# Run LLM-NAS with seeds 0 and 1337 on all 5 RTDL datasets.
# Seed 42 already exists in experiments_v9/.
# Results will land in experiments_multiseed/<dataset>_batch_s<seed>/
#
# Usage:
#   export OPENAI_API_KEY=sk-proj-...
#   bash scripts/run_multiseed.sh
#
# Optional: override output dir and device
#   OUT_DIR=experiments_ms DEVICE=cuda bash scripts/run_multiseed.sh
#
# Requirements: same GPU machine + Python env as the original experiments.
# Estimated time: ~2-4 GPU-hours per dataset per seed (same as original runs).
# API cost: ~$0.09 × 5 datasets × 2 seeds = ~$0.90 total.

set -e

# ── Config ────────────────────────────────────────────────────────────────────
SEEDS=(0 1337)
OUT_DIR="${OUT_DIR:-experiments_multiseed}"
DEVICE="${DEVICE:-cuda}"
BUDGET=40
WARMUP=12
BATCH_N=20
BATCH_K=4
ENSEMBLE_K=7
MODEL="gpt-4o-mini"
# ─────────────────────────────────────────────────────────────────────────────

if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY is not set. Run: export OPENAI_API_KEY=sk-proj-..."
    exit 1
fi

mkdir -p "$OUT_DIR"

run_dataset() {
    local NAME=$1
    local OPENML_ID=$2
    local TASK=$3
    local BUILTIN=$4   # empty string if openml, otherwise builtin name
    local SEED=$5

    local EXP_DIR="${OUT_DIR}/${NAME}_batch_s${SEED}"

    if [ -f "${EXP_DIR}/ensemble_result.json" ]; then
        echo "[SKIP] ${NAME} seed=${SEED} — ensemble_result.json already exists"
        return
    fi

    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "  Dataset : ${NAME}  |  Seed : ${SEED}  |  Out : ${EXP_DIR}"
    echo "════════════════════════════════════════════════════════"

    # Build data-source args
    if [ -n "$BUILTIN" ]; then
        DATA_ARGS="--builtin ${BUILTIN} --task ${TASK}"
    else
        DATA_ARGS="--openml_id ${OPENML_ID} --task ${TASK}"
    fi

    python scripts/run_nas_v2.py \
        $DATA_ARGS \
        --out_dir "${EXP_DIR}" \
        --budget ${BUDGET} \
        --random_warmup ${WARMUP} \
        --mode batch \
        --batch_n ${BATCH_N} \
        --batch_k ${BATCH_K} \
        --ensemble_k ${ENSEMBLE_K} \
        --seed ${SEED} \
        --llm_model ${MODEL} \
        --device ${DEVICE}

    echo "[DONE] ${NAME} seed=${SEED}"
}

# ── Dataset registry ──────────────────────────────────────────────────────────
# Format: NAME  OPENML_ID  TASK  BUILTIN_NAME(or empty)
# ─────────────────────────────────────────────────────────────────────────────

for SEED in "${SEEDS[@]}"; do
    # Also copy seed-42 results from experiments_v9 to experiments_multiseed
    # so aggregate_results.py can find all 3 seeds in one place
    for DATASET in volkert jannis helena adult; do
        SRC="experiments_v9/${DATASET}_batch_s42"
        DST="${OUT_DIR}/${DATASET}_batch_s42"
        if [ -d "$SRC" ] && [ ! -d "$DST" ]; then
            echo "[COPY] seed-42 results: $SRC → $DST"
            cp -r "$SRC" "$DST"
        fi
    done
    # miniboonee has different dir name
    if [ -d "experiments_v9/miniboonee_batch_s42" ] && [ ! -d "${OUT_DIR}/miniboonee_batch_s42" ]; then
        echo "[COPY] seed-42 results: experiments_v9/miniboonee_batch_s42 → ${OUT_DIR}/miniboonee_batch_s42"
        cp -r "experiments_v9/miniboonee_batch_s42" "${OUT_DIR}/miniboonee_batch_s42"
    fi

    run_dataset "volkert"    41166  "multiclass"  ""           $SEED
    run_dataset "jannis"     45021  "multiclass"  ""           $SEED
    run_dataset "miniboonee" 0      "binary"      "miniboonee" $SEED
    run_dataset "helena"     41169  "multiclass"  ""           $SEED
    run_dataset "adult"      1590   "binary"      ""           $SEED
done

echo ""
echo "════════════════════════════════════════════════════════"
echo "  All seeds done. Now compute test metrics:"
echo "  python scripts/compute_test_metrics.py --exp_dir ${OUT_DIR}"
echo ""
echo "  Then aggregate:"
echo "  python scripts/aggregate_results.py --exp_dir ${OUT_DIR}"
echo "════════════════════════════════════════════════════════"
