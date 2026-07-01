#!/usr/bin/env bash
# =============================================================================
# run_thesis_experiments.sh
#
# Использование:
#   bash scripts/run_thesis_experiments.sh sk-proj-...
#   # или
#   export OPENAI_API_KEY=sk-proj-...
#   bash scripts/run_thesis_experiments.sh
#
# Порядок запуска:
#   1. Domain датасеты (ECG + EEG + HAR + ELEC2)  — ~2 часа
#   2. Основные hard датасеты (MiniBooNE + jannis + helena + pol)  — ~4 часа
#   3. Ablation: curve-blind (MiniBooNE без кривых обучения)  — ~40 мин
#   4. Ablation: PROPOSE vs REFINE (MiniBooNE)  — ~40 мин
# =============================================================================

set -e

API_KEY="${1:-${OPENAI_API_KEY}}"
if [ -z "$API_KEY" ]; then
    echo "ERROR: передай API-ключ как аргумент или через OPENAI_API_KEY"
    echo "  bash scripts/run_thesis_experiments.sh sk-proj-..."
    exit 1
fi

MODEL="${LLM_MODEL:-gpt-4o-mini}"
BUDGET=25
COLD=5
RAND_BUDGET=25
ENSEMBLE=5
SEED=42
EXP_DIR="experiments"

echo "================================================================"
echo "ДИПЛОМ: LLM-NAS для табличных данных"
echo "  Модель : $MODEL"
echo "  Бюджет : $BUDGET  (coldstart: $COLD)"
echo "  Папка  : $EXP_DIR"
echo "================================================================"

# ============================================================
# ЧАСТЬ 1: DOMAIN ДАТАСЕТЫ — ECG + EEG + HAR + ELEC2
# ============================================================
echo ""
echo ">>> ЧАСТЬ 1: Domain датасеты (биомедицина + финансы)"
echo "    ECG5000 / EEG Eye State / HAR / ELEC2"
echo ""

python scripts/run_experiment.py \
    --domain biomedical \
    --exp_dir "${EXP_DIR}/domains" \
    --budget $BUDGET \
    --coldstart_n $COLD \
    --random_nas_budget $RAND_BUDGET \
    --ensemble_k $ENSEMBLE \
    --optuna --optuna_trials $BUDGET \
    --naive_llm --naive_llm_trials 3 \
    --seed $SEED \
    --llm_model "$MODEL" \
    --api_key "$API_KEY"

cp "${EXP_DIR}/domains/summary.json" "${EXP_DIR}/summary_biomedical.json" 2>/dev/null || true

python scripts/run_experiment.py \
    --domain finance \
    --exp_dir "${EXP_DIR}/domains" \
    --budget $BUDGET \
    --coldstart_n $COLD \
    --random_nas_budget $RAND_BUDGET \
    --ensemble_k $ENSEMBLE \
    --optuna --optuna_trials $BUDGET \
    --naive_llm --naive_llm_trials 3 \
    --seed $SEED \
    --llm_model "$MODEL" \
    --api_key "$API_KEY"

cp "${EXP_DIR}/domains/summary.json" "${EXP_DIR}/summary_finance.json" 2>/dev/null || true

echo ""
echo ">>> ЧАСТЬ 1 завершена."
echo ""

# ============================================================
# ЧАСТЬ 2: ОСНОВНЫЕ HARD ДАТАСЕТЫ
# jannis / helena / pol
# (MiniBooNE исключён: persistent NaN в raw data)
# ============================================================
echo ">>> ЧАСТЬ 2: Основные hard датасеты"
echo "    jannis + helena + pol"
echo "    (MiniBooNE исключён: persistent NaN в raw data)"
echo ""

# jannis / helena / pol — через OpenML
python scripts/run_experiment.py \
    --datasets 45021 41166 44156 \
    --exp_dir "$EXP_DIR" \
    --budget $BUDGET \
    --coldstart_n $COLD \
    --random_nas_budget $RAND_BUDGET \
    --ensemble_k $ENSEMBLE \
    --optuna --optuna_trials $BUDGET \
    --naive_llm --naive_llm_trials 5 \
    --seed $SEED \
    --llm_model "$MODEL" \
    --api_key "$API_KEY"

cp "$EXP_DIR/summary.json" "$EXP_DIR/summary_main.json" 2>/dev/null || true

echo ""
echo ">>> ЧАСТЬ 2 завершена. Результаты в $EXP_DIR/summary_main.json"
echo ""

# ============================================================
# ЧАСТЬ 3: ABLATION — curve-blind (без кривых обучения)
# ============================================================
echo ">>> ЧАСТЬ 3: Ablation — curve-blind vs curve-aware (MiniBooNE)"
echo ""

python scripts/run_experiment.py \
    --builtin miniboonee \
    --exp_dir "${EXP_DIR}/ablation_no_curves" \
    --budget $BUDGET \
    --coldstart_n $COLD \
    --random_nas_budget $RAND_BUDGET \
    --ensemble_k $ENSEMBLE \
    --no_curves \
    --seed $SEED \
    --llm_model "$MODEL" \
    --api_key "$API_KEY"

cp "${EXP_DIR}/ablation_no_curves/summary.json" "${EXP_DIR}/summary_ablation_no_curves.json" 2>/dev/null || true

echo ""
echo ">>> ЧАСТЬ 3 завершена."
echo ""

# ============================================================
# ЧАСТЬ 4: ABLATION — PROPOSE vs REFINE
# ============================================================
echo ">>> ЧАСТЬ 4: Ablation — PROPOSE vs REFINE (MiniBooNE)"
echo ""

python scripts/run_experiment.py \
    --builtin miniboonee \
    --exp_dir "${EXP_DIR}/ablation_propose" \
    --budget $BUDGET \
    --coldstart_n $COLD \
    --random_nas_budget 0 \
    --ensemble_k $ENSEMBLE \
    --no_refine \
    --seed $SEED \
    --llm_model "$MODEL" \
    --api_key "$API_KEY"

cp "${EXP_DIR}/ablation_propose/summary.json" "${EXP_DIR}/summary_ablation_propose.json" 2>/dev/null || true

echo ""
echo "================================================================"
echo "ВСЕ ЭКСПЕРИМЕНТЫ ЗАВЕРШЕНЫ"
echo ""
echo "Результаты:"
echo "  $EXP_DIR/summary_biomedical.json          ← ECG5000 + HAR  (EEG исключён)"
echo "  $EXP_DIR/summary_finance.json             ← ELEC2"
echo "  $EXP_DIR/summary_main.json                ← jannis + helena + pol  (MiniBooNE исключён)"
echo "  $EXP_DIR/summary_ablation_no_curves.json  ← curve-blind ablation"
echo "  $EXP_DIR/summary_ablation_propose.json    ← propose vs refine ablation"
echo ""
echo "Просмотр таблицы: python scripts/show_results.py"
echo "================================================================"
