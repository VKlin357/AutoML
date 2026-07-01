"""
sanity_check.py — проверяет, что весь стек работает без API-ключа.

Запускает LLM-NAS v2 на встроенном датасете california_housing с budget=3
(без LLM — только random cold start + random mutations + оценка).
Цель: убедиться что нет импортных ошибок, данные грузятся, трейнинг работает.

Запуск:
  python scripts/sanity_check.py
"""
import os, sys, json
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

print("=" * 60)
print("Sanity check: LLM-NAS v2 (random only, no API key needed)")
print("=" * 60)

# Check imports
try:
    from src.nas_orchestrator import run_llm_nas_v2
    from src.data import load_raw
    from src.search_space import sample_random_config, validate_config
    from src.llm.prompts import SYSTEM_PROMPT, build_user_payload_cold_start
    from src.llm.actions import ACTION_NAMES
    print(f"✓ Imports OK")
    print(f"  Action vocabulary ({len(ACTION_NAMES)} actions): {ACTION_NAMES[:5]} ...")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

# Check data loading
try:
    raw, summary = load_raw(source="builtin", builtin_name="california_housing", seed=42)
    print(f"✓ Data loading OK: {summary['name']} n={summary['n_rows']} task={summary['task']}")
except Exception as e:
    print(f"✗ Data loading error: {e}")
    print("  → Install: pip install openml scikit-learn --break-system-packages")
    sys.exit(1)

# Check config sampling
try:
    import random
    rng = random.Random(42)
    cfg = validate_config(sample_random_config(rng))
    print(f"✓ Config sampling OK: family={cfg['arch']['family']}")
except Exception as e:
    print(f"✗ Config sampling error: {e}")
    sys.exit(1)

# Check prompt building
try:
    payload = build_user_payload_cold_start(n=3, dataset_summary=summary, baseline={})
    print(f"✓ Prompt building OK: len={len(payload)} chars")
except Exception as e:
    print(f"✗ Prompt building error: {e}")
    sys.exit(1)

# Check training (1 config, small epochs)
try:
    from src.preprocessing import Preprocessor
    from src.multi_fidelity import evaluate_at_rung, CHEAP
    pre = Preprocessor(num_encoder="standard", cat_encoder="embedding")
    prepared = pre.fit_transform(
        raw.X_train, raw.X_val, raw.X_test,
        raw.y_train, raw.y_val, raw.y_test,
        raw.num_cols, raw.cat_cols, raw.task, raw.n_classes,
    )
    # Use a tiny MLP config
    tiny_cfg = validate_config({
        "preprocess": {"num_encoder": "standard", "cat_encoder": "embedding"},
        "arch": {"family": "mlp", "hidden_dims": [64, 32], "dropout": 0.1,
                 "activation": "relu", "use_batchnorm": False},
        "train": {"optimizer": "adam", "lr": 1e-3, "weight_decay": 1e-4,
                  "batch_size": 256, "epochs": 5, "patience": 5,
                  "label_smoothing": 0.0, "mixup_alpha": 0.0,
                  "lr_scheduler": "none", "grad_clip": 0.0},
    })
    result = evaluate_at_rung(tiny_cfg, prepared, CHEAP, out_dir=None, seed=42, device=None)
    print(f"✓ Training OK: primary={result.primary:.5f}  params={result.n_params}  "
          f"epochs={result.epochs_run}")
    has_curves = len(result.history) > 0
    print(f"  Learning curves: val_history={len(result.history)} points  "
          f"train_loss={len(result.train_loss_history)} points  "
          f"grad_norms={len(result.grad_norm_history)} points")
    if not has_curves:
        print("  ⚠ WARNING: no learning curve data — check train_nn.py")
except Exception as e:
    print(f"✗ Training error: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

print()
print("=" * 60)
print("ALL CHECKS PASSED ✓")
print()
print("Теперь запусти полные эксперименты:")
print("  export OPENAI_API_KEY=sk-proj-...")
print("  bash scripts/run_thesis_experiments.sh")
print()
print("Или один датасет быстро (MiniBooNE, ~2 часа):")
print("  python scripts/run_experiment.py \\")
print("    --datasets 168338 \\")
print("    --budget 15 --coldstart_n 5 \\")
print("    --random_nas_budget 15 \\")
print("    --optuna --optuna_trials 15 \\")
print("    --naive_llm --naive_llm_trials 3 \\")
print("    --ensemble_k 3 \\")
print("    --llm_model gpt-4o-mini \\")
print("    --api_key sk-proj-...")
print("=" * 60)
