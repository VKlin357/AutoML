"""
check_env.py — полная проверка всех компонент перед запуском экспериментов.
Проверяет: GPU, OpenAI API, загрузку всех датасетов, обучение модели на каждом,
           LLM cold start, baselines (CatBoost / LightGBM).

Запуск:
  python scripts/check_env.py --api_key sk-proj-...
  export OPENAI_API_KEY=sk-proj-... && python scripts/check_env.py
"""
import argparse, os, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

ap = argparse.ArgumentParser()
ap.add_argument("--api_key", default=os.getenv("OPENAI_API_KEY", ""))
ap.add_argument("--model", default="gpt-4o-mini")
ap.add_argument("--skip_train", action="store_true", help="Пропустить проверку обучения")
args = ap.parse_args()

OK = "✓"; FAIL = "✗"
passed = 0; failed = 0

def check(label, fn):
    global passed, failed
    t0 = time.time()
    try:
        result = fn()
        elapsed = time.time() - t0
        print(f"  {OK}  {label}: {result}  ({elapsed:.1f}s)")
        passed += 1
        return True
    except Exception as e:
        print(f"  {FAIL}  {label}: {e}")
        failed += 1
        return False

print("\n" + "="*60)
print("ENV CHECK")
print("="*60)

# ── GPU ──────────────────────────────────────────────────────
print("\n[GPU]")
import torch
check("CUDA", lambda: f"{'YES' if torch.cuda.is_available() else 'NO — будет CPU (медленно)'}")
if torch.cuda.is_available():
    check("GPU", lambda: torch.cuda.get_device_name(0))
    check("VRAM", lambda: f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

# ── Packages ─────────────────────────────────────────────────
print("\n[Packages]")
for pkg in ["catboost", "lightgbm", "optuna", "openai", "sklearn", "openml"]:
    check(pkg, lambda p=pkg: __import__(p) and "ok")

# ── OpenAI API ───────────────────────────────────────────────
print("\n[OpenAI API]")
if not args.api_key:
    print(f"  {FAIL}  Нет API ключа (передай --api_key или OPENAI_API_KEY)")
    failed += 1
else:
    def _test_api():
        from openai import OpenAI
        c = OpenAI(api_key=args.api_key)
        r = c.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": "Reply with one word: OK"}],
            max_tokens=5,
        )
        return r.choices[0].message.content.strip()
    check(f"API call ({args.model})", _test_api)

# ── Загрузка датасетов ────────────────────────────────────────
print("\n[Загрузка датасетов]")

DATASETS = [
    # (label,            source,    openml_id, builtin_name, task)
    ("MiniBooNE builtin", "builtin", -1,        "miniboonee",  "binary"),
    ("jannis (45021)",    "openml",  45021,      None,          "multiclass"),
    ("helena (41166)",    "openml",  41166,      None,          "multiclass"),
    ("pol    (44156)",    "openml",  44156,      None,          "binary"),
    ("EEG    (1471)",     "openml",  1471,       None,          "binary"),
    ("ECG5000",           "builtin", -1,         "ecg5000",     "multiclass"),
    ("HAR",               "builtin", -1,         "har",         "multiclass"),
    ("ELEC2",             "builtin", -1,         "elec2",       "binary"),
]

loaded = {}
for label, src, oid, bname, task in DATASETS:
    def _load(s=src, o=oid, b=bname, t=task, lbl=label):
        from src.data import load_raw
        raw, _ = load_raw(source=s, openml_id=o, builtin_name=b, task=t, seed=42)
        n = len(raw.X_train) + len(raw.X_val) + len(raw.X_test)
        loaded[lbl] = raw
        return f"n={n}  shape={raw.X_train.shape}"
    check(label, _load)

# ── Обучение модели на каждом датасете (2 эпохи) ─────────────
if not args.skip_train:
    print("\n[Обучение модели — 2 эпохи на каждом датасете]")

    def _train_quick(label):
        from src.data import load_raw
        from src.preprocessing import make_preprocessor
        from src.search_space import sample_random_config
        from src.train_nn import train_trial
        import random

        # Берём уже загруженный датасет
        raw = loaded.get(label)
        if raw is None:
            raise RuntimeError("датасет не загружен")

        rng = random.Random(42)
        cfg = sample_random_config(rng, family="mlp")
        cfg["train"]["epochs"] = 2
        cfg["train"]["patience"] = 2

        pre = make_preprocessor(cfg["preprocess"])
        p = pre.fit_transform(
            raw.X_train, raw.X_val, raw.X_test,
            raw.y_train, raw.y_val, raw.y_test,
            raw.num_cols, raw.cat_cols, raw.task, raw.n_classes,
        )
        res = train_trial(
            cfg=cfg,
            X_train_num=p.X_train_num, X_train_cat=p.X_train_cat, y_train=p.y_train,
            X_val_num=p.X_val_num,   X_val_cat=p.X_val_cat,   y_val=p.y_val,
            task=p.task, n_classes=p.n_classes,
            cat_cardinalities=p.cat_cardinalities,
            seed=42, save_model=False, max_epochs=2,
        )
        return f"primary={res.primary:.4f}  params={sum(p.numel() for p in __import__('itertools').chain()  if False) or '?'}"

    for label, *_ in DATASETS:
        if label in loaded:
            check(f"train {label}", lambda lbl=label: _train_quick(lbl))

# ── CatBoost baseline на одном датасете ──────────────────────
print("\n[CatBoost baseline — MiniBooNE]")
def _catboost():
    from src.run_baselines_core import run_catboost_baseline
    res = run_catboost_baseline(
        openml_id=-1, task="binary", out_dir="/tmp/cb_check",
        seed=42, source="builtin", builtin_name="miniboonee",
    )
    return f"AUC={res['primary']:.4f}"
check("CatBoost quick", _catboost)

# ── LLM cold start (1 конфиг) ────────────────────────────────
print("\n[LLM cold start — 1 конфиг]")
if not args.api_key:
    print(f"  {FAIL}  Нет API ключа — пропуск")
    failed += 1
else:
    def _llm_coldstart():
        from src.llm.client import OpenAILLM, llm_cold_start
        from src.llm.prompts import build_user_payload_cold_start
        from src.search_space import schema_for_prompt
        from src.llm.prompts import SYSTEM_PROMPT
        llm = OpenAILLM(api_key=args.api_key, model=args.model, temperature=0.7)
        dataset_summary = {"name": "MiniBooNE", "task": "binary",
                           "n_rows": 130064, "n_num": 50, "n_cat": 0, "n_classes": 2}
        baseline = {"catboost": 0.985, "lightgbm": 0.983}
        user_prompt = build_user_payload_cold_start(
            n=1,
            dataset_summary=dataset_summary,
            baseline=baseline,
            schema=schema_for_prompt(),
        )
        configs = llm_cold_start(llm, SYSTEM_PROMPT, user_prompt)
        if not configs:
            raise RuntimeError("LLM вернул пустой список")
        fam = configs[0].get("arch", {}).get("family", "?")
        return f"получен 1 конфиг, family={fam}"
    check("LLM cold start", _llm_coldstart)

# ── Итог ─────────────────────────────────────────────────────
print("\n" + "="*60)
total = passed + failed
print(f"ИТОГ: {passed}/{total} прошло  |  {failed} упало")
if failed == 0:
    print("Всё готово к запуску экспериментов ✓")
else:
    print("Исправь ошибки выше перед запуском полного прогона.")
print("="*60 + "\n")
