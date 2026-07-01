"""
AutoGluon baseline для одного датасета.
Использование:
    python scripts/run_autogluon.py volkert
    python scripts/run_autogluon.py jannis
    python scripts/run_autogluon.py helena
    python scripts/run_autogluon.py adult
"""
import sys, time, shutil, os
import numpy as np
import pandas as pd
sys.path.insert(0, '.')
from src.data import load_raw
from src.metrics import compute_metrics
from src.utils import save_json
from pathlib import Path
from autogluon.tabular import TabularPredictor

DATASETS = {
    "volkert":    (41166, "multiclass", None),
    "jannis":     (45021, "multiclass", None),
    "helena":     (41169, "multiclass", None),
    "adult":      (1590,  "binary",     None),
    "miniboonee": (None,  "binary",     "miniboonee"),
}

name = sys.argv[1]
oid, task, builtin = DATASETS[name]
out_path = Path(f"experiments_v9/autogluon/{name}/autogluon_result.json")

if out_path.exists():
    import json
    r = json.loads(out_path.read_text())
    t = r.get("test_primary", 0)
    if t and t > 0.1:
        print(f"[SKIP] {name} already done: test={t:.5f}")
        sys.exit(0)

if builtin:
    raw, _ = load_raw(source="builtin", builtin_name=builtin, task=task, seed=42)
else:
    raw, _ = load_raw(source="openml", openml_id=oid, task=task, seed=42)

cols = [f"f{i}" for i in range(raw.X_train.shape[1])]
train_df = pd.DataFrame(np.vstack([raw.X_train, raw.X_val]), columns=cols)
train_df["__y__"] = np.concatenate([raw.y_train, raw.y_val])
test_df = pd.DataFrame(raw.X_test, columns=cols)

metric = "roc_auc" if task == "binary" else "accuracy"
ag_path = f"/dev/shm/ag_tmp_{os.getpid()}"
t0 = time.time()

try:
    pred = TabularPredictor(
        label="__y__", eval_metric=metric,
        path=ag_path, verbosity=1
    ).fit(train_df, time_limit=1800)

    proba = pred.predict_proba(test_df)
    if task == "binary":
        arr = proba.iloc[:, 1].values
        m = compute_metrics("binary", raw.y_test, y_pred_proba=arr)
    else:
        m = compute_metrics("multiclass", raw.y_test, y_pred_proba=proba.values)

    result = {
        "test_primary": float(m.primary),
        "metrics": m.metrics,
        "seconds": time.time() - t0,
        "dataset": name,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(out_path, result)
    print(f"{name}: test={m.primary:.5f}  ({result['seconds']:.0f}s)")
finally:
    shutil.rmtree(ag_path, ignore_errors=True)
