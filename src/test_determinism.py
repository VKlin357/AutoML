"""
Проверка детерминизма: одна сетка, два запуска, одни параметры.
Сравниваем val_primary / train_loss по каждой эпохе с точностью до eps.

Запуск:
    python3 scripts/test_determinism.py
    python3 scripts/test_determinism.py --eps 1e-5
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from src.train_nn import train_trial

# ── конфиг — одна небольшая сетка ───────────────────────────────────────────
CFG = {
    "preprocess": {"num_encoder": "quantile", "cat_encoder": "embedding"},
    "arch": {
        "family": "resmlp",
        "n_blocks": 3,
        "block_width": 128,
        "dropout": 0.2,
        "activation": "relu",
        "normalization": "batchnorm",
        "embedding_dim": 16,
    },
    "train": {
        "optimizer": "adamw",
        "lr": 0.003,
        "weight_decay": 1e-4,
        "scheduler": "cosine",
        "batch_size": 256,
        "epochs": 15,
        "patience": 15,       # без early stopping — все 15 эпох
        "label_smoothing": 0.05,
        "grad_clip": 1.0,
        "use_amp": False,
        "feature_noise_std": 0.02,
        "mixup_alpha": 0.1,
    },
}

SEED = 42
EPS  = float(sys.argv[sys.argv.index("--eps") + 1]) if "--eps" in sys.argv else 1e-6

# ── данные: Helena с OpenML ──────────────────────────────────────────────────
print("Загружаем Helena (OpenML 41166)...")
ds = fetch_openml(data_id=41166, as_frame=False, parser="auto")
X = ds.data.astype(np.float32)
y = ds.target.astype(int)
# бинаризуем если нужно
if y.min() != 0:
    y = y - y.min()

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

common = dict(
    cfg=CFG,
    X_train_num=X_train, X_train_cat=np.zeros((len(X_train), 0), dtype=np.int64),
    y_train=y_train,
    X_val_num=X_val,   X_val_cat=np.zeros((len(X_val), 0),   dtype=np.int64),
    y_val=y_val,
    task="multiclass",
    n_classes=int(y.max() + 1),
    cat_cardinalities=[],
    seed=SEED,
    verbose=False,
)

# ── два запуска ──────────────────────────────────────────────────────────────
print("Запуск 1...")
r1 = train_trial(**common)
print(f"  final val_primary = {r1.primary:.8f}")

print("Запуск 2...")
r2 = train_trial(**common)
print(f"  final val_primary = {r2.primary:.8f}")

# ── сравнение ────────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"  СРАВНЕНИЕ (eps={EPS})")
print(f"{'='*55}")

ok = True

# final primary
d = abs(r1.primary - r2.primary)
sym = "✅" if d < 1e-9 else "❌"
print(f"\nfinal val_primary:  {r1.primary:.8f}  vs  {r2.primary:.8f}  delta={d:.2e}  {sym}")
if d >= 1e-9:
    ok = False

# val по эпохам
print("\nval_primary по эпохам:")
for e, (v1, v2) in enumerate(zip(r1.history, r2.history)):
    d = abs(v1 - v2)
    sym = "✅" if d < EPS else "❌"
    flag = f"  ← delta={d:.2e}" if d >= EPS else ""
    print(f"  эпоха {e+1:>2}: {v1:.6f}  {v2:.6f}  {sym}{flag}")
    if d >= EPS:
        ok = False

# train loss по эпохам
if r1.train_loss_history and r2.train_loss_history:
    print("\ntrain_loss по эпохам:")
    for e, (l1, l2) in enumerate(zip(r1.train_loss_history, r2.train_loss_history)):
        d = abs(l1 - l2)
        sym = "✅" if d < EPS else "❌"
        flag = f"  ← delta={d:.2e}" if d >= EPS else ""
        print(f"  эпоха {e+1:>2}: {l1:.6f}  {l2:.6f}  {sym}{flag}")
        if d >= EPS:
            ok = False

print(f"\n{'='*55}")
if ok:
    print(f"  ✅ ДЕТЕРМИНИЗМ ПОДТВЕРЖДЁН")
else:
    print(f"  ❌ ДЕТЕРМИНИЗМ НАРУШЕН — есть расхождения > eps={EPS}")
    sys.exit(1)
print(f"{'='*55}\n")
