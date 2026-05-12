import json
import os
import random
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

def seed_everything(seed: int = 42) -> None:
    """Full determinism — fixes the non-deterministic training bug that caused
    SAME config to produce primary=0.650 in turn7 vs 0.540 in turn8 (helena v6).

    Required pieces (in priority order):
      1. PYTHONHASHSEED       — Python dict/set ordering (affects data loading)
      2. CUBLAS_WORKSPACE_CONFIG — required when use_deterministic_algorithms=True
                                    on CUDA, otherwise it crashes
      3. random/np/torch seeds — basic
      4. cudnn.deterministic + cudnn.benchmark=False — forces cuDNN to pick
                                                       deterministic conv algos
      5. torch.use_deterministic_algorithms(True) — refuse any non-deterministic
                                                    operations (atomicAdd etc)

    Without (4)+(5), the SAME config trained twice on the same data gives
    SIGNIFICANTLY different results (±0.10 on val score) — observed empirically
    on helena turn 7 vs 8 (advisor caught this).
    """
    # 1. Process-level env vars
    os.environ["PYTHONHASHSEED"] = str(seed)
    # CUBLAS_WORKSPACE_CONFIG is MANDATORY when use_deterministic_algorithms(True)
    # is on with CUDA — without it PyTorch raises RuntimeError.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # 2. Basic RNG seeding
    random.seed(seed)
    np.random.seed(seed)

    # 3. PyTorch seeding (CPU + all CUDA devices)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 4. cuDNN settings — disable benchmark, enable deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 5. Hard switch — refuse any non-deterministic ops in PyTorch
    try:
        torch.use_deterministic_algorithms(True)
    except Exception as e:
        # On some PyTorch versions / GPU types this can fail. Log but don't
        # crash — we still have cudnn.deterministic=True which covers most cases.
        print(f"[seed_everything] WARNING: use_deterministic_algorithms failed: {e}")

def now_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_dir(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def to_jsonable(x: Any) -> Any:
    if is_dataclass(x):
        return asdict(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.ndarray,)):
        return x.tolist()
    return x

def save_json(path, obj: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=to_jsonable)

def load_json(path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def getenv(name, default=None):
    v = os.getenv(name)
    return v if v is not None else default
