"""
Forecasting NAS search space.

Defines valid config ranges for each model family and
provides random sampling for warmup phase.
"""
from __future__ import annotations

import random
from typing import Any, Dict, Optional


FORECASTING_FAMILIES = ["linear", "nlinear", "mlp", "patch_mlp", "tcn", "transformer"]

# Family weights for stratified warmup sampling
FAMILY_WEIGHTS = {
    "linear":      2.0,   # DLinear is strong — oversample
    "nlinear":     1.5,
    "mlp":         1.5,
    "patch_mlp":   2.0,   # PatchTST-inspired — promising
    "tcn":         1.5,
    "transformer": 2.0,   # best for complex patterns
}

# Search space bounds per family
SEARCH_SPACE = {
    "linear": {
        "decompose":  [True, False],
        "individual": [True, False],
    },
    "nlinear": {
        "individual": [True, False],
    },
    "mlp": {
        "hidden_size": [128, 256, 512, 1024],
        "n_layers":    [2, 3, 4, 5],
        "dropout":     [0.0, 0.05, 0.1, 0.2, 0.3],
        "activation":  ["relu", "gelu", "silu"],
    },
    "patch_mlp": {
        "patch_size": [8, 16, 24, 32],
        "d_model":    [64, 128, 256],
        "n_layers":   [2, 3, 4],
        "dropout":    [0.0, 0.1, 0.2],
    },
    "tcn": {
        "d_model":     [32, 64, 128, 256],
        "n_layers":    [3, 4, 5, 6],
        "kernel_size": [3, 5, 7],
        "dropout":     [0.0, 0.1, 0.2],
    },
    "transformer": {
        "patch_size": [8, 16, 24],
        "d_model":    [64, 128, 256],
        "n_heads":    [2, 4, 8],
        "n_layers":   [2, 3, 4],
        "dropout":    [0.0, 0.05, 0.1, 0.2],
        "ffn_factor": [2.0, 4.0],
    },
}

# Training hyperparameter space (shared across all families)
TRAIN_SPACE = {
    "lr":            [1e-4, 3e-4, 5e-4, 1e-3, 2e-3, 5e-3],
    "batch_size":    [32, 64, 128, 256],
    "epochs":        [30, 50, 100, 150, 200],
    "patience":      [10, 15, 20],
    "optimizer":     ["adam", "adamw"],
    "weight_decay":  [0.0, 1e-5, 1e-4, 1e-3],
    "scheduler":     ["none", "cosine", "step"],
    "grad_clip":     [0.0, 1.0, 5.0],
    "use_amp":       [True, False],
}


def sample_arch_for_family(rng: random.Random, family: str, lookback: int = 96) -> Dict[str, Any]:
    """Sample arch hyperparameters for a given family."""
    arch = {"family": family}
    for key, choices in SEARCH_SPACE[family].items():
        arch[key] = rng.choice(choices)
    return arch


def sample_random_config(rng: random.Random, lookback: int = 96,
                          family: Optional[str] = None) -> Dict[str, Any]:
    """Sample a random forecasting config."""
    if family is None:
        total_weight = sum(FAMILY_WEIGHTS.values())
        r = rng.random() * total_weight
        cumsum = 0.0
        family = "linear"
        for fam, w in FAMILY_WEIGHTS.items():
            cumsum += w
            if r <= cumsum:
                family = fam
                break

    arch = sample_arch_for_family(rng, family, lookback)

    # Validate patch_size fits in lookback
    if "patch_size" in arch:
        valid_patches = [p for p in SEARCH_SPACE[family]["patch_size"] if lookback % p == 0]
        if valid_patches:
            arch["patch_size"] = rng.choice(valid_patches)
        else:
            arch["patch_size"] = 16 if lookback >= 16 else 8

    # n_heads must divide d_model evenly
    if family == "transformer":
        valid_heads = [h for h in SEARCH_SPACE["transformer"]["n_heads"]
                       if arch["d_model"] % h == 0]
        arch["n_heads"] = rng.choice(valid_heads) if valid_heads else 4

    train = {
        "lr":           rng.choice(TRAIN_SPACE["lr"]),
        "batch_size":   rng.choice(TRAIN_SPACE["batch_size"]),
        "epochs":       rng.choice(TRAIN_SPACE["epochs"]),
        "patience":     rng.choice(TRAIN_SPACE["patience"]),
        "optimizer":    rng.choice(TRAIN_SPACE["optimizer"]),
        "weight_decay": rng.choice(TRAIN_SPACE["weight_decay"]),
        "scheduler":    rng.choice(TRAIN_SPACE["scheduler"]),
        "grad_clip":    rng.choice(TRAIN_SPACE["grad_clip"]),
        "use_amp":      rng.choice(TRAIN_SPACE["use_amp"]),
    }

    return {"arch": arch, "train": train}


def validate_config(cfg: Dict[str, Any], lookback: int = 96) -> Dict[str, Any]:
    """Clamp config values to valid ranges."""
    arch  = cfg.get("arch", {})
    train = cfg.get("train", {})
    family = arch.get("family", "linear")

    if family not in FORECASTING_FAMILIES:
        arch["family"] = "mlp"

    if family in ("patch_mlp", "transformer") and "patch_size" in arch:
        ps = arch["patch_size"]
        if lookback % ps != 0:
            valid = [p for p in [8, 16, 24, 32] if lookback % p == 0]
            arch["patch_size"] = valid[0] if valid else 16

    if family == "transformer":
        dm = arch.get("d_model", 128)
        nh = arch.get("n_heads", 4)
        if dm % nh != 0:
            arch["n_heads"] = 4 if dm % 4 == 0 else 2

    train["lr"] = float(max(1e-5, min(train.get("lr", 1e-3), 1e-1)))
    train["epochs"] = int(max(10, min(train.get("epochs", 100), 500)))
    train["batch_size"] = int(max(16, min(train.get("batch_size", 64), 512)))
    train["patience"] = int(max(5, min(train.get("patience", 15), 50)))

    return {"arch": arch, "train": train}
