"""
Formal search space for tabular neural architecture search.

This module is the SINGLE SOURCE OF TRUTH for what configurations are
legal. The LLM agent, the random-search baseline, the surrogate model
and the evolution operators all read from here.

Design choices:

- The space is *structured*: a config has top-level "preprocess",
  "arch" (with arch.family), "train" (optimizer, scheduler, regularization).
  This is much closer to how a human ML engineer reasons than a flat
  dict of hyper-parameters, and lets the LLM make architecture-level
  decisions instead of just numeric tweaks.

- Every field has explicit bounds / choice set, used for both validation
  and uniform random sampling. Out-of-bounds values from the LLM are
  clamped (not rejected), so a single bad number never wastes a trial.

- Mutation operators are LOCAL: change one (rarely two) fields in a
  parent config. This is what AmoebaNet / Regularized Evolution does and
  what we ask the LLM to do. Crossover swaps whole sub-trees
  (preprocess, arch, train) between two parents, in the spirit of
  Puzzle's "library of alternatives".

References:
- Real et al., "Regularized Evolution for Image Classifier Architecture Search", AAAI 2019
- Bingbin et al., "Puzzle: Distillation-Based NAS for Inference-Optimized LLMs", 2024
- Gorishniy et al., "Revisiting Deep Learning Models for Tabular Data", NeurIPS 2021
"""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Choice sets
# ---------------------------------------------------------------------------

ARCH_FAMILIES = ("mlp", "resmlp", "tabm", "ft_transformer", "gated_tab", "autoint")

ACTIVATIONS = ("relu", "gelu", "silu", "leaky_relu")

NORMALIZATIONS = ("batchnorm", "layernorm", "none")

OPTIMIZERS = ("adamw", "adam", "sgd_momentum")

SCHEDULERS = ("cosine", "onecycle", "plateau", "none")

NUM_ENCODERS = ("standard", "quantile", "none")   # "robust" removed: not implemented in Preprocessor
CAT_ENCODERS = ("embedding", "onehot")

BATCH_SIZES = (128, 256, 512, 1024)

# ---------------------------------------------------------------------------
# Per-family architecture defaults (used by random sampler / cold start)
# ---------------------------------------------------------------------------

ARCH_BOUNDS: Dict[str, Dict[str, Any]] = {
    "mlp": {
        "n_layers": (1, 6),
        "hidden": (16, 1024),
        "dropout": (0.0, 0.6),
        "activation": ACTIVATIONS,
        "normalization": NORMALIZATIONS,
        "embedding_dim": (4, 64),
    },
    "resmlp": {
        "n_blocks": (1, 8),
        "block_width": (64, 1024),
        "dropout": (0.0, 0.5),
        "activation": ACTIVATIONS,
        "normalization": NORMALIZATIONS,
        "embedding_dim": (4, 64),
    },
    "ft_transformer": {
        "n_blocks": (1, 6),
        "d_token": (32, 256),       # token / model dim, must be divisible by n_heads
        "n_heads": (2, 8),
        "ffn_factor": (1.0, 4.0),   # FFN hidden = d_token * ffn_factor
        "attn_dropout": (0.0, 0.4),
        "ffn_dropout": (0.0, 0.4),
        "residual_dropout": (0.0, 0.2),
        "activation": ("gelu", "silu", "relu"),
    },
    "gated_tab": {
        # GLU-style gated MLP: h = (W1 x) ⊙ sigmoid(W2 x)
        "n_blocks": (1, 6),
        "block_width": (64, 1024),
        "dropout": (0.0, 0.5),
        "activation": ACTIVATIONS,
        "normalization": NORMALIZATIONS,
        "embedding_dim": (4, 64),
    },
    "autoint": {
        # Multi-head self-attention over feature tokens, lightweight version.
        "n_blocks": (1, 4),
        "d_token": (16, 128),
        "n_heads": (2, 4),
        "attn_dropout": (0.0, 0.3),
        "ffn_factor": (1.0, 3.0),
        "ffn_dropout": (0.0, 3.0),
        "activation": ("relu", "gelu"),
    },
    "tabm": {
        # Shared trunk + K independent heads with per-head affine perturbation.
        "n_blocks": (2, 6),
        "width": (128, 1024),
        "k": (4, 16),               # number of parallel heads
        "dropout": (0.0, 0.45),     # trunk dropout
        "head_dropout": (0.0, 0.30),# per-head dropout for diversity
        "activation": ("gelu", "silu", "relu"),
        "normalization": ("layernorm", "batchnorm"),
        "embedding_dim": (4, 64),
    },
}

PREPROCESS_BOUNDS = {
    "num_encoder": NUM_ENCODERS,
    "cat_encoder": CAT_ENCODERS,
}

TRAIN_BOUNDS: Dict[str, Any] = {
    "optimizer": OPTIMIZERS,
    "lr": (1e-5, 1e-2),
    "weight_decay": (0.0, 1e-1),
    "scheduler": SCHEDULERS,
    "batch_size": BATCH_SIZES,
    "epochs": (10, 200),
    "patience": (3, 30),
    "label_smoothing": (0.0, 0.2),
    "grad_clip": (0.0, 5.0),
    "use_amp": (False, True),
    "feature_noise_std": (0.0, 0.2),  # Gaussian noise on numeric inputs as cheap aug
    "mixup_alpha": (0.0, 0.4),
}

# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def _sample_int(lo: int, hi: int, rng: random.Random) -> int:
    return rng.randint(int(lo), int(hi))

def _sample_log_uniform(lo: float, hi: float, rng: random.Random) -> float:
    if lo <= 0:
        # treat 0 as a special "off" value half the time
        if rng.random() < 0.25:
            return 0.0
        lo = max(lo, 1e-8)
    a, b = np.log(lo), np.log(hi)
    return float(np.exp(rng.uniform(a, b)))

def _sample_uniform(lo: float, hi: float, rng: random.Random) -> float:
    return float(rng.uniform(lo, hi))

def _sample_choice(choices, rng: random.Random):
    return rng.choice(list(choices))

def _round_div(n: int, d: int) -> int:
    """Round n down to nearest multiple of d (>= d)."""
    return max(d, (n // d) * d)

# ---------------------------------------------------------------------------
# Random samplers per arch family
# ---------------------------------------------------------------------------

def _sample_arch_mlp(rng: random.Random) -> Dict[str, Any]:
    b = ARCH_BOUNDS["mlp"]
    n_layers = _sample_int(*b["n_layers"], rng)
    hidden = []
    cur = _sample_int(*b["hidden"], rng)
    for _ in range(n_layers):
        # gentle pyramid: each next layer width within [0.5x, 1.5x] of previous
        cur = int(np.clip(cur * rng.uniform(0.5, 1.5), b["hidden"][0], b["hidden"][1]))
        hidden.append(cur)
    return {
        "family": "mlp",
        "hidden_dims": hidden,
        "activation": _sample_choice(b["activation"], rng),
        "dropout": round(_sample_uniform(*b["dropout"], rng), 4),
        "normalization": _sample_choice(b["normalization"], rng),
        "embedding_dim": _sample_int(*b["embedding_dim"], rng),
    }

def _sample_arch_resmlp(rng: random.Random) -> Dict[str, Any]:
    b = ARCH_BOUNDS["resmlp"]
    return {
        "family": "resmlp",
        "n_blocks": _sample_int(*b["n_blocks"], rng),
        "block_width": _sample_int(*b["block_width"], rng),
        "dropout": round(_sample_uniform(*b["dropout"], rng), 4),
        "activation": _sample_choice(b["activation"], rng),
        "normalization": _sample_choice(b["normalization"], rng),
        "embedding_dim": _sample_int(*b["embedding_dim"], rng),
    }

def _sample_arch_ftt(rng: random.Random) -> Dict[str, Any]:
    b = ARCH_BOUNDS["ft_transformer"]
    n_heads = _sample_choice([2, 4, 8], rng)
    d_token = _round_div(_sample_int(*b["d_token"], rng), n_heads)
    return {
        "family": "ft_transformer",
        "n_blocks": _sample_int(*b["n_blocks"], rng),
        "d_token": d_token,
        "n_heads": n_heads,
        "ffn_factor": round(_sample_uniform(*b["ffn_factor"], rng), 3),
        "attn_dropout": round(_sample_uniform(*b["attn_dropout"], rng), 4),
        "ffn_dropout": round(_sample_uniform(*b["ffn_dropout"], rng), 4),
        "residual_dropout": round(_sample_uniform(*b["residual_dropout"], rng), 4),
        "activation": _sample_choice(b["activation"], rng),
    }

def _sample_arch_gated(rng: random.Random) -> Dict[str, Any]:
    b = ARCH_BOUNDS["gated_tab"]
    return {
        "family": "gated_tab",
        "n_blocks": _sample_int(*b["n_blocks"], rng),
        "block_width": _sample_int(*b["block_width"], rng),
        "dropout": round(_sample_uniform(*b["dropout"], rng), 4),
        "activation": _sample_choice(b["activation"], rng),
        "normalization": _sample_choice(b["normalization"], rng),
        "embedding_dim": _sample_int(*b["embedding_dim"], rng),
    }

def _sample_arch_autoint(rng: random.Random) -> Dict[str, Any]:
    b = ARCH_BOUNDS["autoint"]
    n_heads = _sample_choice([2, 4], rng)
    d_token = _round_div(_sample_int(*b["d_token"], rng), n_heads)
    return {
        "family": "autoint",
        "n_blocks": _sample_int(*b["n_blocks"], rng),
        "d_token": d_token,
        "n_heads": n_heads,
        "attn_dropout": round(_sample_uniform(*b["attn_dropout"], rng), 4),
        "ffn_factor": round(_sample_uniform(*b["ffn_factor"], rng), 3),
        "ffn_dropout": round(_sample_uniform(*b["ffn_dropout"], rng), 4),
        "activation": _sample_choice(b["activation"], rng),
    }

def _sample_arch_tabm(rng: random.Random) -> Dict[str, Any]:
    b = ARCH_BOUNDS["tabm"]
    return {
        "family": "tabm",
        "n_blocks": _sample_int(*b["n_blocks"], rng),
        "width": _sample_int(*b["width"], rng),
        "k": _sample_int(*b["k"], rng),
        "dropout": round(_sample_uniform(*b["dropout"], rng), 4),
        "head_dropout": round(_sample_uniform(*b["head_dropout"], rng), 4),
        "activation": _sample_choice(b["activation"], rng),
        "normalization": _sample_choice(b["normalization"], rng),
        "embedding_dim": _sample_int(*b["embedding_dim"], rng),
    }

_ARCH_SAMPLERS: Dict[str, Callable[[random.Random], Dict[str, Any]]] = {
    "mlp": _sample_arch_mlp,
    "resmlp": _sample_arch_resmlp,
    "tabm": _sample_arch_tabm,
    "ft_transformer": _sample_arch_ftt,
    "gated_tab": _sample_arch_gated,
    "autoint": _sample_arch_autoint,
}

def sample_random_arch(rng: random.Random, family: str | None = None) -> Dict[str, Any]:
    fam = family or _sample_choice(ARCH_FAMILIES, rng)
    return _ARCH_SAMPLERS[fam](rng)

def sample_random_train(rng: random.Random) -> Dict[str, Any]:
    b = TRAIN_BOUNDS
    return {
        "optimizer": _sample_choice(b["optimizer"], rng),
        "lr": round(_sample_log_uniform(*b["lr"], rng), 6),
        "weight_decay": round(_sample_log_uniform(*b["weight_decay"], rng), 6),
        "scheduler": _sample_choice(b["scheduler"], rng),
        "batch_size": _sample_choice(b["batch_size"], rng),
        "epochs": _sample_int(*b["epochs"], rng),
        "patience": _sample_int(*b["patience"], rng),
        "label_smoothing": round(_sample_uniform(*b["label_smoothing"], rng), 4),
        "grad_clip": round(_sample_uniform(*b["grad_clip"], rng), 3),
        "use_amp": rng.random() < 0.5,
        "feature_noise_std": round(_sample_uniform(*b["feature_noise_std"], rng), 4),
        "mixup_alpha": round(_sample_uniform(*b["mixup_alpha"], rng), 4),
    }

def sample_random_preprocess(rng: random.Random) -> Dict[str, Any]:
    return {
        "num_encoder": _sample_choice(PREPROCESS_BOUNDS["num_encoder"], rng),
        "cat_encoder": _sample_choice(PREPROCESS_BOUNDS["cat_encoder"], rng),
    }

def sample_random_config(rng: random.Random, family: str | None = None) -> Dict[str, Any]:
    return {
        "preprocess": sample_random_preprocess(rng),
        "arch": sample_random_arch(rng, family=family),
        "train": sample_random_train(rng),
    }

# ---------------------------------------------------------------------------
# Validation / clamping
# ---------------------------------------------------------------------------

def _clip(v, lo, hi):
    return max(lo, min(hi, v))

def _coerce_choice(v, choices, default):
    if v in choices:
        return v
    return default

def _validate_arch(arch: Dict[str, Any]) -> Dict[str, Any]:
    fam = arch.get("family")
    if fam not in ARCH_FAMILIES:
        raise ValueError(f"arch.family must be one of {ARCH_FAMILIES}, got {fam!r}")
    a = dict(arch)
    if fam == "mlp":
        b = ARCH_BOUNDS["mlp"]
        hd = a.get("hidden_dims") or [128]
        if not isinstance(hd, list) or len(hd) == 0:
            hd = [128]
        hd = [int(_clip(int(h), b["hidden"][0], b["hidden"][1])) for h in hd][: b["n_layers"][1]]
        a["hidden_dims"] = hd
        a["activation"] = _coerce_choice(a.get("activation"), b["activation"], "relu")
        a["dropout"] = float(_clip(float(a.get("dropout", 0.1)), *b["dropout"]))
        a["normalization"] = _coerce_choice(a.get("normalization"), b["normalization"], "batchnorm")
        a["embedding_dim"] = int(_clip(int(a.get("embedding_dim", 16)), *b["embedding_dim"]))
    elif fam == "resmlp":
        b = ARCH_BOUNDS["resmlp"]
        a["n_blocks"] = int(_clip(int(a.get("n_blocks", 3)), *b["n_blocks"]))
        a["block_width"] = int(_clip(int(a.get("block_width", 256)), *b["block_width"]))
        a["dropout"] = float(_clip(float(a.get("dropout", 0.1)), *b["dropout"]))
        a["activation"] = _coerce_choice(a.get("activation"), b["activation"], "gelu")
        a["normalization"] = _coerce_choice(a.get("normalization"), b["normalization"], "layernorm")
        a["embedding_dim"] = int(_clip(int(a.get("embedding_dim", 16)), *b["embedding_dim"]))
    elif fam == "ft_transformer":
        b = ARCH_BOUNDS["ft_transformer"]
        n_heads = int(_clip(int(a.get("n_heads", 4)), *b["n_heads"]))
        # round n_heads to power-of-2-ish allowed values
        if n_heads not in (2, 4, 8):
            n_heads = min((2, 4, 8), key=lambda v: abs(v - n_heads))
        d_token = int(_clip(int(a.get("d_token", 64)), *b["d_token"]))
        d_token = _round_div(d_token, n_heads)
        a["n_heads"] = n_heads
        a["d_token"] = d_token
        a["n_blocks"] = int(_clip(int(a.get("n_blocks", 3)), *b["n_blocks"]))
        a["ffn_factor"] = float(_clip(float(a.get("ffn_factor", 2.0)), *b["ffn_factor"]))
        a["attn_dropout"] = float(_clip(float(a.get("attn_dropout", 0.1)), *b["attn_dropout"]))
        a["ffn_dropout"] = float(_clip(float(a.get("ffn_dropout", 0.1)), *b["ffn_dropout"]))
        a["residual_dropout"] = float(_clip(float(a.get("residual_dropout", 0.0)), *b["residual_dropout"]))
        a["activation"] = _coerce_choice(a.get("activation"), b["activation"], "gelu")
    elif fam == "gated_tab":
        b = ARCH_BOUNDS["gated_tab"]
        a["n_blocks"] = int(_clip(int(a.get("n_blocks", 3)), *b["n_blocks"]))
        a["block_width"] = int(_clip(int(a.get("block_width", 256)), *b["block_width"]))
        a["dropout"] = float(_clip(float(a.get("dropout", 0.1)), *b["dropout"]))
        a["activation"] = _coerce_choice(a.get("activation"), b["activation"], "gelu")
        a["normalization"] = _coerce_choice(a.get("normalization"), b["normalization"], "layernorm")
        a["embedding_dim"] = int(_clip(int(a.get("embedding_dim", 16)), *b["embedding_dim"]))
    elif fam == "autoint":
        b = ARCH_BOUNDS["autoint"]
        n_heads = int(_clip(int(a.get("n_heads", 2)), *b["n_heads"]))
        if n_heads not in (2, 4):
            n_heads = min((2, 4), key=lambda v: abs(v - n_heads))
        d_token = int(_clip(int(a.get("d_token", 32)), *b["d_token"]))
        d_token = _round_div(d_token, n_heads)
        a["n_heads"] = n_heads
        a["d_token"] = d_token
        a["n_blocks"] = int(_clip(int(a.get("n_blocks", 2)), *b["n_blocks"]))
        a["attn_dropout"] = float(_clip(float(a.get("attn_dropout", 0.1)), *b["attn_dropout"]))
        a["ffn_factor"] = float(_clip(float(a.get("ffn_factor", 2.0)), *b["ffn_factor"]))
        a["ffn_dropout"] = float(_clip(float(a.get("ffn_dropout", 0.1)), 0.0, 0.3))
        a["activation"] = _coerce_choice(a.get("activation"), b["activation"], "gelu")
    elif fam == "tabm":
        b = ARCH_BOUNDS["tabm"]
        a["n_blocks"] = int(_clip(int(a.get("n_blocks", 3)), *b["n_blocks"]))
        a["width"] = int(_clip(int(a.get("width", 256)), *b["width"]))
        a["k"] = int(_clip(int(a.get("k", 8)), *b["k"]))
        a["dropout"] = float(_clip(float(a.get("dropout", 0.15)), *b["dropout"]))
        a["head_dropout"] = float(_clip(float(a.get("head_dropout", 0.10)), *b["head_dropout"]))
        a["activation"] = _coerce_choice(a.get("activation"), b["activation"], "gelu")
        a["normalization"] = _coerce_choice(a.get("normalization"), b["normalization"], "layernorm")
        a["embedding_dim"] = int(_clip(int(a.get("embedding_dim", 16)), *b["embedding_dim"]))
    return a

def _validate_train(t: Dict[str, Any]) -> Dict[str, Any]:
    b = TRAIN_BOUNDS
    out = dict(t or {})
    out["optimizer"] = _coerce_choice(out.get("optimizer"), b["optimizer"], "adamw")
    out["lr"] = float(_clip(float(out.get("lr", 3e-4)), *b["lr"]))
    out["weight_decay"] = float(_clip(float(out.get("weight_decay", 1e-4)), *b["weight_decay"]))
    out["scheduler"] = _coerce_choice(out.get("scheduler"), b["scheduler"], "cosine")
    bs = int(out.get("batch_size", 256))
    if bs not in b["batch_size"]:
        bs = min(b["batch_size"], key=lambda v: abs(v - bs))
    out["batch_size"] = bs
    out["epochs"] = int(_clip(int(out.get("epochs", 60)), *b["epochs"]))
    out["patience"] = int(_clip(int(out.get("patience", 8)), *b["patience"]))
    out["label_smoothing"] = float(_clip(float(out.get("label_smoothing", 0.0)), *b["label_smoothing"]))
    out["grad_clip"] = float(_clip(float(out.get("grad_clip", 1.0)), *b["grad_clip"]))
    out["use_amp"] = bool(out.get("use_amp", True))
    out["feature_noise_std"] = float(_clip(float(out.get("feature_noise_std", 0.0)), *b["feature_noise_std"]))
    out["mixup_alpha"] = float(_clip(float(out.get("mixup_alpha", 0.0)), *b["mixup_alpha"]))
    return out

def _validate_preprocess(p: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(p or {})
    out["num_encoder"] = _coerce_choice(out.get("num_encoder"), PREPROCESS_BOUNDS["num_encoder"], "standard")
    out["cat_encoder"] = _coerce_choice(out.get("cat_encoder"), PREPROCESS_BOUNDS["cat_encoder"], "embedding")
    return out

def validate_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Clamp + coerce; never throws on bounds. Throws only on missing required arch.family."""
    out = {
        "preprocess": _validate_preprocess(cfg.get("preprocess", {})),
        "arch": _validate_arch(cfg.get("arch", {"family": "mlp"})),
        "train": _validate_train(cfg.get("train", {})),
    }
    return out

# ---------------------------------------------------------------------------
# Mutation / Crossover
# ---------------------------------------------------------------------------

# Fields safe to mutate in-place via small perturbations
_NUMERIC_MUTATIONS: Dict[str, Tuple[str, Tuple]] = {
    # path -> (kind, args)  ; kind in {"log", "uni", "int", "choice"}
    "train.lr":                 ("log", TRAIN_BOUNDS["lr"]),
    "train.weight_decay":       ("log", TRAIN_BOUNDS["weight_decay"]),
    "train.batch_size":         ("choice", TRAIN_BOUNDS["batch_size"]),
    "train.epochs":             ("int", TRAIN_BOUNDS["epochs"]),
    "train.patience":           ("int", TRAIN_BOUNDS["patience"]),
    "train.label_smoothing":    ("uni", TRAIN_BOUNDS["label_smoothing"]),
    "train.grad_clip":          ("uni", TRAIN_BOUNDS["grad_clip"]),
    "train.use_amp":            ("bool", ()),
    "train.feature_noise_std":  ("uni", TRAIN_BOUNDS["feature_noise_std"]),
    "train.mixup_alpha":        ("uni", TRAIN_BOUNDS["mixup_alpha"]),
    "train.optimizer":          ("choice", TRAIN_BOUNDS["optimizer"]),
    "train.scheduler":          ("choice", TRAIN_BOUNDS["scheduler"]),
    "preprocess.num_encoder":   ("choice", PREPROCESS_BOUNDS["num_encoder"]),
    "preprocess.cat_encoder":   ("choice", PREPROCESS_BOUNDS["cat_encoder"]),
}

def _get(d: Dict, path: str):
    cur = d
    for p in path.split("."):
        cur = cur[p]
    return cur

def _set(d: Dict, path: str, val):
    parts = path.split(".")
    cur = d
    for p in parts[:-1]:
        cur = cur[p]
    cur[parts[-1]] = val

def _perturb_log(v: float, lo: float, hi: float, rng: random.Random) -> float:
    factor = float(np.exp(rng.uniform(-0.7, 0.7)))   # ~ x0.5 .. x2
    return float(_clip(v * factor, lo, hi))

def _perturb_uni(v: float, lo: float, hi: float, rng: random.Random) -> float:
    span = hi - lo
    return float(_clip(v + rng.uniform(-0.15 * span, 0.15 * span), lo, hi))

def _perturb_int(v: int, lo: int, hi: int, rng: random.Random) -> int:
    return int(_clip(v + rng.choice([-2, -1, 1, 2]), lo, hi))

def mutate_random(cfg: Dict[str, Any], rng: random.Random, n_changes: int = 1) -> Dict[str, Any]:
    """Apply a small random mutation to ``cfg`` (without using LLM).

    Used as the "Random Search baseline" mutator and as a fallback if the LLM
    fails to produce a valid mutation.
    """
    out = copy.deepcopy(cfg)
    paths = list(_NUMERIC_MUTATIONS.keys())
    # 30% of the time mutate something inside arch
    if rng.random() < 0.3:
        out["arch"] = _mutate_arch_random(out["arch"], rng)
        n_changes -= 1
    for _ in range(max(0, n_changes)):
        path = rng.choice(paths)
        kind, args = _NUMERIC_MUTATIONS[path]
        v = _get(out, path)
        if kind == "log":
            _set(out, path, _perturb_log(float(v), *args, rng))
        elif kind == "uni":
            _set(out, path, _perturb_uni(float(v), *args, rng))
        elif kind == "int":
            _set(out, path, _perturb_int(int(v), *args, rng))
        elif kind == "choice":
            _set(out, path, _sample_choice(args, rng))
        elif kind == "bool":
            _set(out, path, not bool(v))
    return validate_config(out)

def _mutate_arch_random(arch: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    """Mutate inside one architecture family OR jump to a different family (rare)."""
    fam = arch["family"]
    # 15% chance: jump to a totally different family (exploration)
    if rng.random() < 0.15:
        new_fam = rng.choice([f for f in ARCH_FAMILIES if f != fam])
        return _ARCH_SAMPLERS[new_fam](rng)
    # otherwise: tweak one numeric / categorical field
    out = copy.deepcopy(arch)
    if fam == "mlp":
        op = rng.choice(["resize", "add_layer", "drop_layer", "act", "drop", "norm", "emb"])
        if op == "resize" and out["hidden_dims"]:
            i = rng.randrange(len(out["hidden_dims"]))
            out["hidden_dims"][i] = int(_clip(int(out["hidden_dims"][i] * rng.uniform(0.5, 1.5)),
                                              *ARCH_BOUNDS["mlp"]["hidden"]))
        elif op == "add_layer" and len(out["hidden_dims"]) < ARCH_BOUNDS["mlp"]["n_layers"][1]:
            out["hidden_dims"].append(out["hidden_dims"][-1] if out["hidden_dims"] else 128)
        elif op == "drop_layer" and len(out["hidden_dims"]) > 1:
            out["hidden_dims"].pop()
        elif op == "act":
            out["activation"] = _sample_choice(ACTIVATIONS, rng)
        elif op == "drop":
            out["dropout"] = round(_perturb_uni(out["dropout"], 0.0, 0.6, rng), 4)
        elif op == "norm":
            out["normalization"] = _sample_choice(NORMALIZATIONS, rng)
        elif op == "emb":
            out["embedding_dim"] = int(_clip(out["embedding_dim"] + rng.choice([-4, -2, 2, 4]), 4, 64))
    elif fam in ("resmlp", "gated_tab"):
        b = ARCH_BOUNDS[fam]
        op = rng.choice(["n_blocks", "width", "drop", "act", "norm", "emb"])
        if op == "n_blocks":
            out["n_blocks"] = _perturb_int(out["n_blocks"], *b["n_blocks"], rng)
        elif op == "width":
            out["block_width"] = int(_clip(int(out["block_width"] * rng.uniform(0.5, 1.5)),
                                           *b["block_width"]))
        elif op == "drop":
            out["dropout"] = round(_perturb_uni(out["dropout"], *b["dropout"], rng), 4)
        elif op == "act":
            out["activation"] = _sample_choice(b["activation"], rng)
        elif op == "norm":
            out["normalization"] = _sample_choice(b["normalization"], rng)
        elif op == "emb":
            out["embedding_dim"] = int(_clip(out["embedding_dim"] + rng.choice([-4, -2, 2, 4]),
                                             *b["embedding_dim"]))
    elif fam == "ft_transformer":
        b = ARCH_BOUNDS["ft_transformer"]
        op = rng.choice(["n_blocks", "d_token", "n_heads", "ffn", "attn_drop", "ffn_drop", "act"])
        if op == "n_blocks":
            out["n_blocks"] = _perturb_int(out["n_blocks"], *b["n_blocks"], rng)
        elif op == "d_token":
            new = int(_clip(int(out["d_token"] * rng.choice([0.75, 1.5])), *b["d_token"]))
            out["d_token"] = _round_div(new, out["n_heads"])
        elif op == "n_heads":
            out["n_heads"] = rng.choice([2, 4, 8])
            out["d_token"] = _round_div(out["d_token"], out["n_heads"])
        elif op == "ffn":
            out["ffn_factor"] = round(_perturb_uni(out["ffn_factor"], *b["ffn_factor"], rng), 3)
        elif op == "attn_drop":
            out["attn_dropout"] = round(_perturb_uni(out["attn_dropout"], *b["attn_dropout"], rng), 4)
        elif op == "ffn_drop":
            out["ffn_dropout"] = round(_perturb_uni(out["ffn_dropout"], *b["ffn_dropout"], rng), 4)
        elif op == "act":
            out["activation"] = _sample_choice(b["activation"], rng)
    elif fam == "autoint":
        b = ARCH_BOUNDS["autoint"]
        op = rng.choice(["n_blocks", "d_token", "n_heads", "ffn", "attn_drop", "ffn_drop", "act"])
        if op == "n_blocks":
            out["n_blocks"] = _perturb_int(out["n_blocks"], *b["n_blocks"], rng)
        elif op == "d_token":
            new = int(_clip(int(out["d_token"] * rng.choice([0.75, 1.5])), *b["d_token"]))
            out["d_token"] = _round_div(new, out["n_heads"])
        elif op == "n_heads":
            out["n_heads"] = rng.choice([2, 4])
            out["d_token"] = _round_div(out["d_token"], out["n_heads"])
        elif op == "ffn":
            out["ffn_factor"] = round(_perturb_uni(out["ffn_factor"], *b["ffn_factor"], rng), 3)
        elif op == "attn_drop":
            out["attn_dropout"] = round(_perturb_uni(out["attn_dropout"], *b["attn_dropout"], rng), 4)
        elif op == "ffn_drop":
            out["ffn_dropout"] = round(_perturb_uni(out["ffn_dropout"], *b["ffn_dropout"], rng), 4)
        elif op == "act":
            out["activation"] = _sample_choice(b["activation"], rng)
    elif fam == "tabm":
        b = ARCH_BOUNDS["tabm"]
        op = rng.choice(["n_blocks", "width", "k", "drop", "head_drop", "act", "norm", "emb"])
        if op == "n_blocks":
            out["n_blocks"] = _perturb_int(out["n_blocks"], *b["n_blocks"], rng)
        elif op == "width":
            out["width"] = int(_clip(int(out.get("width", 256) * rng.uniform(0.6, 1.6)), *b["width"]))
        elif op == "k":
            out["k"] = int(_clip(out.get("k", 8) + rng.choice([-4, -2, 2, 4]), *b["k"]))
        elif op == "drop":
            out["dropout"] = round(_perturb_uni(out.get("dropout", 0.15), *b["dropout"], rng), 4)
        elif op == "head_drop":
            out["head_dropout"] = round(_perturb_uni(out.get("head_dropout", 0.10),
                                                      *b["head_dropout"], rng), 4)
        elif op == "act":
            out["activation"] = _sample_choice(b["activation"], rng)
        elif op == "norm":
            out["normalization"] = _sample_choice(b["normalization"], rng)
        elif op == "emb":
            out["embedding_dim"] = int(_clip(out.get("embedding_dim", 16) + rng.choice([-4, -2, 2, 4]),
                                             *b["embedding_dim"]))
    return out

def crossover(parent_a: Dict[str, Any], parent_b: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    """Sub-tree crossover: independently sample {preprocess, arch, train} from either parent."""
    child = {
        "preprocess": copy.deepcopy(parent_a["preprocess"] if rng.random() < 0.5 else parent_b["preprocess"]),
        "arch":       copy.deepcopy(parent_a["arch"]       if rng.random() < 0.5 else parent_b["arch"]),
        "train":      copy.deepcopy(parent_a["train"]      if rng.random() < 0.5 else parent_b["train"]),
    }
    return validate_config(child)

# ---------------------------------------------------------------------------
# Featurization for surrogate model
# ---------------------------------------------------------------------------

def featurize_config(cfg: Dict[str, Any]) -> Dict[str, float]:
    """Flatten a config into a (mostly) numeric feature dict for the surrogate.

    Categorical fields become one-hot prefixed with ``onehot::``. Numeric
    fields are passed through (with logs for log-scale fields). Architecture-
    family-specific fields are present iff that family is selected (others
    set to 0). The surrogate ignores any keys it didn't see at fit time.
    """
    feats: Dict[str, float] = {}

    # Preprocess one-hots
    for k in PREPROCESS_BOUNDS["num_encoder"]:
        feats[f"onehot::pp.num={k}"] = float(cfg["preprocess"]["num_encoder"] == k)
    for k in PREPROCESS_BOUNDS["cat_encoder"]:
        feats[f"onehot::pp.cat={k}"] = float(cfg["preprocess"]["cat_encoder"] == k)

    # Train numerics + one-hots
    t = cfg["train"]
    feats["train.lr_log"] = float(np.log10(max(t["lr"], 1e-9)))
    feats["train.wd_log"] = float(np.log10(max(t["weight_decay"], 1e-9)))
    feats["train.bs"] = float(t["batch_size"])
    feats["train.epochs"] = float(t["epochs"])
    feats["train.patience"] = float(t["patience"])
    feats["train.label_smoothing"] = float(t["label_smoothing"])
    feats["train.grad_clip"] = float(t["grad_clip"])
    feats["train.use_amp"] = float(t["use_amp"])
    feats["train.feature_noise_std"] = float(t["feature_noise_std"])
    feats["train.mixup_alpha"] = float(t["mixup_alpha"])
    for k in OPTIMIZERS:
        feats[f"onehot::opt={k}"] = float(t["optimizer"] == k)
    for k in SCHEDULERS:
        feats[f"onehot::sched={k}"] = float(t["scheduler"] == k)

    # Arch family one-hot
    a = cfg["arch"]
    fam = a["family"]
    for k in ARCH_FAMILIES:
        feats[f"onehot::fam={k}"] = float(fam == k)

    # Family-specific numerics (zeros if not active)
    if fam == "mlp":
        feats["arch.n_layers"] = float(len(a["hidden_dims"]))
        feats["arch.mean_hidden"] = float(np.mean(a["hidden_dims"]) if a["hidden_dims"] else 0)
        feats["arch.max_hidden"] = float(np.max(a["hidden_dims"]) if a["hidden_dims"] else 0)
        feats["arch.dropout"] = float(a["dropout"])
        feats["arch.embedding_dim"] = float(a["embedding_dim"])
    elif fam in ("resmlp", "gated_tab"):
        feats["arch.n_blocks"] = float(a["n_blocks"])
        feats["arch.block_width"] = float(a["block_width"])
        feats["arch.dropout"] = float(a["dropout"])
        feats["arch.embedding_dim"] = float(a["embedding_dim"])
    elif fam == "ft_transformer":
        feats["arch.n_blocks"] = float(a["n_blocks"])
        feats["arch.d_token"] = float(a["d_token"])
        feats["arch.n_heads"] = float(a["n_heads"])
        feats["arch.ffn_factor"] = float(a["ffn_factor"])
        feats["arch.attn_dropout"] = float(a["attn_dropout"])
        feats["arch.ffn_dropout"] = float(a["ffn_dropout"])
        feats["arch.residual_dropout"] = float(a["residual_dropout"])
    elif fam == "autoint":
        feats["arch.n_blocks"] = float(a["n_blocks"])
        feats["arch.d_token"] = float(a["d_token"])
        feats["arch.n_heads"] = float(a["n_heads"])
        feats["arch.ffn_factor"] = float(a["ffn_factor"])
        feats["arch.attn_dropout"] = float(a["attn_dropout"])
        feats["arch.ffn_dropout"] = float(a["ffn_dropout"])
    elif fam == "tabm":
        feats["arch.n_blocks"] = float(a["n_blocks"])
        feats["arch.block_width"] = float(a.get("width", 256))
        feats["arch.dropout"] = float(a["dropout"])
        feats["arch.head_dropout"] = float(a.get("head_dropout", 0.10))
        feats["arch.tabm_k"] = float(a.get("k", 8))
        feats["arch.embedding_dim"] = float(a["embedding_dim"])

    return feats

# ---------------------------------------------------------------------------
# Distance (for diversity penalty)
# ---------------------------------------------------------------------------

def config_distance(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    """Cheap, family-aware distance in [0, 1]. 0.5 for cross-family; otherwise
    normalized field-wise distance averaged."""
    if a["arch"]["family"] != b["arch"]["family"]:
        return 1.0
    fa = featurize_config(a)
    fb = featurize_config(b)
    keys = set(fa) | set(fb)
    s = 0.0
    n = 0
    for k in keys:
        va = fa.get(k, 0.0)
        vb = fb.get(k, 0.0)
        denom = max(1.0, abs(va) + abs(vb))
        s += abs(va - vb) / denom
        n += 1
    return float(s / max(n, 1))

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

def schema_for_prompt() -> Dict[str, Any]:
    """A compact, human-readable schema we paste into the LLM prompt."""
    return {
        "preprocess": {
            "num_encoder": list(PREPROCESS_BOUNDS["num_encoder"]),
            "cat_encoder": list(PREPROCESS_BOUNDS["cat_encoder"]),
        },
        "arch": {
            "family": list(ARCH_FAMILIES),
            "_per_family": {
                "mlp": {
                    "hidden_dims": "list[int], 1..6 layers, each in [16, 1024]",
                    "activation": list(ACTIVATIONS),
                    "dropout": "float in [0.0, 0.6]",
                    "normalization": list(NORMALIZATIONS),
                    "embedding_dim": "int in [4, 64]",
                },
                "resmlp": {
                    "n_blocks": "int in [1, 8]",
                    "block_width": "int in [64, 1024]",
                    "dropout": "float in [0.0, 0.5]",
                    "activation": list(ACTIVATIONS),
                    "normalization": list(NORMALIZATIONS),
                    "embedding_dim": "int in [4, 64]",
                },
                "ft_transformer": {
                    "n_blocks": "int in [1, 6]",
                    "d_token": "int in [32, 256], must be divisible by n_heads",
                    "n_heads": "one of {2, 4, 8}",
                    "ffn_factor": "float in [1.0, 4.0]",
                    "attn_dropout": "float in [0.0, 0.4]",
                    "ffn_dropout": "float in [0.0, 0.4]",
                    "residual_dropout": "float in [0.0, 0.2]",
                    "activation": ["gelu", "silu", "relu"],
                },
                "gated_tab": {
                    "n_blocks": "int in [1, 6]",
                    "block_width": "int in [64, 1024]",
                    "dropout": "float in [0.0, 0.5]",
                    "activation": list(ACTIVATIONS),
                    "normalization": list(NORMALIZATIONS),
                    "embedding_dim": "int in [4, 64]",
                },
                "autoint": {
                    "n_blocks": "int in [1, 4]",
                    "d_token": "int in [16, 128], divisible by n_heads",
                    "n_heads": "one of {2, 4}",
                    "attn_dropout": "float in [0.0, 0.3]",
                    "ffn_factor": "float in [1.0, 3.0]",
                    "ffn_dropout": "float in [0.0, 0.3]",
                    "activation": ["relu", "gelu"],
                },
                "tabm": {
                    "n_blocks": "int in [2, 6]  (shared trunk depth)",
                    "width": "int in [128, 1024]  (trunk hidden dim)",
                    "k": "int in [4, 16]  (number of parallel heads — internal ensemble)",
                    "dropout": "float in [0.0, 0.45]  (trunk dropout)",
                    "head_dropout": "float in [0.0, 0.30]  (per-head dropout for diversity)",
                    "activation": ["gelu", "silu", "relu"],
                    "normalization": ["layernorm", "batchnorm"],
                    "embedding_dim": "int in [4, 64]",
                },
            },
        },
        "train": {
            "optimizer": list(OPTIMIZERS),
            "lr": "float in [1e-5, 1e-2], log-uniform",
            "weight_decay": "float in [0.0, 1e-1], log-uniform",
            "scheduler": list(SCHEDULERS),
            "batch_size": list(BATCH_SIZES),
            "epochs": "int in [10, 200]",
            "patience": "int in [3, 30]",
            "label_smoothing": "float in [0.0, 0.2]  (classification only)",
            "grad_clip": "float in [0.0, 5.0]  (0 disables)",
            "use_amp": "bool",
            "feature_noise_std": "float in [0.0, 0.2]  (Gaussian noise on numeric inputs)",
            "mixup_alpha": "float in [0.0, 0.4]",
        },
    }
