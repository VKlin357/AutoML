"""
Structured action vocabulary for LLM-guided architecture refinement.

Instead of regenerating entire configs from scratch, the LLM picks ONE
semantic action per iteration and applies it to the current best config.
This makes the search interpretable: we can log *which* actions the LLM
called, count their frequency, and correlate them with metric improvement.

Thesis narrative: "The LLM acts as an ML engineer who diagnoses the
learning curve and applies targeted fixes — e.g. it called
increase_dropout 7 times when overfitting was detected, and reduce_lr 4
times when the loss oscillated."

Actions are family-aware: apply_action() handles MLP, ResMLP,
FT-Transformer, GatedTab, and AutoInt correctly.
"""
from __future__ import annotations

import copy
import random
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Action catalogue
# ---------------------------------------------------------------------------

ACTION_NAMES: List[str] = [
    # --- Local incremental actions (now stochastic — see Prong C / Bug fix #7) ---
    "add_layer",             # +1 block/layer — combat underfitting
    "remove_layer",          # -1 block/layer — combat overfitting / speed up
    "widen",                 # increase hidden / block_width ×rng.uniform(1.3, 1.8)
    "narrow",                # decrease hidden / block_width ×rng.uniform(0.55, 0.78)
    "increase_dropout",      # +dropout (rng.uniform(0.05, 0.20)) — combat overfitting
    "decrease_dropout",      # -dropout — combat underfitting
    "add_normalization",     # none → batchnorm or layernorm
    "remove_normalization",  # batchnorm/layernorm → none (if over-regularised)
    "reduce_lr",             # lr ×rng.uniform(0.3, 0.7)
    "increase_lr",           # lr ×rng.uniform(1.5, 3.0)
    "add_regularization",    # weight_decay ×rng.uniform(1.5, 3.0)
    "reduce_regularization", # weight_decay ×rng.uniform(0.15, 0.4)
    "change_activation",     # try different activation
    "add_augmentation",      # increase feature_noise_std or mixup_alpha
    "change_scheduler",      # switch lr scheduler
    "switch_family",         # jump to different arch family (exploration)
    # --- Prong E (NEW): explicit extreme-corner exploration actions ---
    # These solve the problem that LLM gravitates to "textbook" lr ~1e-3
    # but tabular winners often live at extreme lr (1e-5..1e-4 or 3e-3..3e-2).
    "explore_lr_extreme_low",   # set lr ∈ [1e-5, 1e-4] (random log-uniform)
    "explore_lr_extreme_high",  # set lr ∈ [3e-3, 3e-2] (random log-uniform)
    "explore_deep",             # set depth = rng.choice([5, 6, 7, 8])
    "explore_shallow",          # set depth = 1 or 2
    # --- Mode 2 (refine_extended): actions LLM frequently asks for ---
    # Empirical: in v4 logs LLM mentioned "more epochs" 11 times and
    # "preprocessing" 12 times but had no way to make those changes.
    "extend_training",          # epochs ×rng.uniform(1.3, 1.7), patience ×1.5
    "change_preprocessing",     # cycle preprocess.num_encoder
]

# Human-readable description used in the LLM prompt
ACTION_DESCRIPTIONS: Dict[str, str] = {
    "add_layer":             "Add one layer/block (random size around current). Use when train loss is still decreasing (underfitting).",
    "remove_layer":          "Remove one layer/block. Use when model is overfitting or training is unstable.",
    "widen":                 "Increase hidden dim / block width by ×1.3..1.8 (random). Use for underfitting on large datasets.",
    "narrow":                "Decrease hidden dim by ×0.55..0.78 (random). Use for overfitting on small datasets.",
    "increase_dropout":      "Raise dropout rate by +0.05..+0.20 (random). Use when val loss diverges from train loss (overfitting).",
    "decrease_dropout":      "Lower dropout rate by 0.05..0.15. Use when train loss is high and not converging (underfitting).",
    "add_normalization":     "Switch normalization to batchnorm or layernorm. Stabilises training.",
    "remove_normalization":  "Remove normalization. Use if training is over-regularised.",
    "reduce_lr":             "Multiply lr by 0.3..0.7 (random). Use when loss oscillates, val plateaus 3+ epochs, or you suspect lr is too high for fine convergence (very common on tabular).",
    "increase_lr":           "Multiply lr by 1.5..3.0 (random). Use ONLY when convergence is extremely slow AND you have not used this action recently.",
    "add_regularization":    "Multiply weight_decay by 1.5..3.0 (random). Use when val loss keeps rising after peak.",
    "reduce_regularization": "Multiply weight_decay by 0.15..0.4 (random). Use when train loss is stuck high (underfitting).",
    "change_activation":     "Switch activation function (relu/gelu/silu/leaky_relu).",
    "add_augmentation":      "Increase feature_noise_std or mixup_alpha (random). Use for overfitting.",
    "change_scheduler":      "Switch lr scheduler (cosine/onecycle/plateau/none).",
    "switch_family":         "Change architecture family entirely. Use when current family seems stuck OR for explicit exploration.",
    # New extreme-corner actions
    "explore_lr_extreme_low":  "AGGRESSIVE: jump lr to a random value in [1e-5, 1e-4]. Use when standard lr ~1e-3 has plateaued and you suspect tabular fine-tuning regime.",
    "explore_lr_extreme_high": "AGGRESSIVE: jump lr to a random value in [3e-3, 3e-2]. Use when standard lr is too low (slow convergence) and you want a regime change.",
    "explore_deep":            "AGGRESSIVE: set depth/n_blocks=5..8. Use to escape shallow-architecture local optima.",
    "explore_shallow":         "AGGRESSIVE: set depth/n_blocks=1..2. Use to escape over-deep models that overfit on small tabular data.",
    "extend_training":         "Train longer: epochs ×1.3..1.7, patience ×1.5. Use when val_primary is still climbing at end of training.",
    "change_preprocessing":    "Switch num_encoder (standard ↔ quantile ↔ none). Use when LLM suspects data scale affects training.",
}

# Grouped for prompt display
ACTION_GROUPS = {
    "Capacity":          ["add_layer", "remove_layer", "widen", "narrow"],
    "Regularisation":    ["increase_dropout", "decrease_dropout", "add_normalization",
                          "remove_normalization", "add_regularization", "reduce_regularization",
                          "add_augmentation"],
    "Training dynamics": ["reduce_lr", "increase_lr", "change_scheduler"],
    "Architecture":      ["change_activation", "switch_family"],
    "Extreme exploration (use when stuck OR after 2+ wasted local actions)":
                         ["explore_lr_extreme_low", "explore_lr_extreme_high",
                          "explore_deep", "explore_shallow"],
    "Training control (Mode 2 — new in v6)":
                         ["extend_training", "change_preprocessing"],
}


# ---------------------------------------------------------------------------
# Compat shim – avoid circular import
# ---------------------------------------------------------------------------

def _validate(cfg):
    """Deferred import of validate_config to avoid circular imports."""
    from ..search_space import validate_config as _vc
    return _vc(cfg)


def _sample_choice(choices, rng=None):
    if rng is None:
        return random.choice(list(choices))
    return rng.choice(list(choices))


# ---------------------------------------------------------------------------
# Per-family helpers
# ---------------------------------------------------------------------------

def _apply_mlp(arch: Dict, action: str, params: Dict, rng: random.Random) -> Dict:
    """Bug-fix #7 (Prong C): all numeric perturbations use rng.uniform()
    so that the same parent + same action produces a *different* child every
    time → eliminates seen_hashes dedup-fallback that wasted 50-80% of LLM
    actions in v1/v2/v3 experiments.
    """
    a = copy.deepcopy(arch)
    dims = list(a.get("hidden_dims", [128]))
    if action == "add_layer":
        ref = dims[-1] if dims else 128
        # add a layer with random size near current last layer
        new_size = max(16, int(ref * rng.uniform(0.7, 1.3)))
        dims.append(new_size)
    elif action == "remove_layer" and len(dims) > 1:
        # remove a random layer (not always last) for diversity
        idx = rng.randrange(len(dims))
        dims.pop(idx)
    elif action == "widen":
        factor = rng.uniform(1.3, 1.8)
        dims = [min(1024, int(d * factor)) for d in dims]
    elif action == "narrow":
        factor = rng.uniform(0.55, 0.78)
        dims = [max(16, int(d * factor)) for d in dims]
    elif action == "explore_deep":
        # set depth to 5..8 — aggressive exploration
        target_depth = rng.choice([5, 6, 7, 8])
        ref = dims[-1] if dims else 128
        dims = [max(64, int(ref * rng.uniform(0.7, 1.3))) for _ in range(target_depth)]
    elif action == "explore_shallow":
        target_depth = rng.choice([1, 2])
        # for very shallow MLP, make it wide
        ref = dims[-1] if dims else 128
        dims = [max(128, int(ref * rng.uniform(1.5, 2.5))) for _ in range(target_depth)]
    a["hidden_dims"] = dims
    return a


def _apply_resmlp_gated(arch: Dict, action: str, params: Dict, rng: random.Random) -> Dict:
    """Stochastic. See _apply_mlp docstring."""
    a = copy.deepcopy(arch)
    if action == "add_layer":
        a["n_blocks"] = min(8, a.get("n_blocks", 2) + rng.choice([1, 1, 2]))
    elif action == "remove_layer":
        a["n_blocks"] = max(1, a.get("n_blocks", 2) - 1)
    elif action == "widen":
        factor = rng.uniform(1.3, 1.8)
        a["block_width"] = min(1024, int(a.get("block_width", 256) * factor))
    elif action == "narrow":
        factor = rng.uniform(0.55, 0.78)
        a["block_width"] = max(64, int(a.get("block_width", 256) * factor))
    elif action == "explore_deep":
        a["n_blocks"] = rng.choice([5, 6, 7, 8])
    elif action == "explore_shallow":
        a["n_blocks"] = rng.choice([1, 2])
        # wider shallow networks compensate
        a["block_width"] = min(1024, int(a.get("block_width", 256) * rng.uniform(1.3, 1.8)))
    return a


def _apply_transformer(arch: Dict, action: str, params: Dict, rng: random.Random) -> Dict:
    """Stochastic. See _apply_mlp docstring."""
    a = copy.deepcopy(arch)
    from ..search_space import _round_div
    if action == "add_layer":
        a["n_blocks"] = min(6, a.get("n_blocks", 2) + 1)
    elif action == "remove_layer":
        a["n_blocks"] = max(1, a.get("n_blocks", 2) - 1)
    elif action == "widen":
        factor = rng.uniform(1.3, 1.8)
        new = min(256, int(a.get("d_token", 64) * factor))
        a["d_token"] = _round_div(new, a.get("n_heads", 4))
    elif action == "narrow":
        factor = rng.uniform(0.55, 0.78)
        new = max(32, int(a.get("d_token", 64) * factor))
        a["d_token"] = _round_div(new, a.get("n_heads", 4))
    elif action == "increase_dropout":
        delta = rng.uniform(0.05, 0.20)
        a["attn_dropout"] = round(min(0.4, a.get("attn_dropout", 0.0) + delta), 4)
        a["ffn_dropout"] = round(min(0.4, a.get("ffn_dropout", 0.0) + delta), 4)
    elif action == "decrease_dropout":
        delta = rng.uniform(0.05, 0.15)
        a["attn_dropout"] = round(max(0.0, a.get("attn_dropout", 0.1) - delta), 4)
        a["ffn_dropout"] = round(max(0.0, a.get("ffn_dropout", 0.1) - delta), 4)
    elif action == "explore_deep":
        a["n_blocks"] = rng.choice([5, 6])
    elif action == "explore_shallow":
        a["n_blocks"] = rng.choice([1, 2])
    return a


def _apply_tabm(arch: Dict, action: str, params: Dict, rng: random.Random) -> Dict:
    """Capacity actions for TabM (shared trunk + K heads)."""
    a = copy.deepcopy(arch)
    if action == "add_layer":
        a["n_blocks"] = min(6, int(a.get("n_blocks", 3)) + rng.choice([1, 1, 2]))
    elif action == "remove_layer":
        a["n_blocks"] = max(2, int(a.get("n_blocks", 3)) - 1)
    elif action == "widen":
        factor = rng.uniform(1.25, 1.75)
        a["width"] = min(1024, int(a.get("width", 256) * factor))
    elif action == "narrow":
        factor = rng.uniform(0.55, 0.80)
        a["width"] = max(128, int(a.get("width", 256) * factor))
    elif action == "explore_deep":
        a["n_blocks"] = rng.choice([5, 6])
        a["width"] = min(1024, max(256, int(a.get("width", 256) * rng.uniform(1.1, 1.5))))
    elif action == "explore_shallow":
        a["n_blocks"] = 2
        # wider trunk compensates for fewer blocks
        a["width"] = min(1024, max(384, int(a.get("width", 256) * rng.uniform(1.4, 2.0))))
    elif action == "increase_dropout":
        delta = rng.uniform(0.05, 0.15)
        a["dropout"] = round(min(0.45, a.get("dropout", 0.15) + delta), 4)
        a["head_dropout"] = round(min(0.30, a.get("head_dropout", 0.10) + delta * 0.5), 4)
    elif action == "decrease_dropout":
        delta = rng.uniform(0.05, 0.12)
        a["dropout"] = round(max(0.0, a.get("dropout", 0.15) - delta), 4)
        a["head_dropout"] = round(max(0.0, a.get("head_dropout", 0.10) - delta * 0.5), 4)
    return a


def _apply_autoint(arch: Dict, action: str, params: Dict, rng: random.Random) -> Dict:
    """Stochastic. See _apply_mlp docstring."""
    a = copy.deepcopy(arch)
    from ..search_space import _round_div
    if action == "add_layer":
        a["n_blocks"] = min(4, a.get("n_blocks", 2) + 1)
    elif action == "remove_layer":
        a["n_blocks"] = max(1, a.get("n_blocks", 2) - 1)
    elif action in ("widen", "narrow"):
        factor = rng.uniform(1.3, 1.8) if action == "widen" else rng.uniform(0.55, 0.78)
        new = int(a.get("d_token", 32) * factor)
        new = max(16, min(128, new))
        a["d_token"] = _round_div(new, a.get("n_heads", 2))
    elif action == "increase_dropout":
        delta = rng.uniform(0.05, 0.15)
        a["attn_dropout"] = round(min(0.3, a.get("attn_dropout", 0.0) + delta), 4)
    elif action == "decrease_dropout":
        delta = rng.uniform(0.05, 0.10)
        a["attn_dropout"] = round(max(0.0, a.get("attn_dropout", 0.1) - delta), 4)
    elif action == "explore_deep":
        a["n_blocks"] = 4  # autoint capped at 4
    elif action == "explore_shallow":
        a["n_blocks"] = 1
    return a


# ---------------------------------------------------------------------------
# Main apply function
# ---------------------------------------------------------------------------

def apply_action(
    cfg: Dict[str, Any],
    action: str,
    params: Optional[Dict[str, Any]] = None,
    rng: Optional[random.Random] = None,
) -> Dict[str, Any]:
    """Apply a named action to ``cfg`` and return a new validated config.

    ``params`` can carry optional overrides from the LLM (e.g.
    ``{"target_dropout": 0.3}``). Unknown actions are silently ignored
    and the original config is returned — this prevents a single bad LLM
    response from crashing the search.
    """
    if rng is None:
        rng = random.Random()
    params = params or {}
    out = copy.deepcopy(cfg)
    arch = out["arch"]
    fam = arch["family"]
    train = out["train"]

    # --- Capacity / architecture actions (stochastic in helpers) ---
    if action in ("add_layer", "remove_layer", "widen", "narrow",
                  "explore_deep", "explore_shallow"):
        if fam == "mlp":
            out["arch"] = _apply_mlp(arch, action, params, rng)
        elif fam in ("resmlp", "gated_tab"):
            out["arch"] = _apply_resmlp_gated(arch, action, params, rng)
        elif fam in ("ft_transformer",):
            out["arch"] = _apply_transformer(arch, action, params, rng)
        elif fam == "autoint":
            out["arch"] = _apply_autoint(arch, action, params, rng)
        elif fam == "tabm":
            out["arch"] = _apply_tabm(arch, action, params, rng)

    # --- Dropout actions (Bug-fix #7: rng.uniform deltas) ---
    elif action == "increase_dropout":
        if fam in ("ft_transformer", "autoint"):
            out["arch"] = _apply_transformer(arch, action, params, rng) if fam == "ft_transformer" \
                else _apply_autoint(arch, action, params, rng)
        elif fam == "tabm":
            out["arch"] = _apply_tabm(arch, action, params, rng)
        else:
            target = params.get("target_dropout")
            if target is not None:
                out["arch"]["dropout"] = round(float(min(0.6, max(0.0, target))), 4)
            else:
                cur = arch.get("dropout", 0.1)
                delta = rng.uniform(0.05, 0.20)
                out["arch"]["dropout"] = round(min(0.6, cur + delta), 4)

    elif action == "decrease_dropout":
        if fam in ("ft_transformer", "autoint"):
            out["arch"] = _apply_transformer(arch, action, params, rng) if fam == "ft_transformer" \
                else _apply_autoint(arch, action, params, rng)
        elif fam == "tabm":
            out["arch"] = _apply_tabm(arch, action, params, rng)
        else:
            cur = arch.get("dropout", 0.2)
            delta = rng.uniform(0.05, 0.15)
            out["arch"]["dropout"] = round(max(0.0, cur - delta), 4)

    # --- Normalization actions ---
    elif action == "add_normalization":
        if fam not in ("ft_transformer", "autoint"):
            choice = params.get("normalization", _sample_choice(("batchnorm", "layernorm"), rng))
            out["arch"]["normalization"] = choice

    elif action == "remove_normalization":
        if fam not in ("ft_transformer", "autoint"):
            out["arch"]["normalization"] = "none"

    # --- Learning rate actions (Bug-fix #7: rng.uniform multipliers) ---
    elif action == "reduce_lr":
        factor = rng.uniform(0.3, 0.7)
        out["train"]["lr"] = round(max(1e-6, float(train["lr"]) * factor), 8)

    elif action == "increase_lr":
        factor = rng.uniform(1.5, 3.0)
        out["train"]["lr"] = round(min(3e-2, float(train["lr"]) * factor), 8)

    # --- Regularisation actions (Bug-fix #7: rng.uniform multipliers) ---
    elif action == "add_regularization":
        cur = float(train.get("weight_decay", 1e-4))
        factor = rng.uniform(1.5, 3.0)
        out["train"]["weight_decay"] = round(min(0.1, max(1e-5, cur) * factor), 8)

    elif action == "reduce_regularization":
        cur = float(train.get("weight_decay", 1e-4))
        factor = rng.uniform(0.15, 0.4)
        out["train"]["weight_decay"] = round(max(0.0, cur * factor), 8)

    # --- Augmentation actions (Bug-fix #7: rng.uniform deltas) ---
    elif action == "add_augmentation":
        cur_noise = float(train.get("feature_noise_std", 0.0))
        cur_mixup = float(train.get("mixup_alpha", 0.0))
        out["train"]["feature_noise_std"] = round(min(0.2, cur_noise + rng.uniform(0.02, 0.10)), 4)
        out["train"]["mixup_alpha"] = round(min(0.4, cur_mixup + rng.uniform(0.05, 0.20)), 4)

    # --- Activation action ---
    elif action == "change_activation":
        from ..search_space import ACTIVATIONS
        choices = [a for a in ACTIVATIONS if a != arch.get("activation")]
        if choices:
            target = params.get("activation", _sample_choice(choices, rng))
            if target in ACTIVATIONS:
                out["arch"]["activation"] = target

    # --- Scheduler action ---
    elif action == "change_scheduler":
        from ..search_space import SCHEDULERS
        choices = [s for s in SCHEDULERS if s != train.get("scheduler")]
        target = params.get("scheduler", _sample_choice(choices, rng))
        if target in SCHEDULERS:
            out["train"]["scheduler"] = target

    # --- Switch family with parameter transfer (exploration) ---
    elif action == "switch_family":
        from ..search_space import ARCH_FAMILIES, sample_random_arch
        import math
        choices = [f for f in ARCH_FAMILIES if f != fam]
        target_fam = params.get("family", _sample_choice(choices, rng))
        if target_fam not in ARCH_FAMILIES:
            target_fam = _sample_choice(choices, rng)

        # Extract transferable hyperparams from current arch
        old_depth  = arch.get("n_blocks", len(arch.get("hidden_dims", [None, None, None])))
        old_width  = arch.get("block_width", arch.get("width",
                     arch.get("d_token", arch.get("hidden_dims", [256])[-1])))
        old_drop   = arch.get("dropout", max(
                     arch.get("attn_dropout", 0.1), arch.get("ffn_dropout", 0.1)))
        old_act    = arch.get("activation", "gelu")
        old_emb    = arch.get("embedding_dim", 16)

        if target_fam == "mlp":
            depth = int(max(1, min(8, old_depth)))
            width = int(max(64, min(1024, old_width)))
            new_arch = {
                "family": "mlp",
                "hidden_dims": [width] * depth,
                "dropout": float(max(0.0, min(0.5, old_drop))),
                "activation": old_act if old_act in ("relu", "gelu", "silu", "leaky_relu") else "gelu",
                "normalization": "layernorm",
                "embedding_dim": int(max(4, min(64, old_emb))),
            }
        elif target_fam == "resmlp":
            new_arch = {
                "family": "resmlp",
                "n_blocks": int(max(2, min(8, old_depth))),
                "block_width": int(max(128, min(1024, old_width))),
                "dropout": float(max(0.0, min(0.5, old_drop))),
                "activation": old_act if old_act in ("relu", "gelu", "silu", "leaky_relu") else "gelu",
                "normalization": "layernorm",
                "embedding_dim": int(max(4, min(64, old_emb))),
            }
        elif target_fam == "tabm":
            new_arch = {
                "family": "tabm",
                "n_blocks": int(max(2, min(6, old_depth))),
                "width": int(max(128, min(1024, old_width))),
                "k": rng.choice([4, 8, 12, 16]),
                "dropout": float(max(0.0, min(0.45, old_drop))),
                "head_dropout": round(rng.uniform(0.05, 0.20), 4),
                "activation": "gelu",
                "normalization": "layernorm",
                "embedding_dim": int(max(4, min(64, old_emb))),
            }
        elif target_fam == "ft_transformer":
            d_token = int(max(64, min(256, old_width // 2 if old_width > 128 else old_width)))
            # d_token must be divisible by n_heads
            n_heads = 8 if d_token % 8 == 0 else 4
            new_arch = {
                "family": "ft_transformer",
                "n_blocks": int(max(2, min(6, old_depth))),
                "d_token": d_token,
                "n_heads": n_heads,
                "ffn_factor": 2.0,
                "attn_dropout": float(max(0.0, min(0.35, old_drop * 0.7))),
                "ffn_dropout": float(max(0.0, min(0.35, old_drop))),
                "residual_dropout": 0.05,
                "activation": "gelu",
            }
        elif target_fam == "gated_tab":
            new_arch = {
                "family": "gated_tab",
                "n_blocks": int(max(2, min(8, old_depth))),
                "block_width": int(max(128, min(1024, old_width))),
                "dropout": float(max(0.0, min(0.5, old_drop))),
                "normalization": "layernorm",
                "embedding_dim": int(max(4, min(64, old_emb))),
            }
        elif target_fam == "autoint":
            d_token = int(max(32, min(128, old_width // 4)))
            n_heads = 4 if d_token % 4 == 0 else 2
            new_arch = {
                "family": "autoint",
                "n_blocks": int(max(2, min(6, old_depth))),
                "d_token": d_token,
                "n_heads": n_heads,
                "attn_dropout": float(max(0.0, min(0.35, old_drop))),
                "residual_dropout": 0.05,
                "activation": "gelu",
                "embedding_dim": int(max(4, min(64, old_emb))),
            }
        else:
            new_arch = sample_random_arch(rng, family=target_fam)

        out["arch"] = new_arch

    # --- Prong E NEW: extreme LR exploration (log-uniform sampling) ---
    elif action == "explore_lr_extreme_low":
        # Random lr in [1e-5, 1e-4] log-uniform
        import math
        log_lo, log_hi = math.log10(1e-5), math.log10(1e-4)
        out["train"]["lr"] = round(10 ** rng.uniform(log_lo, log_hi), 8)

    elif action == "explore_lr_extreme_high":
        # Random lr in [3e-3, 3e-2] log-uniform
        import math
        log_lo, log_hi = math.log10(3e-3), math.log10(3e-2)
        out["train"]["lr"] = round(10 ** rng.uniform(log_lo, log_hi), 8)

    # --- Mode 2 NEW: training control actions ---
    elif action == "extend_training":
        # Increase epochs by 1.3-1.7x, patience by 1.5x.
        # Empirically: in v4 LLM said "more epochs" 11 times across datasets
        # but had no way to make this change. Now it can.
        cur_epochs = int(train.get("epochs", 100))
        cur_patience = int(train.get("patience", 10))
        new_epochs = min(300, int(cur_epochs * rng.uniform(1.3, 1.7)))
        new_patience = min(50, int(cur_patience * 1.5))
        out["train"]["epochs"] = new_epochs
        out["train"]["patience"] = new_patience

    elif action == "change_preprocessing":
        # Cycle num_encoder among the available options
        cur = out.get("preprocess", {}).get("num_encoder", "standard")
        choices = ["standard", "quantile", "none"]
        # Random different choice
        others = [c for c in choices if c != cur]
        out.setdefault("preprocess", {})["num_encoder"] = _sample_choice(others, rng)

    return _validate(out)


# ---------------------------------------------------------------------------
# Action statistics helper (for thesis analysis)
# ---------------------------------------------------------------------------

def action_stats(log_records: List[Dict]) -> Dict[str, Any]:
    """Aggregate action call counts and average primary improvement.

    ``log_records`` is the list of dicts from llm_log.jsonl.
    Returns a dict suitable for JSON serialisation.
    """
    from collections import defaultdict
    counts: Dict[str, int] = defaultdict(int)
    improvements: Dict[str, List[float]] = defaultdict(list)

    prev_primary = None
    for rec in log_records:
        action = (rec.get("reasoning") or {}).get("proposed_change", "")
        # logger stores action in reasoning.proposed_change for REFINE calls
        if rec.get("operator") == "refine" and action:
            counts[action] += 1
            p = rec.get("primary_after")
            if p is not None and prev_primary is not None:
                improvements[action].append(p - prev_primary)
        if rec.get("primary_after") is not None:
            prev_primary = rec["primary_after"]

    stats = {}
    for a in ACTION_NAMES:
        n = counts.get(a, 0)
        imps = improvements.get(a, [])
        stats[a] = {
            "count": n,
            "avg_improvement": round(sum(imps) / len(imps), 6) if imps else None,
            "positive_rate": round(sum(1 for x in imps if x > 0) / len(imps), 3) if imps else None,
        }
    return stats
