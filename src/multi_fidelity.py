"""
Multi-fidelity evaluator.

Three rungs (Hyperband / SeqNAS-style):

   cheap   :   5 epochs, 30% data
   medium  :  20 epochs, 100% data
   full    : config's own ``train.epochs``, 100% data, save model

A trial enters at ``cheap``. If it makes the top-K of the current
population (or beats some quantile threshold), it is *promoted* to the
next rung; otherwise it stops there. The orchestrator decides promotion
policy (here we just provide the building block).

We deliberately use a SHALLOW pipeline here — the heavy lifting is in
``train_nn.train_trial`` which already supports ``max_epochs`` and
``data_frac`` overrides.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .preprocessing import PreparedSplit
from .train_nn import TrialResult, train_trial

# ---------------------------------------------------------------------------
# Rung definitions
# ---------------------------------------------------------------------------

@dataclass
class Rung:
    name: str
    max_epochs: Optional[int]   # None = use cfg's epochs
    data_frac: float            # 0..1
    save_model: bool

CHEAP  = Rung(name="cheap",  max_epochs=5,  data_frac=0.30, save_model=False)
MEDIUM = Rung(name="medium", max_epochs=20, data_frac=1.00, save_model=False)
FULL   = Rung(name="full",   max_epochs=None, data_frac=1.00, save_model=True)

DEFAULT_LADDER: List[Rung] = [CHEAP, MEDIUM, FULL]

@dataclass
class FidelityResult:
    rung: str
    primary: float
    search_score: float               # dense proxy for NAS selection (acc + bal_acc + f1 - logloss)
    metrics: Dict[str, float]
    history: List[float]              # val primary per epoch
    train_loss_history: List[float]
    grad_norm_history: List[float]    # L2 gradient norm per epoch — signals instability / vanishing grads
    n_params: int
    epochs_run: int
    seconds: float
    early_stopped: bool

def evaluate_at_rung(
    cfg: Dict[str, Any],
    prepared: PreparedSplit,
    rung: Rung,
    *,
    out_dir: Optional[Path] = None,
    seed: int = 0,
    device: Optional[str] = None,
    verbose: bool = False,
) -> FidelityResult:
    # Only print per-epoch for full-fidelity runs — cheap rung is too noisy
    epoch_verbose = verbose and rung.name == "full"
    res: TrialResult = train_trial(
        cfg=cfg,
        X_train_num=prepared.X_train_num, X_train_cat=prepared.X_train_cat, y_train=prepared.y_train,
        X_val_num=prepared.X_val_num, X_val_cat=prepared.X_val_cat, y_val=prepared.y_val,
        task=prepared.task, n_classes=prepared.n_classes,
        cat_cardinalities=prepared.cat_cardinalities,
        out_dir=out_dir, max_epochs=rung.max_epochs, data_frac=rung.data_frac,
        seed=seed, save_model=rung.save_model, device=device,
        verbose=epoch_verbose,
    )
    return FidelityResult(
        rung=rung.name,
        primary=res.primary,
        search_score=res.search_score,
        metrics=res.metrics,
        history=res.history,
        train_loss_history=res.train_loss_history or [],
        grad_norm_history=res.grad_norm_history or [],
        n_params=res.n_params,
        epochs_run=res.epochs_run,
        seconds=res.seconds,
        early_stopped=res.early_stopped,
    )

# ---------------------------------------------------------------------------
# Promotion policy
# ---------------------------------------------------------------------------

def successive_halving_threshold(scores: List[float], promote_top_frac: float = 0.5) -> float:
    """Return a primary-metric threshold above which a config gets promoted.

    Used after a few cheap evals are already done to decide whether a new
    trial's cheap score is worth running at the next rung.
    """
    if len(scores) < 4:
        return -float("inf")  # too few comparisons → promote everything early
    q = float(np.quantile(scores, 1.0 - promote_top_frac))
    return q
