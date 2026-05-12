"""
Ensemble top-K architectures found by LLM-NAS.

After the NAS search completes we re-train the K best configs at full fidelity
and average their val-set probability predictions.  This is the same trick
AutoGluon uses (weighted ensemble) and typically yields +0.5..2% AUC over
the single best model.

Two weighting schemes:
  - "uniform"  : simple average (robust, no extra fitting)
  - "greedy"   : greedy ensemble selection (Caruana et al. 2004) — iteratively
                 adds the model that maximises ensemble metric on val set.
                 More powerful but needs >=3 good models.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np

from .data import RawDataset
from .metrics import compute_metrics
from .preprocessing import make_preprocessor
from .train_nn import train_trial
from .utils import ensure_dir, save_json

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable row-wise softmax. Ensures probabilities sum to 1."""
    x = np.array(x, dtype=np.float64)
    if x.ndim == 1:
        x = x - x.max()
        e = np.exp(x)
        return e / e.sum()
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)

def _normalize_probas(p: np.ndarray, task: str) -> np.ndarray:
    """Normalize probability arrays from train_trial output.

    IMPORTANT: train_trial already returns softmax/sigmoid probabilities,
    NOT raw logits. Applying softmax again would double-squash the distribution
    and distort ensemble decisions. We only clip and renormalize for safety.
    """
    if p is None:
        return None
    p = np.asarray(p, dtype=np.float64)
    if task == "binary":
        if p.ndim == 2:
            # Shape (n, 2) — take class-1 column (already probability)
            p = p[:, 1]
        # Clip to valid probability range
        return np.clip(p, 1e-9, 1.0 - 1e-9)
    else:
        # Multiclass: shape (n, C) — already softmax from train_trial.
        # Just clip negatives (numerical noise) and renormalize rows.
        p = np.clip(p, 1e-12, 1.0)
        row_sums = p.sum(axis=1, keepdims=True)
        return p / np.where(row_sums == 0, 1.0, row_sums)

def _retrain(cfg: Dict, raw: RawDataset, seed: int, device: Optional[str]) -> Optional[np.ndarray]:
    """Re-train a config at full fidelity; return val probabilities."""
    pre = make_preprocessor(cfg.get("preprocess", {}))
    prepared = pre.fit_transform(
        raw.X_train, raw.X_val, raw.X_test,
        raw.y_train, raw.y_val, raw.y_test,
        raw.num_cols, raw.cat_cols, raw.task, raw.n_classes,
    )
    try:
        res = train_trial(
            cfg=cfg,
            X_train_num=prepared.X_train_num,
            X_train_cat=prepared.X_train_cat,
            y_train=prepared.y_train,
            X_val_num=prepared.X_val_num,
            X_val_cat=prepared.X_val_cat,
            y_val=prepared.y_val,
            task=prepared.task,
            n_classes=prepared.n_classes,
            cat_cardinalities=prepared.cat_cardinalities,
            seed=seed,
            device=device,
            save_model=False,
        )
        # Normalize: val_probas may be raw logits or sigmoid outputs
        return _normalize_probas(res.val_probas, prepared.task)
    except Exception as e:
        print(f"  [Ensemble] retrain failed: {e}")
        return None

def _ensemble_score(probas_list: List[np.ndarray], weights: np.ndarray,
                    y_val: np.ndarray, task: str) -> float:
    """Score weighted ensemble on val set.

    probas_list entries are already normalized (via _normalize_probas):
      binary     → shape (n,)   — probability of class 1
      multiclass → shape (n, C) — proper softmax distribution
    """
    weights = np.array(weights, dtype=np.float64)
    weights = weights / weights.sum()           # ensure weights sum to 1
    avg = np.tensordot(weights, np.stack(probas_list, axis=0), axes=[[0], [0]])
    if task == "binary":
        m = compute_metrics("binary", y_val, y_pred_proba=avg.reshape(-1))
    else:
        # Re-normalize after weighted average (numerical safety)
        row_sums = avg.sum(axis=1, keepdims=True)
        avg = avg / np.where(row_sums == 0, 1.0, row_sums)
        m = compute_metrics("multiclass", y_val, y_pred_proba=avg)
    return m.primary

# ---------------------------------------------------------------------------
# Greedy ensemble selection (Caruana et al. 2004)
# ---------------------------------------------------------------------------

def _greedy_weights(probas_list: List[np.ndarray], y_val: np.ndarray,
                    task: str, n_rounds: int = 50) -> np.ndarray:
    """
    Greedy ensemble: start with uniform weights, repeatedly pick the model
    that maximally improves the ensemble metric when added one more time.
    Returns normalised weights for each model.
    """
    n = len(probas_list)
    counts = np.zeros(n, dtype=np.float64)
    best_score = -1e18

    for _ in range(n_rounds):
        best_i, best_s = -1, -1e18
        for i in range(n):
            trial_counts = counts.copy()
            trial_counts[i] += 1
            w = trial_counts / trial_counts.sum()
            s = _ensemble_score(probas_list, w, y_val, task)
            if s > best_s:
                best_s, best_i = s, i
        if best_s > best_score:
            best_score = best_s
            counts[best_i] += 1
        else:
            break  # no improvement

    if counts.sum() == 0:
        return np.ones(n) / n
    return counts / counts.sum()

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ensemble_top_k(
    all_trials: List[Dict],
    raw: RawDataset,
    *,
    k: int = 5,
    method: str = "greedy",   # "uniform" | "greedy"
    seed: int = 42,
    device: Optional[str] = None,
    out_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Pick top-K unique configs by primary score, re-train each, ensemble.

    Returns dict with keys: primary, method, k, weights, member_scores.
    """
    t0 = time.time()

    # De-duplicate by config — keep best per unique config.
    # Only consider full/medium trials (skip cheap_pruned/error).
    _SKIP_RUNGS = {"cheap_pruned", "medium_pruned", "error"}

    def _candidate_score(rec: Dict) -> float:
        if rec.get("rung") in _SKIP_RUNGS:
            return -1e18
        return float(rec.get("search_score", rec.get("primary", -1e18)))

    seen_cfgs: List[Dict] = []
    seen_keys = set()
    for rec in sorted(all_trials, key=_candidate_score, reverse=True):
        if rec.get("rung") in _SKIP_RUNGS:
            continue
        key = str(rec["config"])
        if key not in seen_keys:
            seen_keys.add(key)
            seen_cfgs.append(rec)
        if len(seen_cfgs) >= k * 2:   # candidate pool
            break

    print(f"\n[Ensemble] Re-training top-{k} configs (method={method}) …")
    probas_list: List[np.ndarray] = []
    member_scores: List[float] = []
    member_cfgs: List[Dict] = []

    for rec in seen_cfgs:
        if len(probas_list) >= k:
            break
        cfg = rec["config"]
        family = cfg.get("arch", {}).get("family", "?")
        print(f"  retrain trial_{rec['trial_id']:03d}  family={family}  "
              f"primary={rec['primary']:.5f}", flush=True)
        p = _retrain(cfg, raw, seed=seed, device=device)
        if p is not None:
            probas_list.append(p)
            member_scores.append(rec["primary"])
            member_cfgs.append(cfg)

    if not probas_list:
        print("[Ensemble] No models retrained successfully.")
        return {"primary": None, "method": method, "k": 0}

    y_val = raw.y_val

    if method == "greedy" and len(probas_list) >= 2:
        weights = _greedy_weights(probas_list, y_val, raw.task)
    else:
        weights = np.ones(len(probas_list)) / len(probas_list)

    final_score = _ensemble_score(probas_list, weights, y_val, raw.task)
    elapsed = time.time() - t0

    result = {
        "primary": float(final_score),
        "method": method,
        "k": len(probas_list),
        "weights": weights.tolist(),
        "member_scores": member_scores,
        "seconds": elapsed,
    }

    print(f"[Ensemble] primary={final_score:.5f}  "
          f"(best_single={max(member_scores):.5f})  {elapsed:.0f}s")

    if out_dir is not None:
        save_json(ensure_dir(out_dir) / "ensemble_result.json", result)

    return result
