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


def _retrain(cfg: Dict, raw: RawDataset, seed: int, device: Optional[str]) -> Optional[Dict[str, Any]]:
    """Re-train a config; return validation and untouched-test predictions."""
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
            X_test_num=prepared.X_test_num,
            X_test_cat=prepared.X_test_cat,
            y_test=prepared.y_test,
            task=prepared.task,
            n_classes=prepared.n_classes,
            cat_cardinalities=prepared.cat_cardinalities,
            seed=seed,
            device=device,
            save_model=False,
        )
        return {
            "val_probas": _normalize_probas(res.val_probas, prepared.task),
            "test_probas": _normalize_probas(res.test_probas, prepared.task),
            "val_primary": res.primary,
            "test_primary": res.test_primary,
            "test_metrics": res.test_metrics,
        }
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


def _ensemble_metrics(probas_list: List[np.ndarray], weights: np.ndarray,
                      y_true: np.ndarray, task: str) -> Dict[str, Any]:
    """Return complete metrics for a fixed weighted ensemble."""
    weights = np.array(weights, dtype=np.float64)
    weights = weights / weights.sum()
    avg = np.tensordot(weights, np.stack(probas_list, axis=0), axes=[[0], [0]])
    if task == "binary":
        m = compute_metrics("binary", y_true, y_pred_proba=avg.reshape(-1))
    else:
        row_sums = avg.sum(axis=1, keepdims=True)
        avg = avg / np.where(row_sums == 0, 1.0, row_sums)
        m = compute_metrics("multiclass", y_true, y_pred_proba=avg)
    return {"primary": float(m.primary), **m.metrics}


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
    val_probas_list: List[np.ndarray] = []
    test_probas_list: List[np.ndarray] = []
    member_scores: List[float] = []
    member_cfgs: List[Dict] = []

    for rec in seen_cfgs:
        if len(val_probas_list) >= k:
            break
        cfg = rec["config"]
        family = cfg.get("arch", {}).get("family", "?")
        print(f"  retrain trial_{rec['trial_id']:03d}  family={family}  "
              f"primary={rec['primary']:.5f}", flush=True)
        predictions = _retrain(cfg, raw, seed=seed, device=device)
        if predictions is not None:
            val_probas_list.append(predictions["val_probas"])
            test_probas_list.append(predictions["test_probas"])
            member_scores.append(rec["primary"])
            member_cfgs.append(cfg)

    if not val_probas_list:
        print("[Ensemble] No models retrained successfully.")
        return {"primary": None, "method": method, "k": 0}

    y_val = raw.y_val

    if method == "greedy" and len(val_probas_list) >= 2:
        weights = _greedy_weights(val_probas_list, y_val, raw.task)
    else:
        weights = np.ones(len(val_probas_list)) / len(val_probas_list)

    val_score = _ensemble_score(val_probas_list, weights, y_val, raw.task)
    test_score = _ensemble_score(test_probas_list, weights, raw.y_test, raw.task)
    val_metrics = _ensemble_metrics(val_probas_list, weights, y_val, raw.task)
    test_metrics = _ensemble_metrics(test_probas_list, weights, raw.y_test, raw.task)
    elapsed = time.time() - t0

    result = {
        "primary": float(test_score),
        "test_primary": float(test_score),
        "val_primary": float(val_score),
        "test_metrics": test_metrics,
        "val_metrics": val_metrics,
        "evaluation_split": "test",
        "weight_selection_split": "validation",
        "method": method,
        "k": len(val_probas_list),
        "weights": weights.tolist(),
        "member_scores": member_scores,
        "seconds": elapsed,
    }

    print(f"[Ensemble] test_primary={test_score:.5f}  val_primary={val_score:.5f}  "
          f"(member_best_val={max(member_scores):.5f})  {elapsed:.0f}s")

    if out_dir is not None:
        save_json(ensure_dir(out_dir) / "ensemble_result.json", result)

    return result


def evaluate_config_on_holdout(
    cfg: Dict,
    raw: RawDataset,
    *,
    seed: int = 42,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate one validation-selected config once on the untouched test set."""
    result = _retrain(cfg, raw, seed=seed, device=device)
    if result is None:
        return {"test_primary": None, "val_primary": None, "evaluation_split": "test"}
    return {
        "test_primary": result["test_primary"],
        "val_primary": result["val_primary"],
        "test_metrics": result["test_metrics"],
        "evaluation_split": "test",
        "selection_split": "validation",
    }
