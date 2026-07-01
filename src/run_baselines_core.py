"""
Baseline runner: CatBoost + LightGBM + Random NN Search.

All three are evaluated on the same train/val split (loaded via the
legacy ``load_openml_dataset`` that applies standard preprocessing).

CatBoost  : strong gradient-boosted trees baseline (de facto SOTA for tabular).
LightGBM  : another GBDT, fast; usually on par with CatBoost.
Random NN : random search over the full NAS search space (same budget as
            LLM-NAS). This is the key ablation: it tells us how much of the
            LLM-NAS gain comes from the architecture search space itself
            vs. the LLM guidance.
"""
from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .data import load_openml_dataset, load_openml_raw, load_raw, load_builtin_raw
from .metrics import compute_metrics
from .preprocessing import make_preprocessor
from .search_space import sample_random_config, validate_config
from .train_nn import train_trial
from .multi_fidelity import evaluate_at_rung, FULL
from .utils import ensure_dir, save_json, seed_everything


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _concat_num_cat(data, ts_dataset: Optional[str] = None) -> tuple:
    """Return train/validation/test arrays with cat columns appended as float.

    If ts_dataset is given (e.g. 'harth', 'pamap2'), applies time-series
    feature engineering on top of the raw flat features before returning.
    """
    Xtr = np.concatenate(
        [data.X_train_num, data.X_train_cat.astype(np.float32)], axis=1
    )
    Xva = np.concatenate(
        [data.X_val_num, data.X_val_cat.astype(np.float32)], axis=1
    )
    Xte = np.concatenate(
        [data.X_test_num, data.X_test_cat.astype(np.float32)], axis=1
    )
    if ts_dataset:
        from .timeseries_features import make_ts_features_combined, TS_CONFIGS
        ds = ts_dataset.lower()
        if ds in TS_CONFIGS:
            print(f"[TS-FE] Applying time-series feature engineering for '{ds}' ...")
            Xtr = make_ts_features_combined(Xtr, ds)
            Xva = make_ts_features_combined(Xva, ds)
            Xte = make_ts_features_combined(Xte, ds)
            print(f"[TS-FE] Feature shape: {Xtr.shape[1]} (raw + engineered)")
        else:
            print(f"[TS-FE] Warning: '{ds}' not in TS_CONFIGS, skipping FE")
    return Xtr, Xva, Xte, data.y_train, data.y_val, data.y_test


# ---------------------------------------------------------------------------
# CatBoost
# ---------------------------------------------------------------------------

def _load_tabular_data(openml_id, task, seed,
                       source="openml", builtin_name=None, csv_path=None):
    """Load data from any source and return (TabularData-like, summary)."""
    from .preprocessing import Preprocessor
    if source == "openml":
        return load_openml_dataset(openml_id=openml_id, task=task, seed=seed)
    # Non-OpenML: load raw, then apply default preprocessing
    raw, summary = load_raw(
        source=source, openml_id=openml_id,
        builtin_name=builtin_name, csv_path=csv_path,
        task=task, seed=seed,
    )
    from .data import TabularData
    pre = Preprocessor(num_encoder="standard", cat_encoder="embedding")
    sp = pre.fit_transform(
        raw.X_train, raw.X_val, raw.X_test,
        raw.y_train, raw.y_val, raw.y_test,
        raw.num_cols, raw.cat_cols, raw.task, raw.n_classes,
    )
    data = TabularData(
        X_train_num=sp.X_train_num, X_val_num=sp.X_val_num, X_test_num=sp.X_test_num,
        X_train_cat=sp.X_train_cat, X_val_cat=sp.X_val_cat, X_test_cat=sp.X_test_cat,
        y_train=sp.y_train, y_val=sp.y_val, y_test=sp.y_test,
        num_cols=raw.num_cols, cat_cols=raw.cat_cols,
        cat_cardinalities=sp.cat_cardinalities,
        task=raw.task, n_classes=raw.n_classes,
    )
    return data, summary


def run_catboost_baseline(openml_id: int, task: str, out_dir: str, seed: int = 42,
                          source: str = "openml", builtin_name=None, csv_path=None,
                          ts_dataset: Optional[str] = None) -> Dict:
    seed_everything(seed)
    out_dir_p = ensure_dir(out_dir)
    data, summary = _load_tabular_data(openml_id, task, seed, source, builtin_name, csv_path)
    save_json(out_dir_p / "dataset_summary.json", summary)

    from catboost import CatBoostClassifier, CatBoostRegressor

    Xtr, Xva, Xte, ytr, yva, yte = _concat_num_cat(data, ts_dataset=ts_dataset)
    t0 = time.time()

    cb_common = dict(
        iterations=2000, learning_rate=0.05, depth=8,
        random_seed=seed, verbose=100,   # print every 100 iterations
    )
    if data.task == "regression":
        m = CatBoostRegressor(loss_function="RMSE", **cb_common)
        m.fit(Xtr, ytr, eval_set=(Xva, yva), use_best_model=True)
        val_met = compute_metrics("regression", yva, y_pred=m.predict(Xva))
        test_met = compute_metrics("regression", yte, y_pred=m.predict(Xte))
    elif data.task == "binary":
        m = CatBoostClassifier(loss_function="Logloss", **cb_common)
        m.fit(Xtr, ytr, eval_set=(Xva, yva), use_best_model=True)
        val_met = compute_metrics("binary", yva, y_pred_proba=m.predict_proba(Xva)[:, 1])
        test_met = compute_metrics("binary", yte, y_pred_proba=m.predict_proba(Xte)[:, 1])
    else:
        m = CatBoostClassifier(loss_function="MultiClass", **cb_common)
        m.fit(Xtr, ytr, eval_set=(Xva, yva), use_best_model=True)
        val_met = compute_metrics("multiclass", yva, y_pred_proba=m.predict_proba(Xva))
        test_met = compute_metrics("multiclass", yte, y_pred_proba=m.predict_proba(Xte))

    result = {
        "primary": test_met.primary,
        **test_met.metrics,
        "test_primary": test_met.primary,
        "test_metrics": test_met.metrics,
        "val_primary": val_met.primary,
        "val_metrics": val_met.metrics,
        "evaluation_split": "test",
        "early_stopping_split": "validation",
        "seconds": time.time() - t0,
    }
    save_json(out_dir_p / "baseline_catboost.json", result)
    print(f"[CatBoost] test_primary={test_met.primary:.5f}  "
          f"val_primary={val_met.primary:.5f}  ({time.time()-t0:.1f}s)")
    return result


# ---------------------------------------------------------------------------
# LightGBM
# ---------------------------------------------------------------------------

def run_lightgbm_baseline(openml_id: int, task: str, out_dir: str, seed: int = 42,
                          source: str = "openml", builtin_name=None, csv_path=None,
                          ts_dataset: Optional[str] = None) -> Dict:
    seed_everything(seed)
    out_dir_p = ensure_dir(out_dir)
    data, summary = _load_tabular_data(openml_id, task, seed, source, builtin_name, csv_path)
    save_json(out_dir_p / "dataset_summary.json", summary)

    try:
        import lightgbm as lgb
    except ImportError:
        print("[LightGBM] lightgbm not installed; skipping.")
        return {"primary": None, "note": "lightgbm not installed"}

    Xtr, Xva, Xte, ytr, yva, yte = _concat_num_cat(data, ts_dataset=ts_dataset)
    t0 = time.time()

    common = dict(
        n_estimators=2000, learning_rate=0.05, num_leaves=31,
        random_state=seed, verbose=-1, n_jobs=-1,  # verbose=-1 silences LGBM spam; callbacks print instead
    )

    # Print every 50 iterations; early stop prints itself
    lgb_callbacks = [lgb.early_stopping(50, verbose=True), lgb.log_evaluation(50)]

    if data.task == "regression":
        from lightgbm import LGBMRegressor
        m = LGBMRegressor(**common)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=lgb_callbacks)
        val_met = compute_metrics("regression", yva, y_pred=m.predict(Xva))
        test_met = compute_metrics("regression", yte, y_pred=m.predict(Xte))
    elif data.task == "binary":
        from lightgbm import LGBMClassifier
        m = LGBMClassifier(objective="binary", **common)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=lgb_callbacks)
        val_met = compute_metrics("binary", yva, y_pred_proba=m.predict_proba(Xva)[:, 1])
        test_met = compute_metrics("binary", yte, y_pred_proba=m.predict_proba(Xte)[:, 1])
    else:
        from lightgbm import LGBMClassifier
        m = LGBMClassifier(objective="multiclass", num_class=data.n_classes, **common)
        m.fit(Xtr, ytr.astype(int), eval_set=[(Xva, yva.astype(int))], callbacks=lgb_callbacks)
        val_met = compute_metrics("multiclass", yva, y_pred_proba=m.predict_proba(Xva))
        test_met = compute_metrics("multiclass", yte, y_pred_proba=m.predict_proba(Xte))

    result = {
        "primary": test_met.primary,
        **test_met.metrics,
        "test_primary": test_met.primary,
        "test_metrics": test_met.metrics,
        "val_primary": val_met.primary,
        "val_metrics": val_met.metrics,
        "evaluation_split": "test",
        "early_stopping_split": "validation",
        "seconds": time.time() - t0,
    }
    save_json(out_dir_p / "baseline_lightgbm.json", result)
    print(f"[LightGBM] test_primary={test_met.primary:.5f}  "
          f"val_primary={val_met.primary:.5f}  ({time.time()-t0:.1f}s)")
    return result


# ---------------------------------------------------------------------------
# Random Neural Architecture Search (ablation baseline)
# ---------------------------------------------------------------------------

def run_random_nas_baseline(
    openml_id: int,
    task: str,
    out_dir: str,
    budget: int = 30,
    seed: int = 42,
    device: Optional[str] = None,
    source: str = "openml",
    builtin_name=None,
    csv_path=None,
) -> Dict:
    """
    Random search over the NAS search space.

    Same budget, same multi-fidelity evaluation as the LLM-NAS run.
    This isolates the contribution of the LLM guidance from the benefit
    of the search space itself.
    """
    seed_everything(seed)
    rng = random.Random(seed)
    out_dir_p = ensure_dir(out_dir)
    trials_dir = ensure_dir(out_dir_p / "trials_random")

    raw, summary = load_raw(source=source, openml_id=openml_id,
                            builtin_name=builtin_name, csv_path=csv_path,
                            task=task, seed=seed)
    save_json(out_dir_p / "dataset_summary.json", summary)

    all_trials = []
    best_primary = -1e18
    cheap_scores = []

    from .multi_fidelity import successive_halving_threshold

    for trial_id in range(budget):
        cfg = validate_config(sample_random_config(rng))
        family = cfg["arch"]["family"]
        t0 = time.time()

        pre_cfg = cfg.get("preprocess", {"num_encoder": "standard", "cat_encoder": "embedding"})
        pre = make_preprocessor(pre_cfg)
        prepared = pre.fit_transform(
            raw.X_train, raw.X_val, raw.X_test,
            raw.y_train, raw.y_val, raw.y_test,
            raw.num_cols, raw.cat_cols, raw.task, raw.n_classes,
        )

        tdir = ensure_dir(trials_dir / f"trial_{trial_id:03d}")
        save_json(tdir / "config.json", cfg)

        try:
            # Cheap rung first
            from .multi_fidelity import CHEAP, FULL, evaluate_at_rung
            cheap_res = evaluate_at_rung(cfg, prepared, CHEAP, out_dir=None,
                                          seed=seed, device=device)
            cheap_scores.append(cheap_res.primary)
            threshold = successive_halving_threshold(cheap_scores[:-1], 0.5)
            if cheap_res.primary >= threshold or len(cheap_scores) <= 4:
                result = evaluate_at_rung(cfg, prepared, FULL, out_dir=tdir,
                                           seed=seed, device=device)
            else:
                result = cheap_res
            primary = result.primary
        except Exception as e:
            print(f"  [Random NAS] trial {trial_id} error: {e}")
            primary = -1e9
            result = type("R", (), {
                "primary": primary, "metrics": {}, "rung": "error",
                "n_params": 0, "epochs_run": 0, "seconds": 0.0, "early_stopped": False,
            })()

        best_primary = max(best_primary, primary)
        rec = {
            "trial_id": trial_id,
            "primary": primary,
            "metrics": getattr(result, "metrics", {}),
            "rung": getattr(result, "rung", "?"),
            "n_params": getattr(result, "n_params", 0),
            "arch_family": family,
            "config": cfg,
            "seconds": time.time() - t0,
        }
        all_trials.append(rec)
        print(f"  [Random NAS] trial={trial_id:03d}  family={family}  "
              f"primary={primary:.5f}  best={best_primary:.5f}")
        save_json(out_dir_p / "trials_random_index.json", {"trials": all_trials})

    best = max(all_trials, key=lambda x: x["primary"])
    selected_cfg = best["config"]
    pre = make_preprocessor(selected_cfg.get("preprocess", {}))
    prepared = pre.fit_transform(
        raw.X_train, raw.X_val, raw.X_test,
        raw.y_train, raw.y_val, raw.y_test,
        raw.num_cols, raw.cat_cols, raw.task, raw.n_classes,
    )
    final_eval = train_trial(
        cfg=selected_cfg,
        X_train_num=prepared.X_train_num, X_train_cat=prepared.X_train_cat,
        y_train=prepared.y_train,
        X_val_num=prepared.X_val_num, X_val_cat=prepared.X_val_cat,
        y_val=prepared.y_val,
        X_test_num=prepared.X_test_num, X_test_cat=prepared.X_test_cat,
        y_test=prepared.y_test,
        task=prepared.task, n_classes=prepared.n_classes,
        cat_cardinalities=prepared.cat_cardinalities,
        seed=seed, save_model=False, device=device,
    )
    result_dict = {
        "primary": final_eval.test_primary,
        "test_primary": final_eval.test_primary,
        "test_metrics": final_eval.test_metrics,
        "validation_best_primary": best_primary,
        "selected_val_primary": final_eval.primary,
        "evaluation_split": "test",
        "selection_split": "validation",
        "best_trial": best,
        "all_trials": all_trials,
        "budget": budget,
    }
    save_json(out_dir_p / "baseline_random_search.json", result_dict)
    print(f"[Random NAS] test_primary={final_eval.test_primary:.5f}  "
          f"validation_best={best_primary:.5f}")
    return result_dict


# ---------------------------------------------------------------------------
# Unified runner (all baselines at once)
# ---------------------------------------------------------------------------

def run_baselines(
    openml_id: int,
    task: str,
    out_dir: str,
    seed: int = 42,
    run_lgbm: bool = True,
    run_random_nas: bool = False,
    random_nas_budget: int = 20,
    device: Optional[str] = None,
    source: str = "openml",
    builtin_name=None,
    csv_path=None,
    ts_dataset: Optional[str] = None,
) -> Dict[str, Any]:
    """Run CatBoost + optional LightGBM + optional random NAS, save all results."""
    kw = dict(source=source, builtin_name=builtin_name, csv_path=csv_path,
              ts_dataset=ts_dataset)
    results = {}
    results["catboost"] = run_catboost_baseline(openml_id, task, out_dir, seed, **kw)
    if run_lgbm:
        results["lightgbm"] = run_lightgbm_baseline(openml_id, task, out_dir, seed, **kw)
    if run_random_nas:
        results["random_nas"] = run_random_nas_baseline(
            openml_id, task, out_dir, budget=random_nas_budget,
            seed=seed, device=device,
            source=source, builtin_name=builtin_name, csv_path=csv_path,
        )
    out_dir_p = ensure_dir(out_dir)
    save_json(out_dir_p / "baselines.json", results)
    return results
