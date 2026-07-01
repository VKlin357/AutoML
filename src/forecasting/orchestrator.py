"""
NAS Orchestrator for Forecasting.

Algorithm:
  1. Random warmup: train N random configs, collect results
  2. LLM phase: propose N candidates → quick screen (5 epochs) → train top-K fully
  3. Save all trials to trials_index.json
  4. Return best config + metrics

Differs from tabular NAS:
  - Metric: val_mse (lower is better), stored as primary = -val_mse
  - Cheap screen: 10 epochs on 30% of data (not 5 epochs)
  - LLM prompt is forecasting-aware
  - Search space has forecasting-specific architectures
"""
from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..forecasting.models import build_forecasting_model
from ..forecasting.search_space import (
    FAMILY_WEIGHTS, sample_random_config, validate_config
)
from ..forecasting.train import ForecastResult, train_forecasting
from ..forecasting.llm_agent import build_forecasting_prompt, call_llm_for_forecasting


def _cfg_hash(cfg: dict) -> str:
    s = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.md5(s.encode()).hexdigest()[:10]


def _save_index(out_dir: Path, trials: List[dict]):
    (out_dir / "trials_index.json").write_text(
        json.dumps({"trials": trials}, indent=2, default=str)
    )


def _cheap_eval(
    cfg: dict,
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    lookback: int, n_channels: int, horizon: int,
    device: str, seed: int,
    target_idx: int = -1,
    cheap_frac: float = 0.3,
    cheap_epochs: int = 10,
) -> float:
    """Quick cheap evaluation: 10 epochs on 30% of data. Returns -val_mse."""
    try:
        n = int(len(X_train) * cheap_frac)
        idx = np.random.choice(len(X_train), n, replace=False)
        cheap_cfg = dict(cfg["train"])
        cheap_cfg["epochs"] = cheap_epochs
        cheap_cfg["patience"] = cheap_epochs + 1   # no early stop
        cheap_cfg["use_amp"] = False

        model = build_forecasting_model(cfg["arch"], lookback, horizon, n_channels, target_idx=target_idx)
        result = train_forecasting(
            model,
            X_train[idx], y_train[idx],
            X_val, y_val,
            X_test, y_test,
            lookback=lookback, n_channels=n_channels,
            train_cfg=cheap_cfg,
            device=device, seed=seed, verbose=False,
        )
        return result.primary
    except Exception as e:
        print(f"    [CHEAP] Error: {e}")
        return -1e9


def run_forecasting_nas(
    dataset_name: str,
    X_train: np.ndarray, y_train: np.ndarray,
    X_val:   np.ndarray, y_val:   np.ndarray,
    X_test:  np.ndarray, y_test:  np.ndarray,
    lookback: int,
    horizon: int,
    n_channels: int,
    out_dir: str,
    target_idx: int = -1,
    dataset_info: dict = None,
    budget: int = 30,
    random_warmup: int = 10,
    batch_n: int = 10,
    batch_k: int = 3,
    seed: int = 42,
    device: Optional[str] = None,
    api_key: str = "",
    llm_model: str = "gpt-4o",
) -> dict:
    """
    Run LLM-guided NAS for forecasting.

    Parameters
    ----------
    budget        : total number of fully-trained trials
    random_warmup : first N trials are random (warmup for LLM)
    batch_n       : LLM proposes N candidates per batch
    batch_k       : top-K from N are trained fully
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    rng = random.Random(seed)
    np.random.seed(seed)

    all_trials: List[dict] = []
    seen_hashes = set()
    trial_id = 0
    best_primary = -1e9

    print(f"\n{'='*60}")
    print(f"  Forecasting NAS: {dataset_name.upper()}")
    print(f"  Budget={budget}  Warmup={random_warmup}  Batch={batch_n}→top{batch_k}")
    print(f"  Lookback={lookback}  Horizon={horizon}  Channels={n_channels}")
    print(f"  Device={device}  LLM={llm_model}")
    print(f"{'='*60}\n")

    def run_trial(cfg: dict, op: str, trial_dir: Path) -> dict:
        nonlocal best_primary
        trial_dir.mkdir(parents=True, exist_ok=True)

        arch  = cfg.get("arch", {})
        train = cfg.get("train", {})
        family = arch.get("family", "?")

        print(f"  [TRIAL {trial_id:03d}] family={family}  "
              f"lr={train.get('lr')}  bs={train.get('batch_size')}  "
              f"epochs={train.get('epochs')}")

        t0 = time.time()
        try:
            model = build_forecasting_model(arch, lookback, horizon, n_channels)
            result = train_forecasting(
                model,
                X_train, y_train, X_val, y_val, X_test, y_test,
                lookback=lookback, n_channels=n_channels,
                train_cfg=train, device=device, seed=seed, verbose=True,
            )
            elapsed = time.time() - t0
            primary = result.primary

            print(f"  [TRIAL {trial_id:03d}] val_mse={result.val_mse:.6f}  "
                  f"val_mae={result.val_mae:.6f}  test_mse={result.test_mse:.6f}  "
                  f"primary={primary:.6f}  {elapsed:.1f}s"
                  + (" ★ NEW BEST" if primary > best_primary else ""))

            if primary > best_primary:
                best_primary = primary

            trial_record = {
                "trial_id": trial_id,
                "op": op,
                "config": cfg,
                "primary": primary,
                "val_mse": result.val_mse,
                "val_mae": result.val_mae,
                "test_mse": result.test_mse,
                "test_mae": result.test_mae,
                "val_history": result.val_history,
                "epochs_run": result.epochs_run,
                "n_params": result.n_params,
                "seconds": result.seconds,
                "early_stopped": result.early_stopped,
                "error": None,
            }

        except Exception as e:
            print(f"  [TRIAL {trial_id:03d}] ERROR: {e}")
            trial_record = {
                "trial_id": trial_id,
                "op": op,
                "config": cfg,
                "primary": -1e9,
                "val_mse": 1e9,
                "val_mae": 1e9,
                "test_mse": 1e9,
                "test_mae": 1e9,
                "val_history": [],
                "epochs_run": 0,
                "n_params": 0,
                "seconds": time.time() - t0,
                "early_stopped": False,
                "error": str(e),
            }

        (trial_dir / "trial_result.json").write_text(
            json.dumps(trial_record, indent=2, default=str)
        )
        return trial_record

    # ── Phase 1: Random warmup ────────────────────────────────────────────────
    print(f"\n[NAS] === Random Warmup ({random_warmup} trials) ===\n")

    # Stratified by family
    families = list(FAMILY_WEIGHTS.keys())
    total_w = sum(FAMILY_WEIGHTS.values())
    family_counts = {f: max(1, round(random_warmup * FAMILY_WEIGHTS[f] / total_w))
                     for f in families}
    # Adjust to exact count
    warmup_families = []
    for f, c in family_counts.items():
        warmup_families.extend([f] * c)
    warmup_families = warmup_families[:random_warmup]
    while len(warmup_families) < random_warmup:
        warmup_families.append(rng.choice(families))
    rng.shuffle(warmup_families)

    for fam in warmup_families:
        if trial_id >= budget:
            break
        cfg = sample_random_config(rng, lookback=lookback, family=fam)
        cfg = validate_config(cfg, lookback=lookback)

        h = _cfg_hash(cfg)
        if h in seen_hashes:
            cfg = sample_random_config(rng, lookback=lookback)
            h = _cfg_hash(cfg)
        seen_hashes.add(h)

        t_dir = out_dir / f"trial_{trial_id:03d}"
        record = run_trial(cfg, "random_warmup", t_dir)
        all_trials.append(record)
        _save_index(out_dir, all_trials)
        trial_id += 1

    # ── Phase 2: LLM-guided batch search ─────────────────────────────────────
    if not api_key:
        print("\n[NAS] No API key — skipping LLM phase")
    else:
        batch_num = 0
        while trial_id < budget:
            batch_num += 1
            remaining = budget - trial_id
            k = min(batch_k, remaining)
            print(f"\n[NAS] === LLM Batch {batch_num} (propose {batch_n} → train top {k}) ===\n")

            # History for LLM prompt
            prompt_history = []
            for t in all_trials:
                if t.get("primary", -1e9) > -1e8:
                    prompt_history.append({
                        "arch": t["config"]["arch"],
                        "train": t["config"]["train"],
                        "val_mse": t.get("val_mse", 1e9),
                        "trial_id": t["trial_id"],
                    })

            prompt = build_forecasting_prompt(
                dataset_info=dataset_info,
                history=prompt_history,
                n_propose=batch_n,
            )

            # LLM proposes N configs
            candidates = call_llm_for_forecasting(
                prompt=prompt,
                api_key=api_key,
                model=llm_model,
                n_propose=batch_n,
                lookback=lookback,
            )

            if not candidates:
                print("  [BATCH] LLM returned no valid configs — using random fallback")
                candidates = [sample_random_config(rng, lookback=lookback)
                              for _ in range(batch_n)]

            # Filter duplicates
            unique = []
            for cfg in candidates:
                h = _cfg_hash(cfg)
                if h not in seen_hashes:
                    seen_hashes.add(h)
                    unique.append(cfg)

            if not unique:
                unique = [sample_random_config(rng, lookback=lookback)
                          for _ in range(batch_n)]

            # Cheap screen: rank by 10-epoch val_mse
            print(f"  [SCREEN] cheap-eval {len(unique)} candidates ...")
            cheap_scores = []
            for cfg in unique:
                score = _cheap_eval(
                    cfg, X_train, y_train, X_val, y_val, X_test, y_test,
                    lookback=lookback, n_channels=n_channels, horizon=horizon,
                    device=device, seed=seed, target_idx=target_idx,
                )
                cheap_scores.append((score, cfg))
                fam = cfg["arch"].get("family", "?")
                print(f"    family={fam}  cheap_score={score:.5f}")

            # Select top-K
            cheap_scores.sort(key=lambda x: x[0], reverse=True)
            top_cfgs = [cfg for _, cfg in cheap_scores[:k]]
            print(f"  [SCREEN] selected top-{k}: "
                  f"{[c['arch']['family'] for c in top_cfgs]}")

            # Full training of top-K
            for cfg in top_cfgs:
                if trial_id >= budget:
                    break
                t_dir = out_dir / f"trial_{trial_id:03d}"
                record = run_trial(cfg, f"batch:propose", t_dir)
                all_trials.append(record)
                _save_index(out_dir, all_trials)
                trial_id += 1

    # ── Final summary ─────────────────────────────────────────────────────────
    valid = [t for t in all_trials if t.get("primary", -1e9) > -1e8]
    if valid:
        best = min(valid, key=lambda t: t.get("val_mse", 1e9))
        print(f"\n{'='*60}")
        print(f"  NAS DONE — {len(all_trials)} trials, best val_mse={best['val_mse']:.6f}")
        print(f"  Best: trial={best['trial_id']}  family={best['config']['arch']['family']}")
        print(f"  Test: mse={best['test_mse']:.6f}  mae={best['test_mae']:.6f}")
        print(f"{'='*60}\n")

        (out_dir / "best_trial.json").write_text(json.dumps(best, indent=2, default=str))
    else:
        print("\n[NAS] No valid trials completed!")
        best = {}

    return {"best": best, "all_trials": all_trials}
