"""
AutoML baselines: AutoGluon, LightAutoML, and Optuna.

These are optional — only run if the library is installed.
Install in Colab:
    !pip install autogluon.tabular -q
    !pip install lightautoml -q
    !pip install optuna -q
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .utils import ensure_dir, save_json


# ---------------------------------------------------------------------------
# AutoGluon
# ---------------------------------------------------------------------------

def run_autogluon_baseline(
    openml_id: int,
    task: str,
    out_dir: str,
    seed: int = 42,
    time_limit: int = 120,      # seconds — same wall-clock budget as ~25 NAS trials
    presets: str = "medium_quality",
) -> Dict:
    """
    Run AutoGluon TabularPredictor and return primary metric on val set.

    time_limit=120 is a fair comparison: our NAS also takes ~2-3 min on
    credit_g with budget=25.  Increase to 300 for a stronger baseline.
    """
    try:
        from autogluon.tabular import TabularDataset, TabularPredictor
    except ImportError:
        print("[AutoGluon] not installed — pip install autogluon.tabular")
        return {"primary": None, "note": "not installed"}

    import openml
    import pandas as pd
    from .metrics import infer_task_type

    ds = openml.datasets.get_dataset(openml_id)
    X, y, _, _ = ds.get_data(
        dataset_format="dataframe",
        target=ds.default_target_attribute,
    )
    X = X.dropna(axis=1, how="all")
    df = X.copy()
    df["__target__"] = y

    # Same 80/20 split as our pipeline
    train_df = df.sample(frac=0.8, random_state=seed)
    val_df   = df.drop(train_df.index)

    task_t = task if task != "auto" else infer_task_type(y.to_numpy())

    eval_metric = {
        "binary":     "roc_auc",
        "multiclass": "accuracy",
        "regression": "rmse",
    }.get(task_t, "roc_auc")

    out_path = str(ensure_dir(out_dir) / "autogluon_model")
    t0 = time.time()

    predictor = TabularPredictor(
        label="__target__",
        eval_metric=eval_metric,
        path=out_path,
        verbosity=0,
    ).fit(
        train_df,
        time_limit=time_limit,
        presets=presets,
        random_state=seed,
    )

    scores = predictor.evaluate(val_df, silent=True)
    primary = abs(scores.get(eval_metric, list(scores.values())[0]))
    # For regression rmse: we store negative so higher=better (like our pipeline)
    if task_t == "regression":
        primary = -primary

    elapsed = time.time() - t0
    result = {
        "primary":    float(primary),
        "scores":     scores,
        "time_limit": time_limit,
        "presets":    presets,
        "seconds":    elapsed,
    }
    save_json(ensure_dir(out_dir) / "baseline_autogluon.json", result)
    print(f"[AutoGluon] primary={primary:.5f}  ({elapsed:.0f}s, presets={presets})")
    return result


# ---------------------------------------------------------------------------
# LightAutoML
# ---------------------------------------------------------------------------

def run_lightautoml_baseline(
    openml_id: int,
    task: str,
    out_dir: str,
    seed: int = 42,
    time_limit: int = 120,
) -> Dict:
    """Run LightAutoML and return primary metric on val set."""
    try:
        from lightautoml.automl.presets.tabular_presets import TabularAutoML
        from lightautoml.tasks import Task as LAMATask
    except ImportError:
        print("[LightAutoML] not installed — pip install lightautoml")
        return {"primary": None, "note": "not installed"}

    import openml
    import pandas as pd
    from .metrics import infer_task_type

    ds = openml.datasets.get_dataset(openml_id)
    X, y, _, _ = ds.get_data(
        dataset_format="dataframe",
        target=ds.default_target_attribute,
    )
    X = X.dropna(axis=1, how="all")
    df = X.copy()
    TARGET = "__target__"
    df[TARGET] = y

    train_df = df.sample(frac=0.8, random_state=seed)
    val_df   = df.drop(train_df.index)

    task_t = task if task != "auto" else infer_task_type(y.to_numpy())
    lama_task_name = {
        "binary":     "binary",
        "multiclass": "multiclass",
        "regression": "reg",
    }.get(task_t, "binary")

    lama_task = LAMATask(lama_task_name)
    automl = TabularAutoML(
        task=lama_task,
        timeout=time_limit,
        general_params={"use_algos": [["lgb", "cb", "linear_l2"]]},
    )

    t0 = time.time()
    oof_pred = automl.fit_predict(train_df, roles={"target": TARGET})
    test_pred = automl.predict(val_df)
    elapsed = time.time() - t0

    from .metrics import compute_metrics
    y_val = val_df[TARGET].to_numpy()

    if task_t == "binary":
        proba = test_pred.data[:, 0]
        met = compute_metrics("binary", y_val, y_pred_proba=proba)
    elif task_t == "multiclass":
        proba = test_pred.data
        met = compute_metrics("multiclass", y_val, y_pred_proba=proba)
    else:
        pred = test_pred.data[:, 0]
        met = compute_metrics("regression", y_val, y_pred=pred)

    result = {
        "primary": float(met.primary),
        "metrics": met.metrics,
        "seconds": elapsed,
        "time_limit": time_limit,
    }
    save_json(ensure_dir(out_dir) / "baseline_lightautoml.json", result)
    print(f"[LightAutoML] primary={met.primary:.5f}  ({elapsed:.0f}s)")
    return result


# ---------------------------------------------------------------------------
# Optuna baseline
# ---------------------------------------------------------------------------

def run_optuna_baseline(
    openml_id: Optional[int],
    task: str,
    out_dir: str,
    seed: int = 42,
    n_trials: int = 25,          # match LLM-NAS budget for fair comparison
    timeout: int = 3600,         # hard wall-clock cap in seconds
    source: str = "openml",
    builtin_name: Optional[str] = None,
    csv_path: Optional[str] = None,
    device: Optional[str] = None,
) -> Dict:
    """
    Run Optuna TPE search over the SAME neural-net search space used by LLM-NAS.

    Fair apples-to-apples comparison:
      - Same 5 arch families, same field names as validate_config expects
      - Same train/val split and preprocessing options
      - Same PyTorch training loop (train_trial)
      - Same n_trials budget as LLM-NAS budget

    Optuna uses Tree-structured Parzen Estimator (TPE): Bayesian optimisation
    that learns a probabilistic model of the search space. Unlike LLM-NAS it has
    no semantic knowledge of what the parameters mean — no learning curve analysis,
    no cross-domain priors, no structured reasoning.

    Install: pip install optuna
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("[Optuna] not installed — pip install optuna")
        return {"primary": None, "note": "not installed"}

    from .data import load_raw
    from .preprocessing import Preprocessor
    from .train_nn import train_trial
    from .search_space import ARCH_FAMILIES, validate_config

    print(f"[Optuna] Starting TPE search: n_trials={n_trials}, timeout={timeout}s")
    t0 = time.time()

    raw, summary = load_raw(
        source=source, openml_id=openml_id,
        builtin_name=builtin_name, csv_path=csv_path,
        task=task, seed=seed,
    )

    best_trials_log = []
    trial_configs = {}

    def objective(trial: "optuna.Trial") -> float:
        # ---- preprocess ----
        # num_encoder valid values: "standard", "quantile", "none"
        num_enc = trial.suggest_categorical("num_encoder", ["standard", "quantile", "none"])
        cat_enc = trial.suggest_categorical("cat_encoder", ["embedding", "onehot"])

        # ---- arch family ----
        family = trial.suggest_categorical("family", list(ARCH_FAMILIES))

        # Shared params (dropout / normalization used by mlp / resmlp / gated_tab)
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        normalization = trial.suggest_categorical("normalization", ["batchnorm", "layernorm", "none"])
        emb_dim = trial.suggest_int("embedding_dim", 4, 32)
        activation = trial.suggest_categorical("activation", ["relu", "gelu", "silu"])

        if family == "mlp":
            n_layers = trial.suggest_int("mlp_n_layers", 1, 6)
            hidden_dim = trial.suggest_int("mlp_hidden_dim", 32, 512)
            arch_cfg = {
                "family": "mlp",
                "hidden_dims": [hidden_dim] * n_layers,
                "activation": activation,
                "dropout": dropout,
                "normalization": normalization,
                "embedding_dim": emb_dim,
            }
        elif family == "resmlp":
            # CORRECT field names: n_blocks + block_width (not n_layers / hidden_dim)
            n_blocks = trial.suggest_int("resmlp_n_blocks", 1, 8)
            block_width = trial.suggest_int("resmlp_block_width", 64, 1024)
            arch_cfg = {
                "family": "resmlp",
                "n_blocks": n_blocks,
                "block_width": block_width,
                "activation": activation,
                "dropout": dropout,
                "normalization": normalization,
                "embedding_dim": emb_dim,
            }
        elif family == "tabm":
            n_blocks = trial.suggest_int("tabm_n_blocks", 2, 6)
            width = trial.suggest_int("tabm_width", 128, 1024)
            k = trial.suggest_int("tabm_k", 4, 16)
            head_dropout = trial.suggest_float("tabm_head_dropout", 0.0, 0.30)
            arch_cfg = {
                "family": "tabm",
                "n_blocks": n_blocks,
                "width": width,
                "k": k,
                "dropout": dropout,
                "head_dropout": head_dropout,
                "activation": activation,
                "normalization": normalization,
                "embedding_dim": emb_dim,
            }
        elif family == "ft_transformer":
            n_heads = trial.suggest_categorical("ftt_n_heads", [2, 4, 8])
            # d_token must be divisible by n_heads
            d_token_mult = trial.suggest_int("ftt_d_token_mult", 4, 32)
            d_token = (d_token_mult // n_heads) * n_heads
            d_token = max(d_token, n_heads)
            n_blocks = trial.suggest_int("ftt_n_blocks", 1, 6)
            attn_dropout = trial.suggest_float("ftt_attn_dropout", 0.0, 0.4)
            ffn_dropout = trial.suggest_float("ftt_ffn_dropout", 0.0, 0.4)
            ffn_factor = trial.suggest_float("ftt_ffn_factor", 1.0, 4.0)
            arch_cfg = {
                "family": "ft_transformer",
                "n_blocks": n_blocks,
                "d_token": d_token,
                "n_heads": n_heads,
                "ffn_factor": round(ffn_factor, 3),
                "attn_dropout": round(attn_dropout, 4),
                "ffn_dropout": round(ffn_dropout, 4),
                "residual_dropout": 0.0,
                "activation": activation,
            }
        elif family == "gated_tab":
            # CORRECT field names: n_blocks + block_width
            n_blocks = trial.suggest_int("gated_n_blocks", 1, 6)
            block_width = trial.suggest_int("gated_block_width", 64, 1024)
            arch_cfg = {
                "family": "gated_tab",
                "n_blocks": n_blocks,
                "block_width": block_width,
                "activation": activation,
                "dropout": dropout,
                "normalization": normalization,
                "embedding_dim": emb_dim,
            }
        else:  # autoint
            n_heads = trial.suggest_categorical("ai_n_heads", [2, 4])
            d_token_mult = trial.suggest_int("ai_d_token_mult", 2, 16)
            d_token = (d_token_mult // n_heads) * n_heads
            d_token = max(d_token, n_heads)
            n_blocks = trial.suggest_int("ai_n_blocks", 1, 4)
            attn_dropout = trial.suggest_float("ai_attn_dropout", 0.0, 0.3)
            ffn_factor = trial.suggest_float("ai_ffn_factor", 1.0, 3.0)
            ffn_dropout = trial.suggest_float("ai_ffn_dropout", 0.0, 0.3)
            arch_cfg = {
                "family": "autoint",
                "n_blocks": n_blocks,
                "d_token": d_token,
                "n_heads": n_heads,
                "attn_dropout": round(attn_dropout, 4),
                "ffn_factor": round(ffn_factor, 3),
                "ffn_dropout": round(ffn_dropout, 4),
                "activation": activation,
            }

        # ---- train ----
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True)
        # batch_size valid values: 128, 256, 512, 1024
        batch_size = trial.suggest_categorical("batch_size", [128, 256, 512, 1024])
        optimizer = trial.suggest_categorical("optimizer", ["adamw", "adam"])
        scheduler = trial.suggest_categorical("scheduler", ["cosine", "plateau", "none"])

        cfg = {
            "preprocess": {"num_encoder": num_enc, "cat_encoder": cat_enc},
            "arch": arch_cfg,
            "train": {
                "optimizer": optimizer,
                "lr": lr,
                "weight_decay": weight_decay,
                "scheduler": scheduler,
                "batch_size": batch_size,
                "epochs": 60,
                "patience": 10,
                "label_smoothing": 0.0,
                "grad_clip": 1.0,
                "use_amp": True,
                "feature_noise_std": 0.0,
                "mixup_alpha": 0.0,
            },
        }

        # Clamp any out-of-range values (same robustness as LLM-NAS pipeline)
        cfg = validate_config(cfg)
        trial_configs[trial.number] = cfg

        try:
            pre = Preprocessor(
                num_encoder=cfg["preprocess"]["num_encoder"],
                cat_encoder=cfg["preprocess"]["cat_encoder"],
            )
            prepared = pre.fit_transform(
                raw.X_train, raw.X_val, raw.X_test,
                raw.y_train, raw.y_val, raw.y_test,
                raw.num_cols, raw.cat_cols, raw.task, raw.n_classes,
            )
            res = train_trial(
                cfg=cfg,
                X_train_num=prepared.X_train_num, X_train_cat=prepared.X_train_cat,
                y_train=prepared.y_train,
                X_val_num=prepared.X_val_num, X_val_cat=prepared.X_val_cat,
                y_val=prepared.y_val,
                task=prepared.task, n_classes=prepared.n_classes,
                cat_cardinalities=prepared.cat_cardinalities,
                seed=seed, save_model=False, device=device,
                max_epochs=cfg["train"]["epochs"],
            )
            primary = float(res.primary)
        except Exception as e:
            print(f"  [Optuna] trial {trial.number} failed: {e}")
            return -1e9

        best_trials_log.append({
            "trial_number": trial.number,
            "primary": primary,
            "family": family,
            "params": trial.params,
        })
        print(f"  [Optuna] trial {trial.number:03d}  family={family}  primary={primary:.5f}",
              flush=True)
        return primary

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

    best = study.best_trial
    best_cfg = trial_configs[best.number]
    pre = Preprocessor(
        num_encoder=best_cfg["preprocess"]["num_encoder"],
        cat_encoder=best_cfg["preprocess"]["cat_encoder"],
    )
    prepared = pre.fit_transform(
        raw.X_train, raw.X_val, raw.X_test,
        raw.y_train, raw.y_val, raw.y_test,
        raw.num_cols, raw.cat_cols, raw.task, raw.n_classes,
    )
    final_eval = train_trial(
        cfg=best_cfg,
        X_train_num=prepared.X_train_num, X_train_cat=prepared.X_train_cat,
        y_train=prepared.y_train,
        X_val_num=prepared.X_val_num, X_val_cat=prepared.X_val_cat,
        y_val=prepared.y_val,
        X_test_num=prepared.X_test_num, X_test_cat=prepared.X_test_cat,
        y_test=prepared.y_test,
        task=prepared.task, n_classes=prepared.n_classes,
        cat_cardinalities=prepared.cat_cardinalities,
        seed=seed, save_model=False, device=device,
        max_epochs=best_cfg["train"]["epochs"],
    )
    elapsed = time.time() - t0
    result = {
        "primary": float(final_eval.test_primary),
        "test_primary": float(final_eval.test_primary),
        "test_metrics": final_eval.test_metrics,
        "validation_best_primary": float(best.value),
        "selected_val_primary": float(final_eval.primary),
        "evaluation_split": "test",
        "selection_split": "validation",
        "best_params": best.params,
        "best_config": best_cfg,
        "n_trials_completed": len(study.trials),
        "seconds": elapsed,
        "n_trials_budget": n_trials,
        "trial_log": best_trials_log,
    }
    save_json(ensure_dir(out_dir) / "baseline_optuna.json", result)
    print(f"[Optuna] test_primary={final_eval.test_primary:.5f}  "
          f"validation_best={best.value:.5f}  "
          f"({len(study.trials)} trials, {elapsed:.0f}s)  "
          f"family={best.params.get('family', '?')}")
    return result


# ---------------------------------------------------------------------------
# Naive LLM baseline
# ---------------------------------------------------------------------------

_NAIVE_LLM_SYSTEM = """\
You are an ML engineer. Given a dataset description, propose a neural network \
configuration to maximise validation performance on the task.

OUTPUT: Strict JSON only — no markdown, no prose, no code fences.

The JSON must have exactly this structure (use ONLY the keys listed for the chosen family):

{
  "preprocess": {
    "num_encoder": "<standard|quantile|none>",
    "cat_encoder": "<embedding|onehot>"
  },
  "arch": {
    "family": "<mlp|resmlp|ft_transformer|gated_tab|autoint>",

    // --- mlp only ---
    "hidden_dims": [128, 256, 128],       // list of layer widths, 1..6 layers, each in [16,1024]

    // --- resmlp or gated_tab only ---
    "n_blocks": 4,                        // int in [1,8] for resmlp, [1,6] for gated_tab
    "block_width": 256,                   // int in [64,1024]

    // --- ft_transformer or autoint only ---
    "n_blocks": 3,                        // int in [1,6] for ft_transformer, [1,4] for autoint
    "d_token": 64,                        // int, must be divisible by n_heads
    "n_heads": 4,                         // one of {2,4,8} for ft_transformer, {2,4} for autoint
    "ffn_factor": 2.0,                    // float in [1.0,4.0]
    "attn_dropout": 0.1,                  // float in [0.0,0.4] for ft_transformer, [0.0,0.3] for autoint
    "ffn_dropout": 0.1,                   // float in [0.0,0.4]
    "residual_dropout": 0.0,              // ft_transformer only, float in [0.0,0.2]

    // --- shared (all families except ft_transformer / autoint) ---
    "dropout": 0.1,                       // float in [0.0,0.5]
    "normalization": "<batchnorm|layernorm|none>",
    "embedding_dim": 16,                  // int in [4,64]
    "activation": "<relu|gelu|silu|leaky_relu>"
  },
  "train": {
    "optimizer": "<adamw|adam|sgd_momentum>",
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "scheduler": "<cosine|plateau|none>",
    "batch_size": 256,                    // one of {128,256,512,1024}
    "epochs": 60,
    "patience": 8,
    "label_smoothing": 0.0,
    "grad_clip": 1.0,
    "use_amp": true,
    "feature_noise_std": 0.0,
    "mixup_alpha": 0.0
  }
}

Include ONLY the arch keys relevant to the chosen family (omit others).
"""

_NAIVE_LLM_USER_TMPL = """\
Dataset: {name}
Task: {task}
Rows: {n_rows}
Features: {n_features}  (numeric: {n_num}, categorical: {n_cat})
Classes: {n_classes}

Propose ONE configuration that you expect to work well for this dataset.
"""


def run_naive_llm_baseline(
    openml_id: int,
    task: str,
    out_dir: str,
    api_key: str,
    llm_model: str = "gpt-4o-mini",
    seed: int = 42,
    n_trials: int = 5,
    source: str = "openml",
    builtin_name: Optional[str] = None,
    csv_path: Optional[str] = None,
) -> Dict:
    """
    Naive LLM baseline: the LLM receives only a plain-English dataset summary
    (no search-space schema, no architecture families listing, no trial history)
    and is asked to propose a configuration.  We run up to ``n_trials`` proposals
    and keep the best.

    This isolates the value of the structured search space + evolutionary loop
    used in LLM-NAS v2: if the naive baseline scores similarly, the schema and
    evolution add little; if it scores worse, they add real value.

    Supports all data sources (openml, builtin, csv) via the ``source`` parameter.
    """
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from .data import load_raw
    from .preprocessing import Preprocessor
    from .train_nn import train_trial
    from .metrics import infer_task_type
    from .search_space import validate_config
    from .llm.client import OpenAILLM, _extract_json

    if not api_key:
        print("[NaiveLLM] No API key — skipping")
        return {"primary": None, "note": "no api_key"}

    print(f"[NaiveLLM] Starting naive LLM baseline: n_trials={n_trials}, source={source}")
    t0 = time.time()

    raw, summary = load_raw(
        source=source, openml_id=openml_id,
        builtin_name=builtin_name, csv_path=csv_path,
        task=task, seed=seed,
    )
    task_t = summary["task"]

    llm = OpenAILLM(api_key=api_key, model=llm_model, temperature=0.8)

    user_msg = _NAIVE_LLM_USER_TMPL.format(
        name=summary.get("name", f"openml_{openml_id}"),
        task=task_t,
        n_rows=summary.get("n_rows", "?"),
        n_features=summary.get("n_features", "?"),
        n_num=summary.get("n_num", "?"),
        n_cat=summary.get("n_cat", "?"),
        n_classes=summary.get("n_classes", "?"),
    )

    best_primary: Optional[float] = None
    best_cfg: Optional[Dict] = None
    trial_log = []

    for trial_idx in range(n_trials):
        try:
            raw_text = llm.complete(_NAIVE_LLM_SYSTEM, user_msg)
            cfg = _extract_json(raw_text)
            if not isinstance(cfg, dict) or "arch" not in cfg:
                print(f"  [NaiveLLM] trial {trial_idx}: LLM returned invalid JSON, skipping")
                continue

            # Clamp / coerce out-of-range values to valid ranges (same as LLM-NAS)
            cfg = validate_config(cfg)

            num_enc = cfg.get("preprocess", {}).get("num_encoder", "standard")
            cat_enc = cfg.get("preprocess", {}).get("cat_encoder", "embedding")

            pre = Preprocessor(num_encoder=num_enc, cat_encoder=cat_enc)
            prepared = pre.fit_transform(
                raw.X_train, raw.X_val, raw.X_test,
                raw.y_train, raw.y_val, raw.y_test,
                raw.num_cols, raw.cat_cols, raw.task, raw.n_classes,
            )
            res = train_trial(
                cfg=cfg,
                X_train_num=prepared.X_train_num, X_train_cat=prepared.X_train_cat,
                y_train=prepared.y_train,
                X_val_num=prepared.X_val_num, X_val_cat=prepared.X_val_cat,
                y_val=prepared.y_val,
                task=prepared.task, n_classes=prepared.n_classes,
                cat_cardinalities=prepared.cat_cardinalities,
                seed=seed, save_model=False, device=None,
                max_epochs=50,
            )
            primary = float(res.primary)
            print(f"  [NaiveLLM] trial {trial_idx:02d}  family={cfg['arch'].get('family','?')}  primary={primary:.5f}")
            trial_log.append({"trial": trial_idx, "primary": primary, "config": cfg})

            if best_primary is None or primary > best_primary:
                best_primary = primary
                best_cfg = cfg

        except Exception as e:
            print(f"  [NaiveLLM] trial {trial_idx}: failed — {e}")

    elapsed = time.time() - t0
    result = {
        "primary": float(best_primary) if best_primary is not None else None,
        "best_config": best_cfg,
        "n_trials": n_trials,
        "n_successful": len(trial_log),
        "seconds": elapsed,
        "trial_log": trial_log,
    }
    save_json(ensure_dir(out_dir) / "baseline_naive_llm.json", result)
    if best_primary is not None:
        print(f"[NaiveLLM] best_primary={best_primary:.5f}  ({len(trial_log)}/{n_trials} successful, {elapsed:.0f}s)")
    else:
        print(f"[NaiveLLM] all trials failed ({elapsed:.0f}s)")
    return result
