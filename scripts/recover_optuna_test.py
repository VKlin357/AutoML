"""
Recover test_primary for existing baseline_optuna.json files that have
best_params saved but no test_primary (old code bug).

Instead of re-running 40 Optuna trials, this script:
  1. Reads best_params from the existing JSON
  2. Reconstructs the full nested config (same logic as the Optuna objective)
  3. Retrains the best model once and evaluates on test
  4. Patches test_primary into the JSON (does not change trial_log or other fields)

Usage (run from repo root on GPU server):
    python scripts/recover_optuna_test.py --exp_dir experiments_v9 --device cuda

    # Specific datasets only:
    python scripts/recover_optuna_test.py --exp_dir experiments_v9 \
        --names adult_baselines helena_baselines har_baselines miniboonee_baselines \
        --device cuda
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Rebuild full cfg from flat best_params dict
# ---------------------------------------------------------------------------
def params_to_cfg(p: dict) -> dict:
    """Convert flat Optuna best_params → nested cfg expected by train_trial."""
    family    = p["family"]
    num_enc   = p["num_encoder"]
    cat_enc   = p["cat_encoder"]
    dropout   = p["dropout"]
    norm      = p["normalization"]
    emb_dim   = p["embedding_dim"]
    activation = p["activation"]

    if family == "mlp":
        arch_cfg = {
            "family": "mlp",
            "hidden_dims": [p["mlp_hidden_dim"]] * p["mlp_n_layers"],
            "activation": activation,
            "dropout": dropout,
            "normalization": norm,
            "embedding_dim": emb_dim,
        }
    elif family == "resmlp":
        arch_cfg = {
            "family": "resmlp",
            "n_blocks": p["resmlp_n_blocks"],
            "block_width": p["resmlp_block_width"],
            "activation": activation,
            "dropout": dropout,
            "normalization": norm,
            "embedding_dim": emb_dim,
        }
    elif family == "tabm":
        arch_cfg = {
            "family": "tabm",
            "n_blocks": p["tabm_n_blocks"],
            "width": p["tabm_width"],
            "k": p["tabm_k"],
            "dropout": dropout,
            "head_dropout": p["tabm_head_dropout"],
            "activation": activation,
            "normalization": norm,
            "embedding_dim": emb_dim,
        }
    elif family == "ft_transformer":
        n_heads = p["ftt_n_heads"]
        d_token_mult = p["ftt_d_token_mult"]
        d_token = (d_token_mult // n_heads) * n_heads
        d_token = max(d_token, n_heads)
        arch_cfg = {
            "family": "ft_transformer",
            "n_blocks": p["ftt_n_blocks"],
            "d_token": d_token,
            "n_heads": n_heads,
            "ffn_factor": round(p["ftt_ffn_factor"], 3),
            "attn_dropout": round(p["ftt_attn_dropout"], 4),
            "ffn_dropout": round(p["ftt_ffn_dropout"], 4),
            "residual_dropout": 0.0,
            "activation": activation,
        }
    elif family == "gated_tab":
        arch_cfg = {
            "family": "gated_tab",
            "n_blocks": p["gated_n_blocks"],
            "block_width": p["gated_block_width"],
            "activation": activation,
            "dropout": dropout,
            "normalization": norm,
            "embedding_dim": emb_dim,
        }
    elif family == "autoint":
        n_heads = p["ai_n_heads"]
        d_token_mult = p["ai_d_token_mult"]
        d_token = (d_token_mult // n_heads) * n_heads
        d_token = max(d_token, n_heads)
        arch_cfg = {
            "family": "autoint",
            "n_blocks": p["ai_n_blocks"],
            "d_token": d_token,
            "n_heads": n_heads,
            "attn_dropout": round(p["ai_attn_dropout"], 4),
            "ffn_factor": round(p["ai_ffn_factor"], 3),
            "ffn_dropout": round(p["ai_ffn_dropout"], 4),
            "activation": activation,
        }
    else:
        raise ValueError(f"Unknown family: {family}")

    cfg = {
        "preprocess": {"num_encoder": num_enc, "cat_encoder": cat_enc},
        "arch": arch_cfg,
        "train": {
            "optimizer": p["optimizer"],
            "lr": p["lr"],
            "weight_decay": p["weight_decay"],
            "scheduler": p["scheduler"],
            "batch_size": p["batch_size"],
            "epochs": 60,
            "patience": 10,
            "label_smoothing": 0.0,
            "grad_clip": 1.0,
            "use_amp": True,
            "feature_noise_std": 0.0,
            "mixup_alpha": 0.0,
        },
    }
    return cfg


# ---------------------------------------------------------------------------
# Dataset registry — same as compute_test_metrics.py
# ---------------------------------------------------------------------------
DATASETS = {
    "volkert":    ("openml", 41166, None,          "multiclass"),
    "jannis":     ("openml", 45021, None,          "multiclass"),
    "miniboonee": ("builtin", None, "miniboonee",  "binary"),
    "helena":     ("openml", 41169, None,          "multiclass"),
    "adult":      ("openml", 1590,  None,          "binary"),
    "har":        ("builtin", None, "har",         "multiclass"),
    "harth":      ("openml", 43921, None,          "multiclass"),
    "pamap2":     ("openml", 44994, None,          "multiclass"),
    "eeg":        ("openml", 1471,  None,          "binary"),
    "pol":        ("openml", 722,   None,          "multiclass"),
}

DIR_PREFIX = {
    "volkert": "volkert", "jannis": "jannis", "miniboonee": "miniboonee",
    "helena": "helena", "adult": "adult", "har_": "har", "harth": "harth",
    "pamap2": "pamap2", "eeg_": "eeg", "pol_": "pol",
}


def resolve_dataset(dirname: str):
    for prefix, key in DIR_PREFIX.items():
        if dirname.startswith(prefix):
            return DATASETS[key]
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_dir", default="experiments_v9")
    ap.add_argument("--names", nargs="*", default=None,
                    help="Specific baseline dirs (e.g. adult_baselines helena_baselines)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    ap.add_argument("--force", action="store_true",
                    help="Re-evaluate even if test_primary already exists")
    args = ap.parse_args()

    exp_root = Path(args.exp_dir)

    if args.names:
        dirs = [exp_root / n for n in args.names]
    else:
        # Auto-discover: all dirs with baseline_optuna.json missing test_primary
        dirs = sorted(exp_root.glob("*/baseline_optuna.json"))
        dirs = [p.parent for p in dirs]

    from src.data import load_raw
    from src.preprocessing import Preprocessor
    from src.train_nn import train_trial
    from src.search_space import validate_config

    results = {}

    for d in dirs:
        optuna_path = d / "baseline_optuna.json"
        if not optuna_path.exists():
            print(f"[SKIP] {d.name} — нет baseline_optuna.json")
            continue

        existing = json.loads(optuna_path.read_text())

        if not args.force and existing.get("test_primary") is not None:
            # Also skip harth_optuna32-style "valid" results
            val = existing.get("validation_best_primary",
                               existing.get("selected_val_primary",
                               existing.get("primary")))
            test = existing["test_primary"]
            print(f"[SKIP] {d.name} — test_primary={test:.5f} уже есть")
            results[d.name] = {"status": "already_done", "test_primary": test}
            continue

        if "best_params" not in existing:
            print(f"[SKIP] {d.name} — нет best_params, нужно полное перепрогон")
            results[d.name] = {"status": "no_best_params"}
            continue

        dataset_info = resolve_dataset(d.name)
        if dataset_info is None:
            print(f"[SKIP] {d.name} — неизвестный датасет")
            results[d.name] = {"status": "unknown_dataset"}
            continue

        source, openml_id, builtin_name, task = dataset_info
        print(f"\n{'='*55}")
        print(f"Восстанавливаем тест: {d.name}")
        print(f"  датасет: source={source}, openml_id={openml_id}, task={task}")

        try:
            best_params = existing["best_params"]
            cfg = params_to_cfg(best_params)
            cfg = validate_config(cfg)

            print(f"  family={cfg['arch']['family']}, "
                  f"val_best={existing.get('primary', '?'):.5f}")

            raw, _ = load_raw(
                source=source, openml_id=openml_id,
                builtin_name=builtin_name, task=task, seed=args.seed,
            )

            pre = Preprocessor(
                num_encoder=cfg["preprocess"]["num_encoder"],
                cat_encoder=cfg["preprocess"]["cat_encoder"],
            )
            prepared = pre.fit_transform(
                raw.X_train, raw.X_val, raw.X_test,
                raw.y_train, raw.y_val, raw.y_test,
                raw.num_cols, raw.cat_cols, raw.task, raw.n_classes,
            )

            eval_result = train_trial(
                cfg=cfg,
                X_train_num=prepared.X_train_num, X_train_cat=prepared.X_train_cat,
                y_train=prepared.y_train,
                X_val_num=prepared.X_val_num, X_val_cat=prepared.X_val_cat,
                y_val=prepared.y_val,
                X_test_num=prepared.X_test_num, X_test_cat=prepared.X_test_cat,
                y_test=prepared.y_test,
                task=prepared.task, n_classes=prepared.n_classes,
                cat_cardinalities=prepared.cat_cardinalities,
                seed=args.seed, save_model=False, device=args.device,
                max_epochs=cfg["train"]["epochs"],
            )

            val_score  = float(eval_result.primary)
            test_score = float(eval_result.test_primary)
            print(f"  ✓ val={val_score:.5f}  test={test_score:.5f}")

            # Patch the JSON — preserve all existing fields
            existing["test_primary"] = test_score
            existing["test_metrics"] = eval_result.test_metrics
            existing["validation_best_primary"] = existing.get("primary", val_score)
            existing["selected_val_primary"] = val_score
            existing["evaluation_split"] = "test"
            existing["selection_split"] = "validation"
            existing["best_config"] = cfg

            optuna_path.write_text(json.dumps(existing, indent=2))
            print(f"  Сохранено → {optuna_path}")
            results[d.name] = {"status": "computed", "val": val_score, "test": test_score}

        except Exception as e:
            import traceback
            print(f"  ОШИБКА: {e}")
            traceback.print_exc()
            results[d.name] = {"status": "error", "error": str(e)}

    # Summary
    print(f"\n{'='*55}")
    print(f"{'Директория':<30} {'Val':>8} {'Test':>8}  Статус")
    print("-" * 55)
    for name, info in sorted(results.items()):
        val  = f"{info['val']:.5f}"  if info.get("val")  else "  N/A  "
        test = f"{info['test']:.5f}" if info.get("test") else "  N/A  "
        print(f"{name:<30} {val:>8} {test:>8}  {info['status']}")


if __name__ == "__main__":
    main()
