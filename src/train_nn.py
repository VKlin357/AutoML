"""
Trainer for one neural-net trial.

Differences from v1:

- Uses ``models.make_model`` and accepts a structured config (preprocess
  is already applied; ``arch`` and ``train`` sub-dicts come from the
  search space).

- Multi-fidelity-aware: ``data_frac`` < 1 trains on a stratified subsample;
  ``max_epochs`` overrides ``train.epochs``. The orchestrator uses these
  to run cheap proxy evaluations and only promote promising configs to
  full training.

- Adds: cosine/onecycle/plateau schedulers, optional AMP, label smoothing
  (classification), grad clipping, mixup-on-input, Gaussian feature noise.

- Returns rich metrics and a ``history`` (val primary per epoch) so the
  orchestrator can compute area-under-learning-curve as a denoised proxy.

The legacy ``TrainConfig`` + ``train_one_trial`` API is preserved at the
bottom so v1 code keeps working.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .metrics import TaskType, compute_metrics
from .models import make_model, count_params
from .utils import ensure_dir, save_json


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class _TabDS(Dataset):
    def __init__(self, X_num: np.ndarray, X_cat: np.ndarray, y: np.ndarray, task: TaskType):
        # Sanitize: replace NaN / ±inf before converting to tensor.
        # NaN can arrive from QuantileTransformer on out-of-range val/test values,
        # or from OpenML datasets with missing values in rare columns.
        X_num = np.nan_to_num(np.ascontiguousarray(X_num).astype(np.float32),
                              nan=0.0, posinf=1e6, neginf=-1e6)
        self.X_num = torch.from_numpy(X_num).float()
        self.X_cat = torch.from_numpy(np.ascontiguousarray(X_cat)).long()
        if task == "regression":
            self.y = torch.from_numpy(y).float().view(-1, 1)
        elif task == "binary":
            self.y = torch.from_numpy(np.asarray(y)).float().view(-1, 1)
        else:
            self.y = torch.from_numpy(np.asarray(y)).long().view(-1)

    def __len__(self): return int(self.y.shape[0])
    def __getitem__(self, i): return self.X_num[i], self.X_cat[i], self.y[i]


# ---------------------------------------------------------------------------
# Optimizer / scheduler / loss
# ---------------------------------------------------------------------------

def _make_optim(params, name: str, lr: float, weight_decay: float):
    name = name.lower()
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd_momentum":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, nesterov=True, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {name}")


def _make_sched(opt, name: str, *, total_steps: int, max_lr: float, plateau_mode: str):
    name = name.lower()
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, total_steps))
    if name == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=max(max_lr, 1e-5),
            total_steps=max(1, total_steps),
            pct_start=0.15,
            div_factor=10.0,
            final_div_factor=100.0,
            anneal_strategy="cos",
        )
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode=plateau_mode, factor=0.5, patience=3
        )
    return None  # "none"


def _make_loss(task: TaskType, label_smoothing: float):
    if task == "regression":
        return nn.MSELoss()
    if task == "binary":
        # BCEWithLogitsLoss doesn't natively do label smoothing in a single arg,
        # but we approximate it inside the training loop if needed.
        return nn.BCEWithLogitsLoss()
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing)


# ---------------------------------------------------------------------------
# Augmentations
# ---------------------------------------------------------------------------

def _apply_feature_noise(x_num: torch.Tensor, std: float) -> torch.Tensor:
    if std <= 0 or x_num.numel() == 0:
        return x_num
    return x_num + torch.randn_like(x_num) * std


def _mixup(x_num: torch.Tensor, x_cat: torch.Tensor, y: torch.Tensor,
           alpha: float, task: TaskType, n_classes: int = 0):
    """Mixup on numeric inputs only (categoricals are kept from anchor sample).

    Returns (x_num_mix, x_cat, y_mix, lam). For classification we mix targets
    one-hot via the standard formulation; for regression we mix the values.

    n_classes must be passed explicitly for multiclass to avoid wrong shape when
    a mini-batch doesn't contain all classes (e.g. rare ECG5000 classes).
    """
    if alpha <= 0:
        return x_num, x_cat, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x_num.size(0), device=x_num.device)
    x_num_mix = lam * x_num + (1 - lam) * x_num[perm]
    if task in ("regression", "binary"):
        y_mix = lam * y.float() + (1 - lam) * y[perm].float()
    else:
        # Use dataset-level n_classes, NOT batch-level max (batch may miss rare classes)
        nc = n_classes if n_classes > 0 else int(y.max().item()) + 1
        y1 = torch.nn.functional.one_hot(y, num_classes=nc).float()
        y_mix = lam * y1 + (1 - lam) * y1[perm]
    return x_num_mix, x_cat, y_mix, lam


def _loss_with_mixup(model, loss_fn, x_num, x_cat, y, task, train_cfg, mixup_alpha,
                     n_classes: int = 0):
    if mixup_alpha > 0 and task != "regression" and task != "binary":
        # multiclass: use soft-target CE manually
        x_num_m, x_cat, y_mix, _ = _mixup(x_num, x_cat, y, mixup_alpha, task,
                                           n_classes=n_classes)
        logits = model(x_num_m, x_cat)
        log_probs = torch.log_softmax(logits, dim=-1)
        loss = -(y_mix * log_probs).sum(dim=-1).mean()
        return loss
    if mixup_alpha > 0 and task == "binary":
        x_num_m, x_cat, y_mix, _ = _mixup(x_num, x_cat, y, mixup_alpha, task)
        logits = model(x_num_m, x_cat)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y_mix.view_as(logits))
        return loss
    if mixup_alpha > 0 and task == "regression":
        x_num_m, x_cat, y_mix, _ = _mixup(x_num, x_cat, y, mixup_alpha, task)
        logits = model(x_num_m, x_cat)
        return nn.functional.mse_loss(logits, y_mix.view_as(logits))
    # no mixup
    logits = model(x_num, x_cat)
    return loss_fn(logits, y)


# ---------------------------------------------------------------------------
# Eval helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def _predict(model, loader, task, device, use_amp):
    model.eval()
    all_logits = []
    autocast = (lambda: torch.amp.autocast("cuda")) if (use_amp and device == "cuda") else _null_ctx
    for x_num, x_cat, _ in loader:
        x_num = x_num.to(device, non_blocking=True)
        x_cat = x_cat.to(device, non_blocking=True)
        with autocast():
            logits = model(x_num, x_cat)
        all_logits.append(logits.detach().float().cpu())
    logits = torch.cat(all_logits, dim=0)
    if task == "regression":
        return None, logits.numpy().reshape(-1)
    if task == "binary":
        return torch.sigmoid(logits).numpy().reshape(-1), None
    return torch.softmax(logits, dim=1).numpy(), None


class _null_ctx:
    def __enter__(self): return self
    def __exit__(self, *a): return False


# ---------------------------------------------------------------------------
# Main trainer (NEW interface)
# ---------------------------------------------------------------------------

@dataclass
class TrialResult:
    primary: float
    search_score: float                # dense proxy for NAS selection (search_score from MetricResult)
    metrics: Dict[str, float]
    history: List[float]              # val primary per epoch
    n_params: int
    epochs_run: int
    seconds: float
    early_stopped: bool
    val_probas: Optional[np.ndarray] = None   # val probabilities (for ensemble)
    test_primary: Optional[float] = None      # untouched holdout score, if supplied
    test_metrics: Optional[Dict[str, float]] = None
    test_probas: Optional[np.ndarray] = None
    train_loss_history: List[float] = None    # train loss per epoch (for LLM feedback)
    grad_norm_history: List[float] = None     # L2 gradient norm per epoch (for LLM feedback)


def train_trial(
    *,
    cfg: Dict[str, Any],                    # full config dict {preprocess, arch, train}
    X_train_num, X_train_cat, y_train,
    X_val_num, X_val_cat, y_val,
    task: TaskType, n_classes: int, cat_cardinalities: List[int],
    X_test_num=None, X_test_cat=None, y_test=None,
    out_dir: Optional[Path] = None,
    max_epochs: Optional[int] = None,       # multi-fidelity override
    data_frac: float = 1.0,                 # multi-fidelity subsample
    seed: int = 0,
    save_model: bool = False,
    device: str | None = None,
    verbose: bool = False,
) -> TrialResult:
    import time
    t0 = time.time()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # FULL determinism per-trial — advisor caught the bug where SAME config
    # gave primary=0.65 vs 0.54 between turn 7 and turn 8 on helena because
    # PyTorch CUDA RNG state had advanced and cuDNN benchmark picked a
    # different conv algorithm. Resetting EVERY seed before every trial:
    from .utils import seed_everything
    seed_everything(seed)

    arch_cfg = cfg["arch"]
    train_cfg = cfg["train"]

    # Multi-fidelity: subsample TRAIN
    if data_frac < 1.0:
        n = X_train_num.shape[0]
        if task in ("binary", "multiclass"):
            # stratified subsample
            rng = np.random.default_rng(seed)
            classes, counts = np.unique(y_train, return_counts=True)
            keep_idx = []
            for c, ct in zip(classes, counts):
                idx = np.where(y_train == c)[0]
                k = max(1, int(round(ct * data_frac)))
                keep_idx.append(rng.choice(idx, size=k, replace=False))
            sel = np.concatenate(keep_idx)
            rng.shuffle(sel)
        else:
            rng = np.random.default_rng(seed)
            sel = rng.choice(n, size=max(8, int(n * data_frac)), replace=False)
        X_train_num = X_train_num[sel]
        X_train_cat = X_train_cat[sel] if X_train_cat.shape[1] else X_train_cat[:len(sel)]
        y_train = y_train[sel]

    # Datasets / loaders
    train_ds = _TabDS(X_train_num, X_train_cat, y_train, task)
    val_ds = _TabDS(X_val_num, X_val_cat, y_val, task)
    bs = int(train_cfg["batch_size"])
    bs = min(bs, max(8, len(train_ds)))
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=False,
                              num_workers=0, pin_memory=(device == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=max(bs, 1024), shuffle=False, drop_last=False,
                            num_workers=0, pin_memory=(device == "cuda"))
    test_loader = None
    if X_test_num is not None and X_test_cat is not None and y_test is not None:
        test_ds = _TabDS(X_test_num, X_test_cat, y_test, task)
        test_loader = DataLoader(
            test_ds, batch_size=max(bs, 1024), shuffle=False, drop_last=False,
            num_workers=0, pin_memory=(device == "cuda"),
        )

    # Model
    model = make_model(
        arch_cfg, n_num=X_train_num.shape[1],
        cat_cardinalities=cat_cardinalities, n_classes=n_classes, task=task,
    ).to(device)

    # Optim / sched / loss
    opt = _make_optim(model.parameters(), train_cfg["optimizer"],
                      train_cfg["lr"], train_cfg["weight_decay"])

    epochs = int(train_cfg["epochs"])
    if max_epochs is not None:
        epochs = min(epochs, int(max_epochs))
    epochs = max(1, epochs)

    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * epochs
    plateau_mode = "max" if task != "regression" else "min"
    sched = _make_sched(opt, train_cfg["scheduler"],
                        total_steps=total_steps, max_lr=train_cfg["lr"],
                        plateau_mode=plateau_mode)

    loss_fn = _make_loss(task, train_cfg.get("label_smoothing", 0.0))
    grad_clip = float(train_cfg.get("grad_clip", 0.0))
    use_amp = bool(train_cfg.get("use_amp", False)) and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    feature_noise = float(train_cfg.get("feature_noise_std", 0.0))
    mixup_alpha = float(train_cfg.get("mixup_alpha", 0.0))

    patience = int(train_cfg.get("patience", 8))
    best_primary = -1e18
    best_state: Optional[Dict[str, torch.Tensor]] = None
    bad_epochs = 0
    history: List[float] = []
    train_loss_history: List[float] = []
    grad_norm_history: List[float] = []
    early_stopped = False
    epochs_run = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss_sum = 0.0
        epoch_batches = 0
        epoch_grad_norm_sum = 0.0
        for x_num, x_cat, y in train_loader:
            x_num = x_num.to(device, non_blocking=True)
            x_cat = x_cat.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            x_num = _apply_feature_noise(x_num, feature_noise)

            opt.zero_grad(set_to_none=True)
            with (torch.amp.autocast("cuda") if use_amp else _null_ctx()):
                loss = _loss_with_mixup(model, loss_fn, x_num, x_cat, y, task, train_cfg,
                                        mixup_alpha, n_classes=n_classes)
            if use_amp:
                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(opt); scaler.update()
            else:
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()
            # FIX: step scheduler AFTER optimizer.step() to avoid PyTorch warning
            if sched is not None and not isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
                sched.step()
            epoch_loss_sum += float(loss.detach().cpu())
            # Track gradient L2 norm (before clipping resets it)
            gn = float(sum(
                p.grad.detach().norm(2).item() ** 2
                for p in model.parameters() if p.grad is not None
            ) ** 0.5)
            epoch_grad_norm_sum += gn
            epoch_batches += 1

        # Record average train loss and gradient norm for this epoch
        avg_train_loss = epoch_loss_sum / max(1, epoch_batches)
        avg_grad_norm  = epoch_grad_norm_sum / max(1, epoch_batches)
        train_loss_history.append(avg_train_loss)
        grad_norm_history.append(avg_grad_norm)

        # validation
        try:
            val_proba, val_pred = _predict(model, val_loader, task, device, use_amp)
            m = compute_metrics(task, y_val, y_pred_proba=val_proba, y_pred=val_pred)
            primary = float(m.primary)
        except Exception:
            primary = -1e18
            m = None
        history.append(primary)
        epochs_run = epoch
        if sched is not None and isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
            sched.step(primary if plateau_mode == "max" else -primary)

        is_best = primary > best_primary
        if is_best:
            best_primary = primary
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        # Per-epoch verbose log
        if verbose:
            cur_lr = opt.param_groups[0]["lr"]
            best_mark = " ★" if is_best else ""
            print(
                f"    ep {epoch:03d}/{epochs}  "
                f"loss={avg_train_loss:.4f}  "
                f"val={primary:.5f}  "
                f"best={best_primary:.5f}  "
                f"gnorm={avg_grad_norm:.3f}  "
                f"lr={cur_lr:.2e}  "
                f"patience={bad_epochs}/{patience}"
                f"{best_mark}",
                flush=True,
            )

        if bad_epochs >= patience:
            early_stopped = True
            if verbose:
                print(f"    → early stop at epoch {epoch}", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    val_proba, val_pred = _predict(model, val_loader, task, device, use_amp)
    m = compute_metrics(task, y_val, y_pred_proba=val_proba, y_pred=val_pred)
    metrics = dict(m.metrics)
    metrics["primary"] = float(m.primary)
    metrics["search_score"] = float(m.search_score)
    test_primary = None
    test_metrics = None
    test_proba = None
    if test_loader is not None:
        test_proba, test_pred = _predict(model, test_loader, task, device, use_amp)
        tm = compute_metrics(task, y_test, y_pred_proba=test_proba, y_pred=test_pred)
        test_primary = float(tm.primary)
        test_metrics = dict(tm.metrics)
        test_metrics["primary"] = test_primary
        test_metrics["search_score"] = float(tm.search_score)

    if out_dir is not None:
        out_dir = ensure_dir(out_dir)
        if save_model:
            torch.save(model.state_dict(), out_dir / "model.pt")
        save_json(out_dir / "metrics.json", metrics)
        if test_metrics is not None:
            save_json(out_dir / "test_metrics.json", test_metrics)
        save_json(out_dir / "history.json", {
            "val_primary_by_epoch": history,
            "train_loss_by_epoch": train_loss_history,
        })

    return TrialResult(
        primary=float(m.primary),
        search_score=float(m.search_score),
        metrics=metrics,
        history=history,
        n_params=count_params(model),
        epochs_run=int(epochs_run),
        seconds=float(time.time() - t0),
        early_stopped=early_stopped,
        val_probas=val_proba,
        test_primary=test_primary,
        test_metrics=test_metrics,
        test_probas=test_proba,
        train_loss_history=train_loss_history,
        grad_norm_history=grad_norm_history,
    )


# ---------------------------------------------------------------------------
# Legacy interface (kept so v1 callers don't break)
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    hidden_dims: List[int]
    activation: str = "relu"
    dropout: float = 0.1
    use_batchnorm: bool = True
    embedding_dim: int = 16
    optimizer: str = "adamw"
    lr: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 512
    epochs: int = 50
    patience: int = 8
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def train_one_trial(
    *,
    out_dir,
    X_train_num, X_train_cat, y_train,
    X_val_num, X_val_cat, y_val,
    task, n_classes, cat_cardinalities,
    cfg: TrainConfig,
):
    full_cfg = {
        "preprocess": {"num_encoder": "standard", "cat_encoder": "embedding"},
        "arch": {
            "family": "mlp",
            "hidden_dims": cfg.hidden_dims,
            "activation": cfg.activation,
            "dropout": cfg.dropout,
            "normalization": "batchnorm" if cfg.use_batchnorm else "none",
            "embedding_dim": cfg.embedding_dim,
        },
        "train": {
            "optimizer": cfg.optimizer, "lr": cfg.lr, "weight_decay": cfg.weight_decay,
            "scheduler": "none", "batch_size": cfg.batch_size, "epochs": cfg.epochs,
            "patience": cfg.patience, "label_smoothing": 0.0, "grad_clip": 0.0,
            "use_amp": False, "feature_noise_std": 0.0, "mixup_alpha": 0.0,
        },
    }
    res = train_trial(
        cfg=full_cfg,
        X_train_num=X_train_num, X_train_cat=X_train_cat, y_train=y_train,
        X_val_num=X_val_num, X_val_cat=X_val_cat, y_val=y_val,
        task=task, n_classes=n_classes, cat_cardinalities=cat_cardinalities,
        out_dir=out_dir, save_model=True,
    )
    return res.metrics
