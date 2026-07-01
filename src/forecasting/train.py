"""
Training loop for forecasting models.

Returns ForecastResult with MSE, MAE, best val score, training curves.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class ForecastResult:
    val_mse: float
    val_mae: float
    test_mse: float
    test_mae: float
    primary: float          # -val_mse (higher is better, for NAS compatibility)
    search_score: float     # same as primary
    val_history: List[float] = field(default_factory=list)
    epochs_run: int = 0
    seconds: float = 0.0
    n_params: int = 0
    early_stopped: bool = False
    error: Optional[str] = None


def _make_loader(X: np.ndarray, y: np.ndarray, batch_size: int,
                 shuffle: bool = True) -> DataLoader:
    X_t = torch.from_numpy(X).float()
    y_t = torch.from_numpy(y).float()
    # Reshape X to (N, L, C) from flat (N, L*C) — done outside
    ds = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=True, drop_last=False)


def _make_scheduler(optimizer, scheduler_name: str, epochs: int):
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs // 3, gamma=0.5)
    return None


def train_forecasting(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    lookback: int,
    n_channels: int,
    train_cfg: dict,
    device: str = "cuda",
    seed: int = 42,
    verbose: bool = True,
) -> ForecastResult:
    """
    Train a forecasting model and evaluate on val + test.

    Parameters
    ----------
    X_train, X_val, X_test : (N, lookback * n_channels) flat arrays
    y_train, y_val, y_test : (N,) or (N, horizon) targets
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    lr           = float(train_cfg.get("lr", 1e-3))
    epochs       = int(train_cfg.get("epochs", 100))
    patience     = int(train_cfg.get("patience", 15))
    batch_size   = int(train_cfg.get("batch_size", 64))
    optimizer_name = train_cfg.get("optimizer", "adamw")
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    scheduler_name = train_cfg.get("scheduler", "cosine")
    grad_clip    = float(train_cfg.get("grad_clip", 1.0))
    use_amp      = bool(train_cfg.get("use_amp", True)) and device.type == "cuda"

    # Reshape flat arrays to (N, L, C)
    def reshape(X):
        return X.reshape(-1, lookback, n_channels)

    X_tr_3d = reshape(X_train)
    X_va_3d = reshape(X_val)
    X_te_3d = reshape(X_test)

    train_loader = _make_loader(X_tr_3d, y_train, batch_size, shuffle=True)
    val_loader   = _make_loader(X_va_3d, y_val,   batch_size, shuffle=False)
    test_loader  = _make_loader(X_te_3d, y_test,  batch_size, shuffle=False)

    # Optimizer
    if optimizer_name == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    sched = _make_scheduler(opt, scheduler_name, epochs)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    criterion = nn.MSELoss()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    best_val_mse  = float("inf")
    best_state    = None
    no_improve    = 0
    val_history   = []
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            opt.zero_grad()
            if use_amp and scaler:
                with torch.cuda.amp.autocast():
                    pred = model(X_batch)
                    loss = criterion(pred.squeeze(-1), y_batch)
                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(opt)
                scaler.update()
            else:
                pred = model(X_batch)
                loss = criterion(pred.squeeze(-1), y_batch)
                loss.backward()
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()

        if sched:
            sched.step()

        # Validate
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                pred = model(X_batch.to(device)).squeeze(-1).cpu().numpy()
                val_preds.append(pred)
                val_targets.append(y_batch.numpy())

        val_pred_all = np.concatenate(val_preds)
        val_true_all = np.concatenate(val_targets)
        val_mse = float(np.mean((val_pred_all - val_true_all) ** 2))
        val_mae = float(np.mean(np.abs(val_pred_all - val_true_all)))
        val_history.append(-val_mse)   # higher is better for NAS

        if verbose:
            lr_now = opt.param_groups[0]["lr"]
            print(f"    ep {epoch:03d}/{epochs}  val_mse={val_mse:.6f}  "
                  f"val_mae={val_mae:.6f}  patience={no_improve}/{patience}  "
                  f"lr={lr_now:.2e}", flush=True)

        if val_mse < best_val_mse - 1e-7:
            best_val_mse = val_mse
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve   = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"    → early stop at epoch {epoch}", flush=True)
                break

    elapsed = time.time() - t0

    # Load best model and evaluate on test
    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    def evaluate(loader):
        preds, targets = [], []
        with torch.no_grad():
            for X_batch, y_batch in loader:
                pred = model(X_batch.to(device)).squeeze(-1).cpu().numpy()
                preds.append(pred)
                targets.append(y_batch.numpy())
        p = np.concatenate(preds)
        t = np.concatenate(targets)
        mse = float(np.mean((p - t) ** 2))
        mae = float(np.mean(np.abs(p - t)))
        return mse, mae

    val_mse_final, val_mae_final = evaluate(val_loader)
    test_mse, test_mae = evaluate(test_loader)

    primary = -val_mse_final   # higher is better

    return ForecastResult(
        val_mse=val_mse_final,
        val_mae=val_mae_final,
        test_mse=test_mse,
        test_mae=test_mae,
        primary=primary,
        search_score=primary,
        val_history=val_history,
        epochs_run=len(val_history),
        seconds=elapsed,
        n_params=n_params,
        early_stopped=no_improve >= patience,
    )
