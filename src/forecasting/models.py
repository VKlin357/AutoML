"""
Forecasting model zoo for NAS search space.

Models:
  Linear       - DLinear: channel-independent linear projection (strong baseline)
  NLinear      - NLinear: with instance normalization trick
  MLP          - flat MLP treating lookback window as tabular features
  PatchMLP     - split into patches, MLP on concatenated patch embeddings
  TCN          - Temporal Convolutional Network with residual blocks
  ForecastTransformer - Transformer with temporal + channel attention
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# 1. Linear (DLinear) — very strong baseline for forecasting
# ─────────────────────────────────────────────────────────────────────────────

class LinearForecast(nn.Module):
    """
    DLinear: Decomposition Linear.
    Trend + seasonal decomposition, then independent linear per channel.
    Zeng et al. 2023 — AAAI.
    """
    def __init__(self, lookback: int, horizon: int, n_channels: int,
                 decompose: bool = True, individual: bool = True,
                 target_idx: int = -1):
        super().__init__()
        self.lookback = lookback
        self.horizon = horizon
        self.n_channels = n_channels
        self.decompose = decompose
        self.individual = individual
        self.target_idx = target_idx if target_idx >= 0 else n_channels - 1

        if decompose:
            kernel = 25
            self.avg = nn.AvgPool1d(kernel_size=kernel, stride=1,
                                     padding=kernel // 2, count_include_pad=False)

        if individual:
            self.trend_proj    = nn.ModuleList([nn.Linear(lookback, horizon) for _ in range(n_channels)])
            self.seasonal_proj = nn.ModuleList([nn.Linear(lookback, horizon) for _ in range(n_channels)])
        else:
            self.trend_proj    = nn.Linear(lookback, horizon)
            self.seasonal_proj = nn.Linear(lookback, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, lookback, C)
        if self.decompose:
            trend = self.avg(x.permute(0, 2, 1)).permute(0, 2, 1)  # (B, L, C)
            # fix length mismatch from padding
            if trend.shape[1] != x.shape[1]:
                trend = trend[:, :x.shape[1], :]
            seasonal = x - trend
        else:
            trend = x
            seasonal = x

        if self.individual:
            trend_out    = torch.stack([self.trend_proj[i](trend[:, :, i])
                                        for i in range(self.n_channels)], dim=-1)   # (B, H, C)
            seasonal_out = torch.stack([self.seasonal_proj[i](seasonal[:, :, i])
                                        for i in range(self.n_channels)], dim=-1)
        else:
            trend_out    = self.trend_proj(trend.permute(0, 2, 1)).permute(0, 2, 1)
            seasonal_out = self.seasonal_proj(seasonal.permute(0, 2, 1)).permute(0, 2, 1)

        return (trend_out + seasonal_out)[:, :, self.target_idx]  # (B, H) — predict target channel


# ─────────────────────────────────────────────────────────────────────────────
# 2. NLinear — with last-value normalization
# ─────────────────────────────────────────────────────────────────────────────

class NLinearForecast(nn.Module):
    """
    NLinear: subtract last value, apply linear, add last value back.
    Very simple but effective for non-stationary series.
    """
    def __init__(self, lookback: int, horizon: int, n_channels: int,
                 individual: bool = True, target_idx: int = -1):
        super().__init__()
        self.lookback = lookback
        self.horizon = horizon
        self.n_channels = n_channels
        self.individual = individual
        self.target_idx = target_idx if target_idx >= 0 else n_channels - 1

        if individual:
            self.linear = nn.ModuleList([nn.Linear(lookback, horizon)
                                          for _ in range(n_channels)])
        else:
            self.linear = nn.Linear(lookback, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C)
        last = x[:, -1:, :]   # (B, 1, C) — last observed value
        x_norm = x - last

        if self.individual:
            out = torch.stack([self.linear[i](x_norm[:, :, i])
                                for i in range(self.n_channels)], dim=-1)  # (B, H, C)
        else:
            out = self.linear(x_norm.permute(0, 2, 1)).permute(0, 2, 1)

        out = out + last   # add back last value
        return out[:, :, self.target_idx]  # (B, H) — predict target channel


# ─────────────────────────────────────────────────────────────────────────────
# 3. MLP — standard tabular MLP on flat window
# ─────────────────────────────────────────────────────────────────────────────

class MLPForecast(nn.Module):
    """MLP treating flattened lookback window as tabular features."""
    def __init__(self, lookback: int, horizon: int, n_channels: int,
                 hidden_size: int = 512, n_layers: int = 3,
                 dropout: float = 0.1, activation: str = "relu"):
        super().__init__()
        input_dim = lookback * n_channels
        act = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}[activation]

        layers = [nn.Linear(input_dim, hidden_size), act(), nn.Dropout(dropout)]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), act(), nn.Dropout(dropout)]
        layers.append(nn.Linear(hidden_size, horizon))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C)
        return self.net(x.flatten(1))   # (B, H)


# ─────────────────────────────────────────────────────────────────────────────
# 4. PatchMLP — divide time series into patches, MLP on patch embeddings
# ─────────────────────────────────────────────────────────────────────────────

class PatchMLPForecast(nn.Module):
    """
    Divide lookback window into non-overlapping patches,
    embed each patch, then MLP over all patch embeddings.
    Inspired by PatchTST (Nie et al. 2023).
    """
    def __init__(self, lookback: int, horizon: int, n_channels: int,
                 patch_size: int = 16, d_model: int = 128,
                 n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches  = lookback // patch_size
        self.n_channels = n_channels

        # Patch embedding per channel
        self.patch_embed = nn.Linear(patch_size, d_model)

        # MLP over all patch embeddings
        mlp_input = self.n_patches * d_model * n_channels
        layers = [nn.Linear(mlp_input, d_model * 4), nn.GELU(), nn.Dropout(dropout)]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(d_model * 4, d_model * 4), nn.GELU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(d_model * 4, horizon))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C)
        B, L, C = x.shape
        n = self.n_patches
        p = self.patch_size

        # Truncate to exact multiple of patch_size
        x = x[:, :n * p, :]                              # (B, n*p, C)
        x = x.reshape(B, n, p, C)                        # (B, n, p, C)
        x = x.permute(0, 3, 1, 2)                        # (B, C, n, p)
        patches = self.patch_embed(x)                     # (B, C, n, d_model)
        flat = patches.flatten(1)                         # (B, C*n*d_model)
        return self.mlp(flat)                             # (B, H)


# ─────────────────────────────────────────────────────────────────────────────
# 5. TCN — Temporal Convolutional Network
# ─────────────────────────────────────────────────────────────────────────────

class _TCNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation, padding=pad)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, dilation=dilation, padding=pad)
        self.drop  = nn.Dropout(dropout)
        self.relu  = nn.ReLU()
        self.norm1 = nn.LayerNorm(out_ch)
        self.norm2 = nn.LayerNorm(out_ch)
        self.skip  = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        res = self.skip(x)
        h = self.conv1(x)[..., :x.shape[-1]]
        h = self.relu(self.norm1(h.transpose(1, 2)).transpose(1, 2))
        h = self.drop(h)
        h = self.conv2(h)[..., :x.shape[-1]]
        h = self.relu(self.norm2(h.transpose(1, 2)).transpose(1, 2))
        h = self.drop(h)
        return self.relu(h + res)


class TCNForecast(nn.Module):
    """Temporal Convolutional Network for time-series forecasting."""
    def __init__(self, lookback: int, horizon: int, n_channels: int,
                 d_model: int = 64, n_layers: int = 4,
                 kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        channels = [d_model] * n_layers
        blocks = []
        in_ch = n_channels
        for i, out_ch in enumerate(channels):
            dilation = 2 ** i
            blocks.append(_TCNBlock(in_ch, out_ch, kernel_size, dilation, dropout))
            in_ch = out_ch
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Linear(d_model, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C)
        h = self.tcn(x.permute(0, 2, 1))   # (B, d_model, L)
        h = h[:, :, -1]                     # take last timestep (B, d_model)
        return self.head(h)                  # (B, H)


# ─────────────────────────────────────────────────────────────────────────────
# 6. ForecastTransformer — Transformer with temporal attention
# ─────────────────────────────────────────────────────────────────────────────

class ForecastTransformer(nn.Module):
    """
    Channel-independent Transformer for forecasting.
    Each channel is processed independently with shared weights.
    Uses patch tokenization + positional encoding.
    """
    def __init__(self, lookback: int, horizon: int, n_channels: int,
                 patch_size: int = 16, d_model: int = 128, n_heads: int = 4,
                 n_layers: int = 2, dropout: float = 0.1, ffn_factor: float = 4.0,
                 target_idx: int = -1):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches  = lookback // patch_size
        self.n_channels = n_channels
        self.d_model    = d_model
        self.target_idx = target_idx if target_idx >= 0 else n_channels - 1

        self.input_proj = nn.Linear(patch_size, d_model)
        self.pos_enc    = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)
        self.dropout    = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=int(d_model * ffn_factor),
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

        # Head: predict horizon from all patch representations
        self.head = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(self.n_patches * d_model, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C) → process each channel independently
        B, L, C = x.shape
        p = self.patch_size
        n = self.n_patches

        x = x[:, :n * p, :]                    # truncate (B, n*p, C)
        x = x.reshape(B, n, p, C)              # (B, n, p, C)
        x = x.permute(0, 3, 1, 2)             # (B, C, n, p)
        x = x.reshape(B * C, n, p)            # (B*C, n, p) — treat each channel separately
        x = self.input_proj(x) + self.pos_enc  # (B*C, n, d_model)
        x = self.dropout(x)
        x = self.transformer(x)               # (B*C, n, d_model)
        x = self.norm(x)
        x = x.reshape(B, C, n, self.d_model)  # (B, C, n, d_model)
        # Average over channels for single-channel target prediction
        x = x[:, self.target_idx]              # use target channel
        return self.head(x)                    # (B, H)


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_forecasting_model(config: dict, lookback: int, horizon: int,
                             n_channels: int, target_idx: int = -1) -> nn.Module:
    """Build forecasting model from NAS config dict."""
    family = config.get("family", "linear")

    if family == "linear":
        return LinearForecast(
            lookback=lookback, horizon=horizon, n_channels=n_channels,
            decompose=config.get("decompose", True),
            individual=config.get("individual", True),
            target_idx=target_idx,
        )
    elif family == "nlinear":
        return NLinearForecast(
            lookback=lookback, horizon=horizon, n_channels=n_channels,
            individual=config.get("individual", True),
            target_idx=target_idx,
        )
    elif family == "mlp":
        return MLPForecast(
            lookback=lookback, horizon=horizon, n_channels=n_channels,
            hidden_size=config.get("hidden_size", 512),
            n_layers=config.get("n_layers", 3),
            dropout=config.get("dropout", 0.1),
            activation=config.get("activation", "relu"),
        )
    elif family == "patch_mlp":
        return PatchMLPForecast(
            lookback=lookback, horizon=horizon, n_channels=n_channels,
            patch_size=config.get("patch_size", 16),
            d_model=config.get("d_model", 128),
            n_layers=config.get("n_layers", 2),
            dropout=config.get("dropout", 0.1),
        )
    elif family == "tcn":
        return TCNForecast(
            lookback=lookback, horizon=horizon, n_channels=n_channels,
            d_model=config.get("d_model", 64),
            n_layers=config.get("n_layers", 4),
            kernel_size=config.get("kernel_size", 3),
            dropout=config.get("dropout", 0.1),
        )
    elif family == "transformer":
        return ForecastTransformer(
            lookback=lookback, horizon=horizon, n_channels=n_channels,
            patch_size=config.get("patch_size", 16),
            d_model=config.get("d_model", 128),
            n_heads=config.get("n_heads", 4),
            n_layers=config.get("n_layers", 2),
            dropout=config.get("dropout", 0.1),
            ffn_factor=config.get("ffn_factor", 4.0),
            target_idx=target_idx,
        )
    else:
        raise ValueError(f"Unknown forecasting model family: {family!r}")
