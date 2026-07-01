"""
Feature engineering for time-series forecasting (CatBoost/LightGBM).

Design philosophy:
  - LESS is MORE for boosting: 150-200 well-chosen features beat 1000+ correlated ones
  - Lag selection: specific lags matching dataset frequency (not all 96)
  - Channel asymmetry: more features for target channel, summaries for others
  - No raw window dump: raw 96×7=672 numbers cause overfitting
  - Calendar features: critical for hourly/15-min data (hour of day, weekday)

Features produced (per dataset with 7 channels, lookback=96):
  Selected lags per channel    : 8 lags × 7 channels = 56
  Rolling stats (mean/std/min/max): 4 windows × 4 stats × 7 ch = 112
  Velocity (last diff)         : 7
  Acceleration (last 2nd diff) : 7
  Autocorrelations (lag 1,2,3) : 3 × 7 = 21
  FFT top-5 per channel        : 5 × 7 = 35
  Calendar (hourly)            : 8
  Total ≈ 246 features (vs 1072 before)
"""
from __future__ import annotations

from typing import List, Optional
import numpy as np
import pandas as pd


# Lag positions to use per frequency
# These match natural seasonality cycles
LAG_CONFIGS = {
    "h":     [1, 2, 3, 6, 12, 24, 48, 96],   # hourly: last hour, 6h, 12h, 1day, 2day, 4day
    "15min": [1, 2, 4, 8, 16, 32, 64, 96],   # 15min: 15m, 30m, 1h, 2h, 4h, 8h, 16h, 24h
    "d":     [1, 2, 3, 7, 14, 21, 30, 60],   # daily: 1d, 1w, 2w, 1m, 2m
    "default": [1, 2, 3, 6, 12, 24, 48, 96],
}

ROLLING_WINDOWS = {
    "h":     [4, 12, 24, 96],    # 4h, 12h, 1day, 4days
    "15min": [4, 16, 48, 96],    # 1h, 4h, 12h, 24h
    "d":     [3, 7, 14, 30],     # 3d, 1w, 2w, 1m
    "default": [4, 12, 24, 96],
}


def make_forecasting_features(
    X: np.ndarray,
    n_channels: int,
    lookback: int,
    freq: str = "h",
    n_fft: int = 5,
    timestamps: Optional[pd.DatetimeIndex] = None,
    include_raw: bool = False,          # False by default now!
) -> np.ndarray:
    """
    Compact, high-quality feature engineering for CatBoost/LightGBM.

    Parameters
    ----------
    X          : (N, lookback * n_channels) flat window matrix
    n_channels : number of input channels
    lookback   : number of past timesteps
    freq       : 'h' | '15min' | 'd' — controls lag selection
    n_fft      : FFT components per channel (default 5, not 10)
    timestamps : optional DatetimeIndex for calendar features
    include_raw: if True, append raw X (default False — avoids overfitting)

    Returns
    -------
    X_fe : (N, ~200) engineered features
    """
    N = X.shape[0]
    assert X.shape[1] == lookback * n_channels

    # Reshape: (N, lookback, n_channels) — time axis first for easy indexing
    W = X.reshape(N, lookback, n_channels)  # (N, T, C)

    lags    = [l for l in LAG_CONFIGS.get(freq, LAG_CONFIGS["default"]) if l < lookback]
    windows = [w for w in ROLLING_WINDOWS.get(freq, ROLLING_WINDOWS["default"]) if w < lookback]

    blocks = []

    # ── 1. Selected lag values (NOT all lookback, just key lags) ─────────────
    # For each channel, extract value at selected lag positions from the end
    for lag in lags:
        lag_vals = W[:, -lag, :]   # (N, C) — value at t-lag
        blocks.append(lag_vals)

    # ── 2. Rolling window statistics ──────────────────────────────────────────
    for w in windows:
        window_data = W[:, -w:, :]              # (N, w, C)
        blocks.append(window_data.mean(axis=1)) # mean
        blocks.append(window_data.std(axis=1))  # std
        blocks.append(window_data.min(axis=1))  # min
        blocks.append(window_data.max(axis=1))  # max

    # ── 3. Velocity and acceleration (last differences) ───────────────────────
    diff1 = np.diff(W, n=1, axis=1)            # (N, T-1, C)
    blocks.append(diff1[:, -1, :])              # last velocity per channel
    blocks.append(diff1[:, -3:, :].mean(axis=1))  # avg velocity last 3 steps

    if lookback > 2:
        diff2 = np.diff(W, n=2, axis=1)
        blocks.append(diff2[:, -1, :])          # last acceleration

    # ── 4. Global statistics (capture distribution of window) ────────────────
    mean = W.mean(axis=1)   # (N, C)
    std  = W.std(axis=1) + 1e-8
    blocks.append(mean)
    blocks.append(std)
    blocks.append(W.max(axis=1) - W.min(axis=1))   # range
    # Position of last value relative to window distribution
    last = W[:, -1, :]
    blocks.append((last - mean) / std)   # z-score of last value

    # ── 5. Autocorrelation at key lags ────────────────────────────────────────
    centered = W - mean[:, None, :]
    for lag in [1, 2, 3]:
        if lookback > lag:
            ac = (centered[:, :-lag, :] * centered[:, lag:, :]).mean(axis=1) / (std ** 2)
            blocks.append(ac)

    # ── 6. FFT — top-N frequency magnitudes per channel ──────────────────────
    if lookback >= 16 and n_fft > 0:
        k = min(n_fft, lookback // 2)
        fft_mag = np.abs(np.fft.rfft(W, axis=1))[:, 1:k+1, :]  # (N, k, C)
        blocks.append(fft_mag.reshape(N, -1))                    # (N, k*C)
        blocks.append((fft_mag ** 2).sum(axis=1))                # spectral energy (N, C)

    # ── 7. Trend features ─────────────────────────────────────────────────────
    # Linear trend coefficient per channel (slope of last half vs first half)
    first_half = W[:, :lookback//2, :].mean(axis=1)
    second_half = W[:, lookback//2:, :].mean(axis=1)
    blocks.append(second_half - first_half)  # (N, C) trend direction

    # ── 8. Calendar features ──────────────────────────────────────────────────
    if timestamps is not None:
        ts = pd.DatetimeIndex(timestamps)
        if freq in ("h", "15min"):
            blocks.append(np.sin(2*np.pi*ts.hour.values/24)[:, None].astype(np.float32))
            blocks.append(np.cos(2*np.pi*ts.hour.values/24)[:, None].astype(np.float32))
        if freq == "15min":
            blocks.append((ts.minute.values[:, None]/45).astype(np.float32))
        blocks.append(np.sin(2*np.pi*ts.dayofweek.values/7)[:, None].astype(np.float32))
        blocks.append(np.cos(2*np.pi*ts.dayofweek.values/7)[:, None].astype(np.float32))
        blocks.append(np.sin(2*np.pi*ts.month.values/12)[:, None].astype(np.float32))
        blocks.append(np.cos(2*np.pi*ts.month.values/12)[:, None].astype(np.float32))
        blocks.append((ts.dayofweek.values >= 5)[:, None].astype(np.float32))

    X_eng = np.concatenate(blocks, axis=1).astype(np.float32)

    if include_raw:
        return np.concatenate([X, X_eng], axis=1).astype(np.float32)
    return X_eng


def feature_count_forecasting(n_channels: int, lookback: int,
                               freq: str = "h", n_fft: int = 5,
                               has_calendar: bool = False) -> dict:
    C = n_channels
    lags = [l for l in LAG_CONFIGS.get(freq, LAG_CONFIGS["default"]) if l < lookback]
    windows = [w for w in ROLLING_WINDOWS.get(freq, ROLLING_WINDOWS["default"]) if w < lookback]
    k = min(n_fft, lookback // 2) if lookback >= 16 else 0

    lag_feats     = len(lags) * C
    rolling_feats = len(windows) * 4 * C
    diff_feats    = C * 3 + (C if lookback > 2 else 0)
    global_feats  = C * 4
    autocorr      = 3 * C
    fft_feats     = (k * C + C) if k > 0 else 0
    trend_feats   = C
    calendar      = 8 if has_calendar else 0

    eng = lag_feats + rolling_feats + diff_feats + global_feats + autocorr + fft_feats + trend_feats + calendar
    return {
        "raw_features": lookback * C,
        "engineered_features": eng,
        "breakdown": {
            "selected_lags": lag_feats,
            "rolling_stats": rolling_feats,
            "differences": diff_feats,
            "global_stats": global_feats,
            "autocorr": autocorr,
            "fft": fft_feats,
            "trend": trend_feats,
            "calendar": calendar,
        }
    }
