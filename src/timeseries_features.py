"""
Feature engineering for time-series datasets to boost CatBoost/LightGBM.

Transforms flat window vectors [n_samples, n_channels * n_steps] into
rich feature matrices by computing per-channel statistics, FFT components,
and temporal difference features.

Usage:
    from src.timeseries_features import make_ts_features, TS_CONFIGS

    X_train_fe = make_ts_features(X_train, dataset="harth")
    X_val_fe   = make_ts_features(X_val,   dataset="harth")
    X_test_fe  = make_ts_features(X_test,  dataset="harth")
"""

import numpy as np
from typing import Optional

# Dataset-specific channel/step configs
TS_CONFIGS = {
    "harth": {
        "n_channels": 6,
        "n_steps": 128,
        "channel_names": ["back_x", "back_y", "back_z", "thigh_x", "thigh_y", "thigh_z"],
    },
    "pamap2": {
        "n_channels": 9,
        "n_steps": 128,
        "channel_names": ["hand_x", "hand_y", "hand_z",
                          "chest_x", "chest_y", "chest_z",
                          "ankle_x", "ankle_y", "ankle_z"],
    },
    "ecg5000": {
        "n_channels": 1,
        "n_steps": 140,
        "channel_names": ["ecg"],
    },
    "emg_gestures": {
        "n_channels": 8,
        "n_steps": 8,
        "channel_names": [f"emg_{i}" for i in range(8)],
    },
}


def make_ts_features(
    X: np.ndarray,
    dataset: str,
    n_fft_components: int = 10,
    add_cross_channel: bool = True,
) -> np.ndarray:
    """
    Convert flat window matrix to feature-engineered matrix.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_channels * n_steps)
    dataset : one of 'harth', 'pamap2', 'ecg5000', 'emg_gestures'
    n_fft_components : number of FFT magnitude components per channel
    add_cross_channel : add cross-channel correlation features

    Returns
    -------
    X_fe : ndarray of shape (n_samples, n_features_engineered)
    """
    cfg = TS_CONFIGS[dataset.lower()]
    n_channels = cfg["n_channels"]
    n_steps = cfg["n_steps"]
    n_samples = X.shape[0]

    assert X.shape[1] == n_channels * n_steps, (
        f"Expected {n_channels * n_steps} features, got {X.shape[1]}"
    )

    # Reshape to (n_samples, n_steps, n_channels)
    # Original flat order: t0_ch0, t0_ch1, ..., t0_chN, t1_ch0, ...
    # Actually in data.py: flat_cols = [f"t{t}_{c}" for t in range(window_size) for c in sensor_cols]
    # So order is: (t=0,ch=0), (t=0,ch=1), ..., (t=0,ch=N-1), (t=1,ch=0), ...
    # → reshape to (n_samples, n_steps, n_channels)
    W = X.reshape(n_samples, n_steps, n_channels)  # (N, T, C)

    feature_blocks = []

    # ── 1. Per-channel statistical features ──────────────────────────────────
    # mean, std, min, max, range, median, IQR, skewness, kurtosis, RMS
    mean = W.mean(axis=1)                          # (N, C)
    std  = W.std(axis=1)                           # (N, C)
    mn   = W.min(axis=1)                           # (N, C)
    mx   = W.max(axis=1)                           # (N, C)
    rng  = mx - mn                                 # (N, C)
    med  = np.median(W, axis=1)                    # (N, C)
    q25  = np.percentile(W, 25, axis=1)            # (N, C)
    q75  = np.percentile(W, 75, axis=1)            # (N, C)
    iqr  = q75 - q25                               # (N, C)
    rms  = np.sqrt((W ** 2).mean(axis=1))          # (N, C)

    # Skewness: E[(x-mu)^3] / std^3
    centered = W - mean[:, None, :]
    skew = (centered ** 3).mean(axis=1) / (std ** 3 + 1e-8)   # (N, C)

    # Kurtosis: E[(x-mu)^4] / std^4 - 3
    kurt = (centered ** 4).mean(axis=1) / (std ** 4 + 1e-8) - 3  # (N, C)

    feature_blocks += [mean, std, mn, mx, rng, med, iqr, rms, skew, kurt]

    # ── 2. Temporal difference features ──────────────────────────────────────
    # First differences: capture velocity/rate of change
    diff1 = np.diff(W, n=1, axis=1)               # (N, T-1, C)
    diff1_mean = diff1.mean(axis=1)                # (N, C)
    diff1_std  = diff1.std(axis=1)                 # (N, C)
    diff1_abs_mean = np.abs(diff1).mean(axis=1)    # (N, C) — mean absolute velocity

    feature_blocks += [diff1_mean, diff1_std, diff1_abs_mean]

    # Second differences: acceleration
    if n_steps > 2:
        diff2 = np.diff(W, n=2, axis=1)            # (N, T-2, C)
        diff2_std = diff2.std(axis=1)              # (N, C)
        feature_blocks.append(diff2_std)

    # ── 3. Zero crossing rate ─────────────────────────────────────────────────
    signs = np.sign(W - mean[:, None, :])          # zero-mean crossings
    zcr = (np.diff(signs, axis=1) != 0).mean(axis=1).astype(np.float32)  # (N, C)
    feature_blocks.append(zcr)

    # ── 4. FFT frequency features ─────────────────────────────────────────────
    if n_steps >= 8 and n_fft_components > 0:
        k = min(n_fft_components, n_steps // 2)
        fft_mag = np.abs(np.fft.rfft(W, axis=1))[:, 1:k+1, :]  # (N, k, C)
        # dominant frequency magnitude and index
        fft_flat = fft_mag.reshape(n_samples, -1)               # (N, k*C)
        feature_blocks.append(fft_flat)

        # Spectral energy per channel
        spectral_energy = (fft_mag ** 2).sum(axis=1)            # (N, C)
        feature_blocks.append(spectral_energy)

        # Dominant frequency index per channel (argmax)
        dom_freq_idx = fft_mag.argmax(axis=1).astype(np.float32)  # (N, C)
        feature_blocks.append(dom_freq_idx)

    # ── 5. Autocorrelation at lags 1, 2, 3 ───────────────────────────────────
    for lag in [1, 2, 3]:
        if n_steps > lag:
            ac = (centered[:, :-lag, :] * centered[:, lag:, :]).mean(axis=1) / (std ** 2 + 1e-8)
            feature_blocks.append(ac)  # (N, C)

    # ── 6. Cross-channel features ─────────────────────────────────────────────
    if add_cross_channel and n_channels > 1:
        # Signal magnitude area (SMA): sum of |channel| means → scalar per group of 3
        n_triplets = n_channels // 3
        for i in range(n_triplets):
            sl = slice(i * 3, (i + 1) * 3)
            sma = np.abs(W[:, :, sl]).mean(axis=(1, 2), keepdims=False)  # (N,)
            feature_blocks.append(sma[:, None])

        # Correlation between adjacent channels (first pair only to avoid explosion)
        if n_channels >= 2:
            for i in range(min(n_channels - 1, 6)):  # at most 6 pairs
                corr = (centered[:, :, i] * centered[:, :, i + 1]).mean(axis=1) / (
                    std[:, i] * std[:, i + 1] + 1e-8
                )
                feature_blocks.append(corr[:, None])

    # ── Concatenate all features ──────────────────────────────────────────────
    X_fe = np.concatenate(feature_blocks, axis=1).astype(np.float32)
    return X_fe


def make_ts_features_combined(
    X: np.ndarray,
    dataset: str,
    include_raw: bool = True,
    **kwargs,
) -> np.ndarray:
    """
    Optionally concatenate engineered features with original raw features.

    include_raw=True → [raw features | engineered features]
    include_raw=False → [engineered features only]
    """
    X_fe = make_ts_features(X, dataset, **kwargs)
    if include_raw:
        return np.concatenate([X, X_fe], axis=1).astype(np.float32)
    return X_fe


def feature_count(dataset: str, n_fft_components: int = 10) -> dict:
    """Report expected feature counts for a dataset."""
    cfg = TS_CONFIGS[dataset.lower()]
    C = cfg["n_channels"]
    T = cfg["n_steps"]
    k = min(n_fft_components, T // 2)

    stats_per_ch = 10        # mean, std, min, max, range, med, iqr, rms, skew, kurt
    diff_per_ch  = 4         # diff1_mean, diff1_std, diff1_abs, diff2_std
    zcr_per_ch   = 1
    fft_per_ch   = k + 2     # k magnitudes + spectral_energy + dom_freq_idx
    autocorr     = 3         # lags 1-3
    cross        = C // 3 + min(C - 1, 6)

    total_eng = C * (stats_per_ch + diff_per_ch + zcr_per_ch + fft_per_ch + autocorr) + cross
    total_raw = C * T

    return {
        "raw_features": total_raw,
        "engineered_features": total_eng,
        "combined_features": total_raw + total_eng,
    }


if __name__ == "__main__":
    for ds in TS_CONFIGS:
        counts = feature_count(ds)
        print(f"{ds}: raw={counts['raw_features']} → +{counts['engineered_features']} eng → {counts['combined_features']} total")
