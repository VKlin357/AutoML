"""
LLM Agent for Forecasting NAS.

Sends trial history to GPT and gets new architecture configs back.
The prompt is forecasting-aware: tells LLM about channels, lookback,
horizon, and the temporal nature of the task.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from ..forecasting.search_space import (
    FORECASTING_FAMILIES, SEARCH_SPACE, TRAIN_SPACE, validate_config
)


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builder
# ─────────────────────────────────────────────────────────────────────────────

def _format_trial(t: dict, rank: int) -> str:
    arch  = t.get("arch", {})
    train = t.get("train", {})
    val_mse = t.get("val_mse", "?")
    lines = [
        f"  Rank #{rank+1}: family={arch.get('family')}  val_mse={val_mse:.6f}",
        f"    arch:  {json.dumps({k:v for k,v in arch.items() if k!='family'})}",
        f"    train: lr={train.get('lr')}  bs={train.get('batch_size')}  "
        f"epochs={train.get('epochs')}  sched={train.get('scheduler')}",
    ]
    return "\n".join(lines)


def build_forecasting_prompt(
    dataset_info: dict,
    history: List[dict],
    n_propose: int = 10,
) -> str:
    """Build LLM prompt for forecasting NAS."""
    name       = dataset_info.get("dataset_name", "unknown")
    n_channels = dataset_info.get("n_channels", 7)
    lookback   = dataset_info.get("lookback", 96)
    horizon    = dataset_info.get("horizon", 1)
    freq       = dataset_info.get("freq", "h")
    target_col = dataset_info.get("target_col", "target")
    n_train    = dataset_info.get("n_train", 0)

    freq_desc = {"h": "hourly", "15min": "15-minute", "d": "daily"}.get(freq, freq)

    # Sort history by val_mse (ascending — lower is better)
    valid = [t for t in history if t.get("val_mse") is not None and t["val_mse"] < 1e8]
    top5  = sorted(valid, key=lambda t: t["val_mse"])[:5]
    recent = valid[-3:] if len(valid) > 3 else valid

    history_str = ""
    if top5:
        history_str += "TOP-5 by val_mse (lower = better):\n"
        for i, t in enumerate(top5):
            history_str += _format_trial(t, i) + "\n"
    if recent:
        history_str += "\nMOST RECENT trials:\n"
        for i, t in enumerate(recent):
            history_str += _format_trial(t, i) + "\n"

    families_desc = """
Available model families:
  linear      - DLinear: decomposes trend+seasonal, linear projection per channel. Very fast, strong baseline.
  nlinear     - NLinear: subtracts last value, linear, adds back. Handles non-stationarity well.
  mlp         - Flat MLP on concatenated lookback features. Simple but can overfit.
  patch_mlp   - Divides time series into patches (like PatchTST), then MLP. Good for local patterns.
  tcn         - Temporal Convolutional Network with dilated convolutions. Captures multi-scale patterns.
  transformer - Channel-independent Transformer with patch tokenization. Best for complex temporal patterns.

Key insight: linear/nlinear are often VERY competitive with complex models on forecasting tasks.
             Start with linear, then explore patch_mlp and transformer for potential gains.
"""

    arch_space = """
Architecture hyperparameters:
  linear/nlinear:
    decompose:  true/false (DLinear only) — whether to decompose trend+seasonal
    individual: true/false — separate linear per channel vs shared

  mlp:
    hidden_size: 128 | 256 | 512 | 1024
    n_layers:    2 | 3 | 4 | 5
    dropout:     0.0 | 0.05 | 0.1 | 0.2 | 0.3
    activation:  relu | gelu | silu

  patch_mlp:
    patch_size: 8 | 16 | 24 | 32  (must divide lookback evenly)
    d_model:    64 | 128 | 256
    n_layers:   2 | 3 | 4
    dropout:    0.0 | 0.1 | 0.2

  tcn:
    d_model:     32 | 64 | 128 | 256
    n_layers:    3 | 4 | 5 | 6
    kernel_size: 3 | 5 | 7
    dropout:     0.0 | 0.1 | 0.2

  transformer:
    patch_size: 8 | 16 | 24  (must divide lookback evenly)
    d_model:    64 | 128 | 256  (must be divisible by n_heads)
    n_heads:    2 | 4 | 8
    n_layers:   2 | 3 | 4
    dropout:    0.0 | 0.05 | 0.1 | 0.2
    ffn_factor: 2.0 | 4.0

Training hyperparameters:
    lr:           1e-4 | 3e-4 | 5e-4 | 1e-3 | 2e-3 | 5e-3
    batch_size:   32 | 64 | 128 | 256
    epochs:       30 | 50 | 100 | 150 | 200
    patience:     10 | 15 | 20
    optimizer:    adam | adamw
    weight_decay: 0.0 | 1e-5 | 1e-4 | 1e-3
    scheduler:    none | cosine | step
    grad_clip:    0.0 | 1.0 | 5.0
    use_amp:      true | false
"""

    prompt = f"""You are an expert in time-series deep learning and Neural Architecture Search (NAS).
Your task: find the best neural architecture for forecasting the dataset below.

=== DATASET: {name.upper()} ===
  Task:       Time-series forecasting (univariate target, multivariate input)
  Frequency:  {freq_desc}
  Channels:   {n_channels} input channels
  Target:     '{target_col}' (channel index 0 in the model input)
  Lookback:   {lookback} timesteps (input window)
  Horizon:    {horizon} timestep(s) ahead (what we predict)
  Train size: {n_train} windows

=== TRIAL HISTORY ===
{history_str if history_str else "No trials yet — this is the first batch."}

=== SEARCH SPACE ===
{families_desc}
{arch_space}

=== YOUR TASK ===
Propose exactly {n_propose} diverse architecture configurations to try next.
Focus on minimizing val_mse (lower is better).

Guidelines:
1. If no trials yet — start diverse: 2 linear, 1 nlinear, 2 mlp, 2 patch_mlp, 1 tcn, 2 transformer
2. If linear/nlinear are winning — explore their hyperparameter variations
3. If transformer is competitive — try larger d_model, more layers, smaller patch_size
4. Avoid configs that obviously failed in history (same family + similar params)
5. For lookback={lookback}: valid patch sizes are {[p for p in [8,16,24,32] if lookback % p == 0]}
6. Transformer d_model must be divisible by n_heads

Respond ONLY with valid JSON array of {n_propose} configs:
[
  {{
    "arch": {{"family": "...", ...arch params...}},
    "train": {{"lr": ..., "batch_size": ..., "epochs": ..., "patience": ...,
               "optimizer": "...", "weight_decay": ..., "scheduler": "...",
               "grad_clip": ..., "use_amp": true}}
  }},
  ...
]
"""
    return prompt


# ─────────────────────────────────────────────────────────────────────────────
# LLM call
# ─────────────────────────────────────────────────────────────────────────────

def call_llm_for_forecasting(
    prompt: str,
    api_key: str,
    model: str = "gpt-4o",
    n_propose: int = 10,
    lookback: int = 96,
    max_retries: int = 3,
) -> List[Dict[str, Any]]:
    """Call LLM and parse proposed forecasting configs."""
    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=4000,
            )
            content = resp.choices[0].message.content
            if content is None:
                print(f"  [LLM] Got None content (attempt {attempt+1})")
                continue
            raw = content.strip()

            # Extract JSON array — try multiple strategies
            import re
            configs = None

            # Strategy 1: direct parse
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    configs = parsed
                elif isinstance(parsed, dict):
                    for key in ("configs", "configurations", "proposals", "results", "architectures"):
                        if key in parsed and isinstance(parsed[key], list):
                            configs = parsed[key]
                            break
                    if configs is None:
                        # try values
                        for v in parsed.values():
                            if isinstance(v, list) and len(v) > 0:
                                configs = v
                                break
            except json.JSONDecodeError:
                pass

            # Strategy 2: extract array with regex
            if configs is None:
                arr_match = re.search(r'\[[\s\S]*\]', raw)
                if arr_match:
                    try:
                        configs = json.loads(arr_match.group(0))
                    except json.JSONDecodeError:
                        pass

            if configs is None:
                print(f"  [LLM] Could not parse JSON (attempt {attempt+1}), raw[:200]={raw[:200]}")
                continue

            if not isinstance(configs, list):
                print(f"  [LLM] Warning: got {type(configs)}, expected list")
                continue

            # Validate and clean each config
            valid = []
            for cfg in configs:
                if not isinstance(cfg, dict):
                    continue
                if "arch" not in cfg or "train" not in cfg:
                    continue
                try:
                    cfg = validate_config(cfg, lookback=lookback)
                    valid.append(cfg)
                except Exception as e:
                    print(f"  [LLM] Config validation error: {e}")
                    continue

            if valid:
                print(f"  [LLM] Got {len(valid)}/{n_propose} valid configs (attempt {attempt+1})")
                return valid

        except json.JSONDecodeError as e:
            print(f"  [LLM] JSON parse error (attempt {attempt+1}): {e}")
        except Exception as e:
            print(f"  [LLM] Error (attempt {attempt+1}): {e}")

    print(f"  [LLM] All retries exhausted — returning empty list")
    return []
