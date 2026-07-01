"""
LLM-guided Neural Architecture Search orchestrator — v2.

Wires together:
  data.load_raw              -> raw splits (OpenML / built-in / CSV)
  preprocessing.Preprocessor -> per-trial preprocessing (part of search space)
  surrogate.ConfigSurrogate  -> cheap pre-filter of LLM proposals
  multi_fidelity             -> cheap / full evaluation rungs (Hyperband-style)
  llm.client (v2)            -> cold-start / propose / reflect / refine
  llm.prompts (v2)           -> structured prompt builders

Algorithm (two modes, controlled by use_refine_loop):

  Phase 0 – Cold Start (coldstart_n trials)
    LLM proposes coldstart_n diverse configs spanning all 5 arch families.
    Each is evaluated at FULL fidelity to seed the surrogate.

  Phase 1A – REFINE loop (default, use_refine_loop=True)
    LLM diagnoses the current best trial's learning curve (val/train loss,
    gradient norms, overfitting signal) and picks ONE targeted action from
    a fixed vocabulary (add_layer, increase_dropout, reduce_lr, …).
    Every explore_pulse_every steps: inject one PROPOSE step for diversity.
    Action call counts are logged → interpretable thesis narrative.

  Phase 1B – PROPOSE loop (ablation, use_refine_loop=False)
    LLM sees full trial history + learning curves → freely proposes n new
    configs → surrogate ranks them → best is evaluated.

  All trial results saved to out_dir/trials/trial_NNN/.
  Running trials_index.json and best_trial.json are kept up to date.
  All LLM calls logged to out_dir/llm_log.jsonl.
"""
from __future__ import annotations

import copy
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .data import load_openml_raw, load_raw, RawDataset
from .preprocessing import Preprocessor, PreparedSplit
from .search_space import (
    ARCH_FAMILIES,
    sample_random_config,
    validate_config,
    mutate_random,
)
from .evolution import cold_start_random
from .surrogate import ConfigSurrogate
from .multi_fidelity import (
    evaluate_at_rung,
    successive_halving_threshold,
    CHEAP, MEDIUM, FULL,
    FidelityResult,
)
from .metrics import TaskType
from .utils import ensure_dir, load_json, save_json, seed_everything
from .llm.client import (
    OpenAILLM,
    AnthropicLLM,
    make_llm,
    ProxyChatOpenAILLM,   # kept for backward compat with existing run scripts
    llm_cold_start,
    llm_reflect,
    llm_propose,
    llm_refine_action,
    llm_multi_refine_action,
    llm_critic,
    LR_ACTIONS,
)
from .llm.prompts import (
    SYSTEM_PROMPT,
    build_user_payload_cold_start,
    build_user_payload_reflect,
    build_user_payload_propose,
    build_user_payload_refine,
    build_user_payload_multi_refine,
    build_critic_payload,
    build_corrector_payload,
    build_batch_propose_payload,
)
from .llm.logger import LLMLogger
from .llm.actions import apply_action, action_stats, ACTION_NAMES


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _prep_from_cfg(cfg: Dict, raw: RawDataset) -> PreparedSplit:
    """Fit-transform the preprocessor on the raw splits for this trial's config."""
    pre_cfg = cfg.get("preprocess", {"num_encoder": "standard", "cat_encoder": "embedding"})
    pre = Preprocessor(
        num_encoder=pre_cfg.get("num_encoder", "standard"),
        cat_encoder=pre_cfg.get("cat_encoder", "embedding"),
    )
    return pre.fit_transform(
        raw.X_train, raw.X_val, raw.X_test,
        raw.y_train, raw.y_val, raw.y_test,
        raw.num_cols, raw.cat_cols, raw.task, raw.n_classes,
    )


def _trial_record(trial_id: int, cfg: Dict, result: FidelityResult,
                  op: str = "?", rationale: str = "") -> Dict:
    return {
        "trial_id": trial_id,
        "op": op,
        "rationale": rationale,
        "primary": result.primary,           # accuracy / auc — for final reporting
        "search_score": result.search_score, # dense proxy — for NAS selection
        "metrics": result.metrics,
        "rung": result.rung,
        "n_params": result.n_params,
        "epochs_run": result.epochs_run,
        "seconds": result.seconds,
        "config": cfg,
        # Learning curves — all fields live in FidelityResult
        "val_history": result.history or [],
        "train_loss_history": result.train_loss_history or [],
        "grad_norm_history": result.grad_norm_history or [],
    }


def _sel_score(trial: Dict) -> float:
    """NAS selection score (dense proxy). Used for: parent selection, surrogate, batch selection.
    Pruned/error trials return -1e18 so they never become best parent."""
    if trial.get("rung") in ("cheap_pruned", "medium_pruned", "error"):
        return -1e18
    return float(trial.get("search_score", trial.get("primary", -1e18)))


def _primary_score(trial: Dict) -> float:
    """Raw accuracy score for final honest comparison with CatBoost/baselines.
    NEVER use this for NAS search decisions — use _sel_score instead."""
    if trial.get("rung") in ("cheap_pruned", "medium_pruned", "error"):
        return -1e18
    return float(trial.get("primary", -1e18))


def _learning_curve_summary(trial: Dict) -> Dict:
    """Compact learning curve digest to pass to LLM without flooding context.

    Includes:
    - val / train loss curves (subsampled to 10 points)
    - gradient norm stats (mean, max, final, trend) for instability detection
    - overfitting / underfitting signals
    - convergence speed (epochs to reach 95% of best val)
    """
    val_h = trial.get("val_history", [])
    trn_h = trial.get("train_loss_history", [])
    grd_h = trial.get("grad_norm_history", [])
    if not val_h:
        return {}

    # Subsample to at most 10 points
    def _subsample(lst, n=10):
        if len(lst) <= n:
            return [round(v, 4) for v in lst]
        step = max(1, len(lst) // n)
        return [round(lst[i], 4) for i in range(0, len(lst), step)][:n]

    best_val_epoch = int(np.argmax(val_h))
    final_val = val_h[-1]
    best_val = val_h[best_val_epoch]

    # Overfitting signal: val degrading while train loss still falling
    overfit_signal = "none"
    if len(val_h) >= 5 and len(trn_h) >= 5:
        q = max(1, len(val_h) // 4)
        early_avg = float(np.mean(val_h[:q]))
        late_avg = float(np.mean(val_h[-q:]))
        train_still_dropping = len(trn_h) >= 4 and trn_h[-1] < trn_h[-4] * 0.98
        if late_avg < early_avg - 0.005 and train_still_dropping:
            overfit_signal = "overfitting: val_primary degraded in last quarter while train_loss still dropping"
        elif late_avg < early_avg - 0.002:
            overfit_signal = "mild_overfit: val_primary plateaued or regressed"

    # Underfitting signal: best_val_primary too low relative to max in history
    underfit_signal = "none"
    if len(val_h) >= 3:
        # If val never exceeded 0.5 (binary/multiclass) it's likely underfitting
        if best_val < 0.55 and trial.get("config", {}).get("arch", {}).get("family") != "regression":
            underfit_signal = f"possible underfitting: best_val={best_val:.4f} is very low"
        # Val still rising at last epoch → more epochs needed
        if not trial.get("early_stopped", False) and len(val_h) >= 3 and val_h[-1] > val_h[-3]:
            underfit_signal = "val still rising at end: more epochs or higher lr may help"

    # Convergence speed: epochs until val first exceeded 95% of best_val
    convergence_epoch = best_val_epoch
    if best_val > 0:
        threshold_95 = best_val * 0.95
        for i, v in enumerate(val_h):
            if v >= threshold_95:
                convergence_epoch = i
                break

    # Gradient norm statistics (only if available)
    grad_norm_stats = {}
    if grd_h:
        grd_arr = np.array(grd_h)
        grad_norm_stats = {
            "mean": round(float(grd_arr.mean()), 4),
            "max": round(float(grd_arr.max()), 4),
            "final": round(float(grd_arr[-1]), 4),
            "trend": "increasing" if len(grd_h) >= 3 and grd_h[-1] > grd_h[0] * 1.5
                     else "decreasing" if len(grd_h) >= 3 and grd_h[-1] < grd_h[0] * 0.5
                     else "stable",
        }
        # Gradient instability signals
        if grad_norm_stats["max"] > 50:
            grad_norm_stats["signal"] = "EXPLODING: grad_norm > 50 — reduce lr or add grad_clip"
        elif grad_norm_stats["mean"] < 0.001:
            grad_norm_stats["signal"] = "VANISHING: grad_norm < 0.001 — model too deep or lr too low"
        elif grad_norm_stats["trend"] == "increasing" and grad_norm_stats["final"] > 10:
            grad_norm_stats["signal"] = "GROWING: gradient norms increasing — possible instability"
        else:
            grad_norm_stats["signal"] = "normal"

    summary = {
        "epochs_run": trial.get("epochs_run", len(val_h)),
        "best_val_epoch": best_val_epoch,
        "best_val_primary": round(best_val, 5),
        "final_val_primary": round(final_val, 5),
        "early_stopped": trial.get("early_stopped", False),
        "convergence_epoch_95pct": convergence_epoch,
        "val_curve_sampled": _subsample(val_h),
        "train_loss_curve_sampled": _subsample(trn_h),
        "overfit_signal": overfit_signal,
        "underfit_signal": underfit_signal,
    }
    if grad_norm_stats:
        summary["grad_norm_stats"] = grad_norm_stats

    # Prong G (per advisor request): add a TEXTUAL NARRATIVE of how the curves
    # behaved. LLMs reason about text more reliably than about raw arrays.
    # Pre-digesting the trajectory into a 1-2 sentence summary gives gpt-4o-mini
    # a much easier signal than 10 floats.
    narrative_parts = []
    if trn_h and len(trn_h) >= 3:
        t_start, t_end, t_min = trn_h[0], trn_h[-1], min(trn_h)
        if t_min < t_start * 0.5 and t_end > t_min * 1.05:
            narrative_parts.append(
                f"train_loss: started high {t_start:.3f}, dropped to min {t_min:.3f}, then bounced back to {t_end:.3f} → instability or overshoot")
        elif t_end < t_start * 0.7:
            narrative_parts.append(
                f"train_loss: clean descent {t_start:.3f}→{t_end:.3f} ({(1-t_end/t_start)*100:.0f}% reduction)")
        elif t_end > t_start * 0.95:
            narrative_parts.append(
                f"train_loss: barely moved ({t_start:.3f}→{t_end:.3f}) — model not learning, likely lr too low or arch too weak")
        else:
            narrative_parts.append(
                f"train_loss: slow descent {t_start:.3f}→{t_end:.3f}")

    if val_h and len(val_h) >= 3:
        v_start, v_end, v_max, v_max_idx = val_h[0], val_h[-1], max(val_h), int(np.argmax(val_h))
        if v_end < v_max - 0.01 and v_max_idx < len(val_h) * 0.7:
            narrative_parts.append(
                f"val_primary: peaked at {v_max:.4f} on epoch {v_max_idx}, then declined to {v_end:.4f} → OVERFITTING after epoch {v_max_idx}")
        elif v_end > v_start + 0.01:
            narrative_parts.append(
                f"val_primary: still climbing at end ({v_start:.4f}→{v_end:.4f}, peak {v_max:.4f}) — underfitting, more epochs or higher lr would help")
        else:
            narrative_parts.append(
                f"val_primary: plateaued near {v_max:.4f} (started {v_start:.4f}, ended {v_end:.4f})")

    if grad_norm_stats:
        sig = grad_norm_stats.get("signal", "normal")
        if sig != "normal":
            narrative_parts.append(f"gradient {sig.split(':')[0].lower()}")

    if best_val_epoch < 5 and len(val_h) > 10:
        narrative_parts.append(
            f"early peak (best at epoch {best_val_epoch}/{len(val_h)}) → lr likely too high; reduce_lr or explore_lr_extreme_low recommended")

    if narrative_parts:
        summary["curve_narrative"] = " | ".join(narrative_parts)

    return summary


def _history_for_prompt(trials: List[Dict], top_k: int = 8, recent_k: int = 5,
                        include_curves: bool = True) -> List[Dict]:
    """Compact history: top-K by primary + last recent_k, deduplicated.

    Sorted ASCENDING by primary (worst → best), following OPRO [Yang et al. 2023]
    which empirically shows ascending order outperforms descending due to LLM
    recency bias: the LLM gives more weight to items at the END of the prompt,
    so the best trials should appear last.

    include_curves: if False, omit learning_curve data (ablation mode — simulates
    systems like GENIUS/EvoPrompting that only see final validation scores).
    """
    seen = set()
    out = []
    # Collect top-K by primary and recent_k
    by_primary = sorted(trials, key=lambda x: x["primary"], reverse=True)
    candidates = {t["trial_id"]: t for t in by_primary[:top_k]}
    for t in trials[-recent_k:]:
        candidates[t["trial_id"]] = t

    for t in candidates.values():
        if t["trial_id"] not in seen:
            seen.add(t["trial_id"])
            entry = {
                "trial_id": t["trial_id"],
                "primary": round(t["primary"], 6),
                "op": t.get("op", "?"),
                "arch_family": t["config"].get("arch", {}).get("family", "?"),
                "config": t["config"],
            }
            if include_curves:
                entry["learning_curve"] = _learning_curve_summary(t)
            out.append(entry)

    # OPRO finding: ascending order (worst→best) is empirically best.
    # LLM recency bias means it attends more to the end of the context,
    # so the best trials should appear LAST in the prompt.
    out.sort(key=lambda x: x["primary"])
    return out


# ---------------------------------------------------------------------------
# LLM operator wrappers (with fallback to random)
# ---------------------------------------------------------------------------

def _cfg_hash(cfg: Dict) -> str:
    """Stable hash of a config dict — used to detect duplicate proposals."""
    import hashlib
    s = json.dumps(cfg, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(s.encode()).hexdigest()[:12]


def _llm_cold_start_configs(
    llm: ProxyChatOpenAILLM,
    n: int,
    summary: Dict,
    baseline: Dict,
    rng: random.Random,
) -> List[Dict]:
    """Ask LLM for n diverse cold-start configs. Fall back to random on failure.

    Bug-fix #5: temporarily bump temperature for cold-start to encourage
    extreme/corner-case proposals. LLMs at default temperature gravitate to
    safe textbook configs (lr=1e-3, dropout=0.2) which empirically lose to
    random search on tabular NAS.
    """
    saved_temp = getattr(llm, "temperature", 0.7)
    try:
        # Bump temperature for cold-start diversity
        # Anthropic caps at 1.0, OpenAI allows up to 2.0
        max_temp = 1.0 if isinstance(llm, AnthropicLLM) else 1.1
        if hasattr(llm, "temperature"):
            llm.temperature = max(saved_temp, max_temp)
        user = build_user_payload_cold_start(n=n, dataset_summary=summary, baseline=baseline)
        print(f"  [LLM→] cold_start prompt len={len(user)} chars  T={llm.temperature}")
        configs = llm_cold_start(llm, SYSTEM_PROMPT, user, expected_n=n)
        valid = []
        for c in configs:
            try:
                vc = validate_config(c)
                print(f"  [←LLM] cold_start config: family={vc['arch']['family']}  "
                      f"lr={vc['train'].get('lr','?'):.2e}  "
                      f"num_enc={vc['preprocess'].get('num_encoder','?')}")
                valid.append(vc)
            except Exception:
                pass
        if len(valid) >= max(1, n // 2):
            while len(valid) < n:
                valid.append(validate_config(sample_random_config(rng)))
            return valid[:n]
    except Exception as e:
        print(f"[NAS] LLM cold-start failed ({type(e).__name__}: {e}); using random init.")
    finally:
        # Always restore temperature
        if hasattr(llm, "temperature"):
            llm.temperature = saved_temp
    return cold_start_random(n, rng)



def _llm_reflect_hints(
    llm: ProxyChatOpenAILLM,
    history: List[Dict],
    summary: Dict,
    surrogate: ConfigSurrogate,
    k: int,
) -> Dict:
    """Ask LLM to reflect on the trajectory and return search hints."""
    try:
        importances = surrogate.feature_importances(top_k=10) if surrogate.fitted else []
        user = build_user_payload_reflect(
            history=history, k=k,
            dataset_summary=summary,
            surrogate_importances=importances,
        )
        print(f"  [LLM→] reflect prompt len={len(user)} chars")
        result = llm_reflect(llm, SYSTEM_PROMPT, user)
        if result and isinstance(result.get("hints"), dict):
            hints_out = result.get("hints", {})
            print(f"  [NAS] Reflection analysis: {result.get('analysis', '')[:300]}")
            print(f"  [←LLM] reflect hints: {json.dumps(hints_out, ensure_ascii=False)}")
            return hints_out
    except Exception as e:
        print(f"[NAS] LLM reflect failed ({type(e).__name__}: {e}).")
    return {}


def _llm_propose_configs(
    llm: OpenAILLM,
    history: List[Dict],
    summary: Dict,
    hints: Dict,
    n: int,
    rng: random.Random,
    seen_hashes: Optional[set] = None,
    llm_logger: Optional["LLMLogger"] = None,
) -> List[Dict]:
    """Simplified PROPOSE operator — LLM sees full history and proposes n new configs.

    Returns list of validated config dicts (may be shorter than n if LLM fails).
    Falls back to random configs on parse failure.
    """
    user = build_user_payload_propose(
        history=history,
        dataset_summary=summary,
        hints=hints,
        n=n,
    )
    print(f"  [LLM→] propose prompt len={len(user)} chars  n={n}  "
          f"hints={json.dumps({k: v for k, v in hints.items() if k in ('preferred_families','avoid_families','overfit_risk','next_priority')}, ensure_ascii=False)}")
    raw_resp = ""
    parsed_json: Dict = {}
    results: List[Dict] = []
    error_msg = ""
    try:
        proposals, raw_resp, parsed_json = llm_propose(llm, SYSTEM_PROMPT, user, expected_n=n)
        for p in proposals:
            cfg_raw = p.get("config", {})
            strategy = p.get("strategy", "?")
            reasoning_text = p.get("reasoning", "")[:150]
            try:
                cfg = validate_config(cfg_raw)
                h = _cfg_hash(cfg)
                if seen_hashes is not None and h in seen_hashes:
                    print(f"  [←LLM] DUPLICATE proposal (strategy={strategy}), skipping")
                    continue
                family = cfg.get("arch", {}).get("family", "?")
                lr = cfg.get("train", {}).get("lr", 0)
                print(f"  [←LLM] proposal strategy={strategy}  family={family}  "
                      f"lr={lr:.2e}  reasoning={reasoning_text}")
                results.append(cfg)
            except Exception as ve:
                print(f"  [←LLM] proposal invalid ({ve}), skipping")
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        print(f"[NAS] LLM propose failed ({error_msg}); falling back to random.")

    # Log to file
    if llm_logger is not None:
        diagnosis = parsed_json.get("diagnosis", {})
        llm_logger.log_call(
            operator="propose",
            prompt=user,
            raw_response=raw_resp,
            parsed={
                "reasoning": {
                    "failure_mode": diagnosis.get("main_bottleneck", ""),
                    "curve_obs": diagnosis.get("key_observation", ""),
                    "proposed_change": f"proposals: {[p.get('strategy') for p in parsed_json.get('proposals', [])]}",
                    "expected_effect": f"{len(results)} valid configs",
                },
                "rationale": diagnosis.get("key_observation", ""),
            },
            error=error_msg,
        )

    # Fallback: fill missing slots with random configs
    while len(results) < n:
        cfg = validate_config(sample_random_config(rng))
        results.append(cfg)
        print(f"  [Fallback] random config (family={cfg['arch']['family']})")

    return results[:n]


# ---------------------------------------------------------------------------
# REFINE operator helper (tool-call mode)
# ---------------------------------------------------------------------------

def _apply_freeform_changes(parent_cfg: Dict, changes: Dict[str, Any]) -> Dict:
    """Mode 3 helper: apply a dict of dotted-path → value updates to parent_cfg.

    Examples of supported paths:
      "train.lr"             → cfg["train"]["lr"] = value
      "arch.family"          → cfg["arch"]["family"] = value (triggers re-init
                               of family-specific keys if needed)
      "arch.hidden_dims"     → cfg["arch"]["hidden_dims"] = value
      "preprocess.num_encoder" → cfg["preprocess"]["num_encoder"] = value

    Returns a NEW validated config dict.
    """
    import copy
    out = copy.deepcopy(parent_cfg)
    family_changed = False
    new_family = None

    for path, value in changes.items():
        if not isinstance(path, str):
            continue
        parts = path.split(".")
        if len(parts) != 2:
            print(f"  [freeform] skipping invalid path: {path}")
            continue
        section, key = parts
        if section not in ("preprocess", "arch", "train"):
            print(f"  [freeform] skipping invalid section: {section}")
            continue
        # Detect family change — handled specially below
        if section == "arch" and key == "family":
            family_changed = True
            new_family = value
            continue
        out.setdefault(section, {})[key] = value

    # If family changed, re-init the arch sub-tree from a random sample of the
    # new family, then overlay any other arch.* changes from changes dict.
    if family_changed and new_family:
        from .search_space import ARCH_FAMILIES, sample_random_arch
        import random as _r
        if new_family in ARCH_FAMILIES:
            rng_local = _r.Random()
            out["arch"] = sample_random_arch(rng_local, family=new_family)
            # Re-apply any other arch.* changes from `changes`
            for path, value in changes.items():
                if not isinstance(path, str): continue
                parts = path.split(".")
                if len(parts) == 2 and parts[0] == "arch" and parts[1] != "family":
                    out["arch"][parts[1]] = value
        else:
            print(f"  [freeform] unknown family: {new_family}, ignoring family change")

    return validate_config(out)


def _build_parent_candidates(all_trials: List[Dict], top_k: int = 5) -> List[Dict]:
    """Prong D: pick top-K parents from population for LLM to choose from.

    Returns a compact summary per parent so the prompt stays small.
    """
    # Sort by primary descending and take top-K unique configs by hash
    sorted_trials = sorted(all_trials, key=lambda t: -_sel_score(t))
    seen_h = set()
    chosen = []
    for t in sorted_trials:
        h = _cfg_hash(t["config"])
        if h in seen_h: continue
        seen_h.add(h)
        chosen.append(t)
        if len(chosen) >= top_k: break

    out = []
    for i, t in enumerate(chosen):
        cfg = t["config"]
        arch = cfg.get("arch", {})
        train = cfg.get("train", {})
        # Compute "depth" generically across families
        if "hidden_dims" in arch:
            depth = len(arch["hidden_dims"])
        else:
            depth = arch.get("n_blocks", "?")
        # Curve summary (only top-2 keys for brevity)
        curve = _learning_curve_summary(t)
        curve_short = {
            "best_val": curve.get("best_val_primary"),
            "overfit": curve.get("overfit_signal", "?"),
            "underfit": curve.get("underfit_signal", "?"),
            "convergence_epoch_95pct": curve.get("convergence_epoch_95pct"),
            "early_stopped": curve.get("early_stopped"),
        }
        out.append({
            "parent_id": "best" if i == 0 else f"parent_{chr(ord('B') + i - 1)}",
            "trial_id": t.get("trial_id"),
            "primary": round(float(t.get("primary", 0)), 5),
            "family": arch.get("family", "?"),
            "depth": depth,
            "lr": train.get("lr"),
            "weight_decay": train.get("weight_decay"),
            "dropout": arch.get("dropout") or arch.get("attn_dropout") or 0.0,
            "n_params": t.get("n_params"),
            "age": len(all_trials) - all_trials.index(t),  # how many trials ago
            "cfg_hash": _cfg_hash(cfg),
            "curve_summary": curve_short,
        })
    return out


def _llm_freeform_step(
    llm: OpenAILLM,
    best_cfg: Dict,
    best_curve: Dict,
    refine_history: List[Dict],
    summary: Dict,
    rng: random.Random,
    seen_hashes: Optional[set] = None,
    llm_logger: Optional["LLMLogger"] = None,
    all_trials: Optional[List[Dict]] = None,
    action_history: Optional[List[Dict]] = None,
    conversation: Optional[List[Dict[str, str]]] = None,
    iteration_idx: int = 0,
    iteration_total: int = 25,
    first_turn_kickoff: str = "",
) -> Tuple[Dict, str]:
    """Mode 3: freeform refine — LLM returns a `changes` dict, not an action.

    Returns (new_cfg, op_label). Falls back to random mutation if LLM
    response is malformed or produces a duplicate.
    """
    from .llm.prompts import build_user_payload_freeform

    parent_candidates = _build_parent_candidates(all_trials, top_k=5) if all_trials else []

    user = build_user_payload_freeform(
        best_cfg=best_cfg,
        best_curve=best_curve,
        refine_history=refine_history[-6:],
        dataset_summary=summary,
        parent_candidates=parent_candidates,
        action_history=action_history or [],
    )
    iteration_remaining = max(0, iteration_total - iteration_idx)
    header = (
        f"=== ITERATION {iteration_idx + 1} of {iteration_total} "
        f"({iteration_remaining} remaining; freeform mode — no fixed action menu) ===\n"
        f"You MUST propose at least 1 concrete change this turn.\n"
    )
    if first_turn_kickoff:
        user = first_turn_kickoff + "\n" + header + "\n" + user
    else:
        user = header + "\n" + user

    print(f"  [LLM→] freeform prompt len={len(user)} chars  "
          f"parents={len(parent_candidates)}  history={len(action_history or [])}")

    raw_resp = ""
    parsed_json: Dict = {}
    error_msg = ""
    new_cfg: Optional[Dict] = None
    parent_id_picked = "best"
    parent_cfg_used = best_cfg
    changes: Dict = {}

    try:
        # Multi-turn chat using conversation history
        if conversation is None:
            raw_resp = llm.complete(SYSTEM_PROMPT, user)
        else:
            if not conversation:
                conversation.append({"role": "system", "content": SYSTEM_PROMPT})
            conversation.append({"role": "user", "content": user})
            raw_resp = llm.chat(conversation)
            conversation.append({"role": "assistant", "content": raw_resp})

        from .llm.client import _extract_json

        # Freeform validation: retry if history_check too short OR
        # LLM only touched safe fields 3+ consecutive times in action_history.
        _SAFE_FIELDS = {"train.lr", "train.epochs", "arch.dropout",
                        "train.weight_decay", "train.patience"}
        _STRUCTURAL_FIELDS = {"arch.family", "arch.n_blocks", "arch.block_width",
                              "arch.hidden_dims", "arch.d_token", "arch.n_heads",
                              "preprocess.num_encoder", "preprocess.cat_encoder"}

        def _safe_only_streak(ah):
            """Count how many of the last 4 action_history entries touched only safe fields."""
            streak = 0
            for entry in reversed((ah or [])[-4:]):
                fields = set(entry.get("changes_keys", []))
                if fields and fields.issubset(_SAFE_FIELDS):
                    streak += 1
                else:
                    break
            return streak

        _MAX_FREEFORM_RETRIES = 2
        for _retry in range(_MAX_FREEFORM_RETRIES + 1):
            parsed_json = _extract_json(raw_resp) or {}
            if not isinstance(parsed_json, dict):
                parsed_json = {}

            diag = parsed_json.get("diagnosis") or {}
            hc = diag.get("history_check", "")
            _changes_raw = parsed_json.get("changes") or {}
            _changes_keys = set(_changes_raw.keys()) if isinstance(_changes_raw, dict) else set()
            _safe_streak = _safe_only_streak(action_history)

            _hc_ok = isinstance(hc, str) and len(hc) >= 80
            _struct_needed = _safe_streak >= 3 and not _changes_keys.intersection(_STRUCTURAL_FIELDS)

            if (_hc_ok and not _struct_needed) or _retry == _MAX_FREEFORM_RETRIES:
                if not _hc_ok:
                    print(f"  [freeform] history_check too short ({len(hc)} chars) — using anyway (retries exhausted)")
                if _struct_needed:
                    print(f"  [freeform] structural field required (streak={_safe_streak}) — using anyway (retries exhausted)")
                break

            # Build retry message
            _retry_reasons = []
            if not _hc_ok:
                _retry_reasons.append(
                    f"history_check is too short ({len(hc)} chars, need >= 80). "
                    "List EVERY train.lr and arch.dropout value you tried with its result, "
                    "then state explicitly what you will NOT repeat this turn."
                )
            if _struct_needed:
                _retry_reasons.append(
                    f"Your last {_safe_streak} turns ONLY changed safe fields (lr/dropout/epochs). "
                    "R7 requires you to change at least one STRUCTURAL field this turn: "
                    "arch.family, arch.n_blocks, arch.block_width, arch.d_token, or preprocess.num_encoder."
                )
            _retry_msg = (
                "RETRY REQUIRED — your previous response violated mandatory rules:\n"
                + "\n".join(f"  {i+1}. {r}" for i, r in enumerate(_retry_reasons))
                + "\nPlease output a corrected JSON response now."
            )
            print(f"  [freeform] retry {_retry+1}/{_MAX_FREEFORM_RETRIES}: {'; '.join(_retry_reasons)}")
            if conversation is not None:
                conversation.append({"role": "user", "content": _retry_msg})
                raw_resp = llm.chat(conversation)
                conversation.append({"role": "assistant", "content": raw_resp})
            else:
                raw_resp = llm.complete(SYSTEM_PROMPT, _retry_msg)

        # Parent selection
        parent_id_picked = (parsed_json.get("parent_id") or "best").strip()
        if parent_id_picked != "best" and parent_candidates:
            for pc in parent_candidates:
                if pc.get("parent_id") == parent_id_picked:
                    for t in (all_trials or []):
                        if t.get("trial_id") == pc.get("trial_id"):
                            parent_cfg_used = t["config"]
                            break
                    break

        # Changes dict
        changes = parsed_json.get("changes") or {}
        if not isinstance(changes, dict):
            changes = {}

        if changes:
            print(f"  [←LLM] freeform changes: {list(changes.keys())}")
            new_cfg = _apply_freeform_changes(parent_cfg_used, changes)
            # Dedup check (very rare with freeform — LLM picks unique values)
            h = _cfg_hash(new_cfg)
            if seen_hashes is not None and h in seen_hashes:
                print(f"  [←LLM] duplicate after freeform changes, fall back to random_mutation")
                new_cfg = validate_config(mutate_random(parent_cfg_used, rng))
                op_label = "freeform_dedup_fallback"
            else:
                op_label = "freeform:" + "+".join(sorted(changes.keys()))[:80]
        else:
            # Empty changes — LLM tried to "pass" — apply random mutation
            print(f"  [←LLM] empty changes, fall back to random_mutation")
            new_cfg = validate_config(mutate_random(parent_cfg_used, rng))
            op_label = "freeform_empty_fallback"

    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        print(f"[NAS] LLM freeform failed ({error_msg}); random mutation.")

    if new_cfg is None:
        new_cfg = validate_config(mutate_random(parent_cfg_used, rng))
        op_label = "freeform_error_fallback"

    # Log
    if llm_logger is not None:
        diagnosis = (parsed_json.get("diagnosis") or {})
        llm_logger.log_call(
            operator="freeform" if changes else "freeform_fallback",
            prompt=user,
            raw_response=raw_resp,
            parsed={
                "reasoning": {
                    "failure_mode": diagnosis.get("failure_mode", ""),
                    "curve_obs": str(diagnosis.get("observation", ""))[:300],
                    "hypothesis": str(diagnosis.get("hypothesis", ""))[:300],
                    "proposed_change": ",".join(sorted(changes.keys())) if changes else "(empty)",
                    "expected_effect": str(diagnosis.get("prediction", ""))[:200],
                    "parent_id_picked": parent_id_picked,
                },
                "rationale": parsed_json.get("rationale", ""),
                "changes": changes,
            },
            error=error_msg,
        )

    return new_cfg, op_label


# ---------------------------------------------------------------------------
# LR-throttle helper (shared by multi_refine and critic_corrector)
# ---------------------------------------------------------------------------

def _check_lr_throttled(action_history: Optional[List[Dict]], window: int = 3, threshold: int = 2) -> bool:
    """Return True if LR actions dominated the recent action_history."""
    if not action_history:
        return False
    recent = action_history[-window:]
    lr_count = sum(
        1 for h in recent
        if h.get("action") in LR_ACTIONS or any(
            a.get("action") in LR_ACTIONS
            for a in (h.get("actions") or [])
            if isinstance(a, dict)
        )
    )
    return lr_count >= threshold


# ---------------------------------------------------------------------------
# Multi-refine step
# ---------------------------------------------------------------------------

def _llm_multi_refine_step(
    llm,
    best_cfg: Dict,
    best_curve: Dict,
    refine_history: List[Dict],
    summary: Dict,
    rng: random.Random,
    seen_hashes: Optional[set] = None,
    llm_logger=None,
    all_trials: Optional[List[Dict]] = None,
    action_history: Optional[List[Dict]] = None,
    conversation: Optional[List[Dict]] = None,
    iteration_idx: int = 0,
    iteration_total: int = 25,
    first_turn_kickoff: str = "",
) -> Tuple[Dict, str]:
    """MULTI-REFINE step: LLM picks 1-3 actions, applied sequentially → 1 trial."""
    parent_candidates = _build_parent_candidates(all_trials, top_k=5) if all_trials else []
    lr_throttled = _check_lr_throttled(action_history)

    user = build_user_payload_multi_refine(
        best_cfg=best_cfg,
        best_curve=best_curve,
        refine_history=refine_history[-6:],
        dataset_summary=summary,
        parent_candidates=parent_candidates,
        action_history=action_history or [],
        lr_throttled=lr_throttled,
    )
    iteration_remaining = max(0, iteration_total - iteration_idx)
    header = (
        f"=== ITERATION {iteration_idx + 1} of {iteration_total} "
        f"({iteration_remaining} remaining; multi-refine — pick 1-3 distinct actions) ===\n"
        f"You MUST propose at least 1 action. Do NOT declare the model 'good enough'.\n"
    )
    if first_turn_kickoff:
        user = first_turn_kickoff + "\n" + header + "\n" + user
    else:
        user = header + "\n" + user

    print(f"  [LLM→] multi_refine prompt len={len(user)} chars  "
          f"lr_throttled={lr_throttled}  parents={len(parent_candidates)}")

    raw_resp = ""
    parsed_json: Dict = {}
    new_cfg: Optional[Dict] = None
    parent_cfg_used = best_cfg
    parent_id_picked = "best"
    actions_applied: List[str] = []
    error_msg = ""

    try:
        actions, raw_resp, parsed_json = llm_multi_refine_action(
            llm, SYSTEM_PROMPT, user, conversation=conversation, lr_throttled=lr_throttled,
        )

        # Resolve parent
        parent_id_picked = (parsed_json.get("parent_id") or "best").strip()
        if parent_id_picked != "best" and parent_candidates:
            for pc in parent_candidates:
                if pc.get("parent_id") == parent_id_picked:
                    for t in (all_trials or []):
                        if t.get("trial_id") == pc.get("trial_id"):
                            parent_cfg_used = t["config"]
                            break
                    break

        if actions:
            # Apply actions sequentially to same parent
            working_cfg = copy.deepcopy(parent_cfg_used)
            for action_name, params in actions:
                working_cfg = apply_action(working_cfg, action_name, params, rng)
                actions_applied.append(action_name)
                print(f"    → applied: {action_name}")

            h = _cfg_hash(working_cfg)
            if seen_hashes is not None and h in seen_hashes:
                # Dedup: try escalating with switch_family
                working_cfg = apply_action(parent_cfg_used, "switch_family", {}, rng)
                h2 = _cfg_hash(working_cfg)
                if seen_hashes is not None and h2 in seen_hashes:
                    working_cfg = validate_config(mutate_random(parent_cfg_used, rng))
                actions_applied = ["switch_family_dedup"]

            new_cfg = working_cfg

    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        print(f"[NAS] LLM multi_refine failed ({error_msg}); random mutation.")

    if new_cfg is None:
        new_cfg = validate_config(mutate_random(parent_cfg_used, rng))
        actions_applied = ["random_fallback"]

    op_label = "multi_refine:" + "+".join(actions_applied) if actions_applied else "multi_refine_fallback"

    # Log
    if llm_logger is not None:
        diagnosis = parsed_json.get("diagnosis") or {}
        llm_logger.log_call(
            operator="multi_refine",
            prompt=user,
            raw_response=raw_resp,
            parsed={
                "actions": actions_applied,
                "failure_mode": diagnosis.get("failure_mode", ""),
                "observation": diagnosis.get("observation", ""),
                "rationale": parsed_json.get("rationale", ""),
                "parent_id_picked": parent_id_picked,
                "lr_throttled": lr_throttled,
            },
            error=error_msg,
        )

    # Update action_history with multi-action entry
    if action_history is not None and actions_applied and actions_applied != ["random_fallback"]:
        action_history.append({
            "step": iteration_idx,
            "mode": "multi_refine",
            "actions": [{"action": a} for a in actions_applied],
            "action": actions_applied[0],  # primary action for LR-throttle check
            "parent_id": parent_id_picked,
            "child_hash": _cfg_hash(new_cfg)[:8],
            "lr_throttled": lr_throttled,
        })

    return new_cfg, op_label


# ---------------------------------------------------------------------------
# Critic-Corrector step
# ---------------------------------------------------------------------------

def _llm_critic_corrector_step(
    llm,
    best_cfg: Dict,
    best_curve: Dict,
    refine_history: List[Dict],
    summary: Dict,
    rng: random.Random,
    all_trials: Optional[List[Dict]] = None,
    seen_hashes: Optional[set] = None,
    llm_logger=None,
    action_history: Optional[List[Dict]] = None,
    conversation: Optional[List[Dict]] = None,
    iteration_idx: int = 0,
    iteration_total: int = 25,
) -> Tuple[Dict, str]:
    """Two-stage pipeline: Critic analyses history → Corrector picks actions."""
    parent_candidates = _build_parent_candidates(all_trials, top_k=5) if all_trials else []
    lr_throttled = _check_lr_throttled(action_history)

    # ── Stage 1: Critic ──────────────────────────────────────────────────
    critic_user = build_critic_payload(
        all_trials=all_trials or [],
        dataset_summary=summary,
    )
    print(f"  [CRITIC→] prompt len={len(critic_user)} chars")
    critic_result: Dict = {}
    try:
        critic_result = llm_critic(llm, SYSTEM_PROMPT, critic_user)
    except Exception as e:
        print(f"  [CRITIC] failed ({e}), proceeding with empty verdict")

    # ── Stage 2: Corrector ───────────────────────────────────────────────
    corrector_user = build_corrector_payload(
        critic_result=critic_result,
        best_cfg=best_cfg,
        best_curve=best_curve,
        refine_history=refine_history[-6:],
        dataset_summary=summary,
        parent_candidates=parent_candidates,
        action_history=action_history or [],
        lr_throttled=lr_throttled,
    )
    iteration_remaining = max(0, iteration_total - iteration_idx)
    header = (
        f"=== ITERATION {iteration_idx + 1} of {iteration_total} "
        f"({iteration_remaining} remaining) ===\n"
    )
    corrector_user = header + "\n" + corrector_user

    print(f"  [CORRECTOR→] prompt len={len(corrector_user)} chars")

    raw_resp = ""
    parsed_json: Dict = {}
    new_cfg: Optional[Dict] = None
    parent_cfg_used = best_cfg
    parent_id_picked = "best"
    actions_applied: List[str] = []
    error_msg = ""

    try:
        actions, raw_resp, parsed_json = llm_multi_refine_action(
            llm, SYSTEM_PROMPT, corrector_user,
            conversation=conversation,
            lr_throttled=lr_throttled,
        )

        parent_id_picked = (parsed_json.get("parent_id") or "best").strip()
        if parent_id_picked != "best" and parent_candidates:
            for pc in parent_candidates:
                if pc.get("parent_id") == parent_id_picked:
                    for t in (all_trials or []):
                        if t.get("trial_id") == pc.get("trial_id"):
                            parent_cfg_used = t["config"]
                            break
                    break

        if actions:
            working_cfg = copy.deepcopy(parent_cfg_used)
            for action_name, params in actions:
                working_cfg = apply_action(working_cfg, action_name, params, rng)
                actions_applied.append(action_name)
                print(f"    → applied: {action_name}")
            h = _cfg_hash(working_cfg)
            if seen_hashes is not None and h in seen_hashes:
                working_cfg = apply_action(parent_cfg_used, "switch_family", {}, rng)
                actions_applied = ["switch_family_dedup"]
            new_cfg = working_cfg

    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        print(f"[NAS] Corrector failed ({error_msg}); random mutation.")

    if new_cfg is None:
        new_cfg = validate_config(mutate_random(parent_cfg_used, rng))
        actions_applied = ["random_fallback"]

    op_label = "critic_corrector:" + "+".join(actions_applied) if actions_applied else "critic_corrector_fallback"

    if llm_logger is not None:
        diagnosis = parsed_json.get("diagnosis") or {}
        llm_logger.log_call(
            operator="critic_corrector",
            prompt=corrector_user,
            raw_response=raw_resp,
            parsed={
                "critic_verdict": critic_result.get("verdict", ""),
                "actions": actions_applied,
                "failure_mode": diagnosis.get("failure_mode", ""),
                "rationale": parsed_json.get("rationale", ""),
                "lr_throttled": lr_throttled,
            },
            error=error_msg,
        )

    if action_history is not None and actions_applied and actions_applied != ["random_fallback"]:
        action_history.append({
            "step": iteration_idx,
            "mode": "critic_corrector",
            "actions": [{"action": a} for a in actions_applied],
            "action": actions_applied[0],
            "parent_id": parent_id_picked,
            "child_hash": _cfg_hash(new_cfg)[:8],
        })

    return new_cfg, op_label


# ---------------------------------------------------------------------------
# Batch round: LLM proposes N → surrogate filters → train top-K
# ---------------------------------------------------------------------------

def _llm_batch_round(
    llm,
    all_trials: List[Dict],
    surrogate,
    summary: Dict,
    raw,          # RawDataset — needed to build per-config PreparedSplit
    out_dir,
    rng: random.Random,
    seen_hashes: set,
    hints: Dict,
    batch_n: int = 15,
    batch_k: int = 3,
    seed: int = 42,
    device=None,
    verbose: bool = False,
    llm_logger=None,
    cheap_scores: Optional[List[float]] = None,
    medium_scores: Optional[List[float]] = None,
    cheap_scores_by_family: Optional[Dict[str, List[float]]] = None,
    medium_scores_by_family: Optional[Dict[str, List[float]]] = None,
    use_multi_fidelity: bool = True,
    n_train_rows: int = 100_000,
    require_llm: bool = False,
) -> List[Dict]:
    """One batch round: critic → LLM proposes batch_n → surrogate top-batch_k → train."""
    if cheap_scores is None:
        cheap_scores = []
    if medium_scores is None:
        medium_scores = []
    if cheap_scores_by_family is None:
        cheap_scores_by_family = {}
    if medium_scores_by_family is None:
        medium_scores_by_family = {}

    # ── Critic call ──────────────────────────────────────────────────────
    critic_user = build_critic_payload(all_trials=all_trials, dataset_summary=summary)
    critic_result: Dict = {}
    try:
        critic_result = llm_critic(llm, SYSTEM_PROMPT, critic_user)
    except Exception as e:
        print(f"  [BATCH-CRITIC] failed ({e})")

    # ── Propose N candidates ──────────────────────────────────────────────
    propose_user = build_batch_propose_payload(
        all_trials=all_trials,
        dataset_summary=summary,
        critic_result=critic_result,
        hints=hints,
        n=batch_n,
        batch_k=batch_k,
    )
    print(f"  [BATCH-PROPOSE→] len={len(propose_user)}  n={batch_n}  k={batch_k}")
    raw_resp = ""
    valid_cfgs: List[Dict] = []
    try:
        raw_resp = llm.complete(SYSTEM_PROMPT, propose_user)
        from .llm.client import _extract_json
        parsed = _extract_json(raw_resp) or {}
        configs_raw = parsed.get("configs", [])
        if not isinstance(configs_raw, list):
            configs_raw = []
        print(f"  [BATCH-PROPOSE←] got {len(configs_raw)} raw configs")
        for cfg in configs_raw:
            if not isinstance(cfg, dict):
                continue
            try:
                cfg = validate_config(cfg)
                h = _cfg_hash(cfg)
                if h not in seen_hashes:
                    valid_cfgs.append(cfg)
            except Exception:
                pass
        print(f"  [BATCH-PROPOSE] valid+unique: {len(valid_cfgs)}")
    except Exception as e:
        if require_llm:
            raise RuntimeError(f"LLM proposal failed in --require_llm mode: {e}") from e
        print(f"  [BATCH-PROPOSE] LLM failed ({e}), falling back to random")

    # Fallback: random configs if LLM gave too few
    if require_llm and len(valid_cfgs) < batch_k:
        raise RuntimeError(
            f"LLM returned only {len(valid_cfgs)} valid unique configs; "
            f"{batch_k} are required in --require_llm mode"
        )
    while len(valid_cfgs) < batch_k:
        # Use _sel_score so fallback mutates best full/medium trial, not cheap-pruned junk
        best = max(all_trials, key=_sel_score) if all_trials else {}
        base_cfg = best.get("config", sample_random_config(rng))
        cfg = validate_config(mutate_random(base_cfg, rng))
        h = _cfg_hash(cfg)
        if h not in seen_hashes:
            valid_cfgs.append(cfg)

    # ── Cheap screening: run cheap eval on ALL candidates, pick top-K ─────
    # Replaces surrogate-only selection which is unreliable at <40 trials.
    # Cheap rung (5 epochs, 30% data) gives real training signal.
    surrogate.fit()  # still fit for tie-breaker at >30 full trials
    if len(valid_cfgs) > batch_k:
        top_cfgs, pre_cheap_results = _cheap_screen_candidates(
            valid_cfgs=valid_cfgs,
            raw=raw,
            batch_k=batch_k,
            all_trials=all_trials,
            surrogate=surrogate,
            seed=seed,
            device=device,
        )
    else:
        top_cfgs = valid_cfgs[:batch_k]
        pre_cheap_results = [None] * len(top_cfgs)

    # ── Train top-K ───────────────────────────────────────────────────────
    new_trials: List[Dict] = []
    for cfg, pre_cheap in zip(top_cfgs, pre_cheap_results):
        trial_id = len(all_trials)
        print(f"  [BATCH] training trial #{trial_id}  family={cfg.get('arch',{}).get('family')}")
        # Each config may have its own preprocess settings — refit preprocessor per-trial
        prepared = _prep_from_cfg(cfg, raw)
        try:
            result = _evaluate_config(
                cfg, prepared, out_dir, trial_id,
                use_multi_fidelity=use_multi_fidelity,
                cheap_scores=cheap_scores,
                medium_scores=medium_scores,
                cheap_scores_by_family=cheap_scores_by_family,
                medium_scores_by_family=medium_scores_by_family,
                n_train_rows=n_train_rows,
                pre_cheap_result=pre_cheap,
                seed=seed, device=device, verbose=verbose,
            )
        except Exception as e:
            print(f"  [BATCH] eval failed ({e}), skipping")
            continue

        h = _cfg_hash(cfg)
        seen_hashes.add(h)
        # Use search_score for surrogate (dense proxy — better signal than raw accuracy)
        surrogate.add(cfg, result.search_score)

        best_so_far = max((t.get("primary", -999) for t in all_trials), default=-999)
        trial_entry: Dict = {
            "trial_id": trial_id,
            "op": "batch:propose",
            "primary": result.primary,
            "search_score": result.search_score,
            "metrics": result.metrics,
            "rung": result.rung,
            "config": cfg,
            "n_params": result.n_params,
            "epochs_run": result.epochs_run,
            "seconds": result.seconds,
            "val_history": result.history,          # FIX: was result.val_history (wrong field)
            "train_loss_history": result.train_loss_history,
            "grad_norm_history": result.grad_norm_history,
        }
        all_trials.append(trial_entry)
        new_trials.append(trial_entry)

        marker = " ★ NEW BEST" if result.primary > best_so_far else ""
        print(f"  [BATCH] trial #{trial_id}: primary={result.primary:.5f}{marker}")

    if llm_logger is not None:
        llm_logger.log_call(
            operator="batch_propose",
            prompt=propose_user,
            raw_response=raw_resp,
            parsed={"n_valid": len(valid_cfgs), "n_trained": len(new_trials),
                    "critic_verdict": critic_result.get("verdict", "")},
            error="",
        )

    return new_trials


def _llm_refine_step(
    llm: OpenAILLM,
    best_cfg: Dict,
    best_curve: Dict,
    refine_history: List[Dict],
    summary: Dict,
    rng: random.Random,
    seen_hashes: Optional[set] = None,
    llm_logger: Optional["LLMLogger"] = None,
    # Prong B/D additions:
    all_trials: Optional[List[Dict]] = None,
    action_history: Optional[List[Dict]] = None,
    # Prong F (conversation history) addition:
    conversation: Optional[List[Dict[str, str]]] = None,
    # Prong G (advisor-driven) additions:
    iteration_idx: int = 0,
    iteration_total: int = 25,
    first_turn_kickoff: str = "",
) -> Tuple[Dict, str]:
    """One REFINE iteration: LLM picks ONE action → apply → return new cfg.

    Prong B (history-aware): pass YOUR_ACTION_HISTORY with hashes & dedup info.
    Prong C (stochastic apply): apply_action now stochastic → child rarely dedupes.
    Prong D (population-aware): pass top-K parents, let LLM pick which to mutate.

    Returns (new_cfg, op_label). On dedup, retries with stochastic re-apply
    (since apply_action is now random) up to 3 times; only falls back to
    random mutation if all retries dedupe (extremely unlikely).
    """
    # Prong D: build parent candidates from population
    parent_candidates = []
    if all_trials:
        parent_candidates = _build_parent_candidates(all_trials, top_k=5)

    user = build_user_payload_refine(
        best_cfg=best_cfg,
        best_curve=best_curve,
        refine_history=refine_history[-6:],   # last 6 refine steps for context
        dataset_summary=summary,
        parent_candidates=parent_candidates,
        action_history=action_history or [],
    )
    # Prong G: prepend explicit iteration counter so the LLM never thinks
    # this is its only/last call. The advisor flagged this as critical:
    # without it, gpt-4o-mini sometimes diagnoses "already_good" on iter 1.
    iteration_remaining = max(0, iteration_total - iteration_idx)
    iteration_header = (
        f"=== ITERATION {iteration_idx + 1} of {iteration_total} "
        f"({iteration_remaining} iterations remaining; budget will be used in full) ===\n"
        f"You MUST propose a concrete improvement action this iteration. "
        f"Do NOT declare the model 'good enough' — the search continues until budget is exhausted.\n"
    )
    # On the first REFINE call, also include cold-start kickoff (advisor:
    # "copy context starting from the very beginning"). This way the LLM sees
    # the 5 cold-start trials as part of its conversation memory.
    if first_turn_kickoff:
        user = first_turn_kickoff + "\n" + iteration_header + "\n" + user
    else:
        user = iteration_header + "\n" + user
    print(f"  [LLM→] refine prompt len={len(user)} chars  "
          f"best_family={best_cfg.get('arch',{}).get('family','?')}  "
          f"parents={len(parent_candidates)}  history={len(action_history or [])}")

    raw_resp = ""
    parsed_json: Dict = {}
    action = None
    params: Dict = {}
    error_msg = ""
    new_cfg: Optional[Dict] = None
    parent_id_picked = "best"
    parent_cfg_used = best_cfg
    n_dedup_retries = 0

    try:
        # Prong F: pass conversation list so the LLM sees its own past responses
        action, params, raw_resp, parsed_json = llm_refine_action(
            llm, SYSTEM_PROMPT, user, conversation=conversation,
        )

        # Prong D: resolve parent_id from LLM response
        parent_id_picked = (parsed_json.get("parent_id") or "best").strip()
        if parent_id_picked != "best" and parent_candidates:
            for pc in parent_candidates:
                if pc.get("parent_id") == parent_id_picked:
                    # Find the actual cfg from all_trials by trial_id
                    for t in (all_trials or []):
                        if t.get("trial_id") == pc.get("trial_id"):
                            parent_cfg_used = t["config"]
                            break
                    break

        if action is not None:
            # Prong C/B: thanks to stochastic apply_action, re-applying produces
            # different child each time — so we can retry on dedup up to 3 times
            # before declaring failure.
            for retry in range(4):
                candidate_cfg = apply_action(parent_cfg_used, action, params, rng)
                h = _cfg_hash(candidate_cfg)
                if seen_hashes is None or h not in seen_hashes:
                    new_cfg = candidate_cfg
                    break
                n_dedup_retries += 1
                # If still deduping after 2 retries, switch to a more aggressive action
                if retry == 2:
                    print(f"  [←LLM] action={action} dedup'd 3x → escalating to switch_family")
                    action = "switch_family"
                    params = {}

    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        print(f"[NAS] LLM refine failed ({error_msg}); random mutation.")

    # Random fallback (very rare now thanks to stochastic apply + dedup retries)
    if new_cfg is None:
        new_cfg = validate_config(mutate_random(parent_cfg_used, rng))
        op_label = "random_refine_fallback"
    else:
        op_label = f"refine:{action}"
        if n_dedup_retries:
            op_label += f"#retry{n_dedup_retries}"

    # Log to file
    if llm_logger is not None:
        diagnosis = (parsed_json.get("diagnosis") or {})
        llm_logger.log_call(
            operator="refine" if action and new_cfg is not None and op_label.startswith("refine:") else "random_refine_fallback",
            prompt=user,
            raw_response=raw_resp,
            parsed={
                "reasoning": {
                    "failure_mode": diagnosis.get("failure_mode", ""),
                    "curve_obs": diagnosis.get("evidence", ""),
                    "history_check": diagnosis.get("history_check", ""),
                    "proposed_change": action or "random_mutation",
                    "expected_effect": parsed_json.get("rationale", ""),
                    "parent_id_picked": parent_id_picked,
                    "n_dedup_retries": n_dedup_retries,
                },
                "rationale": parsed_json.get("rationale", ""),
            },
            error=error_msg,
        )

    return new_cfg, op_label


# ---------------------------------------------------------------------------
# Evaluation with multi-fidelity
# ---------------------------------------------------------------------------
# Cheap screening — replaces surrogate-only candidate selection
# ---------------------------------------------------------------------------

def _cheap_screen_candidates(
    valid_cfgs: List[Dict],
    raw,
    batch_k: int,
    all_trials: List[Dict],
    surrogate,
    seed: int = 42,
    device: Optional[str] = None,
) -> tuple:
    """Run cheap eval (5 epochs, 30% data) on ALL LLM candidates, pick top-K.

    Why this beats surrogate-only selection:
      - Surrogate trained on 12–20 points is very unreliable (picks 0.54, 0.52, 0.51)
      - Cheap rung is a real training signal, not a prediction
      - Overhead: ~5 sec × 20 candidates = 100 sec/round — acceptable

    Returns (selected_cfgs, pre_cheap_results):
      - selected_cfgs[i]: config to train at medium/full
      - pre_cheap_results[i]: FidelityResult from screening (passed to _evaluate_config
        to SKIP redundant cheap re-run for selected configs)
    """
    _PREFERRED = {"tabm", "resmlp", "mlp"}
    n_full = sum(1 for t in all_trials if t.get("rung") == "full")

    screened = []
    print(f"  [SCREEN] cheap-eval {len(valid_cfgs)} candidates …", flush=True)
    for i, cfg in enumerate(valid_cfgs):
        prepared = _prep_from_cfg(cfg, raw)
        try:
            cheap = evaluate_at_rung(
                cfg, prepared, CHEAP, out_dir=None,
                seed=seed, device=device, verbose=False,
            )
            cheap_score = cheap.search_score
        except Exception as e:
            print(f"  [SCREEN] cheap failed cfg {i}: {e}")
            cheap = None
            cheap_score = -1e9

        fam = cfg.get("arch", {}).get("family", "?")

        # Surrogate only as weak tie-breaker once we have enough data
        surr_bonus = 0.0
        if surrogate.fitted and n_full >= 30:
            try:
                surr_bonus = surrogate.score(cfg) * 0.05
            except Exception:
                pass

        screened.append({
            "cfg": cfg,
            "cheap_res": cheap,
            "cheap_score": cheap_score,
            "select_score": cheap_score + surr_bonus,
            "family": fam,
        })

    screened.sort(key=lambda x: -x["select_score"])
    top5 = [(s["family"], round(s["cheap_score"], 4)) for s in screened[:5]]
    print(f"  [SCREEN] top-5 cheap: {top5}")

    # Diversity-aware slot selection (same logic as before, but score = cheap)
    picked = []
    used_families: set = set()
    added_ids: set = set()

    def _add(item: Dict) -> bool:
        iid = id(item["cfg"])
        if iid not in added_ids:
            added_ids.add(iid)
            picked.append(item)
            used_families.add(item["family"])
            return True
        return False

    # Slot 1: best cheap score overall
    if screened:
        _add(screened[0])

    # Slot 2: best preferred family not yet picked
    for s in screened:
        if s["family"] in _PREFERRED and id(s["cfg"]) not in added_ids:
            _add(s)
            break

    # Slot 3: best from a different family
    for s in screened:
        if s["family"] not in used_families and id(s["cfg"]) not in added_ids:
            _add(s)
            break

    # Fill remaining slots by select_score
    for s in screened:
        if len(picked) >= batch_k:
            break
        if id(s["cfg"]) not in added_ids:
            _add(s)

    selected_cfgs = [p["cfg"] for p in picked[:batch_k]]
    pre_cheap_results = [p["cheap_res"] for p in picked[:batch_k]]

    info = [(p["family"], round(p["cheap_score"], 4)) for p in picked[:batch_k]]
    print(f"  [SCREEN] selected {len(selected_cfgs)}: {info}")
    return selected_cfgs, pre_cheap_results


# ---------------------------------------------------------------------------

def _evaluate_config(
    cfg: Dict,
    prepared: PreparedSplit,
    out_dir: Path,
    trial_id: int,
    *,
    use_multi_fidelity: bool,
    cheap_scores: List[float],
    medium_scores: Optional[List[float]] = None,
    cheap_scores_by_family: Optional[Dict[str, List[float]]] = None,
    medium_scores_by_family: Optional[Dict[str, List[float]]] = None,
    promote_frac_cheap: float = 0.75,
    promote_frac_medium: float = 0.50,
    n_train_rows: int = 100_000,
    pre_cheap_result=None,   # FidelityResult from cheap screening — skip cheap rung
    seed: int = 42,
    device: Optional[str] = None,
    verbose: bool = False,
) -> FidelityResult:
    """Evaluate a config using cheap→medium→full promotion ladder.

    cheap   : 5 epochs, 30% data  — quick filter (prune bottom 25%)
    medium  : 20 epochs, 100% data — second filter (prune bottom 50% of survivors)
    full    : full epochs, 100% data — final evaluation

    Promotion is decided by search_score (dense proxy), NOT raw accuracy.
    This prevents killing configs that improve logloss/minority classes but
    haven't yet crossed the argmax accuracy threshold.

    ft_transformer and tabm are always promoted to medium if cheap_scores < 30:
    these families need more epochs to show their true potential.
    """
    tdir = ensure_dir(out_dir / f"trial_{trial_id:03d}")
    if medium_scores is None:
        medium_scores = []

    family = cfg.get("arch", {}).get("family", "")
    _late_bloomers = {"ft_transformer", "tabm"}

    # Per-family score trackers for late-bloomer protection
    if cheap_scores_by_family is None:
        cheap_scores_by_family = {}
    if medium_scores_by_family is None:
        medium_scores_by_family = {}
    fam_cheap  = cheap_scores_by_family.setdefault(family, [])
    fam_medium = medium_scores_by_family.setdefault(family, [])

    if not use_multi_fidelity:
        return evaluate_at_rung(cfg, prepared, FULL, out_dir=tdir, seed=seed, device=device,
                                verbose=verbose)

    # ── Adaptive MF: on small datasets cheap rung is too noisy to prune reliably.
    # cheap rung uses 30% of data → if n_train_rows < 8000 that's < 2400 rows,
    # which gives very unreliable 5-epoch estimates. Soften the threshold sharply.
    #   < 5000 rows  → skip cheap pruning entirely (still run it for score tracking)
    #   5000–15000   → soften promote_frac_cheap to 0.95 (only cut absolute worst 5%)
    #   > 15000      → use the configured promote_frac_cheap (default 0.75)
    effective_promote_frac_cheap = promote_frac_cheap
    skip_cheap_pruning = False
    if n_train_rows < 5_000:
        skip_cheap_pruning = True
    elif n_train_rows < 15_000:
        effective_promote_frac_cheap = max(promote_frac_cheap, 0.95)

    # ── Cheap rung ────────────────────────────────────────────────────────
    # If pre_cheap_result is provided (from cheap screening), skip redundant re-run.
    if pre_cheap_result is not None:
        cheap_res = pre_cheap_result
        print(f"  [MF] using pre-screened cheap: search_score={cheap_res.search_score:.5f}")
    else:
        cheap_res = evaluate_at_rung(cfg, prepared, CHEAP, out_dir=None, seed=seed, device=device,
                                     verbose=False)
    cheap_scores.append(cheap_res.search_score)
    fam_cheap.append(cheap_res.search_score)

    cheap_threshold = successive_halving_threshold(cheap_scores[:-1], effective_promote_frac_cheap)
    # Late-bloomer protection: per-family count. tabm/ft_transformer get at least 8
    # full cheap runs before global threshold applies to them.
    force_promote = skip_cheap_pruning or (family in _late_bloomers and len(fam_cheap) < 8)
    if not force_promote and len(cheap_scores) > 6 and cheap_res.search_score < cheap_threshold:
        print(f"  [MF] trial_{trial_id:03d} pruned at CHEAP "
              f"(search_score={cheap_res.search_score:.5f} < {cheap_threshold:.5f}  "
              f"family={family} fam_evals={len(fam_cheap)})")
        cheap_res.rung = "cheap_pruned"
        save_json(tdir / "metrics.json", {**cheap_res.metrics,
                                           "primary": cheap_res.primary,
                                           "search_score": cheap_res.search_score,
                                           "rung": "cheap_pruned"})
        return cheap_res

    # ── Medium rung ───────────────────────────────────────────────────────
    medium_res = evaluate_at_rung(cfg, prepared, MEDIUM, out_dir=None, seed=seed, device=device,
                                  verbose=False)
    medium_scores.append(medium_res.search_score)
    fam_medium.append(medium_res.search_score)

    med_threshold = successive_halving_threshold(medium_scores[:-1], promote_frac_medium)
    # Late-bloomer: at least 4 per-family medium evaluations before pruning
    force_promote_med = family in _late_bloomers and len(fam_medium) < 4
    if not force_promote_med and len(medium_scores) > 6 and medium_res.search_score < med_threshold:
        print(f"  [MF] trial_{trial_id:03d} pruned at MEDIUM "
              f"(search_score={medium_res.search_score:.5f} < {med_threshold:.5f}  "
              f"family={family} fam_med_evals={len(fam_medium)})")
        medium_res.rung = "medium_pruned"
        save_json(tdir / "metrics.json", {**medium_res.metrics,
                                           "primary": medium_res.primary,
                                           "search_score": medium_res.search_score,
                                           "rung": "medium_pruned",
                                           "cheap_primary": cheap_res.primary,
                                           "cheap_search_score": cheap_res.search_score})
        return medium_res

    # ── Full rung ─────────────────────────────────────────────────────────
    full_res = evaluate_at_rung(cfg, prepared, FULL, out_dir=tdir, seed=seed, device=device,
                                verbose=verbose)
    return full_res


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_llm_nas_v2(
    *,
    openml_id: Optional[int] = None,
    task: str = "auto",
    out_dir: str,
    budget: int = 30,
    coldstart_n: int = 5,
    reflect_every: int = 8,
    surr_candidates_k: int = 3,
    # NEW: Two-phase mode — Phase 0 cold start + Phase 1 REFINE (tool-call)
    # Set False to use legacy PROPOSE loop instead (for ablation comparison)
    use_refine_loop: bool = True,
    # v6: which evolutionary mode to use during REFINE phase.
    #   "refine"          — action-vocab with all v5 guards (default)
    #   "freeform"        — LLM returns {"changes": {dotted-path: value}}, no vocab
    #   "multi_refine"    — LLM picks 1-3 distinct actions per turn (+ LR throttle)
    #   "critic_corrector"— 2-stage: Critic analyses → Corrector picks actions
    #   "batch"           — LLM proposes N candidates → surrogate filters → train top-K
    mode: str = "refine",
    # Batch mode parameters (used when mode="batch")
    batch_n: int = 15,   # how many configs LLM proposes per round
    batch_k: int = 3,    # how many to actually train per round
    # Fix #14 (v7): random warmup BEFORE LLM cold-start.
    # If >0, replace LLM cold-start (coldstart_n trials) with this many random
    # trials. Advisor insight: LLM cold-start produces "textbook" configs;
    # random gives LLM rich diverse history to reason from at first REFINE call.
    random_warmup: int = 0,
    # How often to inject a full PROPOSE step during REFINE phase (exploration pulse)
    explore_pulse_every: int = 6,
    # Ablation: set False to hide learning curve data from LLM prompts
    # (simulates GENIUS / EvoPrompting behaviour — LLM sees only final scores)
    curve_feedback: bool = True,
    # Legacy params kept for backward compat with run_experiment.py
    crossover_p: float = 0.25,
    population_size: int = 20,
    tournament_size: int = 5,
    use_multi_fidelity: bool = True,
    seed: int = 42,
    llm_model: str = "gpt-4o",
    api_key: str = "",
    base_url: str = "",
    llm_temperature: float = 0.7,
    ensemble_k: int = 5,
    device: Optional[str] = None,
    # Non-OpenML data sources
    source: str = "openml",             # "openml" | "builtin" | "csv"
    builtin_name: Optional[str] = None, # "covtype" | "california_housing" | "miniboonee"
    csv_path: Optional[str] = None,
    # Logging verbosity
    verbose: bool = True,               # per-epoch training logs + full LLM prompt/response
    # Resume: skip warmup, load existing trials from trials_index.json
    # Useful when warmup already ran but LLM phase was broken.
    # Keeps only warmup/family_aware_extra trials, discards LLM-phase trials.
    resume: bool = False,
    # Extend: load ALL existing trials from out_dir, run `budget` MORE trials.
    # Unlike --resume (crash recovery), --extend is for intentionally adding
    # more trials after a completed run — e.g. 40 batch → 15 more refine.
    # Model checkpoints stay in same out_dir/trials/, so ensemble still works.
    extend: bool = False,
    require_llm: bool = False,
) -> Dict[str, Any]:
    """
    Run the v2 LLM NAS pipeline.

    Two modes (controlled by use_refine_loop):

    Mode A — REFINE (default, use_refine_loop=True):
      Phase 0 – Cold start (coldstart_n diverse LLM configs at FULL fidelity)
      Phase 1 – REFINE loop: LLM diagnoses current best curve → picks ONE targeted
        action from a fixed vocabulary (add_layer, increase_dropout, reduce_lr, …) →
        action applied to best config → evaluate. Every explore_pulse_every steps,
        inject one PROPOSE step to avoid local optima.
      This mode is interpretable: action call counts are logged to action_stats.json.

    Mode B — PROPOSE (legacy, use_refine_loop=False):
      Phase 0 – Cold start
      Phase 1 – PROPOSE loop: LLM freely generates surr_candidates_k configs per step;
        surrogate picks the best predicted one.

    All LLM interactions logged to out_dir/llm_log.jsonl.
    Returns dict with keys: best_trial, all_trials, ensemble, summary, action_stats.
    """
    t_start = time.time()
    seed_everything(seed)
    rng = random.Random(seed)
    out_dir_p = ensure_dir(out_dir)
    trials_dir = ensure_dir(out_dir_p / "trials")

    # ------------------------------------------------------------------ #
    # 1. Load data
    # ------------------------------------------------------------------ #
    if source == "openml":
        print(f"[NAS] Loading OpenML dataset {openml_id} …")
    else:
        print(f"[NAS] Loading builtin dataset: {builtin_name or csv_path} (source={source}) …")
    raw, summary = load_raw(
        source=source,
        openml_id=openml_id,
        builtin_name=builtin_name,
        csv_path=csv_path,
        task=task, seed=seed,
    )

    # Fix #13 (v7): enrich dataset_summary with per-feature stats,
    # class distribution, and a sample of 5 rows. Advisor's request:
    # «надо дать ллм больше инфы о датасете — distributions + data.sample(5)»
    try:
        import numpy as _np
        import pandas as _pd
        # Compute stats on TRAIN portion only (don't leak val/test)
        X_tr = raw.X_train
        y_tr = raw.y_train
        if isinstance(X_tr, _pd.DataFrame):
            X_df = X_tr
        else:
            X_df = _pd.DataFrame(X_tr, columns=(raw.num_cols + raw.cat_cols))

        # Per-feature stats for NUMERIC features (top 20 by variance for brevity)
        feat_stats: Dict[str, Dict] = {}
        if raw.num_cols:
            num_df = X_df[raw.num_cols]
            try:
                num_df_numeric = num_df.apply(_pd.to_numeric, errors="coerce")
                desc = num_df_numeric.describe().T  # mean, std, min, 25%, 50%, 75%, max
                if len(desc) > 20:
                    # Keep top-20 by variance — gives LLM a sense of scale without flooding
                    variances = num_df_numeric.var().sort_values(ascending=False)
                    top_cols = variances.head(20).index.tolist()
                    desc = desc.loc[top_cols]
                for col in desc.index:
                    row = desc.loc[col]
                    feat_stats[str(col)] = {
                        "mean":   round(float(row.get("mean", 0)),  4),
                        "std":    round(float(row.get("std",  0)),  4),
                        "min":    round(float(row.get("min",  0)),  4),
                        "max":    round(float(row.get("max",  0)),  4),
                        "median": round(float(row.get("50%",  0)),  4),
                    }
            except Exception as _e:
                feat_stats = {"_error": f"could not compute stats: {_e}"}
        summary["feature_stats"] = feat_stats

        # Per-feature stats for CATEGORICAL features (cardinality + top categories)
        cat_stats: Dict[str, Dict] = {}
        if raw.cat_cols:
            for c in raw.cat_cols[:10]:  # cap at 10 cat features
                try:
                    vc = X_df[c].value_counts(dropna=False).head(5)
                    cat_stats[str(c)] = {
                        "n_unique": int(X_df[c].nunique(dropna=False)),
                        "top5": {str(k): int(v) for k, v in vc.items()},
                    }
                except Exception:
                    pass
        if cat_stats:
            summary["categorical_stats"] = cat_stats

        # Class distribution (for classification only)
        if raw.task in ("binary", "multiclass") and y_tr is not None:
            try:
                y_arr = _np.asarray(y_tr).ravel()
                vals, counts = _np.unique(y_arr, return_counts=True)
                total = int(counts.sum())
                summary["class_distribution"] = {
                    int(v): round(float(c) / total, 4) for v, c in zip(vals, counts)
                }
                summary["class_imbalance_ratio"] = round(
                    float(counts.max()) / float(counts.min()), 2
                ) if counts.min() > 0 else None
            except Exception:
                pass

        # Sample 5 random rows from train for LLM to see real data
        try:
            sample = X_df.sample(n=min(5, len(X_df)), random_state=seed)
            # Cast NumPy types → native Python for JSON
            sample_rec = sample.head(5).to_dict(orient="records")
            cleaned = []
            for rec in sample_rec:
                cleaned_rec = {}
                for k, v in rec.items():
                    if hasattr(v, "item"):
                        v = v.item()
                    if isinstance(v, float):
                        v = round(v, 4)
                    cleaned_rec[str(k)] = v
                cleaned.append(cleaned_rec)
            summary["sample_rows"] = cleaned
        except Exception:
            pass

    except Exception as _e:
        print(f"  [Fix #13] could not enrich dataset summary: {_e}")

    save_json(out_dir_p / "dataset_summary.json", summary)
    print(f"  dataset: {summary['name']}  n={summary['n_rows']}  "
          f"num={summary['n_num']}  cat={summary['n_cat']}  task={summary['task']}")

    # ------------------------------------------------------------------ #
    # 2. Load baseline results (if present)
    # ------------------------------------------------------------------ #
    baselines: Dict = {}
    for fname in ("baseline_catboost.json", "baseline_lightgbm.json",
                  "baseline_random_search.json", "baselines.json"):
        bp = out_dir_p / fname
        if bp.exists():
            baselines[fname.replace(".json", "")] = load_json(bp)
    baseline_for_prompt = baselines.get("baseline_catboost", baselines.get(
        next(iter(baselines), ""), {"note": "no baseline found"}))

    # ------------------------------------------------------------------ #
    # 3. Initialize
    # ------------------------------------------------------------------ #
    llm = make_llm(
        api_key=api_key,
        model=llm_model,
        base_url=base_url,
        temperature=llm_temperature,
    )
    llm.verbose = verbose
    surrogate = ConfigSurrogate()
    hints: Dict = {}
    all_trials: List[Dict] = []
    cheap_scores: List[float] = []          # CHEAP search_scores (global threshold)
    medium_scores: List[float] = []         # MEDIUM search_scores (global threshold)
    cheap_scores_by_family: Dict[str, List[float]] = {}   # per-family late-bloomer protection
    medium_scores_by_family: Dict[str, List[float]] = {}  # per-family late-bloomer protection
    seen_hashes: set = set()
    action_counts: Dict[str, int] = {}   # LR-BIAS GUARD: track per-action call counts
    best_primary = -1e18
    trial_id = 0

    # LLM interaction log — paste this file to Claude to debug LLM decisions
    llm_logger = LLMLogger(out_dir_p / "llm_log.jsonl")
    print(f"[NAS] LLM interaction log: {llm_logger.path}")
    print(f"[NAS] curve_feedback={curve_feedback}  "
          f"({'FULL curve-aware mode' if curve_feedback else 'ABLATION: curve-blind mode (scores only)'})")

    # ------------------------------------------------------------------ #
    # 4. Resume: load existing warmup trials, skip re-running them
    # ------------------------------------------------------------------ #
    warmup_ops = {"random_warmup", "cold_start", "family_aware_extra"}
    if resume:
        idx_path = out_dir_p / "trials_index.json"
        if idx_path.exists():
            existing = load_json(idx_path).get("trials", [])
            # Keep only warmup/family_aware_extra trials; drop any broken LLM trials
            warmup_trials = [t for t in existing if t.get("op", "") in warmup_ops]
            if warmup_trials:
                all_trials = warmup_trials
                trial_id = len(all_trials)
                best_primary = max(t.get("primary", -1e18) for t in all_trials)
                for t in all_trials:
                    h = _cfg_hash(t["config"])
                    seen_hashes.add(h)
                    surrogate.add(t["config"], t.get("search_score", t.get("primary", 0.0)))
                    cs = t.get("cheap_score")
                    if cs is not None:
                        cheap_scores.append(cs)
                # Re-save trimmed index
                save_json(out_dir_p / "trials_index.json", {"trials": all_trials})
                print(f"\n[NAS] === RESUME: loaded {len(all_trials)} warmup trials "
                      f"(best={best_primary:.5f}), skipping warmup phase ===")
                # Jump straight to LLM phase — skip warmup block below
                actual_warmup = len([t for t in all_trials if t.get("op") != "family_aware_extra"])
                coldstart_n = actual_warmup
                n_warmup_used = actual_warmup
                family_aware_extras = len([t for t in all_trials if t.get("op") == "family_aware_extra"])
                n_evo_steps = budget - len(all_trials)
                goto_llm_phase = True
            else:
                print(f"[NAS] RESUME: no warmup trials found in {idx_path}, running fresh.")
                goto_llm_phase = False
        else:
            print(f"[NAS] RESUME: {idx_path} not found, running fresh.")
            goto_llm_phase = False
    else:
        goto_llm_phase = False

    # ------------------------------------------------------------------ #
    # 4c. Extend: load ALL existing trials, run `budget` more trials on top
    #     Unlike resume (crash recovery), extend is for intentional continuation.
    #     Usage: run batch 40 trials → extend with 15 refine trials.
    # ------------------------------------------------------------------ #
    if extend and not goto_llm_phase:
        idx_path = out_dir_p / "trials_index.json"
        if idx_path.exists():
            existing = load_json(idx_path).get("trials", [])
            # Load ALL non-error trials (warmup + LLM phase)
            valid = [t for t in existing if t.get("rung") not in ("error",)]
            if valid:
                all_trials = valid
                trial_id = max(t.get("trial_id", 0) for t in all_trials) + 1
                best_primary = max(t.get("primary", -1e18) for t in all_trials)
                for t in all_trials:
                    h = _cfg_hash(t["config"])
                    seen_hashes.add(h)
                    surrogate.add(t["config"],
                                  t.get("search_score", t.get("primary", 0.0)))
                surrogate.fit()
                # Reconstruct cheap_scores from cheap-eval entries (best effort)
                for t in all_trials:
                    cs = t.get("cheap_score")
                    if cs is not None:
                        cheap_scores.append(cs)
                        fam = t.get("config", {}).get("arch", {}).get("family", "")
                        if fam:
                            cheap_scores_by_family.setdefault(fam, []).append(cs)
                n_evo_steps = budget   # budget = number of EXTRA trials to run
                goto_llm_phase = True
                best_fam = max(
                    {t.get("config", {}).get("arch", {}).get("family", "?"): 0
                     for t in all_trials}.keys(),
                    key=lambda f: max(
                        (t.get("primary", -1e9)
                         for t in all_trials
                         if t.get("config", {}).get("arch", {}).get("family") == f),
                        default=-1e9,
                    )
                )
                print(
                    f"\n[NAS] === EXTEND: loaded {len(all_trials)} trials from "
                    f"'{out_dir}' (best_primary={best_primary:.5f}, "
                    f"best_family={best_fam}) ===\n"
                    f"[NAS] Running {budget} MORE trials in {mode.upper()} mode."
                )
            else:
                print(f"[NAS] EXTEND: no valid trials found in {idx_path} — running fresh.")
        else:
            print(f"[NAS] EXTEND: {idx_path} not found — running fresh.")

    # ------------------------------------------------------------------ #
    # 5. Cold start (skipped if resume succeeded)
    #
    # Fix #14: two modes here:
    #   (a) random_warmup == 0  →  LLM cold-start (coldstart_n configs)
    #   (b) random_warmup  > 0  →  random_warmup random configs, NO LLM
    # ------------------------------------------------------------------ #
    if goto_llm_phase:
        pass  # resume: jump over warmup
    elif random_warmup > 0:
        # MODE B — stratified random warmup with family quotas (advisor's idea).
        # Quotas are proportional: resmlp/tabm get double weight (known strong on
        # numeric-heavy tabular data), attention families get fewer trials.
        actual_warmup = random_warmup
        coldstart_n = 0  # we logically have 0 LLM cold-start
        print(f"\n[NAS] === Stratified Random Warmup ({actual_warmup} configs) ===")
        # Base quotas (for n=24): resmlp×6, tabm×6, mlp×4, ft_transformer×4, gated_tab×2, autoint×2
        _quota_weights = {"resmlp": 6, "tabm": 6, "mlp": 4, "ft_transformer": 4,
                          "gated_tab": 2, "autoint": 2}
        _total_weight = sum(_quota_weights.values())
        _fam_list: List[str] = []
        for fam, w in _quota_weights.items():
            count = max(1, round(actual_warmup * w / _total_weight))
            _fam_list.extend([fam] * count)
        # Trim/pad to exact actual_warmup
        rng.shuffle(_fam_list)
        _fam_list = _fam_list[:actual_warmup]
        while len(_fam_list) < actual_warmup:
            _fam_list.append(rng.choice(list(_quota_weights.keys())))
        cold_cfgs = [validate_config(sample_random_config(rng, family=f)) for f in _fam_list]
        family_dist = {}
        for f in _fam_list:
            family_dist[f] = family_dist.get(f, 0) + 1
        print(f"  Family distribution: {family_dist}")
        cs_op_label = "random_warmup"
    else:
        # MODE A — LLM cold-start (default, original behaviour)
        actual_warmup = coldstart_n
        print(f"\n[NAS] === LLM Cold Start ({coldstart_n} configs) ===")
        cold_cfgs = _llm_cold_start_configs(llm, coldstart_n, summary, baseline_for_prompt, rng)
        cs_op_label = "cold_start"

    if not goto_llm_phase:
      for i, cfg in enumerate(cold_cfgs):
        print(f"  [{cs_op_label.upper()} {i+1}/{actual_warmup}] family={cfg['arch']['family']}", flush=True)
        save_json(trials_dir / f"trial_{trial_id:03d}" / "config.json", cfg)
        prepared = _prep_from_cfg(cfg, raw)
        tdir = ensure_dir(trials_dir / f"trial_{trial_id:03d}")
        try:
            result = evaluate_at_rung(cfg, prepared, FULL, out_dir=tdir, seed=seed, device=device,
                                      verbose=verbose)
        except Exception as e:
            print(f"    ERROR: {e}")
            result = type("R", (), {
                "primary": -1e9, "search_score": -1e9, "metrics": {}, "rung": "error",
                "n_params": 0, "epochs_run": 0, "seconds": 0.0, "early_stopped": False,
                "history": [], "train_loss_history": [], "grad_norm_history": [],
            })()
        seen_hashes.add(_cfg_hash(cfg))
        rec = _trial_record(trial_id, cfg, result, op=cs_op_label)
        all_trials.append(rec)
        surrogate.add(cfg, result.search_score)
        best_primary = max(best_primary, result.primary)
        print(f"    primary={result.primary:.5f}  best={best_primary:.5f}")
        save_json(out_dir_p / "trials_index.json", {"trials": all_trials})
        trial_id += 1

      surrogate.fit()

    # ------------------------------------------------------------------ #
    # 4.5. FAMILY-AWARE COLD-START EXTENSION (Fix #11 / v5)
    # ------------------------------------------------------------------ #
    # Insight from v4: helena won because cold-start landed in ft_transformer
    # (the right family for the data). On pol/jannis, cold-start missed the
    # best family → LLM spent 60-80% of budget randomly walking via
    # switch_family. Each switch_family produces a RANDOM new config in the
    # new family — too noisy.
    #
    # Solution: after the 5 diverse LLM cold-starts, pick the best-performing
    # family and add 2 EXTRA random samples in JUST that family. This:
    #   (a) gives the surrogate more data on the winning family
    #   (b) gives REFINE a stronger parent population to mutate from
    #   (c) costs only 2 extra full-fidelity trials (~5% of budget)
    family_aware_extras = 2
    if not goto_llm_phase and all_trials and family_aware_extras > 0:
        # Identify best family from cold-start
        best_by_family: Dict[str, float] = {}
        for t in all_trials:
            fam = t["config"].get("arch", {}).get("family", "?")
            p = t.get("primary", -1e9)
            if fam not in best_by_family or p > best_by_family[fam]:
                best_by_family[fam] = p
        if best_by_family:
            best_family = max(best_by_family.items(), key=lambda x: x[1])[0]
            print(f"\n[NAS] === Family-Aware Extension: best family from cold-start "
                  f"is '{best_family}' (primary={best_by_family[best_family]:.5f}). "
                  f"Adding {family_aware_extras} extra samples in this family. ===")
            for i in range(family_aware_extras):
                extra_cfg = validate_config(sample_random_config(rng, family=best_family))
                # Avoid duplicates
                attempts = 0
                while _cfg_hash(extra_cfg) in seen_hashes and attempts < 5:
                    extra_cfg = validate_config(sample_random_config(rng, family=best_family))
                    attempts += 1
                print(f"  [FAM-EXT {i+1}/{family_aware_extras}] family={extra_cfg['arch']['family']}  "
                      f"lr={extra_cfg['train']['lr']:.2e}", flush=True)
                tdir = ensure_dir(trials_dir / f"trial_{trial_id:03d}")
                save_json(tdir / "config.json", extra_cfg)
                prepared = _prep_from_cfg(extra_cfg, raw)
                try:
                    result = evaluate_at_rung(extra_cfg, prepared, FULL, out_dir=tdir,
                                              seed=seed, device=device, verbose=verbose)
                except Exception as e:
                    print(f"    ERROR: {e}")
                    result = type("R", (), {
                        "primary": -1e9, "search_score": -1e9, "metrics": {}, "rung": "error",
                        "n_params": 0, "epochs_run": 0, "seconds": 0.0,
                        "early_stopped": False, "history": [],
                        "train_loss_history": [], "grad_norm_history": [],
                    })()
                seen_hashes.add(_cfg_hash(extra_cfg))
                rec = _trial_record(trial_id, extra_cfg, result, op="family_aware_extra")
                all_trials.append(rec)
                surrogate.add(extra_cfg, result.search_score)
                best_primary = max(best_primary, result.primary)
                print(f"    primary={result.primary:.5f}  best={best_primary:.5f}")
                save_json(out_dir_p / "trials_index.json", {"trials": all_trials})
                trial_id += 1
            surrogate.fit()  # refit with new data

    # ------------------------------------------------------------------ #
    # 5. Main search loop — REFINE or PROPOSE
    # ------------------------------------------------------------------ #
    # Adjust n_evo_steps to account for family-aware extras
    # Fix #14: account for either LLM cold-start (coldstart_n) OR random
    # warmup (random_warmup) — only one is non-zero by construction.
    if not goto_llm_phase:
        n_warmup_used = random_warmup if random_warmup > 0 else coldstart_n
        n_evo_steps = budget - n_warmup_used - (family_aware_extras if all_trials else 0)
        n_evo_steps = max(1, n_evo_steps)
    # else: n_warmup_used, n_evo_steps, family_aware_extras already set in resume block
    # Bug-fix #8 / Prong B: cap PROPOSE n_proposals at 2 to fit upstream
    # provider's ~2000-char response cap (which truncated all 3-candidate
    # PROPOSE responses in v1/v2/v3, regardless of max_tokens setting).
    n_proposals = min(2, max(2, surr_candidates_k))
    mode_label = mode.upper()
    print(f"\n[NAS] === {mode_label} Loop ({n_evo_steps} steps) ===")

    # ── BATCH MODE (separate loop — rounds of K trials) ────────────────── #
    if mode == "batch":
        cheap_scores_batch: List[float] = []
        medium_scores_batch: List[float] = []
        n_remaining = n_evo_steps
        round_idx = 0
        while n_remaining > 0:
            k_this_round = min(batch_k, n_remaining)
            print(f"\n[BATCH] Round {round_idx + 1}: propose={batch_n}  train={k_this_round}  remaining={n_remaining}")
            new_trials = _llm_batch_round(
                llm=llm,
                all_trials=all_trials,
                surrogate=surrogate,
                summary=summary,
                raw=raw,
                out_dir=out_dir_p,
                rng=rng,
                seen_hashes=seen_hashes,
                hints=hints,
                batch_n=batch_n,
                batch_k=k_this_round,
                seed=seed,
                device=device,
                verbose=verbose,
                llm_logger=llm_logger,
                cheap_scores=cheap_scores_batch,
                medium_scores=medium_scores_batch,
                cheap_scores_by_family=cheap_scores_by_family,
                medium_scores_by_family=medium_scores_by_family,
                use_multi_fidelity=use_multi_fidelity,
                n_train_rows=int(summary.get("n_rows", 100_000)),
                require_llm=require_llm,
            )
            n_remaining -= len(new_trials)
            round_idx += 1
            if not new_trials:
                print("[BATCH] No new trials produced this round — stopping.")
                break
            # Reflect every N rounds
            if round_idx % max(1, reflect_every // batch_k) == 0 and len(all_trials) >= 5:
                surrogate.fit()
                # Use the proper reflect helper (not undefined _do_reflect)
                history_for_reflect = _history_for_prompt(
                    all_trials, top_k=10, recent_k=5, include_curves=True,
                )
                new_hints = _llm_reflect_hints(llm, history_for_reflect, summary, surrogate, reflect_every)
                if new_hints:
                    hints.update(new_hints)

        # Final reporting for batch mode
        full_trials = [t for t in all_trials
                       if t.get("rung") not in ("cheap_pruned", "medium_pruned", "error")]
        rank_pool = full_trials if full_trials else all_trials
        best_trial = max(rank_pool, key=_sel_score)
        save_json(out_dir_p / "trials_index.json", {"trials": all_trials})

        # Two separate "bests":
        #   by_search → used for NAS decisions (search_score proxy)
        #   by_primary → used for honest comparison with CatBoost / baselines
        best_by_search  = max(rank_pool, key=_sel_score)
        best_by_primary = max(rank_pool, key=_primary_score)
        save_json(out_dir_p / "best_trial_by_search.json",  best_by_search)
        save_json(out_dir_p / "best_trial_by_primary.json", best_by_primary)
        # Legacy key — keep for backward compat with downstream scripts
        save_json(out_dir_p / "best_trial.json", best_by_primary)

        # Unbiased single-model reporting: choose using validation search score,
        # then evaluate that selected configuration once on untouched test.
        from .ensemble import evaluate_config_on_holdout
        selected_single_test = evaluate_config_on_holdout(
            best_by_search["config"], raw, seed=seed, device=device,
        )

        # ── Ensemble (batch mode) ─────────────────────────────────────────
        ensemble_result: Dict = {}
        if ensemble_k > 1 and len(full_trials) >= 2:
            from .ensemble import ensemble_top_k
            print(f"\n[BATCH] Running ensemble top-{ensemble_k} …")
            ensemble_result = ensemble_top_k(
                full_trials, raw,
                k=ensemble_k,
                method="greedy",
                seed=seed,
                device=device,
                out_dir=str(out_dir_p),
            )

        best_primary_val = best_by_primary.get("primary", 0.0)
        best_search_val  = best_by_search.get("search_score", 0.0)
        best_ensemble    = ensemble_result.get("primary")
        summary_out = {
            "dataset": summary,
            "mode": mode,
            "seed": seed,
            "n_trials": len(all_trials),
            "n_full_trials": len(full_trials),
            # Legacy validation summary retained for compatibility
            "best_primary": best_primary_val,
            "validation_best_primary": best_primary_val,
            "best_primary_trial_id": best_by_primary.get("trial_id"),
            "best_primary_family": best_by_primary.get("config", {}).get("arch", {}).get("family"),
            # NAS selection metric
            "best_search_score": best_search_val,
            "best_search_trial_id": best_by_search.get("trial_id"),
            # Untouched holdout metrics for final method comparison
            "selected_single_test": selected_single_test,
            "test_primary": selected_single_test.get("test_primary"),
            "evaluation_protocol": "select_on_validation_report_on_test",
            # Ensemble
            "best_ensemble": best_ensemble,
            "ensemble_k": ensemble_result.get("k"),
            "ensemble": ensemble_result,
        }
        save_json(out_dir_p / "final_report.json", summary_out)
        action_stats_out = action_stats(all_trials)
        save_json(out_dir_p / "action_stats.json", action_stats_out)
        print(f"\n[NAS] BATCH done — best_primary={best_primary_val:.6f}"
              + (f"  test_primary={selected_single_test.get('test_primary'):.6f}"
                 if selected_single_test.get("test_primary") is not None else "")
              + (f"  test_ensemble={best_ensemble:.6f}" if best_ensemble else "")
              + f"  trials={len(all_trials)}")
        return {
            "best_trial": best_trial,
            "all_trials": all_trials,
            "ensemble": ensemble_result,
            "summary": summary_out,
            "action_stats": action_stats_out,
        }
    # ── END BATCH MODE ─────────────────────────────────────────────────── #

    # REFINE state: track best config + action call history
    refine_history: List[Dict] = []      # [{step, action, primary_after}]
    # Prong B: rich action history with hashes/dedup info passed to LLM
    action_history: List[Dict] = []      # [{step, action, parent_id, child_hash, dedup, primary_after, improved}]
    # Prong F: multi-turn conversation history for REFINE operator. The LLM
    # sees its own previous responses as actual assistant messages, giving it
    # natural memory of what it tried and how it justified each decision.
    # Initialized lazily on first REFINE call. Windowed to last 6 turns.
    refine_conversation: List[Dict[str, str]] = []
    REFINE_CONV_MAX_TURNS = 6   # 1 system + N (user, assistant) pairs

    for step in range(n_evo_steps):
        global_step = coldstart_n + step

        # --- Best config so far (for REFINE and for PROPOSE context) ---
        best_trial_so_far = max(all_trials, key=_sel_score)
        best_cfg_now = best_trial_so_far["config"]
        best_curve_now = _learning_curve_summary(best_trial_so_far)

        # --- Periodic reflection: update hints ---
        if step > 0 and step % reflect_every == 0:
            print(f"\n  [Reflect] step={global_step}", flush=True)
            history_for_reflect = _history_for_prompt(all_trials, top_k=10, recent_k=5,
                                                      include_curves=curve_feedback)
            new_hints = _llm_reflect_hints(llm, history_for_reflect, summary, surrogate,
                                           k=reflect_every)
            if new_hints:
                hints.update(new_hints)

        # --- Diversity hints (always refresh) ---
        recent_families = [t["config"]["arch"]["family"]
                           for t in all_trials[-6:]] if all_trials else []
        family_counts = {f: recent_families.count(f) for f in set(recent_families)}
        overused = [f for f, c in family_counts.items() if c >= 2]
        untried_recent = [f for f in ARCH_FAMILIES if f not in recent_families]
        hints["avoid_families_recent"] = overused
        hints["explore_untried"] = untried_recent

        # ----------------------------------------------------------------
        # Choose mode for this step
        # ----------------------------------------------------------------
        # In REFINE mode, inject a PROPOSE "exploration pulse" every
        # explore_pulse_every steps to prevent getting stuck.
        is_propose_step = (
            not use_refine_loop
            or (use_refine_loop and explore_pulse_every > 0
                and step > 0 and step % explore_pulse_every == 0)
        )

        print(f"\n  [step {global_step}] mode={'PROPOSE' if is_propose_step else 'REFINE'}", flush=True)

        if is_propose_step:
            # ---- PROPOSE: LLM generates n_proposals free-form configs ----
            history_compact = _history_for_prompt(all_trials, top_k=8, recent_k=4,
                                                  include_curves=curve_feedback)
            candidate_cfgs = _llm_propose_configs(
                llm, history_compact, summary, hints,
                n=n_proposals, rng=rng,
                seen_hashes=seen_hashes,
                llm_logger=llm_logger,
            )
            # Surrogate ranking
            if surrogate.fitted and len(candidate_cfgs) > 1:
                surr_scores = [surrogate.score(c) for c in candidate_cfgs]
                best_idx = int(np.argmax(surr_scores))
                cfg_to_eval = candidate_cfgs[best_idx]
                print(f"  [Surrogate] picked idx={best_idx}  "
                      f"surr_scores={[round(s,4) for s in surr_scores]}")
            else:
                cfg_to_eval = candidate_cfgs[0]
            op_label = "propose"

        else:
            # ---- REFINE: LLM picks ONE action from the fixed vocabulary ----
            # In curve-blind ablation, pass empty curve so LLM must decide without it
            curve_for_refine = best_curve_now if curve_feedback else {}

            # Bug-fix #4 + #9: HARD MODE-COLLAPSE GUARD (looser trigger).
            # Fires when ≥4 of last 5 trials are the SAME arch.family AND none of
            # those last 5 improved best_primary. Looser than the previous
            # "3-strict-in-a-row" rule which missed cases where one anomaly
            # broke the streak but 17/20 trials were still in one family.
            forced_switch = False
            if len(all_trials) >= 5:
                from collections import Counter
                last5 = all_trials[-5:]
                last5_fams = [t["config"].get("arch", {}).get("family", "?") for t in last5]
                last5_primary = [t.get("primary", -1e9) for t in last5]
                fam_counter = Counter(last5_fams)
                dominant_fam, dominant_count = fam_counter.most_common(1)[0]
                if (dominant_count >= 4
                        and max(last5_primary) <= best_primary - 1e-6):
                    other_fams = [f for f in ARCH_FAMILIES if f != dominant_fam]
                    new_family = rng.choice(other_fams)
                    print(f"  [step {global_step}] MODE-COLLAPSE GUARD: "
                          f"{dominant_count}/5 last trials '{dominant_fam}' without improvement "
                          f"→ forcing switch_family → '{new_family}'")
                    cfg_to_eval = apply_action(best_cfg_now, "switch_family",
                                               {"family": new_family}, rng)
                    op_label = "forced:switch_family"
                    forced_switch = True
                    if llm_logger is not None:
                        llm_logger.log_call(
                            operator="forced_switch_family",
                            prompt="<orchestrator hard-rule: mode-collapse guard>",
                            raw_response="",
                            parsed={"reasoning": {
                                "failure_mode": "mode_collapse",
                                "curve_obs": f"{dominant_count}/5 last '{dominant_fam}' no improvement",
                                "proposed_change": f"switch_family→{new_family}",
                                "expected_effect": "regain diversity",
                            }, "rationale": "orchestrator hard-rule"},
                            error="",
                        )

            if not forced_switch and mode == "multi_refine":
                # Multi-refine: LLM picks 1-3 distinct actions per turn + LR throttle
                if len(refine_conversation) > 1 + (REFINE_CONV_MAX_TURNS - 1) * 2:
                    sys_msg = refine_conversation[0]
                    tail = refine_conversation[-(REFINE_CONV_MAX_TURNS - 1) * 2:]
                    refine_conversation[:] = [sys_msg] + tail
                first_turn_kickoff_mr = ""
                if not refine_conversation:
                    cs_summary = []
                    for t in all_trials[:n_warmup_used + family_aware_extras]:
                        cs_summary.append({
                            "trial_id": t.get("trial_id"),
                            "family": t["config"].get("arch", {}).get("family"),
                            "primary": round(t.get("primary", 0), 5),
                        })
                    first_turn_kickoff_mr = (
                        f"=== SEARCH KICK-OFF (MULTI-REFINE) ===\n"
                        f"Dataset: {summary.get('name')} ({summary.get('task')}, "
                        f"{summary.get('n_rows')} rows).\n"
                        f"Initial population:\n{json.dumps(cs_summary, indent=2)}\n"
                    )
                cfg_to_eval, op_label = _llm_multi_refine_step(
                    llm=llm,
                    best_cfg=best_cfg_now,
                    best_curve=curve_for_refine,
                    refine_history=refine_history,
                    summary=summary,
                    rng=rng,
                    seen_hashes=seen_hashes,
                    llm_logger=llm_logger,
                    all_trials=all_trials,
                    action_history=action_history,
                    conversation=refine_conversation,
                    iteration_idx=step,
                    iteration_total=n_evo_steps,
                    first_turn_kickoff=first_turn_kickoff_mr,
                )

            elif not forced_switch and mode == "critic_corrector":
                # Critic-Corrector: 2 LLM calls per step
                cfg_to_eval, op_label = _llm_critic_corrector_step(
                    llm=llm,
                    best_cfg=best_cfg_now,
                    best_curve=curve_for_refine,
                    refine_history=refine_history,
                    summary=summary,
                    rng=rng,
                    all_trials=all_trials,
                    seen_hashes=seen_hashes,
                    llm_logger=llm_logger,
                    action_history=action_history,
                    conversation=refine_conversation,
                    iteration_idx=step,
                    iteration_total=n_evo_steps,
                )

            elif not forced_switch and mode == "freeform":
                # Mode 3: freeform mode — LLM returns changes dict, not action.
                # Window conversation same as in refine mode.
                if len(refine_conversation) > 1 + (REFINE_CONV_MAX_TURNS - 1) * 2:
                    sys_msg = refine_conversation[0]
                    tail = refine_conversation[-(REFINE_CONV_MAX_TURNS - 1) * 2:]
                    refine_conversation[:] = [sys_msg] + tail
                first_turn_kickoff_ff = ""
                if not refine_conversation:
                    coldstart_summary_ff = []
                    n_init_ff = n_warmup_used + family_aware_extras
                    for t in all_trials[:n_init_ff]:
                        coldstart_summary_ff.append({
                            "trial_id": t.get("trial_id"),
                            "family": t["config"].get("arch", {}).get("family"),
                            "primary": round(t.get("primary", 0), 5),
                            "lr": t["config"].get("train", {}).get("lr"),
                        })
                    first_turn_kickoff_ff = (
                        f"=== SEARCH KICK-OFF (FREEFORM MODE) ===\n"
                        f"Dataset: {summary.get('name')} ({summary.get('task')}, "
                        f"{summary.get('n_rows')} rows).\n"
                        f"Budget: {n_evo_steps}.\n"
                        f"Initial population:\n{json.dumps(coldstart_summary_ff, indent=2)}\n"
                    )
                cfg_to_eval, op_label = _llm_freeform_step(
                    llm=llm,
                    best_cfg=best_cfg_now,
                    best_curve=curve_for_refine,
                    refine_history=refine_history,
                    summary=summary,
                    rng=rng,
                    seen_hashes=seen_hashes,
                    llm_logger=llm_logger,
                    all_trials=all_trials,
                    action_history=action_history,
                    conversation=refine_conversation,
                    iteration_idx=step,
                    iteration_total=n_evo_steps,
                    first_turn_kickoff=first_turn_kickoff_ff,
                )

                # Fix #15 (v7): ANTI-CYCLING GUARD for freeform mode.
                # v6 helena showed LLM cycles on lr/dropout/epochs/patience and
                # never touches structural params (family/n_blocks/d_token) →
                # mode collapse 84% in ft_transformer, 5 trials with same
                # primary 0.65188. Solution: if last 4 freeform turns NEVER
                # touched a structural field, this turn force a structural
                # change (switch family randomly).
                STRUCTURAL_FIELDS = {
                    "arch.family", "arch.n_blocks", "arch.hidden_dims",
                    "arch.d_token", "arch.block_width",
                }
                # Look at last 4 freeform op_labels
                last4_ops = [t.get("op", "") for t in all_trials[-4:]]
                last4_freeform = [op for op in last4_ops if op.startswith("freeform:")]
                if len(last4_freeform) >= 4:
                    touched_structural = False
                    for op in last4_freeform:
                        # op_label format: "freeform:arch.dropout+train.lr"
                        paths = op.replace("freeform:", "").split("+")
                        if any(p in STRUCTURAL_FIELDS for p in paths):
                            touched_structural = True
                            break
                    if not touched_structural:
                        cur_family = cfg_to_eval.get("arch", {}).get("family", "")
                        other_fams = [f for f in ARCH_FAMILIES if f != cur_family]
                        new_family = rng.choice(other_fams)
                        print(f"  [step {global_step}] ANTI-CYCLING (freeform): "
                              f"last 4 freeform turns touched only numerics "
                              f"(no family/depth/d_token change). "
                              f"Force switch_family → '{new_family}'.")
                        cfg_to_eval = apply_action(best_cfg_now, "switch_family",
                                                   {"family": new_family}, rng)
                        op_label = "forced:freeform_anti_cycling"
                        if llm_logger is not None:
                            llm_logger.log_call(
                                operator="forced_freeform_anti_cycling",
                                prompt="<orchestrator soft-rule: freeform anti-cycling>",
                                raw_response="",
                                parsed={"reasoning": {
                                    "failure_mode": "cycling_no_structural",
                                    "curve_obs": f"last 4 freeform turns: {last4_freeform}",
                                    "proposed_change": f"switch_family→{new_family}",
                                    "expected_effect": "break structural cycling",
                                }, "rationale": "freeform anti-cycling"},
                                error="",
                            )

            elif not forced_switch:
                # Prong F: window the conversation BEFORE the call so token
                # count stays bounded. Keep the first message (system prompt)
                # plus the last (REFINE_CONV_MAX_TURNS - 1) * 2 messages
                # (user/assistant pairs).
                if len(refine_conversation) > 1 + (REFINE_CONV_MAX_TURNS - 1) * 2:
                    sys_msg = refine_conversation[0]
                    tail = refine_conversation[-(REFINE_CONV_MAX_TURNS - 1) * 2:]
                    refine_conversation[:] = [sys_msg] + tail

                # Prong G: prepare cold-start summary for the FIRST REFINE call.
                # Per advisor: "копить контекст с самого начала" — LLM should
                # see the cold-start trials as the seed of the conversation.
                first_turn_kickoff = ""
                if not refine_conversation:
                    coldstart_summary = []
                    # include both LLM cold-starts AND family-aware extras (Fix #11)
                    n_init = n_warmup_used + family_aware_extras
                    for t in all_trials[:n_init]:
                        coldstart_summary.append({
                            "trial_id": t.get("trial_id"),
                            "family": t["config"].get("arch", {}).get("family"),
                            "primary": round(t.get("primary", 0), 5),
                            "lr": t["config"].get("train", {}).get("lr"),
                            "depth": (len(t["config"]["arch"].get("hidden_dims", []))
                                      or t["config"]["arch"].get("n_blocks", "?")),
                        })
                    first_turn_kickoff = (
                        f"=== SEARCH KICK-OFF ===\n"
                        f"Dataset: {summary.get('name')} ({summary.get('task')}, "
                        f"{summary.get('n_rows')} rows, {summary.get('n_features')} features).\n"
                        f"Total REFINE budget: {n_evo_steps} iterations.\n"
                        f"Cold-start results (5 random initial configs evaluated already):\n"
                        f"{json.dumps(coldstart_summary, ensure_ascii=False, indent=2)}\n"
                    )

                # Prong B/D/F/G: pass all_trials, action_history, multi-turn
                # conversation, iteration counter, and (on first call) cold-
                # start kickoff message.
                cfg_to_eval, op_label = _llm_refine_step(
                    llm=llm,
                    best_cfg=best_cfg_now,
                    best_curve=curve_for_refine,
                    refine_history=refine_history,
                    summary=summary,
                    rng=rng,
                    seen_hashes=seen_hashes,
                    llm_logger=llm_logger,
                    all_trials=all_trials,
                    action_history=action_history,
                    conversation=refine_conversation,
                    iteration_idx=step,
                    iteration_total=n_evo_steps,
                    first_turn_kickoff=first_turn_kickoff,
                )

                # Mode 2 (v6): ANTI-OSCILLATION GUARD.
                # Empirical: on pol LLM kept ping-ponging reduce_lr→increase_lr→
                # reduce_lr→increase_lr because it flipped its diagnosis each
                # iteration based on the LAST result. The structured action
                # vocabulary lets each lr-flip undo the previous one, with no
                # progress.
                # Rule: track last 4 actions; if they form an oscillating LR
                # pattern (≥2 reduce_lr AND ≥2 increase_lr in the last 4)
                # AND best_primary did not improve over those 4 steps,
                # OVERRIDE the LR-action with extend_training (something
                # truly different). LLM keeps its choice of family/arch.
                if op_label in ("refine:reduce_lr", "refine:increase_lr"):
                    last4 = refine_history[-4:] if refine_history else []
                    if len(last4) == 4:
                        last4_actions = [h.get("action") for h in last4]
                        n_red = sum(1 for a in last4_actions if a == "reduce_lr")
                        n_inc = sum(1 for a in last4_actions if a == "increase_lr")
                        last4_max = max(h.get("primary_after", 0) for h in last4)
                        if (n_red >= 2 and n_inc >= 2
                                and last4_max <= best_primary - 1e-6):
                            print(f"  [step {global_step}] ANTI-OSCILLATION: last 4 actions "
                                  f"= {last4_actions}, ping-ponging LR without progress "
                                  f"→ overriding to extend_training")
                            cfg_to_eval = apply_action(best_cfg_now, "extend_training", {}, rng)
                            op_label = "forced:extend_training_antiosc"
                            if llm_logger is not None:
                                llm_logger.log_call(
                                    operator="forced_anti_oscillation",
                                    prompt="<orchestrator soft-rule: anti-LR-oscillation>",
                                    raw_response="",
                                    parsed={"reasoning": {
                                        "failure_mode": "lr_oscillation",
                                        "curve_obs": f"last4_actions={last4_actions}",
                                        "proposed_change": "extend_training",
                                        "expected_effect": "break LR ping-pong, give more epochs",
                                    }, "rationale": "anti-oscillation guard"},
                                    error="",
                                )
                            # Skip the LR-bias guard below since we already overrode
                            op_label_already_handled = True
                        else:
                            op_label_already_handled = False
                    else:
                        op_label_already_handled = False
                else:
                    op_label_already_handled = False

                # Fix #10 (v4 → v5): RELAXED LR-BIAS GUARD.
                # Old logic (LIMIT=2 → forced:switch_family) was way too
                # aggressive — on pol it ate 13/17 LLM-trials. New logic:
                #   - LIMIT increased 2 → 4 (give LLM more room).
                #   - On hit: REPLACE the action with explore_lr_extreme_low
                #     (an aggressive but family-preserving move) rather than
                #     seizing control via switch_family. This keeps LLM's
                #     architectural choice intact while breaking the lr-spiral.
                LR_BIAS_LIMIT = 4
                if (not op_label_already_handled
                        and op_label == "refine:increase_lr"
                        and action_counts.get("increase_lr", 0) >= LR_BIAS_LIMIT):
                    print(f"  [step {global_step}] LR-BIAS GUARD (soft): increase_lr already used "
                          f"{action_counts['increase_lr']}x → "
                          f"replacing action with explore_lr_extreme_low (regime change, "
                          f"keep family)")
                    cfg_to_eval = apply_action(best_cfg_now, "explore_lr_extreme_low",
                                               {}, rng)
                    op_label = "forced:explore_lr_extreme_low_guard"
                    if llm_logger is not None:
                        llm_logger.log_call(
                            operator="forced_explore_lr_extreme_low_guard",
                            prompt="<orchestrator soft-rule: lr-bias guard relaxed (Fix #10)>",
                            raw_response="",
                            parsed={"reasoning": {
                                "failure_mode": "lr_bias",
                                "curve_obs": f"increase_lr used {action_counts['increase_lr']}x",
                                "proposed_change": "explore_lr_extreme_low (preserve family)",
                                "expected_effect": "break lr-spiral via regime change to very low lr",
                            }, "rationale": "orchestrator soft-rule v5"},
                            error="",
                        )

        # --- Evaluate ---
        seen_hashes.add(_cfg_hash(cfg_to_eval))
        family = cfg_to_eval.get("arch", {}).get("family", "?")
        print(f"  [step {global_step}] eval family={family}  op={op_label}", flush=True)
        save_json(trials_dir / f"trial_{trial_id:03d}" / "config.json", cfg_to_eval)
        prepared = _prep_from_cfg(cfg_to_eval, raw)

        try:
            result = _evaluate_config(
                cfg_to_eval, prepared, trials_dir, trial_id,
                use_multi_fidelity=use_multi_fidelity,
                cheap_scores=cheap_scores,
                medium_scores=medium_scores,
                cheap_scores_by_family=cheap_scores_by_family,
                medium_scores_by_family=medium_scores_by_family,
                n_train_rows=int(summary.get("n_rows", 100_000)),
                seed=seed, device=device,
                verbose=verbose,
            )
        except Exception as e:
            print(f"    ERROR during evaluation: {e}")
            result = type("R", (), {
                "primary": -1e9, "search_score": -1e9, "metrics": {}, "rung": "error",
                "n_params": 0, "epochs_run": 0, "seconds": 0.0, "early_stopped": False,
                "history": [], "train_loss_history": [], "grad_norm_history": [],
            })()

        rec = _trial_record(trial_id, cfg_to_eval, result, op=op_label, rationale="")
        all_trials.append(rec)
        surrogate.add(cfg_to_eval, result.search_score)

        # Update LLM log with actual eval result
        if llm_logger._lines:
            llm_logger.update_primary(len(llm_logger._lines) - 1, result.primary)

        # Update REFINE history (for next iteration's context)
        if not is_propose_step:
            # Extract a short action_name from op_label, supporting all 3 modes:
            #   "refine:<action>"            → <action>
            #   "freeform:f1+f2+f3"          → "freeform_changes"
            #   "forced:..."                 → keep full label
            #   "random_refine_fallback"     → keep
            if op_label.startswith("refine:"):
                action_name = op_label.replace("refine:", "").split("#")[0]
                fell_back = False
            elif op_label.startswith("freeform:"):
                action_name = "freeform_changes"
                fell_back = False
            elif op_label.startswith("freeform_") or op_label == "random_refine_fallback":
                action_name = op_label  # *_fallback*
                fell_back = True
            elif op_label.startswith("forced:"):
                action_name = op_label  # forced:* (guard intervention)
                fell_back = False
            else:
                action_name = None
                fell_back = False

            improved = result.primary > best_primary
            refine_history.append({
                "step": global_step,
                "action": action_name,
                "primary_after": round(result.primary, 6),
                "improved": improved,
            })

            # Prong B + Mode 3 rich action_history with cfg_hash + dedup info.
            # For freeform mode: also store the ACTUAL diff so LLM sees what
            # it tried last turn ("you set train.lr=0.0005, train.epochs=200
            # → primary 0.785, regressed −0.003").
            child_h = _cfg_hash(cfg_to_eval)
            n_retries_str = ""
            if "#" in op_label:
                n_retries_str = op_label.split("#", 1)[1]
            parent_id_used = "best"

            # For freeform mode, derive the diff from cfg_to_eval vs best_cfg_now
            freeform_diff: Dict = {}
            if op_label.startswith("freeform:"):
                # The op_label encodes which paths were changed: "freeform:train.lr+arch.dropout"
                paths = op_label.replace("freeform:", "").split("+")
                for path in paths:
                    parts = path.split(".")
                    if len(parts) == 2 and parts[0] in cfg_to_eval:
                        sec, key = parts
                        new_val = cfg_to_eval.get(sec, {}).get(key)
                        old_val = best_cfg_now.get(sec, {}).get(key)
                        if new_val != old_val:
                            freeform_diff[path] = {"from": old_val, "to": new_val}

            entry = {
                "step": global_step,
                "action": action_name if action_name else op_label,
                "parent_id": parent_id_used,
                "child_hash": child_h,
                "dedup_retries": n_retries_str,
                "fell_back_to_random": fell_back,
                "primary_before_best": round(best_primary, 6),
                "primary_after": round(result.primary, 6),
                "delta_vs_best": round(result.primary - best_primary, 6),
                "improved": improved,
            }
            if freeform_diff:
                entry["changes"] = freeform_diff
            # Store changes_keys for freeform structural-diversity validator
            if op_label.startswith("freeform:"):
                paths = op_label.replace("freeform:", "").split("+")
                entry["changes_keys"] = [p.strip() for p in paths if p.strip()]
            action_history.append(entry)
            while len(action_history) > 12:
                action_history.pop(0)

        best_primary = max(best_primary, result.primary)
        print(f"    primary={result.primary:.5f}  best={best_primary:.5f}  "
              f"params={result.n_params}  rung={result.rung}  seconds={result.seconds:.1f}")

        # Update LR-BIAS GUARD counter for the action that was actually applied
        if op_label.startswith("refine:"):
            _act = op_label.split(":", 1)[1]
            action_counts[_act] = action_counts.get(_act, 0) + 1

        # Refit surrogate every 4 steps
        if (step + 1) % 4 == 0:
            surrogate.fit()

        save_json(out_dir_p / "trials_index.json", {"trials": all_trials})
        trial_id += 1

    print(f"\n[NAS] {llm_logger.summary()}")

    # Action statistics (for thesis analysis)
    a_stats = action_stats(llm_logger._lines) if use_refine_loop else {}
    save_json(out_dir_p / "action_stats.json", a_stats)
    if use_refine_loop and refine_history:
        # Print top called actions
        called = [(a, a_stats[a]["count"]) for a in ACTION_NAMES if a_stats.get(a, {}).get("count", 0) > 0]
        called.sort(key=lambda x: -x[1])
        print(f"  [REFINE] Top actions: {called[:8]}")

    # ------------------------------------------------------------------ #
    # 6. Ensemble top-K
    # ------------------------------------------------------------------ #
    ensemble_result: Dict = {}
    full_trials_for_ens = [t for t in all_trials
                           if t.get("rung") not in ("cheap_pruned", "medium_pruned", "error")]
    if ensemble_k > 1 and len(full_trials_for_ens) >= 2:
        from .ensemble import ensemble_top_k
        ensemble_result = ensemble_top_k(
            full_trials_for_ens, raw,
            k=ensemble_k,
            method="greedy",
            seed=seed,
            device=device,
            out_dir=str(out_dir_p),
        )

    # ------------------------------------------------------------------ #
    # 7. Finalize
    # ------------------------------------------------------------------ #
    # Two separate bests — NAS selection vs honest accuracy reporting
    full_trials_final = [t for t in all_trials
                         if t.get("rung") not in ("cheap_pruned", "medium_pruned", "error")]
    rank_pool_final = full_trials_final if full_trials_final else all_trials
    best_by_search_final  = max(rank_pool_final, key=_sel_score)
    best_by_primary_final = max(rank_pool_final, key=_primary_score)
    best_trial = best_by_primary_final   # legacy variable used below
    save_json(out_dir_p / "best_trial_by_search.json",  best_by_search_final)
    save_json(out_dir_p / "best_trial_by_primary.json", best_by_primary_final)
    save_json(out_dir_p / "best_trial.json", best_by_primary_final)
    from .ensemble import evaluate_config_on_holdout
    selected_single_test = evaluate_config_on_holdout(
        best_by_search_final["config"], raw, seed=seed, device=device,
    )

    total_time = time.time() - t_start
    final = {
        "best_trial": best_trial,
        "all_trials": all_trials,
        "ensemble": ensemble_result,
        "selected_single_test": selected_single_test,
        "test_primary": selected_single_test.get("test_primary"),
        "evaluation_protocol": "select_on_validation_report_on_test",
        "summary": summary,
        "baselines": baselines,
        "total_seconds": total_time,
        "n_trials": len(all_trials),
        "action_stats": a_stats,
        "mode": "refine" if use_refine_loop else "propose",
        "curve_feedback": curve_feedback,
    }
    save_json(out_dir_p / "final_report.json", final)

    best_single = best_primary
    best_ensemble = ensemble_result.get("primary")
    print(f"\n[NAS] Done!  best_single={best_single:.6f}  "
          + (f"best_ensemble={best_ensemble:.6f}  " if best_ensemble else "")
          + f"total_time={total_time:.0f}s")
    print(f"  Best arch: {best_trial['config']['arch']['family']}")
    return final


# ---------------------------------------------------------------------------
# Standalone ensemble — run AFTER NAS search (no new LLM calls)
# ---------------------------------------------------------------------------

def run_ensemble_only(
    *,
    out_dir: str,
    ensemble_k: int = 5,
    seed: int = 42,
    device: Optional[str] = None,
    # data source params (same as run_llm_nas_v2)
    openml_id: Optional[int] = None,
    task: str = "auto",
    source: str = "openml",
    builtin_name: Optional[str] = None,
    csv_path: Optional[str] = None,
) -> Dict:
    """Load existing trials_index.json and build ensemble — no NAS, no LLM."""
    import time
    t_start = time.time()

    out_dir_p = Path(out_dir)
    trials_path = out_dir_p / "trials_index.json"

    if not trials_path.exists():
        raise FileNotFoundError(
            f"trials_index.json not found in {out_dir}. "
            "Run NAS search first (without --ensemble_only)."
        )

    raw_data = load_json(trials_path)
    all_trials = raw_data["trials"] if isinstance(raw_data, dict) and "trials" in raw_data else raw_data
    print(f"[Ensemble] Loaded {len(all_trials)} trials from {trials_path}")

    # Load dataset (needed for re-training)
    from .data import load_raw as load_dataset
    raw, _ = load_dataset(
        source=source,
        openml_id=openml_id,
        task=task,
        builtin_name=builtin_name,
        csv_path=csv_path,
    )

    if ensemble_k < 2 or len(all_trials) < ensemble_k:
        print(f"[Ensemble] Not enough trials ({len(all_trials)}) for k={ensemble_k}. Skipping.")
        return {}

    from .ensemble import ensemble_top_k
    ensemble_result = ensemble_top_k(
        all_trials, raw,
        k=ensemble_k,
        method="greedy",
        seed=seed,
        device=device,
        out_dir=str(out_dir_p),
    )

    best_single = max(t["primary"] for t in all_trials)
    best_ensemble = ensemble_result.get("primary")
    elapsed = time.time() - t_start

    result = {
        "best_single": best_single,
        "best_ensemble": best_ensemble,
        "ensemble": ensemble_result,
        "n_trials": len(all_trials),
        "seconds": elapsed,
    }
    save_json(out_dir_p / "ensemble_report.json", result)

    print(f"\n[Ensemble] Done!  best_single={best_single:.6f}  "
          + (f"best_ensemble={best_ensemble:.6f}  " if best_ensemble else "")
          + f"time={elapsed:.0f}s")
    return result
