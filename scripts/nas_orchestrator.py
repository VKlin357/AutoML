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
    ProxyChatOpenAILLM,   # kept for backward compat with existing run scripts
    llm_cold_start,
    llm_reflect,
    llm_propose,
    llm_refine_action,
)
from .llm.prompts import (
    SYSTEM_PROMPT,
    build_user_payload_cold_start,
    build_user_payload_reflect,
    build_user_payload_propose,
    build_user_payload_refine,
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
        "primary": result.primary,
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
        if hasattr(llm, "temperature"):
            llm.temperature = max(saved_temp, 1.1)
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

def _build_parent_candidates(all_trials: List[Dict], top_k: int = 5) -> List[Dict]:
    """Prong D: pick top-K parents from population for LLM to choose from.

    Returns a compact summary per parent so the prompt stays small.
    """
    # Sort by primary descending and take top-K unique configs by hash
    sorted_trials = sorted(all_trials, key=lambda t: -t.get("primary", -1e9))
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

def _evaluate_config(
    cfg: Dict,
    prepared: PreparedSplit,
    out_dir: Path,
    trial_id: int,
    *,
    use_multi_fidelity: bool,
    cheap_scores: List[float],
    promote_frac: float = 0.5,
    seed: int = 42,
    device: Optional[str] = None,
    verbose: bool = False,
) -> FidelityResult:
    tdir = ensure_dir(out_dir / f"trial_{trial_id:03d}")

    if not use_multi_fidelity:
        return evaluate_at_rung(cfg, prepared, FULL, out_dir=tdir, seed=seed, device=device,
                                verbose=verbose)

    # Cheap rung (never verbose — too many epochs)
    cheap_res = evaluate_at_rung(cfg, prepared, CHEAP, out_dir=None, seed=seed, device=device,
                                 verbose=False)
    cheap_scores.append(cheap_res.primary)

    # Promote?
    threshold = successive_halving_threshold(cheap_scores[:-1], promote_frac)
    if cheap_res.primary >= threshold or len(cheap_scores) <= 4:
        # Full rung
        full_res = evaluate_at_rung(cfg, prepared, FULL, out_dir=tdir, seed=seed, device=device,
                                    verbose=verbose)
        return full_res
    else:
        print(f"  [MF] trial_{trial_id:03d} pruned at cheap "
              f"(score={cheap_res.primary:.5f} < threshold={threshold:.5f})")
        save_json(tdir / "metrics.json", {**cheap_res.metrics, "primary": cheap_res.primary,
                                           "rung": "cheap_pruned"})
        return cheap_res


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
    llm = OpenAILLM(
        api_key=api_key,
        model=llm_model,
        temperature=llm_temperature,
        base_url=base_url,
        verbose=verbose,
    )
    surrogate = ConfigSurrogate()
    hints: Dict = {}
    all_trials: List[Dict] = []
    cheap_scores: List[float] = []   # only CHEAP rung scores (for MF threshold)
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
    # 4. Cold start
    # ------------------------------------------------------------------ #
    print(f"\n[NAS] === Cold Start ({coldstart_n} configs) ===")
    cold_cfgs = _llm_cold_start_configs(llm, coldstart_n, summary, baseline_for_prompt, rng)

    for i, cfg in enumerate(cold_cfgs):
        print(f"  [CS {i+1}/{coldstart_n}] family={cfg['arch']['family']}", flush=True)
        save_json(trials_dir / f"trial_{trial_id:03d}" / "config.json", cfg)
        prepared = _prep_from_cfg(cfg, raw)
        tdir = ensure_dir(trials_dir / f"trial_{trial_id:03d}")
        try:
            result = evaluate_at_rung(cfg, prepared, FULL, out_dir=tdir, seed=seed, device=device,
                                      verbose=verbose)
        except Exception as e:
            print(f"    ERROR: {e}")
            result = type("R", (), {
                "primary": -1e9, "metrics": {}, "rung": "error",
                "n_params": 0, "epochs_run": 0, "seconds": 0.0, "early_stopped": False,
                "history": [], "train_loss_history": [], "grad_norm_history": [],
            })()
        seen_hashes.add(_cfg_hash(cfg))
        rec = _trial_record(trial_id, cfg, result, op="cold_start")
        all_trials.append(rec)
        surrogate.add(cfg, result.primary)
        best_primary = max(best_primary, result.primary)
        print(f"    primary={result.primary:.5f}  best={best_primary:.5f}")
        save_json(out_dir_p / "trials_index.json", {"trials": all_trials})
        trial_id += 1

    surrogate.fit()

    # ------------------------------------------------------------------ #
    # 5. Main search loop — REFINE or PROPOSE
    # ------------------------------------------------------------------ #
    n_evo_steps = budget - coldstart_n
    # Bug-fix #8 / Prong B: cap PROPOSE n_proposals at 2 to fit upstream
    # provider's ~2000-char response cap (which truncated all 3-candidate
    # PROPOSE responses in v1/v2/v3, regardless of max_tokens setting).
    n_proposals = min(2, max(2, surr_candidates_k))
    mode_label = "REFINE (tool-call)" if use_refine_loop else "PROPOSE (free-form)"
    print(f"\n[NAS] === {mode_label} Loop ({n_evo_steps} steps) ===")

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
        best_trial_so_far = max(all_trials, key=lambda x: x["primary"])
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

            if not forced_switch:
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
                    for t in all_trials[:coldstart_n]:
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

                # Bug-fix #5: LR-BIAS GUARD.
                # If the LLM keeps picking increase_lr (≥2 times already used),
                # intercept and force switch_family instead.  This prevents the
                # "increase_lr → duplicate → random_refine_fallback" spiral seen
                # in v3 logs where 14/20 steps were wasted fallbacks.
                LR_BIAS_LIMIT = 2
                if (op_label == "refine:increase_lr"
                        and action_counts.get("increase_lr", 0) >= LR_BIAS_LIMIT):
                    cur_family = best_cfg_now.get("arch", {}).get("family", "")
                    other_fams = [f for f in ARCH_FAMILIES if f != cur_family]
                    new_family = rng.choice(other_fams)
                    print(f"  [step {global_step}] LR-BIAS GUARD: increase_lr already used "
                          f"{action_counts['increase_lr']}x → "
                          f"forcing switch_family → '{new_family}'")
                    cfg_to_eval = apply_action(best_cfg_now, "switch_family",
                                               {"family": new_family}, rng)
                    op_label = "forced:switch_family_lr_guard"
                    if llm_logger is not None:
                        llm_logger.log_call(
                            operator="forced_switch_family_lr_guard",
                            prompt="<orchestrator hard-rule: lr-bias guard>",
                            raw_response="",
                            parsed={"reasoning": {
                                "failure_mode": "lr_bias",
                                "curve_obs": f"increase_lr used {action_counts['increase_lr']}x",
                                "proposed_change": f"switch_family→{new_family}",
                                "expected_effect": "diversity instead of lr tuning",
                            }, "rationale": "orchestrator hard-rule"},
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
                seed=seed, device=device,
                verbose=verbose,
            )
        except Exception as e:
            print(f"    ERROR during evaluation: {e}")
            result = type("R", (), {
                "primary": -1e9, "metrics": {}, "rung": "error",
                "n_params": 0, "epochs_run": 0, "seconds": 0.0, "early_stopped": False,
                "history": [], "train_loss_history": [], "grad_norm_history": [],
            })()

        rec = _trial_record(trial_id, cfg_to_eval, result, op=op_label, rationale="")
        all_trials.append(rec)
        surrogate.add(cfg_to_eval, result.primary)

        # Update LLM log with actual eval result
        if llm_logger._lines:
            llm_logger.update_primary(len(llm_logger._lines) - 1, result.primary)

        # Update REFINE history (for next iteration's context)
        if not is_propose_step:
            action_name = op_label.replace("refine:", "").split("#")[0] if op_label.startswith("refine:") else None
            improved = result.primary > best_primary
            refine_history.append({
                "step": global_step,
                "action": action_name,
                "primary_after": round(result.primary, 6),
                "improved": improved,
            })

            # Prong B: rich action_history with cfg_hash + dedup info for the LLM prompt
            child_h = _cfg_hash(cfg_to_eval)
            # Detect if op_label contains "#retryN" (Prong C: stochastic re-apply succeeded after N dedup retries)
            n_retries_str = ""
            if "#" in op_label:
                n_retries_str = op_label.split("#", 1)[1]  # e.g. "retry2"
            # Detect parent_id used (best vs LLM-picked)
            parent_id_used = "best"
            # If forced or fallback, mark accordingly; otherwise, default = best (LLM rarely picks others reliably)
            action_history.append({
                "step": global_step,
                "action": action_name if action_name else op_label,
                "parent_id": parent_id_used,
                "child_hash": child_h,
                "dedup_retries": n_retries_str,
                "fell_back_to_random": op_label == "random_refine_fallback",
                "primary_before_best": round(best_primary, 6),
                "primary_after": round(result.primary, 6),
                "delta_vs_best": round(result.primary - best_primary, 6),
                "improved": improved,
            })
            # Trim action_history if it gets too long (keep last 12 to fit in prompt)
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
    if ensemble_k > 1 and len(all_trials) >= ensemble_k:
        from .ensemble import ensemble_top_k
        ensemble_result = ensemble_top_k(
            all_trials, raw,
            k=ensemble_k,
            method="greedy",
            seed=seed,
            device=device,
            out_dir=str(out_dir_p),
        )

    # ------------------------------------------------------------------ #
    # 7. Finalize
    # ------------------------------------------------------------------ #
    best_trial = max(all_trials, key=lambda x: x["primary"])
    save_json(out_dir_p / "best_trial.json", best_trial)

    total_time = time.time() - t_start
    final = {
        "best_trial": best_trial,
        "all_trials": all_trials,
        "ensemble": ensemble_result,
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
    from .data import load_dataset
    raw = load_dataset(
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
