"""
Prompts for the LLM agent.

One always-on system message and four task operators (each returning STRICT
JSON with a fixed shape):

- ``SYSTEM_PROMPT`` (system message, always sent) tells the LLM what kind of
  artefact it is generating, which architecture families exist, and the
  JSON-only output rule.

- ``COLD_START``  : "Here is a dataset summary. Propose N diverse initial
  candidates spanning multiple architecture families." This is where the
  LLM uses its priors -- it knows that wide tabular data with many
  categoricals likes embeddings + GBM-style nets, that small N likes
  dropout, etc.

- ``MUTATE``      : "Here is the parent and the recent history. Make a
  SMALL modification of the parent that you expect to improve the
  validation score." Corresponds to AmoebaNet's ``mutate()`` operator.

- ``CROSSOVER``   : "Combine the strong points of these two parents."
  Top-level fields (preprocess / arch / train) are treated as units that
  can be inherited from either parent, in the spirit of Puzzle's
  "library of alternatives".

- ``REFLECT``     : "You have observed the following trajectory. Summarise
  what is and isn't working, and recommend a search direction for the
  next K trials." Returned as a JSON object with a free-text ``analysis``
  string + a structured ``hints`` dict that the next mutate calls read.

The compact search-space schema is embedded in every payload via
:mod:`src.llm.schema`, so the LLM always has it regardless of context-
length pressure.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .schema import SEARCH_SPACE, schema_block


def _safe_dataset_summary(dataset_summary: Dict[str, Any]) -> Dict[str, Any]:
    """Remove held-out labels and oversized row samples before LLM prompting."""
    safe = dict(dataset_summary)
    for key in ("class_balance_test", "split_subjects_test", "sample_rows"):
        safe.pop(key, None)
    return safe


# ---------------------------------------------------------------------------
# Backwards-compat: legacy v1 single prompt.
# Used by ``src.llm_nas_core.run_llm_nas`` which still expects a flat MLP
# config schema. New code should use SYSTEM_PROMPT + the v2 operators
# below; this constant is kept only so the import in ``llm_nas_core`` does
# not break.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT_V1 = """You are an ML researcher building neural networks for tabular data.
OUTPUT MUST BE STRICT JSON ONLY. No markdown, no code fences, no extra text.

Return JSON with keys:
model_type, hidden_dims, activation, dropout, use_batchnorm, embedding_dim,
optimizer, lr, weight_decay, batch_size, epochs, patience
"""


# ---------------------------------------------------------------------------
# v2 system prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """ROLE
====
You are the policy of an iterative neural-architecture search algorithm for
TABULAR data. You will be called many times in a single conversation. Each
call is one ITERATION of the search loop. You have a FIXED iteration BUDGET
(it will be told to you in every user message: "iteration N of M"). Your job
is to spend the entire budget improving the model — never declare the model
"good enough" or stop early. The system will stop you when the budget runs
out; until then, every iteration must propose a CONCRETE new action that
genuinely tries to improve the validation score.

CONVERSATION PROTOCOL
=====================
Each user message contains the SAME set of blocks (with updated content):
  - DATASET_SUMMARY  : facts about the data
  - PARENT_CANDIDATES: top-K configs found so far (you can pick any to mutate)
  - YOUR_ACTION_HISTORY: a ground-truth log of every action you have picked
                         in this conversation, with: child config hash, whether
                         it was a duplicate, and the resulting validation score
  - CURRENT_BEST_CURVE: a sampled training curve of the current best config
  - SEARCH-SPACE SCHEMA: the action menu and config schema

Your previous assistant messages (your past reasonings + actions) ARE in this
conversation already. Re-read them before each new turn. Use them to:
  • avoid repeating an action that already produced a duplicate or did not help
  • build on diagnoses you made earlier (don't restart reasoning from scratch)
  • notice patterns ("I called increase_lr 3 times, none improved → switch strategy")

OUTPUT CONTRACT
===============
  1. Output STRICT JSON ONLY. No prose, no markdown, no code fences, no comments.
  2. The exact JSON shape is dictated by the TASK section of the user message
     (e.g. {"configs":[...]}, {"action":"...","params":{...},...}).
  3. Action names must EXACTLY match one of the names listed in the action menu.
     NEVER emit "none", "None", "null", "already_good", "no_action" — these
     are NOT valid actions and will be rejected.
  4. Numeric fields stay within their documented ranges; categorical fields
     come from the listed choice sets verbatim.
  5. arch.family selects family-specific keys; only emit the keys listed under
     arch._per_family[<family>] for the chosen family.
  6. If a constraint cannot be satisfied (e.g. d_token divisible by n_heads),
     adjust the value rather than emit something invalid.

REASONING REQUIREMENTS
======================
Every action you propose must be backed by an explicit chain of thought
in the diagnosis block of your JSON output:

  - Observation: cite SPECIFIC NUMBERS from CURRENT_BEST_CURVE and
                 YOUR_ACTION_HISTORY (e.g. "train_loss 0.45→0.42 in 5 epochs,
                 val_loss 0.48→0.49 — small overfit; my last 2 actions were
                 increase_lr and add_layer, neither improved primary").
  - Inference:   what the observations imply about the model (overfitting,
                 underfitting, capacity_ceiling, lr_too_high, etc.)
  - Hypothesis:  the specific change you want to make and why you expect it
                 to help.
  - Prediction:  what you expect to happen to val_primary on the next eval.

ARCHITECTURE FAMILIES
=====================
  - mlp             : simple feedforward; strong baseline, fast to train.
  - resmlp          : residual MLP with pre-norm; helps when optimal depth is high.
  - ft_transformer  : tokenises each feature; strong on heterogeneous tabular.
  - gated_tab       : GLU-gated MLP; useful for numeric data with interactions.
  - autoint         : multi-head self-attention over feature tokens.
  - tabm            : shared trunk + K independent heads (internal ensemble);
                      parameter-efficient, often outperforms plain MLP/ResMLP
                      at the same param budget. K controls ensemble size (4-16).

Heuristic priors (override these based on YOUR_ACTION_HISTORY when it disagrees):
  - n_rows < 5_000, many categoricals    : small models, high dropout, embedding cat encoder
  - n_rows >= 50_000, mostly numeric     : ResMLP / GatedTab / TabM often beat plain MLP
  - heterogeneous, mid N (5k..50k)       : FT-Transformer often strongest
  - n_features > 100                     : AutoInt / FT-Transformer with small d_token;
                                           TabM with moderate K and width
  - severe class imbalance               : label_smoothing > 0, small mixup
  - regression                           : skip label_smoothing; small mixup can help
  - tabular winning lr is OFTEN at extremes (1e-5..1e-4 OR 3e-3..3e-2),
    NOT at textbook 1e-3. Use explore_lr_extreme_low/high actions to probe.

NEVER:
  - copy the previous config verbatim
  - propose configs whose hash matches one in YOUR_ACTION_HISTORY
  - declare the model "good enough" before the budget is exhausted
  - emit field values outside the schema-documented range

If you are unsure which family is best, prefer FT-Transformer for mixed-type
data and ResMLP/TabM for mostly-numeric data; both are robust defaults.
"""


# ---------------------------------------------------------------------------
# Per-operator instruction blocks
# ---------------------------------------------------------------------------

COLD_START_INSTRUCTION = """\
TASK: COLD-START.
Propose {n} initial configurations to seed the search. They must be DIVERSE
AND COVER EXTREMES of the search space — empirically, tabular NN winners often
live in CORNER CASES (very low LR or very high LR, very deep or very shallow,
very heavy or very light regularisation), NOT at the safe textbook midpoints.

Hard requirements (all must hold across the {n} configs):
  R1. Cover at least min({n}, 6) distinct ``arch.family`` values (one per family
      if n>=6). Available families: mlp, resmlp, ft_transformer, gated_tab, autoint, tabm.
  R2. LR diversity — at least one config with lr in [1e-5, 1e-4] (very low),
      at least one with lr in [3e-3, 1e-2] (high). DO NOT pick all configs in
      the textbook 5e-4 .. 1e-3 range.
  R3. Capacity diversity — at least one tiny model (~10-50k params) AND at
      least one big model (~1M-3M params).
  R4. Depth diversity — include at least one shallow model (1-2 layers/blocks)
      and at least one deep one (5+ layers/blocks).
  R5. Dropout diversity — at least one config with dropout in [0.0, 0.05]
      (almost no regularisation) AND at least one with dropout >= 0.4
      (heavy regularisation).
  R6. Preprocessing — use at least 2 different ``preprocess.num_encoder``
      values across the set.
  R7. DO NOT pick the same arch family more than ceil({n}/2) times.

Why these rules: random search tends to beat naive LLM cold-start because
random naturally hits LR/depth/dropout extremes, while LLMs gravitate to
textbook defaults. To beat random, you MUST explicitly explore extremes.

Return JSON:
  {{"configs": [<cfg1>, <cfg2>, ..., <cfgN>]}}
with exactly {n} items. Each ``<cfg>`` must validate against the SCHEMA below.
"""


MUTATE_INSTRUCTION = """\
TASK: MUTATE.
You are given a PARENT config and the recent HISTORY of (config, validation primary,
learning_curve) triplets. Propose ONE child that is a SMALL modification of the parent
and is expected to improve the validation score.

=== STEP 1: ANALYSE THE LEARNING CURVES (mandatory — do this FIRST) ===

For the PARENT and the best trials in HISTORY, inspect each ``learning_curve`` dict:

  val_curve_sampled          : val primary at sampled epochs
  train_loss_curve_sampled   : train loss at sampled epochs
  overfit_signal             : "none" | "mild_overfit" | "overfitting: ..."
  underfit_signal            : "none" | "possible underfitting" | "val still rising..."
  grad_norm_stats            : {mean, max, final, trend, signal}
  best_val_epoch             : epoch of best val score
  convergence_epoch_95pct    : epoch where val first reached 95% of best
  early_stopped              : did training stop early?

Interpretation rules:
  • overfit_signal = "overfitting"   → raise dropout (+0.1..0.2), shrink model, raise weight_decay ×5
  • overfit_signal = "mild_overfit"  → small regularisation tweak (dropout +0.05, wd ×2)
  • underfit_signal = "val still rising" → more epochs, larger lr, or bigger model
  • train_loss falls fast, val barely moves → severe overfit → strong regularisation
  • best_val_epoch < 5               → lr too high or model unstable → divide lr by 2..4
  • early_stopped = False AND val rising → increase epochs or patience
  • grad_norm_stats.signal = "EXPLODING" → reduce lr by ×0.25 or add grad_clip = 1.0
  • grad_norm_stats.signal = "VANISHING" → switch to residual arch (resmlp) or raise lr ×2
  • grad_norm_stats.trend = "increasing" → gradient instability → add layer norm or reduce depth

=== STEP 2: IDENTIFY THE FAILURE MODE ===

Pick ONE of: [overfitting | underfitting | unstable_gradients | wrong_arch | needs_more_epochs | already_good]
Explain your reasoning in 1-2 sentences before proposing.

=== STEP 3: PROPOSE THE FIX ===

Constraints:
  - change at most 2 fields relative to the parent (unless changing arch.family)
  - the child must validate against the SCHEMA
  - do NOT emit a config already present in HISTORY (compare structurally)
  - if you change ``arch.family``, fully re-pick ALL that family's keys from the SCHEMA
  - keep numeric perturbations local (e.g. lr within ~0.5x..2x) unless REFLECTION_HINTS says otherwise

DIVERSITY RULE (mandatory):
  - If REFLECTION_HINTS["explore_untried"] is non-empty → switch arch.family to one of those
    families at least once every 4 mutations.
  - If REFLECTION_HINTS["avoid_families_recent"] contains the parent's family → switch family.
  - Never propose the same arch.family more than 3 times in a row.

=== OUTPUT FORMAT ===

Return JSON with EXACTLY these keys:
{
  "reasoning": {
    "curve_obs": "<what you see in the parent's learning curve — cite numbers>",
    "failure_mode": "<one of: overfitting | underfitting | unstable_gradients | wrong_arch | needs_more_epochs | already_good>",
    "proposed_change": "<what you changed and why — be specific>",
    "expected_effect": "<what improvement you expect>"
  },
  "config": <cfg>,
  "rationale": "<one sentence summary>"
}
"""


CROSSOVER_INSTRUCTION = """\
TASK: CROSSOVER.
You are given two parent configs with their learning curves. Construct ONE child that
combines the strong points of both. Treat each top-level field (``preprocess`` / ``arch``
/ ``train``) as a unit that can be inherited from either parent. You may then make small
adjustments to harmonise the result (e.g. if you take a deep arch from A and a high lr
from B, you may lower the lr a notch). If either parent shows overfitting in its
learning_curve, prefer the regularisation settings (dropout, weight_decay) from the
better-generalising parent. Do not invent fields the SCHEMA does not list.

Return JSON:
  {"config": <cfg>, "rationale": "<one sentence explaining the combination>"}
"""


REFLECT_INSTRUCTION = """\
TASK: REFLECT.
You have observed the trajectory below, including learning curves and gradient norm stats.
Analyse what is working and recommend a search direction for the next {k} trials.

Pay special attention to:
  1. Which arch families / regularisation settings correlate with HIGH val scores?
  2. Which trials show overfitting (train loss down, val flat/declining) or unstable gradients?
  3. Are gradient norms (grad_norm_stats.signal) exploding or vanishing anywhere? → Adjust lr / arch.
  4. Is convergence speed (convergence_epoch_95pct) fast or slow? → Signals if lr is right.
  5. Is the search still improving or has it converged?

Return JSON with EXACTLY this shape (do not add or rename keys):
{{
  "analysis": "<3-5 sentences: cite concrete trial IDs, primary scores, and learning curve numbers. Identify the best config region found so far and the main failure mode.>",
  "hints": {{
    "preferred_families":      [<one or more of: "mlp", "resmlp", "ft_transformer", "gated_tab", "autoint">],
    "avoid_families":          [<zero or more family names that consistently underperformed>],
    "preferred_lr_range":      [<float lo>, <float hi>],
    "preferred_dropout":       [<float lo>, <float hi>],
    "preferred_weight_decay":  [<float lo>, <float hi>],
    "overfit_risk":            "<none | low | medium | high>",
    "gradient_stability":      "<stable | exploding_risk | vanishing_risk>",
    "explore_more":            "<short phrase, e.g. 'larger models', 'lower lr', 'gated_tab family'>",
    "exploit_more":            "<short phrase, e.g. 'around trial 7 regularisation', 'ft_transformer with d_token=64'>",
    "next_priority":           "<single most impactful change you recommend for the next trial>"
  }}
}}
The hints dict will be passed verbatim to the next batch of MUTATE calls, so be concrete and actionable.
"""


# ---------------------------------------------------------------------------
# Builders that the agent uses
# ---------------------------------------------------------------------------

def _schema_block(schema: Optional[Dict[str, Any]]) -> str:
    """Render the schema as a ``SCHEMA:\\n{...}`` block, defaulting to the
    canonical search space from :mod:`src.llm.schema`."""
    return schema_block(schema if schema is not None else SEARCH_SPACE)


def build_user_payload_cold_start(
    n: int,
    dataset_summary: Dict[str, Any],
    baseline: Dict[str, Any],
    schema: Optional[Dict[str, Any]] = None,
) -> str:
    return "\n\n".join([
        COLD_START_INSTRUCTION.format(n=n),
        f"DATASET_SUMMARY:\n{json.dumps(_safe_dataset_summary(dataset_summary), ensure_ascii=False, indent=2)}",
        f"BASELINE:\n{json.dumps(baseline, ensure_ascii=False, indent=2)}",
        _schema_block(schema),
    ])


def build_user_payload_mutate(
    parent: Dict[str, Any],
    history: List[Dict[str, Any]],
    dataset_summary: Dict[str, Any],
    hints: Dict[str, Any],
    schema: Optional[Dict[str, Any]] = None,
) -> str:
    return "\n\n".join([
        MUTATE_INSTRUCTION,
        f"DATASET_SUMMARY:\n{json.dumps(_safe_dataset_summary(dataset_summary), ensure_ascii=False)}",
        f"PARENT:\n{json.dumps(parent, ensure_ascii=False, indent=2)}",
        f"HISTORY (top-K + recent, primary higher=better):\n"
        f"{json.dumps(history, ensure_ascii=False, indent=2)}",
        f"REFLECTION_HINTS:\n{json.dumps(hints, ensure_ascii=False)}",
        _schema_block(schema),
    ])


def build_user_payload_crossover(
    a: Dict[str, Any],
    b: Dict[str, Any],
    history: List[Dict[str, Any]],
    dataset_summary: Dict[str, Any],
    hints: Dict[str, Any],
    schema: Optional[Dict[str, Any]] = None,
) -> str:
    return "\n\n".join([
        CROSSOVER_INSTRUCTION,
        f"DATASET_SUMMARY:\n{json.dumps(_safe_dataset_summary(dataset_summary), ensure_ascii=False)}",
        f"PARENT_A:\n{json.dumps(a, ensure_ascii=False, indent=2)}",
        f"PARENT_B:\n{json.dumps(b, ensure_ascii=False, indent=2)}",
        f"HISTORY:\n{json.dumps(history, ensure_ascii=False, indent=2)}",
        f"REFLECTION_HINTS:\n{json.dumps(hints, ensure_ascii=False)}",
        _schema_block(schema),
    ])


def build_user_payload_reflect(
    history: List[Dict[str, Any]],
    k: int,
    dataset_summary: Dict[str, Any],
    surrogate_importances: List[Any],
    schema: Optional[Dict[str, Any]] = None,
) -> str:
    return "\n\n".join([
        REFLECT_INSTRUCTION.format(k=k),
        f"DATASET_SUMMARY:\n{json.dumps(_safe_dataset_summary(dataset_summary), ensure_ascii=False)}",
        f"TRAJECTORY:\n{json.dumps(history, ensure_ascii=False, indent=2)}",
        "SURROGATE_FEATURE_IMPORTANCES (top features the surrogate currently weights):\n"
        f"{json.dumps(surrogate_importances, ensure_ascii=False)}",
        _schema_block(schema),
    ])


# ---------------------------------------------------------------------------
# PROPOSE instruction — simplified operator that replaces mutate + crossover
# ---------------------------------------------------------------------------

PROPOSE_INSTRUCTION = """\
TASK: PROPOSE {n} NEW CONFIGURATIONS.

You have seen the full trial history below (sorted: best first, then recent).
Your job: propose {n} configs that you genuinely expect to improve on the current best.
You have FULL FREEDOM — you can:
  • Fine-tune the best config (small tweaks to lr, dropout, width)
  • Switch arch.family entirely if the current best seems to have hit a wall
  • Combine settings from two good trials (manual crossover)
  • Try a regularisation fix if you see consistent overfitting in the curves
  • Explore a family not yet tried

=== STEP 1: DIAGNOSE THE SEARCH SO FAR ===

For each of the top-3 trials, look at its ``learning_curve``:
  • overfit_signal / underfit_signal — is training generalising?
  • grad_norm_stats.signal — are gradients stable?
  • convergence_epoch_95pct — did training converge fast or slow?
  • val_curve_sampled vs train_loss_curve_sampled — gap = overfit severity

Identify the MAIN bottleneck:
  [overfitting | underfitting | unstable_gradients | wrong_arch | capacity_ceiling | good_keep_exploring]

=== STEP 2: PROPOSE {n} DIVERSE CANDIDATES ===

Rules:
  - Each proposal MUST have a different strategy (don't propose {n} near-identical configs)
  - At least one proposal should exploit the best config region (small tweak)
  - At least one proposal should explore a different arch.family or a bigger regularisation shift
  - Every config must validate against the SCHEMA
  - Do NOT duplicate configs already in HISTORY
  - If REFLECTION_HINTS specifies preferred_families or avoid_families, respect them
  - If grad_norm_stats.signal = "EXPLODING" → at least one proposal must reduce lr or add grad_clip
  - If grad_norm_stats.signal = "VANISHING" → at least one proposal must switch to resmlp or raise lr

=== OUTPUT FORMAT (KEEP IT TERSE — upstream caps responses at ~2000 chars) ===

Bug-fix #8: provider truncates responses at ~2000 chars regardless of max_tokens.
KEEP THE OUTPUT COMPACT: short reasoning strings (<= 30 words), only essential
keys per config, no comments, no whitespace beyond JSON minimum.

Return JSON:
{{
  "diagnosis": {{
    "best_trial_id": <int>,
    "main_bottleneck": "<one of: overfitting | underfitting | unstable_gradients | wrong_arch | capacity_ceiling | good_keep_exploring>",
    "key_observation": "<concise, max 200 chars>"
  }},
  "proposals": [
    {{
      "strategy": "<exploit_best | explore_new_family | regularisation_fix | capacity_increase | crossover>",
      "reasoning": "<≤25 words>",
      "config": <cfg>
    }},
    ... (exactly {n} items — keep each cfg minimal)
  ]
}}
"""


def build_user_payload_propose(
    history: List[Dict[str, Any]],
    dataset_summary: Dict[str, Any],
    hints: Dict[str, Any],
    n: int = 3,
    schema: Optional[Dict[str, Any]] = None,
) -> str:
    """Build the user payload for the PROPOSE operator (replaces mutate+crossover)."""
    return "\n\n".join([
        PROPOSE_INSTRUCTION.format(n=n),
        f"DATASET_SUMMARY:\n{json.dumps(_safe_dataset_summary(dataset_summary), ensure_ascii=False)}",
        f"TRIAL_HISTORY (top-K by primary, then recent — primary higher = better):\n"
        f"{json.dumps(history, ensure_ascii=False, indent=2)}",
        f"REFLECTION_HINTS (from last reflect call — may be empty dict at start):\n"
        f"{json.dumps(hints, ensure_ascii=False)}",
        _schema_block(schema),
    ])


# ---------------------------------------------------------------------------
# REFINE instruction — tool-call operator for targeted architecture improvement
# ---------------------------------------------------------------------------

REFINE_INSTRUCTION = """\
TASK: REFINE — pick ONE PARENT and apply ONE targeted action to improve it.

You are acting as an ML engineer who diagnoses learning curves and applies
SINGLE TARGETED FIXES. You have a FIXED set of actions (tools) you can call.
You MUST pick exactly one action from the menu — there is no "no-op" option.

=== STEP 1: REVIEW YOUR PAST DECISIONS — most important step ===

The block YOUR_ACTION_HISTORY below shows EVERY action you have personally
chosen so far, with: action name, parent it was applied to, hash of the
resulting child, and whether the child was a duplicate (DEDUP) or actually
ran. Pay attention to:

  • Which actions you have called MOST frequently (>=3 times)
  • Which actions have produced DEDUPs (your apply produced a config already
    seen — you wasted those trials!)
  • Which actions actually IMPROVED the best score (look at primary_after
    relative to the best at the time)

You MUST avoid:
  - calling an action you've already used 3+ times
  - calling the same action that just produced a DEDUP
  - repeating an action that didn't improve the score in your last 2 calls

=== STEP 2: REVIEW THE PARENT POPULATION ===

The block PARENT_CANDIDATES shows the top-K parents by validation score, plus
their family, lr, depth, and how recently they were tried. You may pick ANY
of these parents to mutate (not just the top-1). Consider:

  • Has the top-1 parent been "milked" too long (last 5 actions all on it)?
    Then pick parent_B or parent_C for diversity.
  • Is there a parent in a completely different family than what you've been
    refining? Mutating it explores a new region.
  • The OLDEST parent (highest age) is often a good source for crossover-like
    exploration.

=== STEP 3: DIAGNOSE THE PICKED PARENT'S LEARNING CURVE ===

Look at the CURVE block for your chosen parent:
  • overfit_signal / underfit_signal
  • grad_norm_stats.signal  (EXPLODING / VANISHING / GROWING / normal)
  • convergence_epoch_95pct — did training converge quickly?
  • val_curve_sampled vs train_loss_curve_sampled — gap = overfit severity

=== STEP 4: SELECT ONE ACTION FROM THE MENU ===

Available actions and when to use them:
{action_menu}

Rules (FOLLOW STRICTLY):
  R1. Pick the action that MOST DIRECTLY addresses the diagnosed failure mode.
  R2. EXPLORATION RULE — if you cannot diagnose a clear problem, pick:
        "switch_family"            → try a different architecture
        "explore_lr_extreme_low"   → if standard lr ~1e-3 plateaued
        "explore_lr_extreme_high"  → if standard lr is too low (slow conv)
        "explore_deep" / "explore_shallow" → escape architectural local optima
      DO NOT return "none", "already_good", "no_action", or null — INVALID.
  R3. LR-BIAS GUARD — Tabular NN winners often live at LR EXTREMES (1e-5..1e-4
      OR 3e-3..3e-2), not at textbook 1e-3. Symmetry rules:
        • val plateaus 3+ epochs without improvement → try "reduce_lr" or
          "explore_lr_extreme_low" (LR may be too high to converge cleanly)
        • best_val_epoch > 60% of total epochs AND val still rising → "increase_lr"
        • DO NOT call "increase_lr" if YOUR_ACTION_HISTORY shows it called 2+ times
        • If you've called increase_lr 3+ times and it never helped — call
          "explore_lr_extreme_low" instead (regime change)
  R4. DIVERSITY RULE — if last 3 trials are the SAME arch.family AND none
      improved best primary, you MUST call "switch_family" with a NEW family.
  R5. NO-REPEAT RULE — avoid calling the SAME action you called last iteration
      if that action did not improve the metric.
  R6. ANTI-BIAS RULE — count how many times YOUR_ACTION_HISTORY shows you said
      "underfitting" or "slow_convergence". If that count >= 2, this turn you
      MUST diagnose differently (try "overfitting", "unstable_gradients", or
      "needs_exploration") AND pick a non-capacity action (regularisation,
      change_scheduler, switch_family, explore_*, NOT add_layer/widen/increase_lr).
  R7. ANTI-DEDUP RULE — if YOUR_ACTION_HISTORY shows your last action was a DEDUP,
      pick an EXTREME action this turn (explore_*, switch_family). Do not pick
      a small local change that will likely dedup again.

=== STEP 5: OUTPUT — STRUCTURED CHAIN OF THOUGHT ===

Return JSON with EXACTLY these keys (full chain of thought, no shortcuts):
{{
  "parent_id":  "<which parent you picked, e.g. 'best' / 'parent_B' / 'parent_C'. Default 'best' if unsure.>",
  "diagnosis": {{
    "observation":  "<2-3 sentences citing CONCRETE NUMBERS: (1) from CURRENT_BEST_CURVE — e.g. 'train_loss dropped 0.45→0.30 by epoch 8, val_loss flat at 0.48' — and (2) from YOUR_ACTION_HISTORY — e.g. 'I called increase_lr at steps 6,9,12; none improved primary above 0.785'.>",
    "inference":    "<2-3 sentences: what those observations IMPLY about the model. State the failure mode explicitly and explain WHY the numbers point to it.>",
    "hypothesis":   "<2-3 sentences: the SPECIFIC change you propose and why you expect it to fix the inferred failure mode. Reference the past actions you DIDN'T try yet.>",
    "prediction":   "<1 sentence: what you expect to happen to val_primary on the next evaluation (e.g. '+0.005..+0.015' or 'will probably regress on val but unlock a better region').>",
    "failure_mode": "<one of: overfitting | underfitting | unstable_gradients | needs_more_capacity | slow_convergence | needs_exploration>",
    "history_check":"<one sentence: what YOUR_ACTION_HISTORY tells you to AVOID this turn — specific past mistakes or patterns.>",
    "confidence":   "<low | medium | high>"
  }},
  "action": "<EXACT action name from the menu — MUST be one of the 20 listed>",
  "params": {{}},
  "rationale": "<one sentence summary of why this action this turn>"
}}

The diagnosis fields (observation/inference/hypothesis/prediction) are NOT
optional. The advisor of this project has explicitly required full chain of
thought. Brief 1-clause answers are insufficient and may be rejected.

CRITICAL RULES FOR "action" FIELD:
  • Must be a STRING EXACTLY matching one of the action names from the menu.
  • NEVER use "none", "None", "already_good", "no_action", null.
  • If unsure → use "switch_family" or one of the four explore_* actions.

For "params", supply hints the action can use:
  increase_dropout   → {{"target_dropout": 0.3}}
  change_activation  → {{"activation": "gelu"}}
  switch_family      → {{"family": "resmlp"}}     # any family except current
  change_scheduler   → {{"scheduler": "cosine"}}
For actions without params just return {{}}.
"""


def build_user_payload_refine(
    best_cfg: Dict[str, Any],
    best_curve: Dict[str, Any],
    refine_history: List[Dict[str, Any]],
    dataset_summary: Dict[str, Any],
    schema: Optional[Dict[str, Any]] = None,
    parent_candidates: Optional[List[Dict[str, Any]]] = None,
    action_history: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Build the user payload for one REFINE (tool-call) step.

    New blocks (Prong B + D, March 2026):
      - parent_candidates : list of dicts {id, family, lr, depth, primary, age, curve_summary}
                           — top-K parents from population, not just best.
      - action_history    : list of dicts {step, action, parent_id, child_hash, dedup, primary_after}
                           — every prior REFINE decision YOU made, with its outcome.
    """
    from .actions import ACTION_GROUPS, ACTION_DESCRIPTIONS

    # Build a compact action menu grouped by category
    lines = []
    for group, names in ACTION_GROUPS.items():
        lines.append(f"  [{group}]")
        for n in names:
            lines.append(f"    {n:30s} — {ACTION_DESCRIPTIONS[n]}")
    action_menu = "\n".join(lines)

    instruction = REFINE_INSTRUCTION.format(action_menu=action_menu)

    blocks = [
        instruction,
        f"DATASET_SUMMARY:\n{json.dumps(_safe_dataset_summary(dataset_summary), ensure_ascii=False)}",
    ]

    # Block: parent population (Prong D)
    if parent_candidates:
        blocks.append(
            "PARENT_CANDIDATES (top-K parents available for mutation; pick one as 'parent_id'):\n"
            f"{json.dumps(parent_candidates, ensure_ascii=False, indent=2)}"
        )

    # Block: current best config (parent_id='best' default)
    blocks.append(
        f"CURRENT_BEST_CONFIG (parent_id='best'):\n"
        f"{json.dumps(best_cfg, ensure_ascii=False, indent=2)}"
    )
    blocks.append(
        f"CURRENT_BEST_CURVE:\n{json.dumps(best_curve, ensure_ascii=False, indent=2)}"
    )

    # Block: YOUR_ACTION_HISTORY (Prong B)
    if action_history is not None:
        blocks.append(
            "YOUR_ACTION_HISTORY (every REFINE decision you've made so far — read this carefully!):\n"
            f"{json.dumps(action_history, ensure_ascii=False, indent=2)}"
        )

    # Backwards compat: keep generic refine_history block
    blocks.append(
        f"REFINE_HISTORY (last {len(refine_history)} refine steps — summary):\n"
        f"{json.dumps(refine_history[-6:], ensure_ascii=False, indent=2)}"
    )

    blocks.append(_schema_block(schema))
    return "\n\n".join(blocks)


# ===========================================================================
# Mode 3 (freeform): no fixed action vocabulary — LLM directly proposes
# the diff against a chosen parent config.
# ===========================================================================

FREEFORM_INSTRUCTION = """\
TASK: FREEFORM — propose a config DIFF to apply to a chosen parent.

You are operating in the EXPERIMENTAL freeform mode: there is NO fixed
action menu. Instead, you DIRECTLY specify which fields of the parent
config should change, and to what values. You may change one field, or
many — whatever you judge will help most.

=== STEP 1: REVIEW STATE (same as REFINE) ===

Look at YOUR_ACTION_HISTORY, PARENT_CANDIDATES, CURRENT_BEST_CURVE.
Pay attention to:
  • diagnoses you already made (re-read your past assistant messages)
  • child_hash values that appeared as DUPLICATEs
  • actions where you ping-ponged (reduce_lr → increase_lr → reduce_lr)
  • curve_narrative (textual interpretation of training trajectory)

=== STEP 2: PICK A PARENT ===

Return parent_id ∈ {"best", "parent_B", "parent_C", "parent_D", "parent_E"}.

=== STEP 3: PROPOSE A CHANGES DICT ===

The "changes" field is a flat dict of dotted-paths → new values. Examples:
  - "train.lr": 0.0005
  - "train.epochs": 150
  - "train.patience": 25
  - "train.weight_decay": 0.001
  - "train.scheduler": "cosine"
  - "train.batch_size": 256
  - "train.feature_noise_std": 0.05
  - "train.mixup_alpha": 0.2
  - "train.label_smoothing": 0.1
  - "arch.dropout": 0.3
  - "arch.activation": "gelu"
  - "arch.hidden_dims": [512, 256, 128]   (mlp only)
  - "arch.n_blocks": 4                    (resmlp/gated_tab/ft_transformer/autoint)
  - "arch.block_width": 256               (resmlp/gated_tab only)
  - "arch.d_token": 96                    (ft_transformer/autoint only — divisible by n_heads)
  - "arch.n_heads": 4                     (ft_transformer/autoint only)
  - "arch.normalization": "batchnorm"     (mlp/resmlp/gated_tab only)
  - "preprocess.num_encoder": "quantile"
  - "preprocess.cat_encoder": "embedding"
  - "arch.family": "ft_transformer"       (full family switch — see below!)

RULES:
  R1. Each value must obey the SCHEMA constraints. If you change arch.family,
      you MUST also specify ALL family-specific keys from scratch (your changes
      replace the family's arch sub-tree entirely).
  R2. You may include 1 to 6 changed fields. More fields = bigger jump.
  R3. Never propose a child that is identical to a config in YOUR_ACTION_HISTORY
      (check the cfg_hash values).
  R4. ALWAYS include enough chain-of-thought (observation, inference, hypothesis,
      prediction) to justify your changes.
  R5. If you observe LR oscillation in YOUR_ACTION_HISTORY (e.g. you flipped
      between lower/higher lr 3+ times), DO NOT change train.lr this turn.
      Try changing something else (epochs, scheduler, dropout, family).
  R6. If your last 3 changes only touched ONE field — try a MULTI-FIELD change
      this turn (e.g. lr + scheduler + dropout together).
  R7. STRUCTURAL DIVERSITY RULE (CRITICAL):
      Count how many of your last 4 changes touched ONLY fields from this
      "safe set": {{train.lr, train.epochs, arch.dropout, train.weight_decay}}.
      If that count >= 3 → you MUST change at least one STRUCTURAL field this
      turn: arch.family, arch.n_blocks, arch.block_width, arch.hidden_dims,
      arch.d_token, arch.n_heads, or preprocess.num_encoder.
      Rationale: small tweaks to lr/dropout/epochs have diminishing returns.
      Architecture and preprocessing changes unlock new regions of search space.
  R8. HISTORY CHECK RULE:
      Before proposing, explicitly list in history_check:
        (a) every train.lr value you already tried and its result
        (b) every arch.dropout value you already tried and its result
        (c) how many consecutive turns you changed ONLY safe fields (lr/dropout/epochs)
      Then state what you will NOT repeat this turn. Minimum 80 characters.

=== STEP 4: OUTPUT ===

Return JSON with EXACTLY these keys:
{{
  "parent_id":  "<best | parent_B | ...>",
  "diagnosis": {{
    "observation":  "<2-3 sentences with concrete numbers from CURRENT_BEST_CURVE>",
    "inference":    "<2-3 sentences: what those numbers imply about the model>",
    "hypothesis":   "<2-3 sentences: what specific changes you propose and why>",
    "prediction":   "<expected delta to val_primary, e.g. '+0.005..+0.015'>",
    "failure_mode": "<one of: overfitting | underfitting | unstable_gradients | needs_more_capacity | slow_convergence | needs_exploration>",
    "history_check":"<MANDATORY >= 80 chars: list past lr/dropout values tried + their outcomes; state what you will NOT repeat; count safe-field-only turns>",
    "confidence":   "<low | medium | high>"
  }},
  "changes": {{
    "train.lr": 0.0005,
    "train.epochs": 150,
    "arch.dropout": 0.25
  }},
  "rationale": "<one sentence summary>"
}}

IMPORTANT: history_check is NOT optional. A response with history_check shorter
than 80 characters or missing lr/dropout history will be rejected and retried.
Each iteration MUST try something concrete — empty "changes" dict is not allowed.
"""


def build_user_payload_freeform(
    best_cfg: Dict[str, Any],
    best_curve: Dict[str, Any],
    refine_history: List[Dict[str, Any]],
    dataset_summary: Dict[str, Any],
    schema: Optional[Dict[str, Any]] = None,
    parent_candidates: Optional[List[Dict[str, Any]]] = None,
    action_history: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Build the user payload for Mode 3 (freeform) — same blocks as
    build_user_payload_refine but with FREEFORM_INSTRUCTION instead.
    """
    blocks = [
        FREEFORM_INSTRUCTION,
        f"DATASET_SUMMARY:\n{json.dumps(_safe_dataset_summary(dataset_summary), ensure_ascii=False)}",
    ]
    if parent_candidates:
        blocks.append(
            "PARENT_CANDIDATES (pick one as 'parent_id'):\n"
            f"{json.dumps(parent_candidates, ensure_ascii=False, indent=2)}"
        )
    blocks.append(
        f"CURRENT_BEST_CONFIG (parent_id='best'):\n"
        f"{json.dumps(best_cfg, ensure_ascii=False, indent=2)}"
    )
    blocks.append(
        f"CURRENT_BEST_CURVE:\n{json.dumps(best_curve, ensure_ascii=False, indent=2)}"
    )
    if action_history is not None:
        blocks.append(
            "YOUR_ACTION_HISTORY (every freeform decision you've made so far):\n"
            f"{json.dumps(action_history, ensure_ascii=False, indent=2)}"
        )
    blocks.append(
        f"REFINE_HISTORY (last {len(refine_history)} steps — summary):\n"
        f"{json.dumps(refine_history[-6:], ensure_ascii=False, indent=2)}"
    )
    blocks.append(_schema_block(schema))
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# MULTI-REFINE instruction — LLM picks up to 3 DISTINCT actions per turn
# ---------------------------------------------------------------------------

MULTI_REFINE_INSTRUCTION = """\
TASK: MULTI-REFINE — pick 1 to 3 DISTINCT actions to apply to the chosen parent.

All actions are applied SEQUENTIALLY to the SAME parent config, producing ONE
new config that is then evaluated. You get one trial per turn regardless of
how many actions you pick.

{lr_throttle_block}
=== RULES ===

  R1. UNIQUENESS: each action name must appear at most ONCE in your list.
      Duplicates are silently dropped — so if you list ["increase_lr","increase_lr"]
      you waste a slot.
  R2. NO-CONTRADICTION: do not pair opposing actions in the same turn:
        BAD:  increase_lr  +  reduce_lr
        BAD:  add_layer    +  remove_layer
        BAD:  increase_dropout  +  decrease_dropout
      These cancel each other and produce noise.
  R3. GOOD COMBOS (orthogonal changes unlock new regions):
        switch_family  +  change_scheduler           — architecture + training fix
        widen          +  increase_dropout            — capacity + regularisation
        change_activation + change_scheduler          — two independent tweaks
        add_layer      +  increase_dropout            — depth + regularisation
  R4. Action 1 = primary fix for the diagnosed failure mode.
      Actions 2-3 = complementary tweaks.
  R5. All standard REFINE rules apply (valid action names, anti-bias, diversity).

=== ACTION MENU ===
{action_menu}

=== OUTPUT FORMAT ===

Return JSON with EXACTLY these keys:
{{
  "parent_id": "<best | parent_B | parent_C | parent_D | parent_E>",
  "diagnosis": {{
    "observation":  "<2-3 sentences citing CONCRETE NUMBERS from curves and history>",
    "inference":    "<what those numbers imply about the model>",
    "hypothesis":   "<why this COMBINATION of actions addresses the failure>",
    "prediction":   "<expected delta to val_primary, e.g. +0.005..+0.015>",
    "failure_mode": "<overfitting|underfitting|unstable_gradients|needs_more_capacity|slow_convergence|needs_exploration>",
    "history_check":"<what YOUR_ACTION_HISTORY says to avoid; cite specific repeated mistakes>",
    "confidence":   "<low | medium | high>"
  }},
  "actions": [
    {{"action": "<action_name_1>", "params": {{}}}},
    {{"action": "<action_name_2>", "params": {{}}}}
  ],
  "rationale": "<one sentence: why this combination this turn>"
}}

The "actions" array: 1, 2, or 3 items. Each "action" must be a DIFFERENT
name from the menu above. Never repeat the same action name in one response.
"""


def build_user_payload_multi_refine(
    best_cfg: Dict[str, Any],
    best_curve: Dict[str, Any],
    refine_history: List[Dict[str, Any]],
    dataset_summary: Dict[str, Any],
    schema: Optional[Dict[str, Any]] = None,
    parent_candidates: Optional[List[Dict[str, Any]]] = None,
    action_history: Optional[List[Dict[str, Any]]] = None,
    lr_throttled: bool = False,
) -> str:
    """Build user payload for MULTI-REFINE (up to 3 actions per turn)."""
    from .actions import ACTION_GROUPS, ACTION_DESCRIPTIONS

    lines = []
    for group, names in ACTION_GROUPS.items():
        lines.append(f"  [{group}]")
        for n in names:
            lines.append(f"    {n:30s} — {ACTION_DESCRIPTIONS[n]}")
    action_menu = "\n".join(lines)

    lr_throttle_block = ""
    if lr_throttled:
        lr_throttle_block = (
            "⚠️  LR THROTTLE ACTIVE: You have used LR-related actions (increase_lr / reduce_lr / "
            "explore_lr_extreme_low / explore_lr_extreme_high) in 2 of the last 3 steps. "
            "You MUST NOT include any LR action this turn. Focus on architecture, scheduler, "
            "dropout, or family instead.\n\n"
        )

    instruction = MULTI_REFINE_INSTRUCTION.format(
        action_menu=action_menu,
        lr_throttle_block=lr_throttle_block,
    )

    blocks = [
        instruction,
        f"DATASET_SUMMARY:\n{json.dumps(_safe_dataset_summary(dataset_summary), ensure_ascii=False)}",
    ]
    if parent_candidates:
        blocks.append(
            "PARENT_CANDIDATES (pick one as 'parent_id'):\n"
            f"{json.dumps(parent_candidates, ensure_ascii=False, indent=2)}"
        )
    blocks.append(
        f"CURRENT_BEST_CONFIG (parent_id='best'):\n"
        f"{json.dumps(best_cfg, ensure_ascii=False, indent=2)}"
    )
    blocks.append(
        f"CURRENT_BEST_CURVE:\n{json.dumps(best_curve, ensure_ascii=False, indent=2)}"
    )
    if action_history is not None:
        blocks.append(
            "YOUR_ACTION_HISTORY (every MULTI-REFINE decision — read carefully!):\n"
            f"{json.dumps(action_history, ensure_ascii=False, indent=2)}"
        )
    blocks.append(
        f"REFINE_HISTORY (last {len(refine_history)} steps — summary):\n"
        f"{json.dumps(refine_history[-6:], ensure_ascii=False, indent=2)}"
    )
    blocks.append(_schema_block(schema))
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# CRITIC instruction — Stage 1 of Critic-Corrector pipeline
# ---------------------------------------------------------------------------

CRITIC_INSTRUCTION = """\
TASK: CRITIC — analyse the full NAS history and write a concise verdict.

You are the CRITIC agent. You do NOT propose new configs. Your job is to
analyse the trajectory of trials and produce a structured summary that the
CORRECTOR agent will use to make better decisions next.

=== WHAT TO ANALYSE ===

1. WHAT WORKED: Which arch families, LR ranges, schedulers, depths correlated
   with HIGH val_primary? Cite specific trial IDs and scores.

2. WHAT FAILED: Which directions were tried and consistently underperformed?
   Include specific numbers (e.g. "increase_lr called 4×, never improved above 0.693").

3. OVERFITTING SIGNALS: Were there trials where train_loss dropped fast but
   val was flat/declining? Which configs had this problem?

4. UNDERFITTING SIGNALS: Were there configs where val was still rising at
   the last epoch? Which ones, and how much room was left?

5. UNEXPLORED DIRECTIONS: Which arch families have NOT been tried or tried
   only once? Which LR extremes (< 1e-4 or > 5e-3) are unexplored?

6. VERDICT: In 2-3 sentences — the single most important thing the next
   Corrector agent should focus on.

=== OUTPUT FORMAT ===

Return JSON:
{{
  "what_worked": "<cite arch families, LR ranges, trial IDs that scored best>",
  "what_failed": "<cite specific actions/configs that repeatedly underperformed>",
  "overfit_trials": [<trial_id>, ...],
  "underfit_trials": [<trial_id>, ...],
  "unexplored": "<families or hyperparameter regions not yet tried>",
  "avoid_actions": ["<action_name>", ...],
  "verdict": "<2-3 sentences: the single most important direction for next step>",
  "recommended_family": "<arch family the Corrector should prefer, or null>",
  "recommended_lr_range": [<lo>, <hi>]
}}
"""


def build_critic_payload(
    all_trials: List[Dict[str, Any]],
    dataset_summary: Dict[str, Any],
    schema: Optional[Dict[str, Any]] = None,
) -> str:
    """Build user payload for the CRITIC call."""
    # Compact trial summary
    trial_summaries = []
    for t in all_trials:
        trial_summaries.append({
            "trial_id": t.get("trial_id"),
            "op": t.get("op"),
            "primary": round(t.get("primary", -999), 5),
            "family": t.get("config", {}).get("arch", {}).get("family"),
            "lr": t.get("config", {}).get("train", {}).get("lr"),
            "n_blocks": t.get("config", {}).get("arch", {}).get("n_blocks"),
            "dropout": t.get("config", {}).get("arch", {}).get("dropout"),
            "scheduler": t.get("config", {}).get("train", {}).get("scheduler"),
            "overfit_signal": t.get("curve_summary", {}).get("overfit_signal", "?"),
            "underfit_signal": t.get("curve_summary", {}).get("underfit_signal", "?"),
            "best_val_epoch": t.get("curve_summary", {}).get("best_val_epoch"),
            "early_stopped": t.get("curve_summary", {}).get("early_stopped"),
        })
    return "\n\n".join([
        CRITIC_INSTRUCTION,
        f"DATASET_SUMMARY:\n{json.dumps(_safe_dataset_summary(dataset_summary), ensure_ascii=False)}",
        f"TRIAL_HISTORY ({len(trial_summaries)} trials):\n"
        f"{json.dumps(trial_summaries, ensure_ascii=False, indent=2)}",
        _schema_block(schema),
    ])


# ---------------------------------------------------------------------------
# CORRECTOR instruction — Stage 2 of Critic-Corrector pipeline
# ---------------------------------------------------------------------------

CORRECTOR_INSTRUCTION = """\
TASK: CORRECTOR — given the Critic's analysis, propose the best action(s).

You are the CORRECTOR agent. The CRITIC has already analysed the history for
you. Your job is to translate its verdict into 1-3 concrete REFINE actions.

=== CRITIC'S VERDICT ===
{critic_json}

=== YOUR JOB ===

Based on the Critic's verdict:
  1. Read "verdict" — what is the single most important direction?
  2. Read "avoid_actions" — do NOT use any of those actions.
  3. Read "unexplored" — consider trying something from there.
  4. Read "recommended_family" and "recommended_lr_range" as strong hints.

Then pick 1-3 actions from the menu below. Same rules as MULTI-REFINE:
  - Each action name must be UNIQUE in your list
  - No contradicting pairs
  - Actions in "avoid_actions" are BANNED this turn

=== ACTION MENU ===
{action_menu}

{lr_throttle_block}

=== OUTPUT FORMAT ===

Return JSON with EXACTLY these keys:
{{
  "parent_id": "<best | parent_B | parent_C | parent_D | parent_E>",
  "diagnosis": {{
    "critic_takeaway": "<1-2 sentences: what the Critic told you to focus on>",
    "hypothesis":      "<why the chosen actions address the Critic's verdict>",
    "prediction":      "<expected delta to val_primary>",
    "failure_mode":    "<overfitting|underfitting|unstable_gradients|needs_more_capacity|slow_convergence|needs_exploration>",
    "confidence":      "<low | medium | high>"
  }},
  "actions": [
    {{"action": "<action_name>", "params": {{}}}}
  ],
  "rationale": "<one sentence summary>"
}}
"""


def build_corrector_payload(
    critic_result: Dict[str, Any],
    best_cfg: Dict[str, Any],
    best_curve: Dict[str, Any],
    refine_history: List[Dict[str, Any]],
    dataset_summary: Dict[str, Any],
    schema: Optional[Dict[str, Any]] = None,
    parent_candidates: Optional[List[Dict[str, Any]]] = None,
    action_history: Optional[List[Dict[str, Any]]] = None,
    lr_throttled: bool = False,
) -> str:
    """Build user payload for the CORRECTOR call."""
    from .actions import ACTION_GROUPS, ACTION_DESCRIPTIONS

    lines = []
    for group, names in ACTION_GROUPS.items():
        lines.append(f"  [{group}]")
        for n in names:
            lines.append(f"    {n:30s} — {ACTION_DESCRIPTIONS[n]}")
    action_menu = "\n".join(lines)

    lr_throttle_block = ""
    if lr_throttled:
        lr_throttle_block = (
            "⚠️  LR THROTTLE ACTIVE: Do NOT include any LR-related action "
            "(increase_lr / reduce_lr / explore_lr_extreme_low / explore_lr_extreme_high) this turn."
        )

    instruction = CORRECTOR_INSTRUCTION.format(
        critic_json=json.dumps(critic_result, ensure_ascii=False, indent=2),
        action_menu=action_menu,
        lr_throttle_block=lr_throttle_block,
    )

    blocks = [
        instruction,
        f"DATASET_SUMMARY:\n{json.dumps(_safe_dataset_summary(dataset_summary), ensure_ascii=False)}",
    ]
    if parent_candidates:
        blocks.append(
            "PARENT_CANDIDATES:\n"
            f"{json.dumps(parent_candidates, ensure_ascii=False, indent=2)}"
        )
    blocks.append(
        f"CURRENT_BEST_CONFIG:\n{json.dumps(best_cfg, ensure_ascii=False, indent=2)}"
    )
    blocks.append(
        f"CURRENT_BEST_CURVE:\n{json.dumps(best_curve, ensure_ascii=False, indent=2)}"
    )
    if action_history is not None:
        blocks.append(
            "YOUR_ACTION_HISTORY:\n"
            f"{json.dumps(action_history, ensure_ascii=False, indent=2)}"
        )
    blocks.append(_schema_block(schema))
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# BATCH-PROPOSE instruction — LLM generates N candidates for surrogate filter
# ---------------------------------------------------------------------------

BATCH_PROPOSE_INSTRUCTION = """\
TASK: BATCH-PROPOSE — generate {n} diverse candidate configs for surrogate-guided search.

You are operating in BATCH mode. You will propose {n} candidate configs.
A surrogate model will score all candidates, and only the top-{k} will be
actually trained. Your goal: generate a set that is BOTH high-quality AND
diverse, so the surrogate has good options to pick from.

=== CRITIC'S ANALYSIS (read this first!) ===
{critic_json}

=== YOUR JOB ===

1. READ the Critic's "verdict", "what_worked", "unexplored", "avoid_actions".
2. PROPOSE {n} configs following these rules:
   - At least {exploit_n} configs must EXPLOIT the best region (small tweaks
     to top-1 or top-2 config — different LR, scheduler, dropout).
   - At least {explore_n} configs must EXPLORE: different arch.family, very
     different LR range, or unexplored preprocessing.
   - At least 1 config with LR in [1e-5, 5e-4] (extreme low).
   - At least 1 config with LR in [5e-3, 1e-2] (extreme high).
   - No two configs may be identical (structurally).
   - Do NOT reproduce configs already in TRIAL_HISTORY (check hashes
     mentally — same family+blocks+lr+dropout ≈ duplicate).

=== OUTPUT FORMAT ===

{{
  "analysis": "<2-3 sentences from the Critic's verdict — what you focus on>",
  "configs": [<cfg1>, <cfg2>, ..., <cfg{n}>]
}}

Each <cfg> must validate against the SCHEMA. Keep configs compact.
Exactly {n} items in "configs".
"""


def build_batch_propose_payload(
    all_trials: List[Dict[str, Any]],
    dataset_summary: Dict[str, Any],
    critic_result: Dict[str, Any],
    hints: Dict[str, Any],
    n: int = 15,
    batch_k: int = 3,
    schema: Optional[Dict[str, Any]] = None,
) -> str:
    """Build user payload for the BATCH-PROPOSE operator."""
    exploit_n = max(1, n // 3)
    explore_n = max(1, n // 3)

    instruction = BATCH_PROPOSE_INSTRUCTION.format(
        n=n,
        k=batch_k,
        exploit_n=exploit_n,
        explore_n=explore_n,
        critic_json=json.dumps(critic_result, ensure_ascii=False, indent=2),
    )

    # Compact trial history
    trial_summaries = []
    for t in sorted(all_trials, key=lambda x: -x.get("primary", -999))[:20]:
        trial_summaries.append({
            "trial_id": t.get("trial_id"),
            "primary": round(t.get("primary", -999), 5),
            "family": t.get("config", {}).get("arch", {}).get("family"),
            "lr": t.get("config", {}).get("train", {}).get("lr"),
            "n_blocks": t.get("config", {}).get("arch", {}).get("n_blocks"),
            "dropout": t.get("config", {}).get("arch", {}).get("dropout"),
        })

    return "\n\n".join([
        instruction,
        f"DATASET_SUMMARY:\n{json.dumps(_safe_dataset_summary(dataset_summary), ensure_ascii=False)}",
        f"TRIAL_HISTORY (top-20 by primary):\n"
        f"{json.dumps(trial_summaries, ensure_ascii=False, indent=2)}",
        f"REFLECTION_HINTS:\n{json.dumps(hints, ensure_ascii=False)}",
        _schema_block(schema),
    ])


__all__ = [
    "SYSTEM_PROMPT",
    "SYSTEM_PROMPT_V1",
    "COLD_START_INSTRUCTION",
    "MUTATE_INSTRUCTION",
    "CROSSOVER_INSTRUCTION",
    "REFLECT_INSTRUCTION",
    "PROPOSE_INSTRUCTION",
    "REFINE_INSTRUCTION",
    "FREEFORM_INSTRUCTION",
    "MULTI_REFINE_INSTRUCTION",
    "CRITIC_INSTRUCTION",
    "CORRECTOR_INSTRUCTION",
    "BATCH_PROPOSE_INSTRUCTION",
    "build_user_payload_cold_start",
    "build_user_payload_mutate",
    "build_user_payload_crossover",
    "build_user_payload_reflect",
    "build_user_payload_propose",
    "build_user_payload_refine",
    "build_user_payload_freeform",
    "build_user_payload_multi_refine",
    "build_critic_payload",
    "build_corrector_payload",
    "build_batch_propose_payload",
]
