"""
LLM client — v2.

Uses the OpenAI Responses API (POST /v1/responses) directly via httpx.
Falls back gracefully: any network / parse error returns a sentinel so the
orchestrator can use its random fallback instead of crashing.

v2 operators (structured JSON responses):
  llm_cold_start  -> List[Dict]            from {"configs": [...]}
  llm_mutate      -> Tuple[Dict, str]      from {"config": ..., "rationale": "..."}
  llm_crossover   -> Tuple[Dict, str]      same shape as mutate
  llm_reflect     -> Dict                  from {"analysis": "...", "hints": {...}}

v1 legacy:
  llm_propose_config  -> Dict  (old flat MLP config, used by llm_nas_core)
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx


# ---------------------------------------------------------------------------
# OpenAI Responses API client
# ---------------------------------------------------------------------------

OPENAI_API_URL      = "https://api.openai.com/v1/chat/completions"
OPENROUTER_API_URL  = "https://openrouter.ai/api/v1/chat/completions"
ANTHROPIC_API_URL   = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION   = "2023-06-01"


class OpenAILLM:
    """
    Thin wrapper around any OpenAI-compatible Chat Completions API.

    Works with:
      - OpenAI directly:  base_url="https://api.openai.com/v1/chat/completions"
      - OpenRouter:       base_url="https://openrouter.ai/api/v1/chat/completions"
                          api_key from openrouter.ai, model e.g. "openai/gpt-4o-mini"

    Usage::

        llm = OpenAILLM(api_key="sk-or-...", base_url=OPENROUTER_API_URL,
                        model="openai/gpt-4o-mini")
        text = llm.complete(system_prompt="You are ...", user_prompt="Hello")
    """

    # Models that require max_completion_tokens instead of max_tokens.
    # GPT-5.x series dropped max_tokens support entirely.
    _MAX_COMPLETION_TOKENS_MODELS = ("gpt-5.",)

    def __init__(
        self,
        api_key: str = "",
        model: str = "openai/gpt-4o-mini",
        temperature: float = 0.7,
        timeout_s: float = 120.0,
        base_url: str = "",
        verbose: bool = False,  # if True: print full prompt + response to stdout
        max_tokens: int = 8192,  # Bug-fix #1: default was unset → API truncated PROPOSE responses at ~1-2k chars
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        # Auto-detect URL: if key starts with "sk-or-" → OpenRouter
        if base_url:
            self.base_url = base_url
        elif self.api_key.startswith("sk-or-"):
            self.base_url = OPENROUTER_API_URL
        else:
            self.base_url = OPENAI_API_URL
        self.model = model
        self.temperature = temperature
        self.timeout_s = timeout_s
        self.verbose = verbose
        self.max_tokens = max_tokens
        # GPT-5.x requires max_completion_tokens instead of max_tokens
        self._use_max_completion_tokens = any(
            model.startswith(p) for p in self._MAX_COMPLETION_TOKENS_MODELS
        )

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        max_retries: int = 6,
        base_delay: float = 5.0,
        verbose: bool = False,
    ) -> str:
        """Send a system + user message; return the assistant's text.

        Retries on 429 (rate limit) and 5xx (server errors) with
        exponential backoff.  Delays: 5s, 10s, 20s, 40s, 80s, 160s.

        For multi-turn conversations (Prong F: REFINE conversation history),
        use ``chat(messages=[...])`` instead.
        """
        return self.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            max_retries=max_retries,
            base_delay=base_delay,
            verbose=verbose,
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_retries: int = 6,
        base_delay: float = 5.0,
        verbose: bool = False,
    ) -> str:
        """Multi-turn chat (Prong F): send a list of {role, content} messages
        and return the assistant's reply text.

        ``messages`` must start with a system message and include all prior
        user/assistant exchanges. The orchestrator maintains the conversation
        state across REFINE turns, so the LLM sees its own previous responses
        as actual assistant messages — much more natural than dumping a
        synthesised history block in the user prompt.
        """
        # GPT-5.x uses max_completion_tokens; older models use max_tokens
        tokens_key = "max_completion_tokens" if self._use_max_completion_tokens else "max_tokens"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            tokens_key: self.max_tokens,  # Bug-fix #1: prevent response truncation
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        _verbose = verbose or self.verbose
        if _verbose:
            sep = "─" * 60
            # Show the last user message and total messages count
            last_user = ""
            for m in reversed(messages):
                if m.get("role") == "user":
                    last_user = m.get("content", "")
                    break
            total_chars = sum(len(m.get("content", "")) for m in messages)
            print(f"\n{sep}")
            print(f"[LLM→] CHAT  turns={len(messages)}  total_chars={total_chars}  last_user_chars={len(last_user)}")
            print(sep)
            print(last_user[-1500:])  # show only tail of last user msg
            print(sep, flush=True)

        last_exc: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                resp = httpx.post(
                    self.base_url,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout_s,
                )

                # 429 Rate Limit — wait and retry
                if resp.status_code == 429:
                    delay = base_delay * (2 ** attempt)
                    retry_after = resp.headers.get("retry-after")
                    if retry_after:
                        try:
                            delay = max(delay, float(retry_after))
                        except ValueError:
                            pass
                    print(f"  [LLM] 429 rate limit — waiting {delay:.0f}s "
                          f"(attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    continue

                # 5xx server errors — retry with backoff
                if resp.status_code >= 500:
                    delay = base_delay * (2 ** attempt)
                    print(f"  [LLM] {resp.status_code} server error — "
                          f"retrying in {delay:.0f}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    continue

                resp.raise_for_status()
                data = resp.json()

                # Chat Completions shape:
                # {"choices": [{"message": {"content": "..."}}]}
                try:
                    content = data["choices"][0]["message"]["content"] or ""
                except (KeyError, IndexError, TypeError):
                    content = ""

                if _verbose:
                    sep = "─" * 60
                    print(f"\n{sep}")
                    print(f"[←LLM] RESPONSE  ({len(content)} chars)")
                    print(sep)
                    print(content)
                    print(sep, flush=True)

                return content

            except httpx.TimeoutException as e:
                delay = base_delay * (2 ** attempt)
                print(f"  [LLM] timeout — retrying in {delay:.0f}s "
                      f"(attempt {attempt + 1}/{max_retries})")
                last_exc = e
                time.sleep(delay)
            except httpx.HTTPStatusError as e:
                # Non-retryable HTTP errors (4xx except 429)
                raise
            except Exception as e:
                last_exc = e
                delay = base_delay * (2 ** attempt)
                time.sleep(delay)

        raise RuntimeError(
            f"LLM request failed after {max_retries} attempts"
        ) from last_exc


# ---------------------------------------------------------------------------
# Native Anthropic API client
# ---------------------------------------------------------------------------

class AnthropicLLM:
    """
    Thin wrapper around the native Anthropic Messages API.

    Works with api_key starting with "sk-ant-".
    Exposes the same .complete() and .chat() interface as OpenAILLM
    so the orchestrator can use either interchangeably.
    """

    def __init__(
        self,
        api_key: str = "",
        model: str = "claude-sonnet-4-5",
        temperature: float = 0.7,
        timeout_s: float = 120.0,
        verbose: bool = False,
        max_tokens: int = 8192,
        **kwargs,  # absorb base_url etc. silently
    ):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self.model = model
        self.temperature = temperature
        self.timeout_s = timeout_s
        self.verbose = verbose
        self.max_tokens = max_tokens

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        max_retries: int = 6,
        base_delay: float = 5.0,
        verbose: bool = False,
    ) -> str:
        return self.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            max_retries=max_retries,
            base_delay=base_delay,
            verbose=verbose,
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_retries: int = 6,
        base_delay: float = 5.0,
        verbose: bool = False,
    ) -> str:
        # Anthropic separates system from user/assistant messages
        system_text = ""
        filtered = []
        for m in messages:
            if m["role"] == "system":
                system_text = m["content"]
            else:
                filtered.append(m)

        payload: Dict[str, Any] = {
            "model":       self.model,
            "messages":    filtered,
            "max_tokens":  self.max_tokens,
            "temperature": self.temperature,
        }
        if system_text:
            payload["system"] = system_text

        headers = {
            "x-api-key":         self.api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type":      "application/json",
        }

        _verbose = verbose or self.verbose
        if _verbose:
            sep = "─" * 60
            last_user = next((m["content"] for m in reversed(filtered) if m["role"] == "user"), "")
            print(f"\n{sep}")
            print(f"[LLM→] ANTHROPIC CHAT  turns={len(filtered)}  model={self.model}")
            print(sep)
            print(last_user[-1500:])
            print(sep, flush=True)

        last_exc: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                resp = httpx.post(
                    ANTHROPIC_API_URL,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout_s,
                )

                if resp.status_code == 429:
                    delay = base_delay * (2 ** attempt)
                    retry_after = resp.headers.get("retry-after")
                    if retry_after:
                        try:
                            delay = max(delay, float(retry_after))
                        except ValueError:
                            pass
                    print(f"  [LLM] 429 rate limit — waiting {delay:.0f}s "
                          f"(attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    continue

                if resp.status_code >= 500:
                    delay = base_delay * (2 ** attempt)
                    print(f"  [LLM] {resp.status_code} server error — "
                          f"retrying in {delay:.0f}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    continue

                if not resp.is_success:
                    print(f"  [AnthropicLLM] HTTP {resp.status_code} error body: {resp.text[:500]}")
                resp.raise_for_status()
                data = resp.json()

                # Anthropic response shape:
                # {"content": [{"type": "text", "text": "..."}]}
                try:
                    content = data["content"][0]["text"] or ""
                except (KeyError, IndexError, TypeError):
                    content = ""

                if _verbose:
                    sep = "─" * 60
                    print(f"\n{sep}")
                    print(f"[←LLM] ANTHROPIC RESPONSE  ({len(content)} chars)")
                    print(sep)
                    print(content)
                    print(sep, flush=True)

                return content

            except httpx.TimeoutException as e:
                delay = base_delay * (2 ** attempt)
                print(f"  [LLM] timeout — retrying in {delay:.0f}s "
                      f"(attempt {attempt + 1}/{max_retries})")
                last_exc = e
                time.sleep(delay)
            except httpx.HTTPStatusError as e:
                raise
            except Exception as e:
                last_exc = e
                delay = base_delay * (2 ** attempt)
                time.sleep(delay)

        raise RuntimeError(
            f"LLM request failed after {max_retries} attempts"
        ) from last_exc


def make_llm(
    api_key: str,
    model: str,
    base_url: str = "",
    temperature: float = 0.7,
    timeout_s: float = 120.0,
    max_tokens: int = 8192,
) -> "OpenAILLM | AnthropicLLM":
    """Factory: picks AnthropicLLM for sk-ant-* keys, OpenAILLM for everything else."""
    key = api_key or os.getenv("ANTHROPIC_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")
    if key.startswith("sk-ant-"):
        # Strip "anthropic/" prefix from model name if present
        model_name = model.replace("anthropic/", "")
        return AnthropicLLM(
            api_key=key,
            model=model_name,
            temperature=temperature,
            timeout_s=timeout_s,
            max_tokens=max_tokens,
        )
    return OpenAILLM(
        api_key=key,
        model=model,
        base_url=base_url,
        temperature=temperature,
        timeout_s=timeout_s,
        max_tokens=max_tokens,
    )


# ---------------------------------------------------------------------------
# Legacy alias (keeps old import paths working)
# ---------------------------------------------------------------------------

# ProxyChatOpenAILLM is kept as an alias so existing imports don't break.
# It ignores proxy_url/proxy_token and reads the key from env or kwarg.
class ProxyChatOpenAILLM(OpenAILLM):
    """Backward-compat wrapper. Use OpenAILLM for new code."""

    def __init__(
        self,
        proxy_url: str = "",       # ignored
        token: str = "",           # treated as api_key if set
        model: str = "gpt-4o-mini",
        verify_ssl: bool = True,   # ignored
        temperature: float = 0.7,
        timeout_s: float = 120.0,
        api_key: str = "",
        max_tokens: int = 8192,
    ):
        # Prefer explicit api_key, then token, then env var
        key = api_key or token or os.getenv("OPENAI_API_KEY", "")
        super().__init__(api_key=key, model=model,
                         temperature=temperature, timeout_s=timeout_s,
                         max_tokens=max_tokens)


# ---------------------------------------------------------------------------
# JSON extraction helpers
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> Any:
    """Try to extract the first valid JSON object or array from ``text``."""
    if not text:
        return None
    # 1. Direct parse
    try:
        return json.loads(text)
    except Exception:
        pass
    # 2. Extract from ```json ... ``` fence
    fence = re.search(r"```(?:json)?\s*([\[{].*?)\s*```", text, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1))
        except Exception:
            pass
    # 3. Find outermost { ... } or [ ... ]
    for opener, closer in [('{', '}'), ('[', ']')]:
        s = text.find(opener)
        e = text.rfind(closer)
        if s != -1 and e != -1 and e > s:
            try:
                return json.loads(text[s:e + 1])
            except Exception:
                pass
    return None


# ---------------------------------------------------------------------------
# v1 legacy helpers
# ---------------------------------------------------------------------------

def safe_json(text: str) -> Dict:
    result = _extract_json(text)
    if isinstance(result, dict):
        return result
    return {}


def llm_propose_config(llm: OpenAILLM, system_prompt: str, user_payload: Any) -> Dict:
    """v1 legacy: propose a flat config dict. Used by the old llm_nas_core."""
    resp = llm.complete(system_prompt, json.dumps(user_payload, ensure_ascii=False))
    return safe_json(resp)


# ---------------------------------------------------------------------------
# v2 operator parsers
# ---------------------------------------------------------------------------

def llm_cold_start(
    llm: OpenAILLM,
    system_prompt: str,
    user_prompt: str,
    expected_n: int = 5,
) -> List[Dict]:
    """Call LLM for cold-start diversity; parse {"configs": [...]}."""
    resp = llm.complete(system_prompt, user_prompt)
    parsed = _extract_json(resp)
    if parsed is None:
        return []
    if isinstance(parsed, dict) and "configs" in parsed:
        configs = parsed["configs"]
        if isinstance(configs, list):
            return [c for c in configs if isinstance(c, dict)]
    if isinstance(parsed, list):
        return [c for c in parsed if isinstance(c, dict)]
    return []


def llm_mutate(
    llm: OpenAILLM,
    system_prompt: str,
    user_prompt: str,
) -> Tuple[Optional[Dict], str]:
    """Call LLM for mutation; parse {"reasoning": {...}, "config": {...}, "rationale": "..."}."""
    resp = llm.complete(system_prompt, user_prompt)
    parsed = _extract_json(resp)
    if parsed is None:
        return None, ""
    if isinstance(parsed, dict):
        cfg = parsed.get("config")
        rationale = str(parsed.get("rationale", ""))
        # Log chain-of-thought reasoning if present
        reasoning = parsed.get("reasoning")
        if isinstance(reasoning, dict):
            fm = reasoning.get("failure_mode", "?")
            pc = reasoning.get("proposed_change", "")[:100]
            print(f"  [CoT] failure_mode={fm}  change={pc}")
        if isinstance(cfg, dict):
            return cfg, rationale
        # LLM returned the config directly (no wrapper)
        if "arch" in parsed or "train" in parsed or "preprocess" in parsed:
            return parsed, ""
    return None, ""


def llm_crossover(
    llm: OpenAILLM,
    system_prompt: str,
    user_prompt: str,
) -> Tuple[Optional[Dict], str]:
    """Call LLM for crossover; same shape as mutate."""
    return llm_mutate(llm, system_prompt, user_prompt)


def llm_reflect(
    llm: OpenAILLM,
    system_prompt: str,
    user_prompt: str,
) -> Dict:
    """Call LLM for reflection; parse {"analysis": "...", "hints": {...}}."""
    resp = llm.complete(system_prompt, user_prompt)
    parsed = _extract_json(resp)
    if isinstance(parsed, dict):
        return parsed
    return {}


def llm_refine_action(
    llm: OpenAILLM,
    system_prompt: str,
    user_prompt: str,
    conversation: Optional[List[Dict[str, str]]] = None,
) -> Tuple[Optional[str], Dict, str, Dict]:
    """Call LLM for the REFINE (tool-call) operator.

    Returns (action_name, params, raw_response, parsed_json).
    ``action_name`` is None on parse failure.

    Prong F (conversation history): if ``conversation`` is provided, this is a
    multi-turn chat — the user_prompt is appended to the conversation, the
    assistant's response is also appended. The orchestrator owns the list and
    can window it (drop oldest user/assistant pair if too long). On the first
    REFINE call, pass conversation=[{"role":"system","content":system_prompt}]
    or just []; we'll initialize.

    Expected LLM output::

        {
          "diagnosis": {"failure_mode": "...", "evidence": "...", "confidence": "..."},
          "action": "<action name>",
          "params": {},
          "rationale": "<one sentence>"
        }
    """
    from .actions import ACTION_NAMES

    if conversation is None:
        # Backwards-compat: single-shot mode (used by some operators)
        resp = llm.complete(system_prompt, user_prompt)
    else:
        # Prong F: multi-turn mode
        if not conversation:
            conversation.append({"role": "system", "content": system_prompt})
        conversation.append({"role": "user", "content": user_prompt})
        resp = llm.chat(conversation)
        # Append assistant message to the conversation so it persists
        conversation.append({"role": "assistant", "content": resp})

    parsed = _extract_json(resp)
    if not isinstance(parsed, dict):
        return None, {}, resp, {}

    action = parsed.get("action", "")
    params = parsed.get("params") or {}
    diagnosis = parsed.get("diagnosis") or {}
    rationale = str(parsed.get("rationale", ""))

    if not isinstance(params, dict):
        params = {}

    # Log to console
    fm = diagnosis.get("failure_mode", "?")
    evidence = diagnosis.get("evidence", "")[:150]
    conf = diagnosis.get("confidence", "?")
    print(f"  [REFINE] failure_mode={fm} ({conf})  action={action}  | {evidence}")
    if rationale:
        print(f"  [REFINE] rationale: {rationale}")

    # Bug-fix #2 (post-processing): coerce common "no-op intent" responses into
    # an exploration action. Empirically, the LLM keeps emitting "None"/"none"/
    # "already_good" even with explicit prompt instructions, especially on datasets
    # where it diagnoses the current best as good. Map these to switch_family so
    # we keep exploring rather than silently falling back to random mutation.
    NOOP_INTENTS = {None, "", "None", "none", "null", "Null",
                    "already_good", "good_keep_exploring", "no_action", "noop"}
    if isinstance(action, str):
        action_normalised = action.strip()
    else:
        action_normalised = action

    if action_normalised in NOOP_INTENTS:
        print(f"  [REFINE] LLM emitted '{action_normalised}' (no-op intent) "
              f"→ post-processing maps to 'switch_family' for exploration")
        action = "switch_family"
        params = {}  # apply_action will pick a different family at random
        # Mark in parsed json so analysis can distinguish coerced vs LLM-picked
        parsed["_coerced_from"] = action_normalised
        return action, params, resp, parsed

    # Validate action name
    if action not in ACTION_NAMES:
        print(f"  [REFINE] Unknown action '{action}' — will use random fallback")
        return None, {}, resp, parsed

    return action, params, resp, parsed


def llm_propose(
    llm: OpenAILLM,
    system_prompt: str,
    user_prompt: str,
    expected_n: int = 3,
) -> tuple[list[Dict], str, Dict]:
    """Call LLM for the PROPOSE operator.

    Returns (proposals, raw_response, parsed_json) where proposals is a list of:
        {"strategy": ..., "reasoning": ..., "config": {...}}
    Falls back to empty list on parse failure.
    """
    resp = llm.complete(system_prompt, user_prompt)
    parsed = _extract_json(resp)
    if not isinstance(parsed, dict):
        return [], resp, {}

    diagnosis = parsed.get("diagnosis", {})
    proposals_raw = parsed.get("proposals", [])
    if not isinstance(proposals_raw, list):
        return [], resp, parsed

    proposals = []
    for p in proposals_raw:
        if isinstance(p, dict) and isinstance(p.get("config"), dict):
            proposals.append(p)

    # Log diagnosis to console
    if diagnosis:
        bottleneck = diagnosis.get("main_bottleneck", "?")
        obs = diagnosis.get("key_observation", "")[:200]
        print(f"  [CoT-Propose] bottleneck={bottleneck}  obs={obs}")

    return proposals, resp, parsed


# ---------------------------------------------------------------------------
# Multi-refine action parser
# ---------------------------------------------------------------------------

LR_ACTIONS: set = {"increase_lr", "reduce_lr", "explore_lr_extreme_low", "explore_lr_extreme_high"}


def llm_multi_refine_action(
    llm: "OpenAILLM",
    system_prompt: str,
    user_prompt: str,
    conversation: Optional[List[Dict[str, str]]] = None,
    lr_throttled: bool = False,
) -> Tuple[List[Tuple[str, Dict]], str, Dict]:
    """Call LLM for MULTI-REFINE; return ([(action, params), ...], raw_resp, parsed).

    Enforces:
      - action names come from the known vocab
      - no duplicate action names in one response
      - LR actions dropped when lr_throttled=True
    """
    from .actions import ACTION_NAMES

    if conversation is None:
        resp = llm.complete(system_prompt, user_prompt)
    else:
        if not conversation:
            conversation.append({"role": "system", "content": system_prompt})
        conversation.append({"role": "user", "content": user_prompt})
        resp = llm.chat(conversation)
        conversation.append({"role": "assistant", "content": resp})

    parsed = _extract_json(resp)
    if not isinstance(parsed, dict):
        return [], resp, {}

    actions_raw = parsed.get("actions", [])
    diagnosis = parsed.get("diagnosis") or {}

    # Log
    fm = diagnosis.get("failure_mode", "?")
    conf = diagnosis.get("confidence", "?")
    obs = str(diagnosis.get("observation", ""))[:150]
    raw_names = [a.get("action") for a in (actions_raw or []) if isinstance(a, dict)]
    print(f"  [MULTI-REFINE] fm={fm} ({conf})  proposed={raw_names}")
    if obs:
        print(f"  [MULTI-REFINE] obs: {obs}")
    if parsed.get("rationale"):
        print(f"  [MULTI-REFINE] rationale: {parsed['rationale']}")

    # Parse and filter
    actions: List[Tuple[str, Dict]] = []
    seen: set = set()
    for item in (actions_raw or [])[:3]:
        if not isinstance(item, dict):
            continue
        name = str(item.get("action", "")).strip()
        params = item.get("params") or {}
        if not isinstance(params, dict):
            params = {}
        if name not in ACTION_NAMES:
            print(f"  [MULTI-REFINE] unknown action '{name}' — skipped")
            continue
        if name in seen:
            print(f"  [MULTI-REFINE] duplicate '{name}' — skipped")
            continue
        if lr_throttled and name in LR_ACTIONS:
            print(f"  [MULTI-REFINE] LR throttle active — '{name}' dropped")
            continue
        seen.add(name)
        actions.append((name, params))

    return actions, resp, parsed


def llm_critic(
    llm: "OpenAILLM",
    system_prompt: str,
    user_prompt: str,
) -> Dict:
    """Call LLM as the CRITIC; parse and return the result dict."""
    resp = llm.complete(system_prompt, user_prompt)
    parsed = _extract_json(resp)
    if isinstance(parsed, dict):
        verdict = parsed.get("verdict", "")[:200]
        print(f"  [CRITIC] verdict: {verdict}")
        return parsed
    return {}
