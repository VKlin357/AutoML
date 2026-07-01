"""
LLM interaction logger.

Writes every LLM call to a JSONL file so you can paste it to Claude
and ask "why did the LLM make these decisions?".

Usage::

    logger = LLMLogger(out_dir / "llm_log.jsonl")
    logger.log_call(
        operator="mutate",
        prompt=user_prompt,
        raw_response=resp_text,
        parsed={"reasoning": ..., "config": ..., "rationale": ...},
    )

Each line in the JSONL file is a self-contained JSON object with:
    ts              ISO timestamp
    operator        cold_start | propose | reflect | random_fallback | ...
    step            global step index
    prompt_chars    length of user prompt (for context budget tracking)
    prompt          FULL user prompt sent to LLM (was missing before)
    reasoning       FULL CoT block from LLM (failure_mode, curve_obs, etc.)
    rationale       LLM's one-liner summary (full)
    family          arch.family of the proposed config (or null)
    primary_after   val primary after evaluation (filled in later via .update_last)
    raw_response    FULL raw LLM text (no truncation)
    error           FULL error message if LLM failed / fallback triggered

Truncation note (Fix v6): previously prompt was not saved at all, raw_response
was capped at 2000 chars, and reasoning fields at 200-300 chars. This made
log analysis impossible. Now everything is saved in full. If JSONL becomes
too large, pass ``max_field_chars=N`` to LLMLogger() to cap fields at N
characters (default = None, no limit).
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _maybe_truncate(s: str, limit: Optional[int]) -> str:
    """Truncate a string to `limit` chars, only if limit is set."""
    if limit is None or not s or len(s) <= limit:
        return s or ""
    return s[:limit] + f"... [TRUNCATED at {limit} chars; original={len(s)}]"


class LLMLogger:
    """Thread-unsafe but simple append-only JSONL logger for LLM calls.

    By default (``max_field_chars=None``) saves the FULL prompt, FULL response,
    and FULL reasoning chain. Pass a positive integer to cap each text field
    (useful if the JSONL grows beyond a few hundred MB).
    """

    def __init__(self, path: str | Path, max_field_chars: Optional[int] = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._step = 0
        self._last_line_idx: Optional[int] = None
        self._lines: list = []  # in-memory copy for .update_last
        self.max_field_chars = max_field_chars  # None = no truncation

    def log_call(
        self,
        *,
        operator: str,
        prompt: str = "",
        raw_response: str = "",
        parsed: Optional[Dict[str, Any]] = None,
        error: str = "",
        family: Optional[str] = None,
    ) -> int:
        """Append one LLM call record. Returns the line index (for update_last).

        Fix v6: saves FULL prompt and FULL raw_response (was previously
        ``prompt=missing``, ``raw_response[:2000]``). Reasoning fields are
        also saved in full. This enables proper post-hoc analysis of every
        LLM call.
        """
        parsed = parsed or {}
        reasoning = parsed.get("reasoning") or {}
        rationale = parsed.get("rationale", "")
        if not family:
            cfg = parsed.get("config") or {}
            family = cfg.get("arch", {}).get("family") if isinstance(cfg, dict) else None

        lim = self.max_field_chars  # None = no truncation
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "step": self._step,
            "operator": operator,
            "prompt_chars": len(prompt),
            # Fix v6: SAVE THE PROMPT (was missing entirely)
            "prompt": _maybe_truncate(prompt, lim),
            "reasoning": {
                "failure_mode": reasoning.get("failure_mode", ""),
                # Fix v6: full reasoning (was [:300] / [:200])
                "curve_obs": _maybe_truncate(reasoning.get("curve_obs", ""), lim),
                "proposed_change": _maybe_truncate(reasoning.get("proposed_change", ""), lim),
                "expected_effect": _maybe_truncate(reasoning.get("expected_effect", ""), lim),
                # Also save Prong G chain-of-thought fields if present
                "observation": _maybe_truncate(reasoning.get("observation", ""), lim),
                "inference": _maybe_truncate(reasoning.get("inference", ""), lim),
                "hypothesis": _maybe_truncate(reasoning.get("hypothesis", ""), lim),
                "prediction": _maybe_truncate(reasoning.get("prediction", ""), lim),
                "history_check": _maybe_truncate(reasoning.get("history_check", ""), lim),
                "parent_id_picked": reasoning.get("parent_id_picked", ""),
                "n_dedup_retries": reasoning.get("n_dedup_retries", 0),
            } if reasoning else None,
            # Fix v6: full rationale (was [:300])
            "rationale": _maybe_truncate(rationale, lim),
            "family": family,
            "primary_after": None,   # filled in by update_last()
            # Fix v6: FULL raw_response (was [:2000])
            "raw_response": _maybe_truncate(raw_response, lim),
            # Fix v6: full error (was [:500])
            "error": _maybe_truncate(error, lim),
            # Fix v6: optionally include any extra parsed fields (changes dict
            # from freeform mode, _coerced_from marker, etc.)
            "parsed_extra": {
                k: v for k, v in parsed.items()
                if k not in ("reasoning", "rationale", "config")
            } if parsed else None,
        }
        self._step += 1
        self._lines.append(record)
        self._flush_last()
        return len(self._lines) - 1

    def update_primary(self, line_idx: int, primary: float) -> None:
        """Fill in the val primary after evaluation is done."""
        if 0 <= line_idx < len(self._lines):
            self._lines[line_idx]["primary_after"] = round(primary, 6)
            self._rewrite()

    def _flush_last(self) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(self._lines[-1], ensure_ascii=False) + "\n")

    def _rewrite(self) -> None:
        """Rewrite the whole file (called rarely, only on update_primary)."""
        with open(self.path, "w", encoding="utf-8") as f:
            for rec in self._lines:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def summary(self) -> str:
        """One-liner summary for end-of-run printing."""
        total = len(self._lines)
        errors = sum(1 for r in self._lines if r.get("error"))
        fallbacks = sum(1 for r in self._lines if "fallback" in r.get("operator", ""))
        return (f"LLM calls: {total}  errors: {errors}  fallbacks: {fallbacks}  "
                f"log: {self.path}")
