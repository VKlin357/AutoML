from __future__ import annotations

import os
from typing import Protocol

from openai import OpenAI


class LLMClient(Protocol):
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        ...


class DummyLLM:
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        return ""


class OpenAILLM:
    """
    Uses OPENAI_API_KEY from environment by default.
    Returns text that should be JSON: {"candidates":[...]}
    """
    def __init__(self, model: str = "gpt-4o", api_key: str | None = None, base_url: str | None = None):
        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url  # optional
        )
        self.model = model

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        return resp.choices[0].message.content or ""