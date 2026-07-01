"""
Search-space schema bridge for the LLM agent.

The single source of truth for the search space is :mod:`src.search_space`.
This module is a thin re-export so that prompt builders in
:mod:`src.llm.prompts` can pull the same description that the validator,
sampler and surrogate use, without depending on the full search-space
machinery (sampling RNGs, mutation operators, featurization, ...).

Usage in prompts::

    from .schema import SEARCH_SPACE, SEARCH_SPACE_JSON

The schema is exposed in three forms:

- ``SEARCH_SPACE`` -- a plain ``dict`` (lazy, computed once on import) that
  mirrors :func:`src.search_space.schema_for_prompt`. Useful if a builder
  wants to inspect or trim it programmatically.

- ``SEARCH_SPACE_JSON`` -- the same dict serialised as a JSON string with
  ``indent=2``. This is what gets pasted verbatim into the LLM prompt.

- ``get_search_space()`` -- function returning a fresh deep copy of the
  schema, for callers that want to mutate it locally without polluting the
  module-level constant.

Architecture families: mlp, resmlp, ft_transformer, gated_tab, autoint, tabm
(see :data:`src.search_space.ARCH_FAMILIES`).
"""
from __future__ import annotations

import copy
import json
from typing import Any, Dict

from ..search_space import (
    ARCH_FAMILIES,
    BATCH_SIZES,
    NUM_ENCODERS,
    CAT_ENCODERS,
    OPTIMIZERS,
    SCHEDULERS,
    schema_for_prompt,
)

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

#: Canonical search space, identical to ``schema_for_prompt()`` but cached.
SEARCH_SPACE: Dict[str, Any] = schema_for_prompt()

#: Pre-rendered JSON string of :data:`SEARCH_SPACE`, ready to embed in a prompt.
SEARCH_SPACE_JSON: str = json.dumps(SEARCH_SPACE, ensure_ascii=False, indent=2)

#: Tuple of legal architecture family names, re-exported for convenience.
ARCH_FAMILIES_TUPLE = tuple(ARCH_FAMILIES)


def get_search_space() -> Dict[str, Any]:
    """Return a fresh deep copy of the search space.

    Use this when you want to pass a (possibly trimmed) schema to the prompt
    builders without mutating the module-level :data:`SEARCH_SPACE`.
    """
    return copy.deepcopy(SEARCH_SPACE)


def schema_block(schema: Dict[str, Any] | None = None) -> str:
    """Render a schema dict as a ``"SCHEMA:\\n{...}"`` block for prompts.

    Defaults to :data:`SEARCH_SPACE`.
    """
    if schema is None:
        return "SCHEMA:\n" + SEARCH_SPACE_JSON
    return "SCHEMA:\n" + json.dumps(schema, ensure_ascii=False, indent=2)


__all__ = [
    "SEARCH_SPACE",
    "SEARCH_SPACE_JSON",
    "ARCH_FAMILIES_TUPLE",
    "get_search_space",
    "schema_block",
]
