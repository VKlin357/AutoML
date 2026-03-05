from __future__ import annotations

import json
import random
from typing import Any, Dict, List, Optional

SYSTEM_PROMPT = """You are an expert AutoML planner for tabular ML.
Propose candidate pipelines as JSON only.

Output schema:
{"candidates":[
  {"drop_cols":[...],
   "cat_encoding":"onehot" or "target",
   "num_scaling": true/false,
   "model":{"name": "...", "params": {...}}
  }, ...
]}
Constraints:
- target encoding is NOT allowed for multiclass.
- prefer robust, not exotic hyperparams.
"""


def build_user_prompt(
    profile: Dict[str, Any],
    task: str,
    metric: str,
    availability: Dict[str, Any],
    history_top: List[Dict[str, Any]],
    n: int
) -> str:
    payload = {
        "task": task,
        "metric": metric,
        "profile": profile,
        "availability": availability,
        "top_history": history_top[:10],
        "request": f"Propose {n} diverse candidates to improve score.",
    }
    return json.dumps(payload, ensure_ascii=False)


def parse_llm_candidates(text: str) -> List[Dict[str, Any]]:
    if not text or not text.strip():
        return []
    try:
        obj = json.loads(text)
        cands = obj.get("candidates", [])
        return cands if isinstance(cands, list) else []
    except Exception:
        return []


def heuristic_candidates(task: str, profile: Dict[str, Any], rng: random.Random, availability: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Strong heuristic generator: gives 3-5 robust candidates.
    LLM improves over this by proposing smarter hyperparams/pipeline combos.
    """
    cat_cols = profile["categorical_cols"]

    high_card = []
    for c, u in profile["nunique_top20"].items():
        if c in cat_cols and u > 200:
            high_card.append(c)

    prefer_target = (task in ("binary", "regression")) and (len(high_card) > 0)

    drop_cols = list(profile.get("id_like_cols", []))[:3]

    # base models: we use only sklearn here to keep dependencies minimal
    if task in ("binary", "multiclass"):
        models = [
            {"name": "hgb", "params": {"learning_rate": rng.choice([0.03, 0.05, 0.1]), "max_leaf_nodes": rng.choice([31, 63, 127]), "l2_regularization": rng.choice([0.0, 0.1, 1.0])}},
            {"name": "rf",  "params": {"n_estimators": rng.choice([300, 600]), "max_depth": rng.choice([None, 12, 20]), "min_samples_leaf": rng.choice([1, 2, 5])}},
            {"name": "et",  "params": {"n_estimators": rng.choice([500, 800]), "max_depth": rng.choice([None, 12, 20]), "min_samples_leaf": rng.choice([1, 2, 5])}},
            {"name": "logreg", "params": {"C": 10 ** rng.uniform(-2, 1), "max_iter": 2000}},
        ]
    else:
        models = [
            {"name": "hgb", "params": {"learning_rate": rng.choice([0.03, 0.05, 0.1]), "max_leaf_nodes": rng.choice([31, 63, 127]), "l2_regularization": rng.choice([0.0, 0.1, 1.0])}},
            {"name": "rf",  "params": {"n_estimators": rng.choice([300, 600]), "max_depth": rng.choice([None, 12, 20]), "min_samples_leaf": rng.choice([1, 2, 5])}},
            {"name": "et",  "params": {"n_estimators": rng.choice([500, 800]), "max_depth": rng.choice([None, 12, 20]), "min_samples_leaf": rng.choice([1, 2, 5])}},
            {"name": "ridge", "params": {"alpha": 10 ** rng.uniform(-3, 2)}},
        ]

    cands = [
        {"drop_cols": drop_cols, "cat_encoding": "onehot", "num_scaling": False, "model": models[0]},
        {"drop_cols": drop_cols, "cat_encoding": "onehot", "num_scaling": True,  "model": next(m for m in models if m["name"] in ("logreg", "ridge"))},
        {"drop_cols": drop_cols, "cat_encoding": "onehot", "num_scaling": False, "model": next(m for m in models if m["name"] in ("rf", "et"))},
    ]

    if prefer_target:
        cands.append({"drop_cols": drop_cols, "cat_encoding": "target", "num_scaling": False, "model": models[0], "target_smoothing": 10.0})

    rng.shuffle(cands)
    return cands


def validate_candidate(task: str, cand: Dict[str, Any]) -> Optional[str]:
    # basic schema checks + safety constraints
    if "model" not in cand or "name" not in cand["model"]:
        return "Candidate missing model.name"
    if cand.get("cat_encoding") not in ("onehot", "target"):
        return "Unknown cat_encoding"
    if task == "multiclass" and cand.get("cat_encoding") == "target":
        return "Target encoding not allowed for multiclass"
    if not isinstance(cand.get("drop_cols", []), list):
        return "drop_cols must be list"
    if not isinstance(cand.get("num_scaling", False), bool):
        return "num_scaling must be bool"
    if not isinstance(cand["model"].get("params", {}), dict):
        return "model.params must be dict"
    return None