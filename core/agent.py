from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import pandas as pd
import joblib

from .config import AgentConfig
from .llm_client import LLMClient
from .profiling import infer_task, default_metric, profile_tabular
from .candidates import (
    SYSTEM_PROMPT, build_user_prompt, parse_llm_candidates,
    heuristic_candidates, validate_candidate
)
from .evaluator import evaluate_cv, evaluate_holdout
from .pipeline_builder import build_pipeline
from .utils import ensure_dir, json_dump, jsonl_append


class LLMAutoMLAgent:
    def __init__(self, cfg: AgentConfig, llm: LLMClient):
        self.cfg = cfg
        self.llm = llm

        ensure_dir(cfg.output_dir)
        self.runs_path = f"{cfg.output_dir}/runs.jsonl"
        self.best_model_path = f"{cfg.output_dir}/best_model.pkl"
        self.best_meta_path = f"{cfg.output_dir}/best_meta.json"

        self.history: List[Dict[str, Any]] = []
        self.best: Optional[Dict[str, Any]] = None

    def fit(self, df: pd.DataFrame):
        cfg = self.cfg
        t_start = time.time()

        task = cfg.task or infer_task(df, cfg.target)
        metric = cfg.metric or default_metric(task)
        prof = profile_tabular(df, cfg.target)

        availability = {
            # сейчас у нас только sklearn-модели; позже добавишь бустинги — расширишь
            "models": ["hgb", "rf", "et", "logreg", "ridge"],
        }

        use_holdout = prof["n_rows"] > cfg.max_rows_for_cv

        for round_i in range(1, cfg.max_rounds + 1):
            if time.time() - t_start >= cfg.time_budget_s:
                break

            # 1) candidates from LLM or heuristics
            candidates: List[Dict[str, Any]] = []
            if cfg.use_llm:
                hist_sorted = sorted(
                    [h for h in self.history if h.get("status") == "ok"],
                    key=lambda r: r["score_mean"],
                    reverse=True
                )
                user_prompt = build_user_prompt(prof, task, metric, availability, hist_sorted[:10], cfg.n_candidates_per_round)
                llm_text = self.llm.complete(SYSTEM_PROMPT, user_prompt)
                candidates = parse_llm_candidates(llm_text)

            if not candidates:
                import random
                candidates = heuristic_candidates(task, prof, random.Random(cfg.seed + round_i), availability)
                candidates = candidates[:cfg.n_candidates_per_round]

            # 2) evaluate each candidate
            for cand in candidates:
                if time.time() - t_start >= cfg.time_budget_s:
                    break

                err = validate_candidate(task, cand)
                rec: Dict[str, Any] = {
                    "ts": time.time(),
                    "round": round_i,
                    "task": task,
                    "metric": metric,
                    "candidate": cand,
                    "status": "ok",
                }

                if err is not None:
                    rec["status"] = "error"
                    rec["error"] = err
                    jsonl_append(self.runs_path, rec)
                    continue

                try:
                    if use_holdout:
                        ev = evaluate_holdout(
                            df=df, target=cfg.target, task=task, metric=metric,
                            cand=cand, seed=cfg.seed, holdout_size=cfg.holdout_size,
                            max_onehot_categories=cfg.max_onehot_categories
                        )
                    else:
                        ev = evaluate_cv(
                            df=df, target=cfg.target, task=task, metric=metric,
                            cand=cand, seed=cfg.seed, cv_folds=cfg.cv_folds,
                            max_onehot_categories=cfg.max_onehot_categories
                        )

                    rec.update(ev)
                    self.history.append(rec)

                    if self.best is None or rec["score_mean"] > self.best["score_mean"]:
                        self.best = rec

                    jsonl_append(self.runs_path, rec)

                except Exception as e:
                    rec["status"] = "error"
                    rec["error"] = repr(e)
                    jsonl_append(self.runs_path, rec)

        if self.best is None:
            raise RuntimeError("No successful runs. Check runs.jsonl")

        # 3) refit best on full data
        best_cand = self.best["candidate"]

        pipe, used_cols = build_pipeline(
            df=df, target=cfg.target, task=task,
            cand=best_cand, seed=cfg.seed,
            max_onehot_categories=cfg.max_onehot_categories
        )
        Xfull = df.drop(columns=[cfg.target])
        yfull = df[cfg.target].to_numpy()

        pipe.fit(Xfull, yfull)
        joblib.dump(pipe, self.best_model_path)

        meta = {
            "task": task,
            "metric": metric,
            "best_score_mean": self.best["score_mean"],
            "best_score_std": self.best.get("score_std", 0.0),
            "best_candidate": best_cand,
            "used_cols": used_cols,
            "profile": prof,
            "use_holdout": use_holdout,
        }
        json_dump(self.best_meta_path, meta)
        return self

    def predict(self, df: pd.DataFrame):
        pipe = joblib.load(self.best_model_path)
        return pipe.predict(df)

    def predict_proba(self, df: pd.DataFrame):
        pipe = joblib.load(self.best_model_path)
        if hasattr(pipe, "predict_proba"):
            return pipe.predict_proba(df)
        raise AttributeError("Best model has no predict_proba")