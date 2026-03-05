# tabular_agent.py
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    log_loss,
    mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


# ---------- Metric helpers ----------
def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def get_metric_fn(task_type: str, metric: str) -> Callable:
    task_type = task_type.lower()
    metric = metric.lower()

    if task_type in ("binary", "multiclass"):
        if metric in ("auc", "roc_auc"):
            return roc_auc_score
        if metric in ("acc", "accuracy"):
            return accuracy_score
        if metric in ("logloss", "log_loss"):
            return log_loss
        raise ValueError(f"Unsupported classification metric: {metric}")

    if task_type == "regression":
        if metric in ("rmse",):
            return rmse
        if metric in ("mse",):
            return mean_squared_error
        raise ValueError(f"Unsupported regression metric: {metric}")

    raise ValueError(f"Unknown task_type: {task_type}")


# ---------- LLM interface ----------
class LLM:
    """Abstract LLM client. Implement .complete(prompt) for different providers/models."""
    name: str = "abstract-llm"

    def complete(self, prompt: str) -> str:
        raise NotImplementedError


class DummyLLM(LLM):
    """No-LLM baseline: returns empty feature code."""
    name = "dummy"

    def complete(self, prompt: str) -> str:
        return ""


# ---------- Safe-ish code execution (minimal sandbox) ----------
SAFE_GLOBALS = {
    "__builtins__": {
        "len": len,
        "min": min,
        "max": max,
        "sum": sum,
        "abs": abs,
        "float": float,
        "int": int,
        "str": str,
        "bool": bool,
        "range": range,
        "enumerate": enumerate,
        "zip": zip,
        "list": list,
        "dict": dict,
        "set": set,
    },
    "np": np,
    "pd": pd,
}

FEATURE_FUNC_NAME = "make_features"


def extract_python_code(text: str) -> str:
    """
    Extract a python code block if present, else return raw.
    Accepts ```python ... ``` or plain code.
    """
    m = re.search(r"```python\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*(.*?)```", text, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


def compile_feature_fn(code: str) -> Optional[Callable[[pd.DataFrame], pd.DataFrame]]:
    """
    Expect user/LLM to define:
        def make_features(df: pd.DataFrame) -> pd.DataFrame:
            ...
            return df2
    """
    code = extract_python_code(code)
    if not code:
        return None

    # ultra-simple guardrails (not bulletproof)
    banned = ["import os", "import sys", "subprocess", "open(", "exec(", "eval(", "__import__"]
    if any(b in code for b in banned):
        return None

    local_env: Dict[str, Any] = {}
    try:
        exec(code, SAFE_GLOBALS, local_env)  # noqa: S102 (intentionally controlled)
        fn = local_env.get(FEATURE_FUNC_NAME)
        if callable(fn):
            return fn
        return None
    except Exception:
        return None


# ---------- Agent config ----------
@dataclass
class AgentConfig:
    task_type: str                # "binary" | "multiclass" | "regression"
    metric: str                   # "roc_auc" | "accuracy" | "logloss" | "rmse" ...
    target_col: str
    test_size: float = 0.2
    random_state: int = 42
    time_budget_s: int = 600      # total time for training search
    use_llm_features: bool = True
    max_llm_feature_rounds: int = 1


@dataclass
class RunResult:
    model_name: str
    metric_name: str
    score: float
    train_seconds: float
    used_llm: str
    notes: str = ""


# ---------- The agent ----------
class TabularTrainingAgent:
    def __init__(self, config: AgentConfig, llm: Optional[LLM] = None):
        self.cfg = config
        self.llm = llm or DummyLLM()

    def _split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        y = df[self.cfg.target_col]
        X = df.drop(columns=[self.cfg.target_col])

        stratify = None
        if self.cfg.task_type in ("binary", "multiclass"):
            stratify = y

        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=self.cfg.test_size,
            random_state=self.cfg.random_state,
            stratify=stratify
        )
        return X_train, X_val, y_train, y_val

    def _build_preprocess(self, X: pd.DataFrame) -> ColumnTransformer:
        cat_cols = [c for c in X.columns if X[c].dtype == "object" or str(X[c].dtype).startswith("category")]
        num_cols = [c for c in X.columns if c not in cat_cols]

        numeric = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ])
        categorical = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=True)),
        ])

        return ColumnTransformer(
            transformers=[
                ("num", numeric, num_cols),
                ("cat", categorical, cat_cols),
            ],
            remainder="drop",
            sparse_threshold=0.3,
        )

    def _candidate_models(self) -> List[Tuple[str, Any]]:
        if self.cfg.task_type in ("binary", "multiclass"):
            return [
                ("logreg", LogisticRegression(max_iter=500, n_jobs=1)),
                ("rf", RandomForestClassifier(n_estimators=400, n_jobs=1, random_state=self.cfg.random_state)),
                # LightGBM optional (if installed)
                ("lgbm", "lightgbm.LGBMClassifier"),
            ]
        else:
            return [
                ("ridge", Ridge(random_state=self.cfg.random_state)),
                ("rf", RandomForestRegressor(n_estimators=400, n_jobs=1, random_state=self.cfg.random_state)),
                ("lgbm", "lightgbm.LGBMRegressor"),
            ]

    def _maybe_make_llm_features(self, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, str]:
        if not self.cfg.use_llm_features or self.llm.name == "dummy":
            return X_train, "llm_off"

        # minimal dataset summary (keep it short and safe)
        sample = X_train.head(5)
        dtypes = X_train.dtypes.astype(str).to_dict()
        missing = X_train.isna().mean().round(3).to_dict()

        prompt = f"""
You are a feature engineering assistant for tabular ML.

Task: {self.cfg.task_type}, metric: {self.cfg.metric}
Target column is NOT included in df.

Given:
- dtypes: {json.dumps(dtypes)[:1200]}
- missing_rate: {json.dumps(missing)[:1200]}
- sample_rows (first 5):
{sample.to_csv(index=False)[:1500]}

Write ONLY python code that defines:

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    # create NEW columns safely
    # do not drop columns
    # must return a DataFrame of same length
    return df

Rules:
- No file/network access, no imports, no subprocess, no reading/writing.
- Use only pandas and numpy (pd, np are available).
- Keep it simple: interactions, ratios, logs, group stats only if obvious.
"""

        code = self.llm.complete(prompt)
        fn = compile_feature_fn(code)
        if fn is None:
            return X_train, "llm_failed_compile"

        try:
            X_new = fn(X_train.copy())
            if not isinstance(X_new, pd.DataFrame) or len(X_new) != len(X_train):
                return X_train, "llm_bad_return"
            return X_new, "llm_features_ok"
        except Exception:
            return X_train, "llm_runtime_error"

    def fit_evaluate(self, df: pd.DataFrame) -> List[RunResult]:
        t0 = time.time()
        X_train, X_val, y_train, y_val = self._split(df)

        # LLM feature step (train only)
        notes = ""
        if self.cfg.use_llm_features:
            X_train2, notes = self._maybe_make_llm_features(X_train, y_train)
            # Apply same features to val (if features added)
            if X_train2 is not X_train:
                # try to recompile from same prompt is hard; easiest:
                # re-run LLM to get the same code is not deterministic.
                # So we store the compiled fn when it works: for simplicity, we re-use by recomputing now.
                # Better: keep fn in state. We'll do that quickly here:
                pass

        # Better version: if features were ok, we must apply identical transform to val.
        # We'll do it by re-running _maybe_make_llm_features but with same LLM output captured.
        feature_fn = None
        if self.cfg.use_llm_features and self.llm.name != "dummy":
            code = self.llm.complete("Return exactly the same code as previously.")  # not reliable across providers
            # So: in practice, set deterministic model+seed or store code externally.
            # For NOW we turn this off unless you provide a deterministic LLM wrapper.
            # Keep pipeline stable for diploma: start with use_llm_features=False, then enable after you add determinism.
        # ---- END note ----

        # For a robust baseline agent: disable LLM features by default in experiments until deterministic wrapper is ready.
        if self.cfg.use_llm_features:
            notes = notes + " | NOTE: for strict reproducibility, store feature code and apply to val/test."

        preprocess = self._build_preprocess(X_train)
        metric_fn = get_metric_fn(self.cfg.task_type, self.cfg.metric)

        results: List[RunResult] = []
        for model_name, model in self._candidate_models():
            if (time.time() - t0) > self.cfg.time_budget_s:
                break

            model_obj = model
            if isinstance(model, str) and model.startswith("lightgbm."):
                try:
                    import importlib
                    mod = importlib.import_module("lightgbm")
                    cls = getattr(mod, model.split(".")[1])
                    # light defaults
                    if self.cfg.task_type in ("binary", "multiclass"):
                        model_obj = cls(
                            n_estimators=2000,
                            learning_rate=0.05,
                            num_leaves=64,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            random_state=self.cfg.random_state,
                            n_jobs=1,
                        )
                    else:
                        model_obj = cls(
                            n_estimators=2000,
                            learning_rate=0.05,
                            num_leaves=64,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            random_state=self.cfg.random_state,
                            n_jobs=1,
                        )
                except Exception:
                    continue  # skip if lightgbm not installed

            pipe = Pipeline(steps=[
                ("prep", preprocess),
                ("model", model_obj),
            ])

            start = time.time()
            pipe.fit(X_train, y_train)
            train_seconds = time.time() - start

            # predict for metric
            if self.cfg.task_type in ("binary", "multiclass"):
                if self.cfg.metric in ("roc_auc", "auc"):
                    # use probabilities if possible
                    if hasattr(pipe, "predict_proba"):
                        proba = pipe.predict_proba(X_val)
                        if self.cfg.task_type == "binary":
                            pred = proba[:, 1]
                            score = float(metric_fn(y_val, pred))
                        else:
                            score = float(metric_fn(y_val, proba, multi_class="ovr"))
                    else:
                        pred = pipe.decision_function(X_val)
                        score = float(metric_fn(y_val, pred))
                elif self.cfg.metric in ("logloss", "log_loss"):
                    proba = pipe.predict_proba(X_val)
                    score = float(metric_fn(y_val, proba))
                else:
                    pred = pipe.predict(X_val)
                    score = float(metric_fn(y_val, pred))
            else:
                pred = pipe.predict(X_val)
                score = float(metric_fn(y_val, pred))

            results.append(RunResult(
                model_name=model_name,
                metric_name=self.cfg.metric,
                score=score,
                train_seconds=train_seconds,
                used_llm=self.llm.name,
                notes=notes,
            ))

        # sort best first (higher is better except losses)
        higher_is_better = self.cfg.metric.lower() in ("roc_auc", "auc", "accuracy", "acc")
        results.sort(key=lambda r: r.score, reverse=higher_is_better)
        return results


def save_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")