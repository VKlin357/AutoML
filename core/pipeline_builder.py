from __future__ import annotations

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor


class TargetEncoder:
    """Simple target encoding with smoothing for binary/regression."""
    def __init__(self, cols: List[str], smoothing: float = 10.0):
        self.cols = cols
        self.smoothing = smoothing
        self.global_mean_: float | None = None
        self.mapping_: dict[str, dict[Any, float]] = {}

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        self.global_mean_ = float(np.mean(y))
        for c in self.cols:
            stats = pd.DataFrame({"x": X[c], "y": y}).groupby("x")["y"].agg(["mean", "count"])
            enc = (stats["count"] * stats["mean"] + self.smoothing * self.global_mean_) / (stats["count"] + self.smoothing)
            self.mapping_[c] = enc.to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.global_mean_ is None:
            raise RuntimeError("TargetEncoder not fitted")
        X = X.copy()
        for c in self.cols:
            mp = self.mapping_.get(c, {})
            X[c] = X[c].map(mp).fillna(self.global_mean_)
        return X


def build_estimator(task: str, model_cfg: Dict[str, Any], seed: int):
    name = model_cfg["name"]
    params = dict(model_cfg.get("params", {}))

    if task in ("binary", "multiclass"):
        if name == "logreg":
            return LogisticRegression(**params, solver="lbfgs", max_iter=params.get("max_iter", 2000), class_weight="balanced")
        if name == "rf":
            return RandomForestClassifier(random_state=seed, n_jobs=-1, **params)
        if name == "et":
            return ExtraTreesClassifier(random_state=seed, n_jobs=-1, **params)
        if name == "hgb":
            return HistGradientBoostingClassifier(random_state=seed, **params)
        raise ValueError(f"Unknown model for classification: {name}")

    # regression
    if name == "ridge":
        return Ridge(**params)
    if name == "rf":
        return RandomForestRegressor(random_state=seed, n_jobs=-1, **params)
    if name == "et":
        return ExtraTreesRegressor(random_state=seed, n_jobs=-1, **params)
    if name == "hgb":
        return HistGradientBoostingRegressor(random_state=seed, **params)
    raise ValueError(f"Unknown model for regression: {name}")


def build_pipeline(
    df: pd.DataFrame,
    target: str,
    task: str,
    cand: Dict[str, Any],
    seed: int,
    max_onehot_categories: int
) -> Tuple[Any, List[str]]:
    """
    Returns (pipeline_like_object, used_feature_columns)
    pipeline_like_object must support .fit(Xdf, y), .predict(Xdf) and optionally .predict_proba(Xdf)
    """
    X = df.drop(columns=[target]).copy()

    drop_cols = [c for c in cand.get("drop_cols", []) if c in X.columns]
    X = X.drop(columns=drop_cols, errors="ignore")

    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]

    cat_encoding = cand.get("cat_encoding", "onehot")
    num_scaling = bool(cand.get("num_scaling", False))

    estimator = build_estimator(task, cand["model"], seed)

    # numeric pipeline
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if num_scaling:
        num_steps.append(("scaler", StandardScaler(with_mean=True)))
    num_tf = Pipeline(steps=num_steps)

    if cat_encoding == "onehot":
        cat_tf = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", max_categories=max_onehot_categories)),
        ])

        pre = ColumnTransformer(
            transformers=[
                ("num", num_tf, num_cols),
                ("cat", cat_tf, cat_cols),
            ],
            remainder="drop",
            sparse_threshold=0.3,
        )

        pipe = Pipeline(steps=[("pre", pre), ("model", estimator)])
        return pipe, list(X.columns)

    if cat_encoding == "target":
        if task == "multiclass":
            raise ValueError("Target encoding disabled for multiclass")

        smoothing = float(cand.get("target_smoothing", 10.0))

        class ManualTEPipeline:
            def __init__(self):
                self.te = TargetEncoder(cols=cat_cols, smoothing=smoothing)
                self.num_imputer = SimpleImputer(strategy="median")
                self.scaler = StandardScaler(with_mean=True) if (num_scaling and len(num_cols) > 0) else None
                self.model = estimator

            def fit(self, Xdf: pd.DataFrame, y: np.ndarray):
                Xdf = Xdf.copy()

                # cats: fill missing marker
                for c in cat_cols:
                    Xdf[c] = Xdf[c].astype("object").where(~Xdf[c].isna(), "__MISSING__")
                if cat_cols:
                    self.te.fit(Xdf[cat_cols], y)

                # nums
                Xnum = Xdf[num_cols]
                if len(num_cols) > 0:
                    Xnum = self.num_imputer.fit_transform(Xnum)
                    if self.scaler is not None:
                        Xnum = self.scaler.fit_transform(Xnum)
                else:
                    Xnum = np.zeros((len(Xdf), 0), dtype=float)

                # cats -> encoded
                if cat_cols:
                    Xcat = self.te.transform(Xdf[cat_cols]).to_numpy(dtype=float)
                else:
                    Xcat = np.zeros((len(Xdf), 0), dtype=float)

                Xmat = np.hstack([Xnum, Xcat])
                self.model.fit(Xmat, y)
                return self

            def predict(self, Xdf: pd.DataFrame):
                Xmat = self._transform(Xdf)
                return self.model.predict(Xmat)

            def predict_proba(self, Xdf: pd.DataFrame):
                Xmat = self._transform(Xdf)
                if hasattr(self.model, "predict_proba"):
                    return self.model.predict_proba(Xmat)
                raise AttributeError("Model has no predict_proba")

            def _transform(self, Xdf: pd.DataFrame):
                Xdf = Xdf.copy()
                for c in cat_cols:
                    Xdf[c] = Xdf[c].astype("object").where(~Xdf[c].isna(), "__MISSING__")

                # nums
                if len(num_cols) > 0:
                    Xnum = self.num_imputer.transform(Xdf[num_cols])
                    if self.scaler is not None:
                        Xnum = self.scaler.transform(Xnum)
                else:
                    Xnum = np.zeros((len(Xdf), 0), dtype=float)

                # cats
                if cat_cols:
                    Xcat = self.te.transform(Xdf[cat_cols]).to_numpy(dtype=float)
                else:
                    Xcat = np.zeros((len(Xdf), 0), dtype=float)

                return np.hstack([Xnum, Xcat])

        return ManualTEPipeline(), list(X.columns)

    raise ValueError(f"Unknown cat_encoding: {cat_encoding}")