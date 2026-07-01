"""Unified experiment tracking for LLM-NAS.

One small interface, four interchangeable backends — pick with `--track` or the
`LLMNAS_TRACKER` env var:

    none      → local JSONL only (default, zero setup)
    mlflow    → MLflow (works fully offline against a local ./mlruns store)
    wandb     → Weights & Biases (needs WANDB_API_KEY)
    clearml   → ClearML (needs clearml.conf / env creds)

Every backend also always writes a local `tracking/<run>.jsonl`, so there is a
real, inspectable record even with no account. Missing packages degrade
gracefully to `none` instead of crashing a training run.

Usage
-----
    from src.tracking import Tracker
    tr = Tracker(backend="mlflow", project="llm-nas", run_name="jannis_llm",
                 config={"budget": 40, "seed": 42})
    tr.log_params({"family": "ft_transformer"})
    tr.log_metrics({"val_acc": 0.79}, step=1)
    tr.log_artifact("outputs/jannis/final_report.json")
    tr.finish()
"""
from __future__ import annotations

import json
import os
import time
import warnings
from pathlib import Path
from typing import Any

_BACKENDS = ("none", "mlflow", "wandb", "clearml", "tensorboard")


class Tracker:
    def __init__(
        self,
        backend: str | None = None,
        project: str = "llm-tabular-nas",
        run_name: str | None = None,
        config: dict[str, Any] | None = None,
        out_dir: str | Path = "tracking",
    ):
        backend = (backend or os.environ.get("LLMNAS_TRACKER", "none")).lower()
        if backend not in _BACKENDS:
            warnings.warn(f"Unknown tracker '{backend}', using 'none'.")
            backend = "none"
        self.backend = backend
        self.project = project
        self.run_name = run_name or f"run_{int(time.time())}"
        self.config = dict(config or {})
        self._impl = None
        self._tb = None

        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._jsonl = (self.out_dir / f"{self.run_name}.jsonl").open("a")
        self._emit({"event": "init", "backend": backend, "project": project,
                    "run_name": self.run_name, "config": self.config})

        self.backend = self._init_backend(backend)  # may downgrade to 'none'

    # ---- backend bring-up (all failures degrade to 'none') ------------------
    def _init_backend(self, backend: str) -> str:
        try:
            if backend == "mlflow":
                import mlflow
                mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns"))
                mlflow.set_experiment(self.project)
                self._impl = mlflow.start_run(run_name=self.run_name)
                if self.config:
                    mlflow.log_params(_flatten(self.config))
                return "mlflow"
            if backend == "wandb":
                import wandb
                self._impl = wandb.init(project=self.project, name=self.run_name, config=self.config)
                return "wandb"
            if backend == "clearml":
                from clearml import Task
                self._impl = Task.init(project_name=self.project, task_name=self.run_name)
                if self.config:
                    self._impl.connect(self.config)
                return "clearml"
            if backend == "tensorboard":
                from torch.utils.tensorboard import SummaryWriter
                self._tb = SummaryWriter(log_dir=str(self.out_dir / "tb" / self.run_name))
                return "tensorboard"
        except Exception as e:  # missing dep / no creds → local only
            warnings.warn(f"Tracker backend '{backend}' unavailable ({e}); using local JSONL only.")
        return "none"

    # ---- public API ---------------------------------------------------------
    def log_params(self, params: dict[str, Any]) -> None:
        self.config.update(params)
        self._emit({"event": "params", **_flatten(params)})
        try:
            if self.backend == "mlflow":
                import mlflow; mlflow.log_params(_flatten(params))
            elif self.backend == "wandb":
                self._impl.config.update(params, allow_val_change=True)
            elif self.backend == "clearml":
                self._impl.connect(params)
        except Exception as e:
            warnings.warn(f"log_params failed: {e}")

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        clean = {k: float(v) for k, v in metrics.items() if _is_num(v)}
        self._emit({"event": "metrics", "step": step, **clean})
        try:
            if self.backend == "mlflow":
                import mlflow; mlflow.log_metrics(clean, step=step)
            elif self.backend == "wandb":
                self._impl.log(clean, step=step)
            elif self.backend == "clearml":
                logger = self._impl.get_logger()
                for k, v in clean.items():
                    logger.report_scalar(k, "value", v, iteration=step or 0)
            elif self.backend == "tensorboard" and self._tb is not None:
                for k, v in clean.items():
                    self._tb.add_scalar(k, v, step or 0)
        except Exception as e:
            warnings.warn(f"log_metrics failed: {e}")

    def log_artifact(self, path: str | Path) -> None:
        path = str(path)
        self._emit({"event": "artifact", "path": path})
        try:
            if self.backend == "mlflow":
                import mlflow; mlflow.log_artifact(path)
            elif self.backend == "wandb":
                self._impl.save(path)
            elif self.backend == "clearml":
                self._impl.upload_artifact(Path(path).name, artifact_object=path)
        except Exception as e:
            warnings.warn(f"log_artifact failed: {e}")

    def finish(self) -> None:
        self._emit({"event": "finish"})
        try:
            if self.backend == "mlflow":
                import mlflow; mlflow.end_run()
            elif self.backend == "wandb":
                self._impl.finish()
            elif self.backend == "tensorboard" and self._tb is not None:
                self._tb.close()
        except Exception as e:
            warnings.warn(f"finish failed: {e}")
        finally:
            self._jsonl.close()

    # ---- context-manager sugar ---------------------------------------------
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.finish()
        return False

    def _emit(self, record: dict) -> None:
        record["t"] = time.time()
        self._jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._jsonl.flush()


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out
