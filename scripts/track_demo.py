#!/usr/bin/env python3
"""End-to-end proof that experiment tracking works.

Trains a small model on synthetic data and logs params, a training curve, and an
artifact through the unified Tracker. Runs fully offline with MLflow:

    python scripts/track_demo.py --track mlflow      # -> ./mlruns + tracking/*.jsonl
    python scripts/track_demo.py --track wandb       # needs WANDB_API_KEY
    python scripts/track_demo.py --track none        # local JSONL only
"""
import argparse
import json
import os
import sys
from pathlib import Path

from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.tracking import Tracker


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="none", help="none|mlflow|wandb|clearml|tensorboard")
    ap.add_argument("--project", default="llm-tabular-nas-demo")
    ap.add_argument("--n_estimators", type=int, default=120)
    ap.add_argument("--max_depth", type=int, default=3)
    args = ap.parse_args()

    X, y = make_classification(n_samples=2000, n_features=20, n_informative=10, random_state=42)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    params = {"model": "gbm", "n_estimators": args.n_estimators, "max_depth": args.max_depth, "seed": 42}

    with Tracker(backend=args.track, project=args.project, run_name="gbm_demo", config=params) as tr:
        clf = GradientBoostingClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=42)
        clf.fit(Xtr, ytr)
        # log a real training curve using staged predictions
        for i, proba in enumerate(clf.staged_predict_proba(Xte)):
            if i % 10 == 0 or i == args.n_estimators - 1:
                tr.log_metrics({"val_acc": accuracy_score(yte, proba.argmax(1)),
                                "val_logloss": log_loss(yte, proba)}, step=i)
        final_acc = float(accuracy_score(yte, clf.predict(Xte)))
        tr.log_metrics({"final_acc": final_acc})

        out = Path("outputs"); out.mkdir(exist_ok=True)
        art = out / "demo_report.json"
        art.write_text(json.dumps({"final_acc": final_acc, **params}, indent=2))
        tr.log_artifact(str(art))
        print(f"[track_demo] backend={tr.backend}  final_acc={final_acc:.4f}  "
              f"jsonl=tracking/{tr.run_name}.jsonl")


if __name__ == "__main__":
    main()
