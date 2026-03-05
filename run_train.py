import argparse
import pandas as pd

from automl.config import AgentConfig
from automl.llm_client import DummyLLM, OpenAILLM
from automl.agent import LLMAutoMLAgent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--task", default=None, choices=[None, "binary", "multiclass", "regression"])
    p.add_argument("--metric", default=None)  # e.g. roc_auc, accuracy, rmse, logloss, mae, r2
    p.add_argument("--time_budget_s", type=int, default=1200)
    p.add_argument("--output_dir", default="runs/exp1")
    p.add_argument("--use_llm", action="store_true")
    args = p.parse_args()

    df = pd.read_csv(args.train)

    cfg = AgentConfig(
        target=args.target,
        task=args.task,
        metric=args.metric,
        time_budget_s=args.time_budget_s,
        output_dir=args.output_dir,
        use_llm=args.use_llm,
    )

    if args.use_llm:
        llm = OpenAILLM(model="gpt-4o")   # можно заменить на "gpt-4.1" или другое
    else:
        llm = DummyLLM()
    agent = LLMAutoMLAgent(cfg, llm=llm)
    agent.fit(df)

    print("DONE.")
    print("Best model:", f"{cfg.output_dir}/best_model.pkl")
    print("Best meta :", f"{cfg.output_dir}/best_meta.json")
    print("All runs  :", f"{cfg.output_dir}/runs.jsonl")


if __name__ == "__main__":
    main()