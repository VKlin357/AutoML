import argparse
import yaml
from pathlib import Path
from datetime import datetime

from core.storage import SimpleRunStorage
from core.schema import SchemaValidator
from core.exec import Executor
from core.llm import LLM
from core.pipeline import Pipeline

from blocks.eda import EDABlock
from blocks.task_spec import TaskSpecBlock

from provider import GeminiProvider, OllamaProvider, OpenAIProvider

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-files", nargs="+", required=True)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text()) or {}
    run_dir = args.run_dir or f"runs/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    storage = SimpleRunStorage(run_dir)
    schema = SchemaValidator()

    prompts = yaml.safe_load(Path("prompt.yaml").read_text())

    model_name = cfg["model_name"]
    api_key = cfg.get("api_key")

    def make_provider(m):
        for cls in [OllamaProvider, OpenAIProvider, GeminiProvider]:
            if cls.provider_instance(m):
                return cls(api_key, m)
        raise ValueError(f"No provider for {m}")

    providers = {
        "ANALYZER": make_provider(cfg.get("analyzer_model", model_name)),
        "PLANNER": make_provider(cfg.get("planner_model", model_name)),
    }

    llm = LLM(providers)
    executor = Executor(exec_dir=f"{run_dir}/exec_env", timeout=cfg.get("execution_timeout", 60))

    blocks = [
        EDABlock(llm, executor, storage, schema, prompts),
        TaskSpecBlock(llm, storage, schema, prompts),
    ]

    pipe = Pipeline(storage, blocks)
    context = {"data_files": args.data_files}
    result = pipe.run(context)
    print("DONE. Saved to", run_dir)

if __name__ == "__main__":
    main()
