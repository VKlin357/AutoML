import json
from pathlib import Path
from typing import Dict, Any
from blocks.base import Block, BlockResult

class EDABlock(Block):
    name = "eda"

    def __init__(self, llm, executor, storage, schema_validator, prompt_templates):
        self.llm = llm
        self.executor = executor
        self.storage = storage
        self.schema_validator = schema_validator
        self.prompt_templates = prompt_templates

    def run(self, context: Dict[str, Any]) -> BlockResult:
        data_path = context["data_files"][0]
        prompt = self.prompt_templates["eda_block"].format(filename=data_path)

        raw = self.llm.call(agent="ANALYZER", prompt=prompt)
        code = self.llm.extract_code(raw)

        stdout = self.executor.execute_and_debug(code, data_files=[data_path], data_desc="EDA block")

        try:
            out = json.loads(stdout)
        except Exception:
            return BlockResult(
                ok=False,
                output={},
                raw_model_output=raw,
                artifacts={"stdout": stdout, "code": code}
            )

        valid, errors = self.schema_validator.validate("eda", out)
        ok = valid

        self.storage.save_json("eda.json", out)
        self.storage.save_text("eda_code.py", code)
        self.storage.save_text("eda_raw.txt", raw)
        self.storage.save_text("eda_stdout.txt", stdout)

        return BlockResult(
            ok=ok,
            output=out,
            raw_model_output=raw,
            artifacts={"schema_errors": errors}
        )
