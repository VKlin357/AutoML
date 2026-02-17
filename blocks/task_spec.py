import json
from typing import Dict, Any
from blocks.base import Block, BlockResult

class TaskSpecBlock(Block):
    name = "task_spec"

    def __init__(self, llm, storage, schema_validator, prompt_templates):
        self.llm = llm
        self.storage = storage
        self.schema_validator = schema_validator
        self.prompt_templates = prompt_templates

    def run(self, context: Dict[str, Any]) -> BlockResult:
        eda = context["eda"]
        prompt = self.prompt_templates["task_block"].format(eda_json=json.dumps(eda, ensure_ascii=False))

        raw = self.llm.call(agent="PLANNER", prompt=prompt)

        try:
            out = json.loads(raw)
        except Exception:
            return BlockResult(False, {}, raw, {"reason": "not_json"})

        valid, errors = self.schema_validator.validate("task", out)
        ok = valid

        self.storage.save_json("task.json", out)
        self.storage.save_text("task_raw.txt", raw)

        return BlockResult(ok, out, raw, {"schema_errors": errors})
