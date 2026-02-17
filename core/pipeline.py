from typing import Dict, Any, List
from evals.eda_eval import eval_eda
from evals.task_eval import eval_task

class Pipeline:
    def __init__(self, storage, blocks: List):
        self.storage = storage
        self.blocks = blocks

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        results: Dict[str, Any] = {"blocks": {}, "evals": {}}

        eda_res = self.blocks[0].run(context)
        results["blocks"]["eda"] = {"ok": eda_res.ok, "output": eda_res.output, "artifacts": eda_res.artifacts}
        context["eda"] = eda_res.output

        results["evals"]["eda"] = eval_eda(eda_res.output)

        task_res = self.blocks[1].run(context)
        results["blocks"]["task"] = {"ok": task_res.ok, "output": task_res.output, "artifacts": task_res.artifacts}
        context["task"] = task_res.output

        results["evals"]["task"] = eval_task(task_res.output, context["eda"])
        self.storage.save_json("evaluations.json", results)
        return results
