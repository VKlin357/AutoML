from typing import Dict, Any, List

def eval_task(task: Dict[str, Any], eda: Dict[str, Any]) -> Dict[str, Any]:
    issues: List[str] = []
    score = 1.0

    target = task.get("target")
    colnames = {c["name"] for c in eda.get("columns", []) if "name" in c}

    if target not in colnames:
        issues.append("target_not_in_columns")
        score -= 0.6

    task_type = task.get("task_type")
    if task_type == "timeseries" and task.get("split", {}).get("type") != "time":
        issues.append("timeseries_without_time_split")
        score -= 0.3

    score = max(0.0, min(1.0, score))
    return {"score": score, "issues": issues}
