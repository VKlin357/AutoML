from typing import Dict, Any, List, Tuple

def eval_eda(eda: Dict[str, Any]) -> Dict[str, Any]:
    issues: List[str] = []
    score = 1.0

    for key in ["columns", "missing", "basic_stats"]:
        if key not in eda:
            issues.append(f"missing_key:{key}")
            score -= 0.3

    n_rows = eda.get("basic_stats", {}).get("n_rows")
    n_cols = eda.get("basic_stats", {}).get("n_cols")
    if isinstance(n_rows, int) and n_rows <= 0:
        issues.append("n_rows_nonpositive")
        score -= 0.2
    if isinstance(n_cols, int) and n_cols <= 0:
        issues.append("n_cols_nonpositive")
        score -= 0.2

    cols = eda.get("columns", [])
    if not cols or len(cols) < 2:
        issues.append("too_few_columns_described")
        score -= 0.2

    score = max(0.0, min(1.0, score))
    return {"score": score, "issues": issues}
