from dataclasses import dataclass
from typing import Optional


@dataclass
class AgentConfig:
    target: str
    task: Optional[str] = None            # "binary" | "multiclass" | "regression" | None(auto)
    metric: Optional[str] = None          # None -> default per task

    time_budget_s: int = 20 * 60
    max_rounds: int = 25
    n_candidates_per_round: int = 4

    cv_folds: int = 5
    seed: int = 42

    output_dir: str = "runs/exp"
    use_llm: bool = False

    # guardrails
    max_onehot_categories: int = 200
    max_rows_for_cv: int = 200_000   # above -> holdout validation
    holdout_size: float = 0.2