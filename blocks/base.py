from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class BlockResult:
    ok: bool
    output: Dict[str, Any]
    raw_model_output: Optional[str]
    artifacts: Dict[str, Any]

class Block:
    name: str = "block"

    def run(self, context: Dict[str, Any]) -> BlockResult:
        raise NotImplementedError
