import json
from pathlib import Path

class SimpleRunStorage:
    def __init__(self, run_dir: str):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def save_text(self, name: str, text: str):
        (self.run_dir / name).write_text(text or "", encoding="utf-8")

    def save_json(self, name: str, obj):
        (self.run_dir / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
