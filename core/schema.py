import json
from pathlib import Path
from jsonschema import Draft7Validator

class SchemaValidator:
    def __init__(self, schemas_dir="schemas"):
        self.schemas_dir = Path(schemas_dir)
        self.validators = {}
        for key, fn in [("eda", "eda.schema.json"), ("task", "task.schema.json")]:
            schema = json.loads((self.schemas_dir / fn).read_text(encoding="utf-8"))
            self.validators[key] = Draft7Validator(schema)

    def validate(self, key: str, obj):
        errs = [e.message for e in self.validators[key].iter_errors(obj)]
        return (len(errs) == 0), errs
