import sys, uuid, subprocess
from pathlib import Path

class Executor:
    def __init__(self, exec_dir: str, timeout: int = 60):
        self.exec_dir = Path(exec_dir)
        self.exec_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout

    def execute(self, code: str, data_files=None):
        exec_id = uuid.uuid4().hex[:8]
        path = self.exec_dir / f"exec_{exec_id}.py"
        path.write_text(code, encoding="utf-8")

        try:
            r = subprocess.run([sys.executable, str(path)], capture_output=True, text=True, timeout=self.timeout)
            if r.returncode == 0:
                return r.stdout, None
            return "", (r.stderr or "unknown error")
        except subprocess.TimeoutExpired:
            return "", f"timeout after {self.timeout}s"

    def execute_and_debug(self, code: str, data_files=None, data_desc=""):
        out, err = self.execute(code, data_files)
        if err:
            raise RuntimeError(err)
        return out
