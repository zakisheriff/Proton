"""Execute generated code against tests, safely-ish, for RL rewards and eval.

Runs code in a subprocess with a wall-clock timeout in a temp dir. This is a
*basic* sandbox: fine for trusted local eval. For untrusted/production RL at
scale, run inside a container / gVisor / firecracker with no network and strict
resource limits — this module's interface stays the same.
"""

import os
import subprocess
import tempfile

TIMEOUT = 10  # seconds


def run_python(code: str, test: str) -> bool:
    """True if `code` + `test` runs without error (test uses asserts)."""
    return _run(code + "\n\n" + test, "prog.py", ["python3", "prog.py"])


def run_node(code: str, test: str) -> bool:
    """True if JS `code` + `test` runs without error under Node."""
    return _run(code + "\n\n" + test, "prog.mjs", ["node", "prog.mjs"])


def _run(source: str, filename: str, cmd: list[str]) -> bool:
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(source)
        try:
            proc = subprocess.run(
                cmd, cwd=d, capture_output=True, text=True, timeout=TIMEOUT,
                env={**os.environ, "NODE_OPTIONS": ""},
            )
            return proc.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return False


RUNNERS = {"python": run_python, "javascript": run_node, "js": run_node}


def check(code: str, test: str, language: str) -> bool:
    runner = RUNNERS.get(language.lower())
    if runner is None:
        raise ValueError(f"no sandbox runner for language {language!r}")
    return runner(code, test)
