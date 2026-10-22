"""Run PR-modified tests through the repository's standard discovery layout."""
import json
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent
modules = json.loads((root / "changed_test_modules.json").read_text())
command = [sys.executable, "-m", "unittest", "discover", "-s", "test", "-v"]
for module in modules:
    command.extend(["-k", module.removeprefix("test.")])
print("Focused discovery:", command, flush=True)
raise SystemExit(subprocess.call(command, cwd="/workspace/TICO"))
