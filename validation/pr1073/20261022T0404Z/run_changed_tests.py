"""Run all PR-modified test modules and the real runtime-independence contract."""
import json
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent
modules = json.loads((root / "changed_test_modules.json").read_text())
raise SystemExit(subprocess.call([sys.executable, "-m", "unittest", "-v", *modules], cwd="/workspace/TICO"))
