"""Run unchanged repository lintrunner config on PR-changed Python source files."""
import pathlib
import subprocess
import sys

repo = pathlib.Path('/workspace/TICO')
paths = subprocess.check_output(['git', 'diff', '--name-only', '--diff-filter=ACMR', 'bb09ca852b0ca1b1a6271cfe1e10c162104c997e', 'HEAD', '--', '*.py', '*.pyi'], cwd=repo, text=True).splitlines()
print('Changed Python files:', len(paths), flush=True)
command = ['lintrunner', '--config', str(repo / '.lintrunner.toml'), *paths]
raise SystemExit(subprocess.call(command, cwd=repo))
