"""Audit original changed-command file selection using the cheap UFMT tool only."""
import pathlib
import subprocess
repo = pathlib.Path('/workspace/TICO')
paths = subprocess.check_output(['git', 'diff', '--name-only', '--diff-filter=ACMR', 'bb09ca852b0ca1b1a6271cfe1e10c162104c997e', 'HEAD', '--', '*.py', '*.pyi'], cwd=repo, text=True).splitlines()
print('Original intended paths:', len(paths), flush=True)
raise SystemExit(subprocess.call(['lintrunner', '-vv', '--take', 'UFMT', '--config', str(repo / '.lintrunner.toml'), *paths], cwd=repo))
