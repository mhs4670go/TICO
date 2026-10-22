"""Check original counterparts of candidate UFMT findings without source edits."""
import json
import pathlib
import subprocess

root = pathlib.Path('/workspace/TICO/out/pr1073_validation/20261022T0404Z')
repo = pathlib.Path('/workspace/TICO-export-baseline')
findings = json.loads((root / 'ufmt_findings.json').read_text())
paths = [item['path'] for item in findings if item['exists_in_baseline']]
print('Baseline existing counterparts:', len(paths), flush=True)
raise SystemExit(subprocess.call(['lintrunner', '--take', 'UFMT', '--config', str(repo / '.lintrunner.toml'), *paths], cwd=repo))
