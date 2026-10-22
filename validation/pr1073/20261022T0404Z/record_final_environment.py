"""Record final checkout integrity and non-secret resource information."""
import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

root = Path(__file__).resolve().parent
def capture(args, cwd):
    result = subprocess.run(args, cwd=cwd, text=True, capture_output=True, check=False)
    return {'command': args, 'cwd': cwd, 'exit_code': result.returncode,
            'stdout': result.stdout, 'stderr': result.stderr}

data = {'recorded_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'python': sys.version, 'executable': sys.executable,
        'disk_bytes': dict(zip(('total', 'used', 'free'), shutil.disk_usage('/workspace'))),
        'cpu_visible': os.cpu_count(), 'memory_info': Path('/proc/meminfo').read_text(),
        'cpu_quota': Path('/sys/fs/cgroup/cpu.max').read_text().strip(),
        'memory_limit': Path('/sys/fs/cgroup/memory.max').read_text().strip(),
        'repositories': {}, 'packages': {}}
for name, cwd, expected in [
    ('candidate','/workspace/TICO','6b2f6a0511458d6a750e2a260d706080a78149dd'),
    ('baseline','/workspace/TICO-export-baseline','bb09ca852b0ca1b1a6271cfe1e10c162104c997e')]:
    head = capture(['git','rev-parse','HEAD'], cwd)
    status = capture(['git','status','--porcelain=v1','--untracked-files=normal'], cwd)
    diff = capture(['git','diff','--exit-code'], cwd)
    cached = capture(['git','diff','--cached','--exit-code'], cwd)
    assert head['stdout'].strip() == expected
    assert status['exit_code'] == 0 and status['stdout'] == ''
    assert diff['exit_code'] == 0 and cached['exit_code'] == 0
    data['repositories'][name] = {'head':head,'status':status,'diff':diff,'cached_diff':cached}
for name in ['torch','torchvision','transformers','circle-schema','datasets','numpy','accelerate','lm-eval','lmms-eval']:
    try:
        data['packages'][name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        data['packages'][name] = None
import tico
import torch
data['tico_file'] = tico.__file__
data['cuda_available'] = torch.cuda.is_available()
assert data['tico_file'] == '/workspace/TICO/tico/__init__.py'
data['dependency_check'] = capture([sys.executable,'-m','pip','check'], '/workspace/TICO')
assert data['dependency_check']['exit_code'] == 0
(root/'final_repository_environment.json').write_text(json.dumps(data,indent=2)+'\n')
print(json.dumps({'candidate':data['repositories']['candidate']['head']['stdout'].strip(),
                  'baseline':data['repositories']['baseline']['head']['stdout'].strip(),
                  'both_worktrees_clean':True, 'pip_check':'PASS',
                  'disk_free_bytes':data['disk_bytes']['free']}))
