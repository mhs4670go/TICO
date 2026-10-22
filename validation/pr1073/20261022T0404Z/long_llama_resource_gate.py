"""Require less than 12 GiB cgroup RAM use before the long eager stress."""
import json
import shutil
import sys
from pathlib import Path
root = Path(__file__).resolve().parent
current = int(Path('/sys/fs/cgroup/memory.current').read_text())
maximum = Path('/sys/fs/cgroup/memory.max').read_text().strip()
result = {'cgroup_current_bytes': current, 'cgroup_max_bytes': maximum, 'threshold_bytes': 12 * 1024**3, 'workspace_free_bytes': shutil.disk_usage('/workspace').free, 'status': 'READY' if current < 12 * 1024**3 else 'DEFERRED'}
(root / 'long_llama_resource_gate.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
sys.exit(0 if result['status'] == 'READY' else 75)
