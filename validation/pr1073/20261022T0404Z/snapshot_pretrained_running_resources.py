"""Persist a live resource observation before a bounded validation may time out."""
import datetime
import json
from pathlib import Path
import shutil
import time
import psutil
ROOT=Path(__file__).resolve().parent
report={'observed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Live partial observation only, not a complete peak measurement','available_memory_bytes':psutil.virtual_memory().available,'disk':dict(zip(['total','used','free'],shutil.disk_usage('/workspace'))),'workers':[]}
for process in psutil.process_iter(['pid','cmdline','create_time','memory_info','cpu_times']):
    try:
        args=process.info['cmdline'] or []
        if '_worker' in args and any('pretrained_tinyllama_quantized_cached.pt' in arg for arg in args):
            report['workers'].append({'pid':process.info['pid'],'elapsed_seconds':time.time()-process.info['create_time'],'rss_bytes':process.info['memory_info'].rss,'cpu_seconds':process.info['cpu_times'].user+process.info['cpu_times'].system})
    except (psutil.NoSuchProcess,psutil.AccessDenied):pass
(ROOT/'pretrained_cached_circle_live_resource_snapshot.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
