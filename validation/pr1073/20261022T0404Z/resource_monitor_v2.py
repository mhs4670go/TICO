"""Run a validation command while recording bounded CPU memory and disk evidence."""
import argparse
import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import psutil
p=argparse.ArgumentParser();p.add_argument('--report',required=True);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
cmd=a.command[1:] if a.command[:1]==['--'] else a.command
report={'command':cmd,'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'disk_before':dict(zip(['total','used','free'],shutil.disk_usage('/workspace'))),'samples':[]}
start=time.monotonic();proc=subprocess.Popen(cmd);root=psutil.Process(proc.pid)
while proc.poll() is None:
    try:
        processes=[root]+root.children(recursive=True)
        rss=sum(item.memory_info().rss for item in processes if item.is_running())
        available=psutil.virtual_memory().available
        report['samples'].append({'elapsed_seconds':round(time.monotonic()-start,2),'process_tree_rss_bytes':rss,'available_memory_bytes':available,'free_disk_bytes':shutil.disk_usage('/workspace').free})
        if len(report['samples'])%10==0:
            report['status']='RUNNING'
            report['max_process_tree_rss_bytes']=max(x['process_tree_rss_bytes'] for x in report['samples'])
            report['minimum_available_memory_bytes']=min(x['available_memory_bytes'] for x in report['samples'])
            Path(a.report).write_text(json.dumps(report,indent=2)+'\n')
    except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    time.sleep(1)
report.update(status='COMPLETE',exit_code=proc.returncode,elapsed_seconds=time.monotonic()-start,disk_after=dict(zip(['total','used','free'],shutil.disk_usage('/workspace'))))
report['max_process_tree_rss_bytes']=max((x['process_tree_rss_bytes'] for x in report['samples']),default=0)
report['minimum_available_memory_bytes']=min((x['available_memory_bytes'] for x in report['samples']),default=None)
Path(a.report).write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='samples'}));sys.exit(proc.returncode)
