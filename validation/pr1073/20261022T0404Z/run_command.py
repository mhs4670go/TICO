"""Capture reproducible commands, output, timing, and exit status."""
import argparse, datetime, json, os, pathlib, shlex, subprocess, sys, time
p=argparse.ArgumentParser(); p.add_argument('--name',required=True); p.add_argument('--cwd',default='/workspace/TICO'); p.add_argument('--timeout',type=int,default=1800); p.add_argument('command',nargs=argparse.REMAINDER); a=p.parse_args()
root=pathlib.Path(__file__).resolve().parent
cmd=a.command[1:] if a.command[:1]==['--'] else a.command
record={'name':a.name,'cwd':a.cwd,'command':cmd,'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'environment':{k:os.environ.get(k) for k in ['VIRTUAL_ENV','PATH','OMP_NUM_THREADS','MKL_NUM_THREADS','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','RESULT_ROOT']}}
with (root/'commands.sh').open('a') as f: f.write('\n# '+a.name+'\n(cd '+shlex.quote(a.cwd)+' && '+shlex.join(cmd)+')\n')
start=time.monotonic()
with (root/(a.name+'.log')).open('w') as f:
 try:
  proc=subprocess.Popen(cmd,cwd=a.cwd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  try: record['exit_code']=proc.wait(timeout=a.timeout)
  except subprocess.TimeoutExpired:
   import signal
   os.killpg(proc.pid,signal.SIGTERM); proc.wait(); record['exit_code']=124; record['timeout']=True
 except Exception as e: record.update(exit_code=127,error=repr(e))
record['elapsed_seconds']=time.monotonic()-start
(root/(a.name+'.command.json')).write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({k:v for k,v in record.items() if k!='environment'})); sys.exit(record['exit_code'])
