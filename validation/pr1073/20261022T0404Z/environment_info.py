"""Record non-secret runtime and package provenance."""
import importlib, importlib.metadata, json, os, pathlib, platform, shutil, subprocess, sys
root=pathlib.Path(__file__).resolve().parent
info={'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'cpu_count':os.cpu_count(),'disk':dict(zip(['total','used','free'],shutil.disk_usage('/workspace'))),'packages':{},'imports':{},'cache_configuration':{k:os.environ.get(k) for k in ['HF_HOME','HF_HUB_CACHE','TRANSFORMERS_CACHE']},'head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'baseline':'bb09ca852b0ca1b1a6271cfe1e10c162104c997e'}
for package in ['tico','torch','torchvision','transformers','circle-schema','datasets','numpy','accelerate','sentencepiece','lm-eval','lmms-eval','lintrunner','mypy','pylint']:
 try: info['packages'][package]=importlib.metadata.version(package)
 except importlib.metadata.PackageNotFoundError: info['packages'][package]=None
for name in ['tico','torch','transformers']:
 try:
  mod=importlib.import_module(name);info['imports'][name]=mod.__file__
 except Exception as exc: info['imports'][name]={'error':repr(exc)}
import torch
info['torch']={'cuda_available':torch.cuda.is_available(),'cuda_version':torch.version.cuda,'threads':torch.get_num_threads()}
info['memory_info']=pathlib.Path('/proc/meminfo').read_text()
info['cgroups']={p:pathlib.Path('/sys/fs/cgroup',p).read_text().strip() for p in ['memory.max','cpu.max'] if pathlib.Path('/sys/fs/cgroup',p).exists()}
(root/'environment.json').write_text(json.dumps(info,indent=2)+'\n');print(json.dumps(info,indent=2))
