"""Inspect only configured repository output and approved Hugging Face caches."""
import json, os, pathlib
repo=pathlib.Path('/workspace/TICO')
roots=[repo/'out',repo/'outputs',repo/'models',repo/'checkpoints',pathlib.Path('/workspace/shared/downloads'),pathlib.Path('/workspace/library-files')]
for key in ['HF_HOME','HF_HUB_CACHE','HUGGINGFACE_HUB_CACHE','TRANSFORMERS_CACHE']:
 if os.environ.get(key): roots.append(pathlib.Path(os.environ[key]))
roots.append(pathlib.Path.home()/'.cache/huggingface')
records=[]
for root in dict.fromkeys(roots):
 files=[]
 if root.exists():
  for p in root.rglob('*'):
   if p.is_file() and (p.suffix in ['.pt','.pth','.bin','.safetensors','.mp4','.jsonl','.parquet'] or p.name in ['config.json','tokenizer.json']): files.append({'path':str(p),'bytes':p.stat().st_size})
 records.append({'root':str(root),'exists':root.exists(),'assets':files})
print(json.dumps({'scope':'Configured project paths and approved caches only; before fixture generation','roots':records},indent=2))
