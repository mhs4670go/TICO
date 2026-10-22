"""Replay an existing frozen Gemma stage and record unchanged quantizer calls."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
import tico
from tico.circle.runtime import CircleReferenceRuntime, ExecutionMode
from tico.circle.runtime.quantization import fake_quantize_array
from tico.quantization.recipes.export.gemma4 import _unwrap_gemma4_components
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from gemma_text_export_recording import _state_contract

ROOT=Path(__file__).resolve().parent
BASE=ROOT/'gemma-text-independent-baseline/before'
CHECKPOINT=ROOT/'fixtures/baseline_gemma_image70.pt'
STAGE='decoder_layer_prefill_0.q.circle'
torch.set_num_threads(1)
started=time.process_time()
if not str(Path(tico.__file__).resolve()).startswith('/workspace/TICO-export-baseline/'):
 raise RuntimeError('Expected baseline source import')
model=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
initial=_state_contract(model)
_,_,_,text=_unwrap_gemma4_components(model)
layer=text.layers[0].wrapped
arrays=np.load(BASE/'records'/f'{STAGE}.npz')
inputs=[arrays[f'p0_input_{i}'].copy() for i in range(4)]
runtime=CircleReferenceRuntime((BASE/'artifacts'/STAGE).read_bytes())
result=runtime.run(inputs,mode=ExecutionMode.FAKE_QUANTIZE,trace=True)
trace=result.tensor_values
names={id(module):name for name,module in model.named_modules()}
calls=[]
original=QuantModuleBase._fq

def capture(owner,value,observer):
 """Delegate exactly to the original implementation and copy diagnostic values."""
 out=original(owner,value,observer)
 calls.append({'module':names.get(id(owner),type(owner).__name__),'observer':observer.name,'input':value.detach().cpu().numpy().copy(),'output':out.detach().cpu().numpy().copy(),'scale':observer._cached_scale.detach().cpu().numpy().copy() if hasattr(observer,'_cached_scale') else None,'zero':observer._cached_zp.detach().cpu().numpy().copy() if hasattr(observer,'_cached_zp') else None})
 return out

QuantModuleBase._fq=capture
try:
 with torch.no_grad():
  adapter=layer.as_export_module('prefill',return_kv=True)
  actual=adapter(torch.from_numpy(inputs[0]),attention_mask=torch.from_numpy(inputs[1]),position_embeddings=(torch.from_numpy(inputs[2]),torch.from_numpy(inputs[3])))
finally:
 QuantModuleBase._fq=original
if _state_contract(model)!=initial:
 raise RuntimeError('Frozen model state changed during read-only tracing')

def metric(a,b):
 d=np.abs(a.astype(np.float64)-b.astype(np.float64)); index=list(map(int,np.unravel_index(np.argmax(d),d.shape))) if d.size else []
 return {'max_abs':float(d.max()) if d.size else 0.,'index':index,'nonzero':int(np.count_nonzero(d)),'left':float(a[tuple(index)]) if index else float(a),'right':float(b[tuple(index)]) if index else float(b)}

replay={'tico_file':tico.__file__,'checkpoint_sha256':hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),'circle_sha256':hashlib.sha256((BASE/'artifacts'/STAGE).read_bytes()).hexdigest(),'circle_replay_exact':[np.array_equal(value,arrays[f'p0_circle_{i}']) for i,value in enumerate(result.outputs)],'torch_replay_exact':[np.array_equal(value.detach().numpy(),arrays[f'p0_torch_{i}']) for i,value in enumerate(actual)],'state_preserved':True,'scope':'Existing binary replay and real frozen eager call only; no conversion/calibration/parameter changes. Diagnostic recorder delegates to original _fq and is restored.'}
records=[]; dumps={f'circle_{index}':value for index,value in trace.items()}
for index,call in enumerate(calls):
 x,y,scale,zp=call['input'],call['output'],call['scale'],call['zero']
 dumps[f'fq{index}_input']=x; dumps[f'fq{index}_output']=y
 candidates=[]
 for tensor in runtime.program.tensors:
  q=tensor.quantization
  if tensor.index not in trace or trace[tensor.index].shape!=y.shape or q is None or scale is None or not scale.size:
   continue
  if not np.array_equal(np.asarray(q.scale,dtype=np.float32).reshape(-1),scale.reshape(-1)) or not np.array_equal(np.asarray(q.zero_point).reshape(-1),zp.reshape(-1)):
   continue
  candidates.append((float(np.max(np.abs(trace[tensor.index]-y))) if y.size else 0.,tensor))
 candidates.sort(key=lambda pair:pair[0])
 row={k:v for k,v in call.items() if k not in ['input','output','scale','zero']}
 row.update(index=index,shape=list(y.shape),scale=None if scale is None else scale.tolist(),zero=None if zp is None else zp.tolist())
 if candidates:
  _,tensor=candidates[0]
  local=fake_quantize_array(x,tensor)
  row.update(circle_tensor=tensor.index,circle_name=tensor.name,matching_tensor_count=len(candidates),trace_difference=metric(y,trace[tensor.index]),isolated_quantizer_difference=metric(y,local))
  if np.any(y!=local):
   d=np.abs(y-local); flat=np.unravel_index(d.argmax(),d.shape)
   row['raw_at_quantizer_difference']=float(x[flat]); row['normalized_raw']=float(x[flat]/scale.reshape(-1)[0]+zp.reshape(-1)[0])
 records.append(row)
replay['fq_calls']=records
replay['process_cpu_seconds']=time.process_time()-started
(ROOT/'GEMMA_TEXT_PARITY_TRACE.json').write_text(json.dumps(replay,indent=2,allow_nan=False)+'\n')
np.savez_compressed(ROOT/'GEMMA_TEXT_PARITY_TRACE.npz',**dumps)
print(json.dumps({k:v for k,v in replay.items() if k!='fq_calls'},indent=2))
print('Divergent isolated quantizer calls:')
for row in records:
 if row.get('isolated_quantizer_difference',{}).get('max_abs',0)>0:
  print(json.dumps(row))
