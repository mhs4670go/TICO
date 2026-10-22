"""Localize baseline softmax rounding and unobserved cache quantization."""
import json
from pathlib import Path
import sys
import numpy as np

root=Path(__file__).resolve().parent
sys.path.insert(0,'/workspace/TICO-export-baseline')
import torch
import tico
from tico.circle.runtime import CircleReferenceRuntime
from tico.circle.runtime.quantization import fake_quantize_array
torch.set_num_threads(1)
model=torch.load(root/'fixtures/baseline_llama_v2.pt',map_location='cpu',weights_only=False)
report={'tico_file':tico.__file__,'past_observers':{},'rounding':{}}
for layer_index,layer in enumerate(model.wrapped.model.wrapped.layers):
    attention=layer.wrapped.self_attn.wrapped
    for key in ('obs_past_key','obs_past_value'):
        observer=getattr(attention,key)
        report['past_observers'][f'layer{layer_index}.{key}']={
            name: {'values':getattr(observer,name).reshape(-1).tolist(),
                   'finite':bool(torch.isfinite(getattr(observer,name)).all())}
            for name in ('min_val','max_val','_cached_scale','_cached_zp')}
phase=root/'tiny-llama-old-new-v2/baseline/before'
runtime=CircleReferenceRuntime((phase/'artifacts/decoder_layer_prefill_0.q.circle').read_bytes())
tensor=runtime.program.tensor(36)
scale=np.asarray(tensor.quantization.scale,dtype=np.float32)[0]
value=np.array([1.0/6.0],dtype=np.float32)
torch_result=torch.fake_quantize_per_tensor_affine(torch.from_numpy(value),float(scale),0,0,255).numpy()
circle_result=fake_quantize_array(value,tensor)
report['rounding']={'source':'uniform six-key softmax row', 'value':value.tolist(),
    'scale':float(scale),'scaled_value':(value/scale).tolist(),
    'torch_fake_quant':torch_result.tolist(),'circle_fake_quant':circle_result.tolist(),
    'max_abs_error':float(np.max(np.abs(torch_result-circle_result))),
    'torch_grid_integer':(torch_result/scale).tolist(),'circle_grid_integer':(circle_result/scale).tolist()}
# Write nonfinite observer values as strings to keep strict JSON.
for observer in report['past_observers'].values():
    for entry in observer.values():
        entry['values']=[float(value) if np.isfinite(value) else repr(value) for value in entry['values']]
(root/'ROUNDING_AND_PAST_DIAGNOSIS.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps(report,indent=2,allow_nan=False))
