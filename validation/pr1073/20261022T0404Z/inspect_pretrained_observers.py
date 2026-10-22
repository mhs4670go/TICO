"""Read frozen observer coverage without recalculating or modifying qparams."""
import hashlib
import json
from pathlib import Path
import torch
import tico
from tico.quantization.wrapq.observers.base import ObserverBase
ROOT=Path(__file__).resolve().parent
report={'tico_file':tico.__file__,'scope':'Read-only frozen observer coverage; no compute_qparams/fake_quant/recalibration is called','checkpoints':{}}
for label,name in [('original','pretrained_tinyllama_quantized.pt'),('cached','pretrained_tinyllama_quantized_cached.pt')]:
    path=ROOT/name
    if not path.exists():continue
    model=torch.load(path,map_location='cpu',weights_only=False)
    items=[]
    for module_name,observer in model.named_modules():
        if not isinstance(observer,ObserverBase):continue
        fields={}
        for attr in ['min_val','max_val','_cached_scale','_cached_zp']:
            tensor=getattr(observer,attr,None)
            if not isinstance(tensor,torch.Tensor):continue
            flat=tensor.detach().reshape(-1)
            finite=torch.isfinite(flat)
            fields[attr]={'shape':list(tensor.shape),'finite':bool(finite.all()),'nonfinite_count':int((~finite).sum()),'first_values':[str(x.item()) for x in flat[:4]]}
        if any(not field['finite'] for field in fields.values()) or 'obs_past' in module_name:
            items.append({'name':module_name,'class':type(observer).__name__,'dtype':str(observer.dtype),'fields':fields})
    report['checkpoints'][label]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'root_class':type(model).__name__,'observer_coverage':items}
(ROOT/'pretrained_observer_coverage.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps(report,indent=2))
