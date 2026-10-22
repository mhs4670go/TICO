"""Record frozen tiny full-model QUANT forwards with real PNG processor inputs."""

import gc
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root=Path(__file__).resolve().parent
    memory=int(Path('/sys/fs/cgroup/memory.current').read_text())
    if memory>=12*1024**3:
        print(json.dumps({'status':'BLOCKED_RESOURCE','memory_current_bytes':memory}))
        return 2
    sys.path.insert(0,'/workspace/TICO')
    import torch
    import tico
    from tico.quantization.recipes.debug.export_regression import _state_contract
    torch.set_num_threads(1)
    output=root/'png_circle_inputs'
    report={'scope':'actual PNG processed by normal HF image processors, tiny random frozen real-HF QUANT full-model single forward per family; no FP comparison, score, pretrained, cache-linked decode, Circle completion, or NPU claim',
        'tico_file':tico.__file__,'models':{},'memory_current_before_bytes':memory}
    def profiles(model):
        config=getattr(model,'config',None) or model.wrapped.config
        return {'hf_config':config.to_dict(),
            'qcfg_repr':{name:repr(module.qcfg) for name,module in model.named_modules() if hasattr(module,'qcfg')},
            'attention_profile_repr':{name:repr(module.attn_options) for name,module in model.named_modules() if hasattr(module,'attn_options')}}
    complete=True
    for family,stem in (('qwen3_vl','baseline_qwen_cached'),('gemma4','baseline_gemma_image70')):
        checkpoint=root/'fixtures'/(stem+'.pt')
        inputs=output/('qwen_actual_png_kwargs.pt' if family=='qwen3_vl' else 'gemma_actual_png_kwargs.pt')
        model=torch.load(checkpoint,map_location='cpu',weights_only=False).eval()
        loaded=torch.load(inputs,map_location='cpu',weights_only=True)
        calls=loaded if isinstance(loaded,list) else [loaded]
        assert len(calls)==1 and calls[0]['use_cache'] is False
        kwargs=calls[0]
        state_before=_state_contract(model)
        profile_before=profiles(model)
        with torch.no_grad():
            result=model(**kwargs)
        logits=result.logits.detach().cpu().numpy()
        state_after=_state_contract(model)
        profile_after=profiles(model)
        destination=output/(family+'_actual_png_quant_logits.npy')
        np.save(destination,logits)
        finite=bool(np.isfinite(logits).all())
        preserved=state_before==state_after and profile_before==profile_after
        complete &= finite and preserved
        item={'status':'PASS' if finite and preserved else 'FAIL',
            'checkpoint':str(checkpoint),'checkpoint_sha256':sha256(checkpoint),
            'inputs':str(inputs),'inputs_sha256':sha256(inputs),
            'calls':1,'use_cache':False,'past_key_values_returned':getattr(result,'past_key_values',None) is not None,
            'logits':{'shape':list(logits.shape),'dtype':str(logits.dtype),'element_count':int(logits.size),
                'finite':finite,'min':float(logits.min()),'max':float(logits.max()),
                'array':str(destination),'array_file_sha256':sha256(destination),
                'raw_bytes_sha256':hashlib.sha256(logits.tobytes()).hexdigest()},
            'frozen_state_preserved':state_before==state_after,
            'profile_qcfg_config_preserved':profile_before==profile_after,
            'all_quantization_modes':sorted(set(state_after['modes'].values())),
            'memory_current_after_bytes':int(Path('/sys/fs/cgroup/memory.current').read_text())}
        (output/(family+'_frozen_before.json')).write_text(json.dumps(state_before,indent=2)+'\n')
        (output/(family+'_frozen_after.json')).write_text(json.dumps(state_after,indent=2)+'\n')
        (output/(family+'_profiles_before.json')).write_text(json.dumps(profile_before,indent=2,default=str)+'\n')
        (output/(family+'_profiles_after.json')).write_text(json.dumps(profile_after,indent=2,default=str)+'\n')
        report['models'][family]=item
        print(json.dumps(item,indent=2))
        del model,result,logits,state_before,state_after,profile_before,profile_after
        gc.collect()
    report['status']='PASS' if complete else 'FAIL'
    (output/'fullmodel_forward_evidence.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return 0 if complete else 1


if __name__=='__main__':
    raise SystemExit(main())
