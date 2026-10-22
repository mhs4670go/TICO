"""Inspect actual Qwen Circle records and frozen cache-observer provenance."""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def numeric_record(left,right):
    error=np.abs(left.astype(np.float64)-right.astype(np.float64))
    maximum=tuple(np.unravel_index(error.argmax(),error.shape)) if error.size else ()
    failed=~np.isclose(left,right,atol=1e-5,rtol=1e-4)
    return {'shape':list(left.shape),'dtype':str(left.dtype),'finite':bool(np.isfinite(left).all() and np.isfinite(right).all()),
        'exact_equal':bool(np.array_equal(left,right)),
        'max_abs_error':float(error.max()) if error.size else 0,
        'max_position':list(map(int,maximum)),
        'torch_value_at_max':float(left[maximum]) if error.size else None,
        'circle_value_at_max':float(right[maximum]) if error.size else None,
        'failing_elements':int(failed.sum()),'first_failing_positions':np.argwhere(failed)[:8].tolist()}


def main():
    root=Path(__file__).resolve().parent
    sys.path.insert(0,'/workspace/TICO-export-baseline')
    import torch
    import tico
    torch.set_num_threads(1)
    checkpoint=root/'fixtures/baseline_qwen_cached.pt'
    model=torch.load(checkpoint,map_location='cpu',weights_only=False).eval()
    run=root/'tiny-qwen-old-new-cached'
    component=json.loads((run/'component_summary.json').read_text())
    report={'scope':'tiny random real-HF frozen Qwen fixture; read-only analysis of actual completed Circle records; no new export and no pretrained validation',
        'tico_file':tico.__file__,'checkpoint':str(checkpoint),
        'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'raw_strict_status':component['raw_status'],
        'component_summary':component,'cache_observers':{},'phases':{}}
    for name,module in model.named_modules():
        if type(module).__name__!='QuantQwen3VLTextAttention':continue
        entry={'mode':module._mode.name,'observers':{}}
        for key in ('obs_past_key','obs_past_value','obs_present_key','obs_present_value'):
            observer=getattr(module,key)
            fields={}
            for field in ('min_val','max_val','_cached_scale','_cached_zp'):
                array=getattr(observer,field).detach().cpu().reshape(-1).numpy()
                fields[field]={'values':[float(value) if np.isfinite(value) else repr(float(value)) for value in array],
                    'finite':bool(np.isfinite(array).all())}
            fields['enabled']=observer.enabled
            entry['observers'][key]=fields
        report['cache_observers'][name]=entry
    for phase_name in ('baseline/before','candidate/before','candidate/after_eager'):
        phase=run/phase_name
        snapshot=json.loads((phase/'snapshot.json').read_text())
        stages={}
        for stage_name,record_name in sorted(snapshot['stages'].items()):
            metadata=json.loads((phase/'records'/(record_name+'.json')).read_text())
            stage={'execution_mode':metadata['mode'],'probes':[]}
            with np.load(phase/'records'/(record_name+'.npz')) as arrays:
                for index,probe in enumerate(metadata['probes']):
                    outputs=[]
                    for output_index in range(len(probe['torch_parity'])):
                        torch_value=arrays[f'p{index}_torch_{output_index}']
                        circle_value=arrays[f'p{index}_circle_{output_index}']
                        role=('hidden' if output_index==0 else 'delta_key' if output_index==1 else 'delta_value') if stage_name.startswith('decoder_layer') else 'stage_output'
                        outputs.append({'output_index':output_index,'role':role,
                            'strict_ok':probe['torch_parity'][output_index]['ok'],**numeric_record(torch_value,circle_value)})
                    stage['probes'].append({'probe':probe['probe'],'outputs':outputs})
            stages[stage_name]=stage
        report['phases'][phase_name]={'stage_count':len(stages),'state_preserved':snapshot['state_preserved'],'stages':stages}
    (root/'QWEN_PARITY_DIAGNOSTIC.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'report':str(root/'QWEN_PARITY_DIAGNOSTIC.json'),
        'cache_observer_owners':len(report['cache_observers']),
        'raw_status':report['raw_strict_status'],
        'decoder_export_example':{name:{'errors':[item['max_abs_error'] for item in stage['probes'][0]['outputs']],
            'positions':[item['max_position'] for item in stage['probes'][0]['outputs']]}
            for name,stage in report['phases']['baseline/before']['stages'].items() if name.startswith('decoder_layer')}},indent=2))


if __name__=='__main__':
    main()
