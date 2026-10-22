"""Read actual Gemma full-text stage evidence without re-export or rerun."""

import hashlib
import json
from pathlib import Path

import numpy as np


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def main():
    root=Path(__file__).resolve().parent
    run=root/'tiny-gemma-full-text-png-masks-retry'
    summary=json.loads((run/'summary.json').read_text())
    components=json.loads((run/'component_summary.json').read_text())
    report={'scope':'tiny real-HF frozen Gemma text-only full-wrapper Circle probe plus one separate actual PNG eager forward; not full multimodal export, pretrained evaluation, linked-cache decode or NPU validation',
        'run':str(run),'raw_status':summary['status'],'comparisons_complete':summary['comparisons_complete'],
        'strict_torch_parity_required':summary['torch_parity_required'],'atol':summary['atol'],'rtol':summary['rtol'],
        'component_summary':components,'phases':{},
        'copied_runner_provenance':json.loads((root/'export_regression_gemma_full_text_masks.provenance.json').read_text()),
        'directory_fix_provenance':json.loads((root/'export_regression_full_text_dirs.provenance.json').read_text()),
        'recorded_host_masks':json.loads((run/'recorded_host_masks.json').read_text()),
        'eager':json.loads((run/'candidate/eager.json').read_text()),
        'original_errors':{}}
    for name,relative in (('missing_full_text_artifact_directory','tiny-qwen-full-text-png-original'),
                          ('gemma_missing_required_static_masks','tiny-gemma-full-text-png-dirs')):
        previous=root/relative
        report['original_errors'][name]={'run':str(previous),
            'summary':json.loads((previous/'summary.json').read_text()),
            'baseline_log_last_lines':(previous/'baseline.log').read_text().splitlines()[-5:]}
    for phase_name in ('baseline/before','candidate/before','candidate/after_eager'):
        phase=run/phase_name
        snapshot=json.loads((phase/'snapshot.json').read_text())
        stages={}
        for stage_name,record_name in snapshot['stages'].items():
            record=json.loads((phase/'records'/(record_name+'.json')).read_text())
            contract=record['contract']
            graph=contract['subgraphs'][0]
            qparams={str(index):tensor['quantization'] for index,tensor in enumerate(graph['tensors']) if tensor.get('quantization') is not None}
            constants={str(index):tensor['constant'] for index,tensor in enumerate(graph['tensors']) if tensor.get('constant') is not None}
            stage={'execution_mode':record['mode'],'graph_input_count':len(graph['inputs']),
                'graph_output_count':len(graph['outputs']),'operator_count':len(graph['operators']),
                'tensor_count':len(graph['tensors']),'qparam_tensor_count':len(qparams),
                'constant_tensor_count':len(constants),'contract_sha256':digest(contract),
                'topology_options_sha256':digest({'inputs':graph['inputs'],'outputs':graph['outputs'],'operators':graph['operators']}),
                'qparams_sha256':digest(qparams),'constants_sha256':digest(constants),'probes':[]}
            with np.load(phase/'records'/(record_name+'.npz')) as arrays:
                stage['all_arrays_finite']=all(bool(np.isfinite(arrays[key]).all()) for key in arrays.files)
                for index,probe in enumerate(record['probes']):
                    outputs=[]
                    circle_keys=sorted(key for key in arrays.files if key.startswith(f'p{index}_circle_'))
                    torch_keys=sorted(key for key in arrays.files if key.startswith(f'p{index}_torch_'))
                    input_keys=sorted(key for key in arrays.files if key.startswith(f'p{index}_input_'))
                    for output_index in range(len(probe['torch_parity'])):
                        left,right=arrays[f'p{index}_torch_{output_index}'],arrays[f'p{index}_circle_{output_index}']
                        delta=np.abs(left.astype(np.float64)-right.astype(np.float64))
                        pos=tuple(np.unravel_index(delta.argmax(),delta.shape))
                        failed=~np.isclose(left,right,atol=1e-5,rtol=1e-4)
                        outputs.append({'output_index':output_index,'role':'last_hidden_state','shape':list(left.shape),
                            'dtype':str(left.dtype),'finite':bool(np.isfinite(left).all() and np.isfinite(right).all()),
                            'max_abs_error':float(delta.max()),'max_position':list(map(int,pos)),
                            'torch_value':float(left[pos]),'circle_value':float(right[pos]),
                            'failing_elements':int(failed.sum()),'first_failing_positions':np.argwhere(failed)[:8].tolist(),
                            'strict_ok':probe['torch_parity'][output_index]['ok'],
                            'torch_raw_sha256':hashlib.sha256(left.tobytes()).hexdigest(),
                            'circle_raw_sha256':hashlib.sha256(right.tobytes()).hexdigest()})
                    stage['probes'].append({'probe':probe['probe'],'input_array_count':len(input_keys),
                        'circle_output_count':len(circle_keys),'torch_output_count':len(torch_keys),'outputs':outputs})
            stages[stage_name]=stage
        report['phases'][phase_name]={'stage_count':len(stages),'state_preserved':snapshot['state_preserved'],'stages':stages}
    (root/'GEMMA_FULL_TEXT_DIAGNOSTIC.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'report':str(root/'GEMMA_FULL_TEXT_DIAGNOSTIC.json'),
        'raw_status':report['raw_status'],'phases':report['phases']},indent=2))


if __name__=='__main__':
    main()
