"""Locate strict Torch/Circle differences in actual saved numerical stage records."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
run=ROOT/'pretrained_selected_stage'
report={'atol':1e-5,'rtol':1e-4,'scope':'Actual selected layer0 numerical outputs; not a full bundle and no expected-RoPE exemption','stages':[]}
for phase in ['before','after_eager']:
    records=run/f'{phase}/records'
    for path in sorted(records.glob('*.json')):
        if not path.with_suffix('.npz').exists():continue
        record=json.loads(path.read_text());arrays=np.load(path.with_suffix('.npz'))
        graph=record['contract']['subgraphs'][0]
        stage={'phase':phase,'stage':path.stem,'runtime_mode':record['mode'],'probes':[]}
        for index,probe in enumerate(record['probes']):
            outputs=[]
            for output_index,tensor_index in enumerate(graph['outputs']):
                left=arrays[f'p{index}_torch_{output_index}'];right=arrays[f'p{index}_circle_{output_index}']
                difference=np.abs(left.astype(np.float64)-right.astype(np.float64));finite=bool(np.isfinite(left).all() and np.isfinite(right).all())
                close=np.isclose(left,right,atol=1e-5,rtol=1e-4,equal_nan=False)
                loc=tuple(int(v) for v in np.unravel_index(np.argmax(difference),difference.shape))
                quantization=graph['tensors'][tensor_index].get('quantization')
                scale=None
                if quantization:scale=quantization['fields'].get('scale',{}).get('values')
                failures=[]
                for row in np.argwhere(~close)[:12]:
                    key=tuple(int(v) for v in row);failures.append({'index':list(key),'torch':float(left[key]),'circle':float(right[key]),'absolute_error':float(difference[key])})
                outputs.append({'output_index':output_index,'shape':list(left.shape),'dtype':str(left.dtype),'all_finite':finite,'strict_parity':bool(close.all() and finite),'failure_count':int((~close).sum()),'max_abs_error':float(difference[loc]),'max_error_index':list(loc),'torch_at_max':float(left[loc]),'circle_at_max':float(right[loc]),'circle_output_scale':scale,'max_error_in_scale_units':float(difference[loc]/scale[0]) if scale and len(scale)==1 else None,'first_failure_locations':failures})
            stage['probes'].append({'probe':probe['probe'],'outputs':outputs})
        report['stages'].append(stage)
summary=run/'summary.json'
execution=ROOT/'pretrained_selected_stage.command.json'
command=json.loads(execution.read_text()) if execution.exists() else {}
report['raw_coordinator_summary_exists']=summary.exists()
report['raw_summary']=json.loads(summary.read_text()) if summary.exists() else {'status':'TIMEOUT' if command.get('exit_code')==124 else 'RUNNING','comparisons_complete':False,'command_exit_code':command.get('exit_code')}
(ROOT/'pretrained_selected_stage_differences.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({'stage_records':len(report['stages']),'strict_fail_stage_records':sum(any(not output['strict_parity'] for probe in stage['probes'] for output in probe['outputs']) for stage in report['stages']),'raw_status':report['raw_summary']['status'],'comparisons_complete':report['raw_summary'].get('comparisons_complete',False)},indent=2))
