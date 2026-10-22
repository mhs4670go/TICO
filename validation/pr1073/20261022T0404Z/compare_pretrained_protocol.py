"""Check completed FP and Quant evaluations share one explicit protocol."""
import copy
import json
from pathlib import Path
import re
ROOT=Path(__file__).resolve().parent
report={'status':'ERROR','scope':'Tiny pretrained Llama only; protocol equality does not require equal FP and Quant quality scores'}
try:
    protocols={label:json.loads((ROOT/f'pretrained_{label}_cli/evaluation_protocol.json').read_text()) for label in ['fp','quant']}
    allowed={'checkpoint','model_class','quantization_modes'}
    normalized={label:{k:v for k,v in p.items() if k not in allowed} for label,p in protocols.items()}
    report['cli_protocol_equal']=normalized['fp']==normalized['quant']
    report['allowed_protocol_differences']={key:{label:p[key] for label,p in protocols.items()} for key in sorted(allowed)}
    report['cli_metrics']={}
    for label in ['fp','quant']:
        command=json.loads((ROOT/f'pretrained_{label}_cli_ppl.command.json').read_text())
        matches=re.findall(r'│\s+([0-9]+\.[0-9]+)',(ROOT/f'pretrained_{label}_cli_ppl.log').read_text())
        report['cli_metrics'][label]={'exit_code':command['exit_code'],'ppl_printed':float(matches[-1]) if matches else None}
        if command['exit_code']!=0 or not matches:raise AssertionError(f'{label}: no completed CLI metric')
    supplemental=json.loads((ROOT/'pretrained_smoke_results.json').read_text())
    report['dataset']=supplemental['dataset']
    report['supplemental_protocol']=supplemental['protocol']
    comparable=['resolved_context','recorded_capacity','parameter_dtypes','seed','source_revision','tokenizer_class','tokenizer_path','dataset_fingerprint','sample_ids','corpus_token_ids_sha256']
    report['supplemental_conditions_equal']=all(supplemental['metrics']['fp'][k]==supplemental['metrics']['quant'][k] for k in comparable)
    samples={label:[{key:sample[key] for key in ['sample_id','prompt','input_ids','input_length','target_tokens']} for sample in supplemental['samples'] if sample['model']==label] for label in ['fp','quant']}
    report['generation_and_likelihood_inputs_equal']=samples['fp']==samples['quant']
    report['cli_exact_ppl_consistency']={label:abs(report['cli_metrics'][label]['ppl_printed']-supplemental['metrics'][label]['ppl_exact'])<=0.0051 for label in ['fp','quant']}
    report['frozen_state_equal']=supplemental['frozen_state_equal']
    report['ppl_quant_minus_fp']=supplemental['metrics']['ppl_delta_quant_minus_fp']
    report['metrics_exact']={label:supplemental['metrics'][label]['ppl_exact'] for label in ['fp','quant']}
    success=report['cli_protocol_equal'] and report['supplemental_conditions_equal'] and report['generation_and_likelihood_inputs_equal'] and all(report['cli_exact_ppl_consistency'].values()) and report['frozen_state_equal']
    report['protocol_status']='PASS' if success else 'FAIL'
    report['generation_status']='PASS' if all(supplemental['metrics'][label]['generation_completed'] for label in ['fp','quant']) else 'FAIL'
    report['status']='PASS' if success and report['generation_status']=='PASS' else 'FAIL'
except Exception as exc:
    report['error']=repr(exc)
(ROOT/'pretrained_protocol_comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps(report,indent=2))
raise SystemExit(0 if report['status']=='PASS' else 1)
