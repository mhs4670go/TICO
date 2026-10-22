"""Build a truthful per-check summary from saved execution evidence."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def read(name):
    path=ROOT/name
    return json.loads(path.read_text()) if path.exists() else None

def circle_status(folder,command):
    summary=read(folder+'/summary.json')
    execution=read(command+'.command.json')
    records={phase:len(list((ROOT/folder/f'candidate/{phase}/records').glob('*.json'))) for phase in ['before','after_eager']}
    return {'raw_summary':summary,'command':execution,'recorded_stages':records,'status':summary['status'] if summary else 'TIMEOUT' if execution and execution.get('exit_code')==124 else 'RUNNING','comparisons_complete':summary.get('comparisons_complete',False) if summary else False}

smoke=read('pretrained_smoke_results.json');protocol=read('pretrained_protocol_comparison.json')
report={'scope':'Tiny pretrained Maykeye/TinyLLama-v0 4.62M only; not Llama-3.2, not NPU validation','model_revision':'298338802ab94432b917bcce11382aa151aee50f','candidate_sha':'6b2f6a0511458d6a750e2a260d706080a78149dd','baseline_sha':'bb09ca852b0ca1b1a6271cfe1e10c162104c997e','original_checkpoint':read('pretrained_checkpoint_identity.json'),'cached_checkpoint':read('pretrained_cached_checkpoint_identity.json'),'quality_smoke_status':smoke.get('status') if smoke else 'NOT_RUN','ppl':{'status':protocol.get('protocol_status') if protocol else 'NOT_RUN','metrics':protocol.get('metrics_exact') if protocol else None,'quant_minus_fp':protocol.get('ppl_quant_minus_fp') if protocol else None,'sample_rows':12,'corpus_tokens':660,'window':64,'stride':32,'evaluation_cap':128,'calibration_length':32,'export_length':16},'likelihood':{'status':'PASS' if smoke and len(smoke['samples'])==4 else 'INCOMPLETE','samples':[{key:sample[key] for key in ['model','sample_id','input_length','nll','target_tokens']} for sample in smoke['samples']] if smoke else []},'generation':{'fp':'PASS' if smoke and smoke['metrics']['fp']['generation_completed'] else 'FAIL','quant':'PASS' if smoke and smoke['metrics']['quant']['generation_completed'] else 'FAIL','failure':'QuantLlamaForCausalLM lacks get_experts_implementation required by Transformers5.18.0','classification':'PREEXISTING_COMPATIBILITY_FAILURE','baseline_evidence':'baseline_pretrained_generation.json'},'frozen_eval_state':{'status':'PASS' if smoke and smoke['frozen_state_equal'] else 'FAIL','coverage':'weights, observers, frozen qparams, modes, full config, qcfg profiles, attention options, return_type'},'original_circle':circle_status('pretrained_tinyllama_isolation_16','pretrained_circle_isolation_16'),'cached_circle':circle_status('pretrained_tinyllama_cached_isolation_16','pretrained_cached_circle_isolation_16'),'selected_stage':read('pretrained_selected_stage/summary.json') or {'status':'NOT_RUN','comparisons_complete':False},'old_new_export':{'status':'BLOCKED','reason':'No compatible user pre-PR pretrained full-module checkpoint; fresh baseline preparation was generation diagnosis only'},'seq2048_export':{'status':'NOT_RUN','reason':'Shape-independent strict Torch/Circle failure at16tokens; not repeated at2048'},'eager4096_plus4':{'status':'BLOCKED','recorded_capacity':2048,'required_capacity':4100,'reason':'Native config/checkpoint capacity was not enlarged'},'product_code_changed':False,'tolerances_changed':False}
(ROOT/'pretrained_summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({key:report[key] for key in ['scope','quality_smoke_status','ppl','generation','frozen_eval_state','seq2048_export','eager4096_plus4']},indent=2))
