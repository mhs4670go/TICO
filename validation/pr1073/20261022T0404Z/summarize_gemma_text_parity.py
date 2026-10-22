"""Summarize existing Gemma stage traces without exporting or changing tensors."""
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np

ROOT=Path(__file__).resolve().parent
BASE=ROOT/'gemma-text-independent-baseline/before'
trace=json.loads((ROOT/'GEMMA_TEXT_PARITY_TRACE.json').read_text())
a=np.load(ROOT/'GEMMA_TEXT_PARITY_TRACE.npz')
stages={}
for file in sorted((BASE/'records').glob('*.json')):
 record=json.loads(file.read_text()); arrays=np.load(file.with_suffix('.npz'))
 stages[file.stem]=record['probes']
record=json.loads((BASE/'records/decoder_layer_prefill_0.q.circle.json').read_text())
graph=record['contract']['subgraphs'][0]

def grid(index):
 t=graph['tensors'][index]
 q=(t.get('quantization')or{}).get('fields',{})
 return {'index':index,'scale':q.get('scale',{}).get('values'),'zero_point':q.get('zeroPoint',{}).get('values'),'shape':t['shape']['values']}

def matching_grid(call):
 return [i for i in range(len(graph['tensors'])) if grid(i)['scale']==[call['scale']] and grid(i)['zero_point']==[call['zero']]]

source_files=['tico/quantization/passes/fold_quant_ops.py','tico/quantization/passes/propagate_qparam_forward.py','tico/quantization/passes/propagate_qparam_backward.py']
source_equal={name:(Path('/workspace/TICO')/name).read_bytes()==subprocess.check_output(['git','show','bb09ca852b0ca1b1a6271cfe1e10c162104c997e:'+name],cwd='/workspace/TICO') for name in source_files}
selected=[r for r in trace['fq_calls'] if r['index'] in [9,10,11,12,14,15,34,35,37,38,50,51,64]]
replay={k:v for k,v in trace.items() if k!='fq_calls'}
result={
 'status':'LOCALISED_EXISTING_CONVERSION_BOUNDARY_MISMATCH','raw_strict_result':'FAIL_UNCHANGED',
 'scope':'One tiny frozen baseline text stage; no pretrained or NPU claim.',
 'original_stages':stages,'replay':replay,
 'profile':{'sequence_length':8,'layer0_sliding_window':8,'layer1_full_attention':True,'mask_equal_between_layers':bool(np.array_equal(np.load(BASE/'records/decoder_layer_prefill_0.q.circle.npz')['p0_input_1'],np.load(BASE/'records/decoder_layer_prefill_1.q.circle.npz')['p0_input_1'])),'PLE':0,'shared_KV':0,'sine_convention':'pre_negated_sin'},
 'proof':{
  'qnorm_output_exact':np.array_equal(a['circle_22'],a['fq9_output']),
  'serialized_split0_equals_unrequantized_first_half':np.array_equal(a['circle_25'],a['circle_22'][...,:8]),
  'serialized_split1_equals_unrequantized_second_half':np.array_equal(a['circle_26'],a['circle_22'][...,8:]),
  'serialized_cat_equals_reordered_unrequantized_halves':np.array_equal(a['circle_27'],np.concatenate([a['circle_22'][...,8:],a['circle_22'][...,:8]],axis=-1)),
  'torch_cat_input_equals_reordered_requantized_halves':np.array_equal(a['fq12_input'],np.concatenate([a['fq11_output'],a['fq10_output']],axis=-1)),
  'q_cat_difference_indices':np.argwhere(a['circle_27']!=a['fq12_output']).tolist(),
  'frozen_q_x1_grid':{'scale':trace['fq_calls'][10]['scale'],'zero_point':trace['fq_calls'][10]['zero']},
  'frozen_q_x2_grid':{'scale':trace['fq_calls'][11]['scale'],'zero_point':trace['fq_calls'][11]['zero']},
  'q_x1_grid_serialized_tensors':matching_grid(trace['fq_calls'][10]),
  'q_x2_grid_serialized_tensors':matching_grid(trace['fq_calls'][11]),
  'serialized_tensor_grids':[grid(i) for i in [22,25,26,27]],
  'example':{'qnorm_index':[0,4,0,4],'qnorm_value':float(a['fq9_output'][0,4,0,4]),'torch_qx1_after_quant':float(a['fq10_output'][0,4,0,4]),'qcat_index':[0,4,0,12],'torch_qcat':float(a['fq12_output'][0,4,0,12]),'circle_qcat':float(a['circle_27'][0,4,0,12])},
  'mapped_quantizer_calls':sum('circle_tensor' in row for row in trace['fq_calls']),
  'isolated_same_input_quantizer_mismatches':sum(row.get('isolated_quantizer_difference',{}).get('max_abs',0)>0 for row in trace['fq_calls']),
 },
 'selected_quantizer_boundaries':selected,
 'source_pipeline_files_unchanged_from_baseline':source_equal,
 'established':'The serialized SPLIT/CONCAT path does not retain two frozen query-half requantization domains; their loss first changes q_cat, then query rotation/attention and row4 hidden output.',
 'not_established':'Exact pass at which boundaries disappear; a corrected export/fix; applicability to pretrained checkpoints or other profiles.',
 'no_changes':'No re-export, recalibration, checkpoint write, product fix, tolerance change, or replacement runtime.',
 'infrastructure_failure_preserved':'Initial helper passed Path to baseline runtime expecting bytes; original failed script/log retained and minimal read_bytes fix rerun separately.',
}
(ROOT/'GEMMA_TEXT_PARITY_DIAGNOSIS.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({'status':result['status'],'proof':result['proof'],'source_files_unchanged':source_equal},indent=2))
