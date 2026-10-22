"""Check actual first-decoder Circle isolation on one full pretrained model instance."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import traceback
import numpy as np
import torch
import tico
from tico.quantization.recipes.debug.export_regression import (
    SCHEMA_VERSION, ExportRecorder, _array_record, _clone_inputs, _file_sha256,
    _first_differences, _host_contract, _seed, _tree_arrays, _write_json, compare_snapshots,
)
from tico.quantization.wrapq.wrappers.llama.static_inputs import prepare_llama_static_inputs
from tico.utils.pytree_utils import register_dynamic_cache, register_dynamic_layer
from pretrained_llama_smoke import ROOT, state as frozen_quant_state

def state(model):
    snapshot=frozen_quant_state(model)
    snapshot['export_controls']={name:{'force_export':getattr(module,'force_export',None),'static_inputs_present':getattr(module,'_static_export_inputs',None) is not None,'temporary_export_buffers':[key for key in module._buffers if '_static_export' in key]} for name,module in model.named_modules()}
    return snapshot

p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
output=Path(a.output)
if output.exists():raise FileExistsError(output)
output.mkdir(parents=True)
report={'status':'ERROR','comparisons_complete':False,'scope':'Selected first decoder prefill/decode stages of one full 4.62M pretrained model only; not full bundle, not pre-PR equivalence, not Llama-3.2 or NPU validation','baseline_comparison':'NOT_RUN','selected_layers':[0],'total_layers':8,'static_capacity':16,'eager_prefill_tokens':64,'teacher_forced_decode_steps':4,'host_source_recomputed_each_phase':True,'atol':1e-5,'rtol':1e-4}
try:
    torch.set_num_threads(1);_seed(42)
    register_dynamic_layer();register_dynamic_cache()
    checkpoint=ROOT/'pretrained_tinyllama_quantized_cached.pt'
    model=torch.load(checkpoint,map_location='cpu',weights_only=False).eval()
    assert all(parameter.dtype==torch.float32 for parameter in model.parameters() if parameter.is_floating_point())
    full=model.wrapped
    layer=full.model.wrapped.layers[0].wrapped
    initial=state(model)
    source=prepare_llama_static_inputs(layer,16)
    hidden=torch.randn(1,16,full.config.hidden_size)
    prefill_mask=source.mask(16,16,0,device=torch.device('cpu'))
    prefill_rope=source.rope(0,16,device=torch.device('cpu'),dtype=hidden.dtype)
    prefill_inputs=((hidden,),{'attention_mask':prefill_mask,'position_embeddings':prefill_rope})
    with torch.no_grad():
        reference_prefill=layer.as_export_module('prefill',return_kv=True).eval()(*_clone_inputs(prefill_inputs[0]),**_clone_inputs(prefill_inputs[1]))
    if len(reference_prefill)!=3:raise AssertionError('Prefill must expose hidden,deltaK,deltaV')
    past=(reference_prefill[1][:,:,:15,:].detach().clone(),reference_prefill[2][:,:,:15,:].detach().clone())
    decode_mask=torch.zeros(1,1,16,dtype=torch.float32)
    report.update(checkpoint_sha256=_file_sha256(checkpoint),tico_file=tico.__file__,python=__import__('sys').executable,versions={name:importlib.metadata.version(name) for name in ['torch','transformers','circle-schema','numpy']},model_class=type(model).__name__,owner_model_id=id(model),owner_layer_id=id(layer),recorded_capacity=full.config.max_position_embeddings)
    _write_json(output/'environment.json',report)
    _write_json(output/'frozen_initial.json',initial)

    def phase(name):
        root=output/name
        artifacts=root/'artifacts';artifacts.mkdir(parents=True)
        _seed(42)
        phase_source=prepare_llama_static_inputs(layer,16)
        phase_prefill_mask=phase_source.mask(16,16,0,device=torch.device('cpu'))
        phase_prefill_rope=phase_source.rope(0,16,device=torch.device('cpu'),dtype=hidden.dtype)
        phase_decode_rope=phase_source.rope(15,1,device=torch.device('cpu'),dtype=hidden.dtype)
        phase_inputs={'prefill':((hidden,),{'attention_mask':phase_prefill_mask,'position_embeddings':phase_prefill_rope}),'decode':((hidden[:,-1:].clone(),),{'attention_mask':decode_mask,'position_embeddings':phase_decode_rope,'past_key_value':past})}
        phase_arrays={'prefill_mask':phase_prefill_mask.numpy(),'prefill_cos':phase_prefill_rope[0].numpy(),'prefill_sin':phase_prefill_rope[1].numpy(),'decode_mask':decode_mask.numpy(),'decode_cos':phase_decode_rope[0].numpy(),'decode_sin':phase_decode_rope[1].numpy()}
        with ExportRecorder(root,atol=1e-5,rtol=1e-4) as recorder:
            for mode,inputs in phase_inputs.items():
                adapter=layer.as_export_module(mode,return_kv=True).eval() if mode=='prefill' else layer.as_export_module(mode).eval()
                converted=tico.convert(adapter,_clone_inputs(inputs[0]),kwargs=_clone_inputs(inputs[1]),strict=False)
                converted.save(artifacts/f'decoder_layer_{mode}_0.q.circle')
        if len(recorder.stages)!=2:raise AssertionError('Both selected stages must produce numerical evidence')
        actual_files={path.name for path in artifacts.glob('*.circle')}
        if actual_files!=set(recorder.stages):raise AssertionError('Saved stages differ from recorded stages')
        np.savez(artifacts/'selected_host_static_inputs.npz',**phase_arrays)
        after=state(model)
        snapshot={'schema_version':SCHEMA_VERSION,'stages':recorder.stages,'state_preserved':after==initial,'state_differences':_first_differences(initial,after),'host':_host_contract(artifacts)}
        _write_json(root/'snapshot.json',snapshot)
        _write_json(root/'frozen_after.json',after)
        return snapshot

    phase('before')
    _seed(43)
    if full.config.max_position_embeddings<68:raise AssertionError('Recorded capacity is insufficient')
    eager=[]
    with torch.no_grad():
        ids=torch.randint(0,full.config.vocab_size,(1,64))
        result=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=True,logits_to_keep=1,return_dict=True)
        cache=result.past_key_values
        def evidence(result,index,length):
            leaves=_tree_arrays(result)
            finite=all(bool(np.isfinite(value).all()) for value in leaves.values())
            entry={'step':index,'context_tokens':length,'all_outputs_finite':finite,'output_arrays':{key:_array_record(value) for key,value in leaves.items()}}
            if not finite:raise AssertionError('Eager result contains NaN/Inf')
            eager.append(entry)
        evidence(result,0,64)
        if cache is None:raise AssertionError('Cached prefill did not return KV')
        for index in range(4):
            token=torch.randint(0,full.config.vocab_size,(1,1))
            result=model(input_ids=token,attention_mask=torch.ones(1,65+index,dtype=torch.long),past_key_values=cache,use_cache=True,logits_to_keep=1,return_dict=True)
            cache=result.past_key_values
            evidence(result,index+1,65+index)
    after_eager=state(model)
    _write_json(output/'eager.json',{'kind':'text_only_teacher_forced_full_model','prefill_tokens':64,'decode_steps':4,'capacity':full.config.max_position_embeddings,'same_model_id':id(model),'state_preserved':initial==after_eager,'state_differences':_first_differences(initial,after_eager),'calls':eager})
    phase('after_eager')
    isolation=compare_snapshots(output/'before',output/'after_eager',atol=1e-5,rtol=1e-4,require_torch_parity=False)
    strict=compare_snapshots(output/'before',output/'after_eager',atol=1e-5,rtol=1e-4,require_torch_parity=True)
    report.update(comparisons_complete=True,before_after_isolation=isolation,strict_torch_circle_validation=strict,frozen_state_preserved=state(model)==initial,status='PASS' if strict['ok'] and state(model)==initial else 'FAIL')
except Exception as exc:
    traceback.print_exc();report.update(error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
_write_json(output/'summary.json',report)
print(json.dumps({'status':report['status'],'comparisons_complete':report['comparisons_complete'],'scope':report['scope'],'frozen_state_preserved':report.get('frozen_state_preserved'),'before_after_equal':report.get('before_after_isolation',{}).get('ok'),'strict_ok':report.get('strict_torch_circle_validation',{}).get('ok'),'error':report.get('error')}))
raise SystemExit({'PASS':0,'FAIL':1,'ERROR':2}[report['status']])
