"""Diagnose existing real Circle parity failures without re-export or mutation."""

import json
from pathlib import Path
import sys

import numpy as np


def error_record(left, right):
    delta=np.abs(left.astype(np.float64)-right.astype(np.float64))
    position=tuple(np.unravel_index(np.argmax(delta),delta.shape)) if delta.size else ()
    bad=~np.isclose(left,right,atol=1e-5,rtol=1e-4)
    return {"shape":list(left.shape),"max_abs_error":float(delta.max()) if delta.size else 0,
        "max_position":list(map(int,position)),"torch_value":float(left[position]) if delta.size else None,
        "circle_value":float(right[position]) if delta.size else None,
        "failing_elements":int(bad.sum()),"first_failing_positions":np.argwhere(bad)[:8].tolist()}


def main():
    root=Path(__file__).resolve().parent
    repo=Path('/workspace/TICO-export-baseline')
    sys.path.insert(0,str(repo))
    import torch
    import tico
    from tico.circle.runtime import CircleReferenceRuntime,ExecutionMode
    from tico.circle.runtime.quantization import fake_quantize_array
    from tico.utils.signature import ModelInputSpec
    from tico.quantization.wrapq.observers.affine_base import AffineObserverBase
    torch.set_num_threads(1)
    model=torch.load(root/'fixtures/baseline_llama_v2.pt',map_location='cpu',weights_only=False).eval()
    result={"repo":str(repo),"tico_file":tico.__file__,"scope":"diagnostic replay of existing baseline Circle files; raw strictFAIL retained", "stages":{}}
    phase=root/'tiny-llama-old-new-v2/baseline/before'
    for stage_path in sorted((phase/'artifacts').glob('*.circle')):
        record_path=phase/'records'/(stage_path.name+'.json')
        metadata=json.loads(record_path.read_text())
        with np.load(phase/'records'/(stage_path.name+'.npz')) as loaded:
            arrays={key:loaded[key].copy() for key in loaded.files}
        stage={"probes":[]}
        result['stages'][stage_path.name]=stage
        for index,probe in enumerate(metadata['probes']):
            outputs=[]
            for output_index in range(len(probe['torch_parity'])):
                a,b=arrays[f'p{index}_torch_{output_index}'],arrays[f'p{index}_circle_{output_index}']
                outputs.append({"output_index":output_index,"role":"hidden" if output_index==0 else 'delta_key' if output_index==1 else 'delta_value',**error_record(a,b)})
            stage['probes'].append({"probe":probe['probe'],"outputs":outputs})
        if 'decoder_layer_' not in stage_path.name:
            continue
        layer_index=int(stage_path.name.split('_')[-1].split('.')[0])
        layer=model.wrapped.model.wrapped.layers[layer_index].wrapped
        adapter=layer.as_export_module('decode' if '_decode_' in stage_path.name else 'prefill',return_kv=True).eval()
        runtime=CircleReferenceRuntime(stage_path.read_bytes())
        specification=ModelInputSpec(stage_path.read_bytes())
        inputs=[arrays[f'p0_input_{i}'].copy() for i in range(len(runtime.program.inputs))]
        names=specification.names
        def run_torch(items):
            values={name:torch.from_numpy(value.copy()) for name,value in zip(names,items)}
            kwargs={"attention_mask":values['attention_mask'],"position_embeddings":(values['position_embeddings_0'],values['position_embeddings_1'])}
            if 'past_key_value_0' in values:
                kwargs['past_key_value']=(values['past_key_value_0'],values['past_key_value_1'])
            with torch.no_grad():
                return tuple(value.numpy() for value in adapter(values['hidden_states'],**kwargs))
        actual=runtime.run(inputs,mode=ExecutionMode.FAKE_QUANTIZE,trace=True)
        eager=run_torch(inputs)
        stage['replay_matches_saved_torch']=[bool(np.array_equal(value,arrays[f'p0_torch_{i}'])) for i,value in enumerate(eager)]
        stage['replay_matches_saved_circle']=[bool(np.array_equal(value,arrays[f'p0_circle_{i}'])) for i,value in enumerate(actual.outputs)]
        stage['input_quantization']=[]
        quantized_inputs=[]
        for name,value,tensor_index in zip(names,inputs,runtime.program.inputs):
            tensor=runtime.program.tensor(tensor_index)
            quantized=fake_quantize_array(value,tensor) if tensor.is_quantized_integer else value
            quantized_inputs.append(quantized)
            stage['input_quantization'].append({"name":name,"quantized_integer":tensor.is_quantized_integer,
                "max_raw_grid_error":float(np.max(np.abs(value-quantized))) if value.size else 0,
                "scale":list(tensor.quantization.scale) if tensor.quantization else None,
                "zero_point":list(tensor.quantization.zero_point) if tensor.quantization else None})
        grid_eager=run_torch(quantized_inputs)
        stage['torch_grid_inputs_vs_circle_raw_inputs']=[error_record(a,b) for a,b in zip(grid_eager,actual.outputs)]
        # Observe genuine observer calls, preserving every original implementation.
        captures=[]
        restorations=[]
        for name,observer in model.named_modules():
            if not isinstance(observer,AffineObserverBase):
                continue
            original=observer.fake_quant
            had_instance='fake_quant' in vars(observer)
            def observed(value, original=original, name=name, observer=observer):
                returned=original(value)
                captures.append((name,observer,returned.detach().numpy().copy()))
                return returned
            restorations.append((observer,original,had_instance))
            observer.fake_quant=observed
        try:
            run_torch(inputs)
        finally:
            for observer,original,had_instance in restorations:
                if had_instance: observer.fake_quant=original
                else: del observer.fake_quant
        matches=[]
        for name,observer,value in captures:
            scale=observer._cached_scale.detach().numpy().reshape(-1)
            zp=observer._cached_zp.detach().numpy().reshape(-1)
            candidates=[]
            for tensor in runtime.program.tensors:
                if tensor.quantization is None or tensor.index not in actual.tensor_values:
                    continue
                actual_value=actual.tensor_values[tensor.index]
                if value.shape!=actual_value.shape:
                    continue
                q=tensor.quantization
                if not np.array_equal(np.asarray(q.scale,dtype=scale.dtype),scale) or not np.array_equal(np.asarray(q.zero_point,dtype=zp.dtype),zp):
                    continue
                candidates.append((float(np.max(np.abs(value-actual_value))) if value.size else 0,tensor))
            if candidates:
                candidates.sort(key=lambda item:item[0])
                best_error,tensor=candidates[0]
                matches.append({"observer":name,"shape":list(value.shape),"candidate_count":len(candidates),
                    "best_tensor":tensor.name,"best_tensor_index":tensor.index,"best_max_abs_error":best_error})
        stage['observer_trace_matching']=matches
    (root/'PARITY_DIAGNOSIS.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({"report":str(root/'PARITY_DIAGNOSIS.json'),"stages":len(result['stages'])}))


if __name__=='__main__':
    main()
