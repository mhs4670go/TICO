"""Save one genuine Gemma Circle before reproducing the runtime failure."""

import hashlib
import json
from pathlib import Path
import resource
import sys
import traceback

import numpy as np


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root=Path(__file__).resolve().parent
    output=root/'gemma-vision-failure-artifact'
    current_memory=int(Path('/sys/fs/cgroup/memory.current').read_text())
    if current_memory >= 12*1024**3:
        print(json.dumps({'status':'BLOCKED_RESOURCE','memory_current_bytes':current_memory}))
        return 2
    output.mkdir(exist_ok=False)
    sys.path.insert(0,'/workspace/TICO')
    import torch
    import tico
    from tico.circle import CircleDocument
    from tico.circle.runtime import CircleReferenceRuntime,ExecutionMode
    from tico.quantization.recipes.debug.export_regression import _state_contract,_tree_arrays,circle_contract,compare_arrays
    from tico.quantization.wrapq.wrappers.gemma4.export_adapters import build_gemma4_vision_prefill_export_module
    assert Path(tico.__file__).resolve().is_relative_to('/workspace/TICO')
    torch.set_num_threads(1)
    checkpoint=root/'fixtures/baseline_gemma_image70.pt'
    input_file=root/'png_circle_inputs/gemma_actual_png_kwargs.pt'
    model=torch.load(checkpoint,map_location='cpu',weights_only=False).eval()
    calls=torch.load(input_file,map_location='cpu',weights_only=True)
    kwargs=calls[0] if isinstance(calls,list) else calls
    before=_state_contract(model)
    (output/'frozen_before.json').write_text(json.dumps(before,indent=2)+'\n')
    gemma_model=model.wrapped.model.wrapped
    adapter=build_gemma4_vision_prefill_export_module(gemma_model,
        pixel_position_ids=kwargs['image_position_ids']).eval()
    pixel_values=kwargs['pixel_values']
    report={'status':'ERROR','scope':'one actual tiny Gemma vision Circle conversion and real reference-runtime reproduction; not completed Circle PASS, pretrained, full VLM, or NPU validation',
        'tico_file':tico.__file__,'checkpoint':str(checkpoint),'checkpoint_sha256':sha256(checkpoint),
        'inputs':str(input_file),'inputs_sha256':sha256(input_file),
        'pixels_shape':list(pixel_values.shape),'image_position_ids_shape':list(kwargs['image_position_ids'].shape),
        'memory_current_before_bytes':current_memory,'atol':1e-5,'rtol':1e-4}
    code=2
    try:
        with torch.no_grad():
            golden=_tree_arrays(adapter(pixel_values))
            assert golden and all(np.isfinite(value).all() for value in golden.values())
            np.savez(output/'torch_outputs.npz',**golden)
            report['torch_outputs']={name:{'shape':list(value.shape),'dtype':str(value.dtype),
                'finite':bool(np.isfinite(value).all()),'sha256':hashlib.sha256(value.tobytes()).hexdigest()}
                for name,value in golden.items()}
            converted=tico.convert(adapter,(pixel_values,),strict=False)
        circle_path=output/'vision_prefill.q.circle'
        converted.save(circle_path)
        report['saved_circle']=str(circle_path)
        report['saved_circle_sha256']=sha256(circle_path)
        report['saved_circle_bytes']=circle_path.stat().st_size
        with CircleDocument.load_mapped(circle_path) as document:
            document.verify(raise_on_error=True)
            contract=circle_contract(document)
        (output/'circle_contract.json').write_text(json.dumps(contract,indent=2)+'\n')
        report['circle_structurally_verified']=True
        runtime=CircleReferenceRuntime(converted.circle_binary)
        mode=ExecutionMode.FAKE_QUANTIZE if runtime.has_quantized_activations() else ExecutionMode.NATIVE
        report['runtime_mode']=mode.value
        actual=runtime.run([pixel_values.detach().numpy()],mode=mode).outputs
        np.savez(output/'circle_outputs.npz',**{str(index):value for index,value in enumerate(actual)})
        comparisons=[compare_arrays(left,right,atol=1e-5,rtol=1e-4) for left,right in zip(golden.values(),actual)]
        report['torch_parity']=comparisons
        success=len(golden)==len(actual) and all(item['ok'] for item in comparisons)
        report['status']='PASS' if success else 'FAIL'
        report['scope']='one actual tiny Gemma vision stage only; no full VLM/pretrained/NPU claim'
        code=0 if success else 1
    except Exception as exc:
        traceback.print_exc()
        report.update(error_type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc(),
            classification='REFERENCE_RUNTIME_UNSUPPORTED_OPERATOR_OR_SHAPE',comparisons_complete=False)
    finally:
        after=_state_contract(model)
        report['frozen_state_preserved']=before==after
        (output/'frozen_after.json').write_text(json.dumps(after,indent=2)+'\n')
        report['memory_current_after_bytes']=int(Path('/sys/fs/cgroup/memory.current').read_text())
        report['process_max_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        (output/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(json.dumps({key:report.get(key) for key in ('status','saved_circle','saved_circle_sha256','error_type','message','runtime_mode','frozen_state_preserved')},indent=2))
    return code


if __name__=='__main__':
    raise SystemExit(main())
