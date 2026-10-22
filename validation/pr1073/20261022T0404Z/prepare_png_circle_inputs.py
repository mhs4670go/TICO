"""Prepare normal HF processor inputs from deterministic PNG files."""

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from transformers.models.gemma4.image_processing_gemma4 import Gemma4ImageProcessor
from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_record(value):
    array=value.detach().cpu().contiguous().numpy()
    return {"shape":list(array.shape),"dtype":str(array.dtype),
        "sha256":hashlib.sha256(array.tobytes()).hexdigest(),
        "finite":bool(np.isfinite(array).all())}


def main():
    torch.set_num_threads(1)
    root=Path(__file__).resolve().parent
    output=root/'png_circle_inputs'
    output.mkdir(exist_ok=False)
    yy,xx=np.meshgrid(np.arange(4,dtype=np.uint8),np.arange(4,dtype=np.uint8),indexing='ij')
    rgb=np.stack((xx*64,yy*64,(xx+yy)*32),axis=-1)
    png=output/'qwen_4x4.png'
    Image.fromarray(rgb).save(png)
    processor=Qwen2VLImageProcessor(patch_size=2,temporal_patch_size=2,merge_size=2,
        size={"shortest_edge":16,"longest_edge":16})
    pixels=processor(images=Image.open(png).convert('RGB'),return_tensors='pt')
    assert tuple(pixels['pixel_values'].shape)==(4,24)
    assert pixels['image_grid_thw'].tolist()==[[1,2,2]]
    source=root/'fixtures/baseline_qwen_cached.image_inputs.pt'
    loaded=torch.load(source,map_location='cpu',weights_only=True)
    calls=loaded if isinstance(loaded,list) else [loaded]
    assert len(calls)==1 and isinstance(calls[0],dict)
    original={key:value.clone() if isinstance(value,torch.Tensor) else value for key,value in calls[0].items()}
    call={**calls[0],"pixel_values":pixels['pixel_values'],"image_grid_thw":pixels['image_grid_thw']}
    for key,value in original.items():
        if key not in ('pixel_values','image_grid_thw') and isinstance(value,torch.Tensor):
            assert torch.equal(value,call[key]),key
    kwargs=output/'qwen_actual_png_kwargs.pt'
    torch.save([call],kwargs)
    processor.save_pretrained(output/'qwen_processor')
    report={"scope":"tiny random frozen Qwen fixture + real PNG + normal HF image processor; one image forward; no metric/pretrained or linked-cache decode claim",
        "source_kwargs":str(source),"source_kwargs_sha256":sha256(source),
        "checkpoint":str(root/'fixtures/baseline_qwen_cached.pt'),
        "checkpoint_sha256":sha256(root/'fixtures/baseline_qwen_cached.pt'),
        "png":str(png),"png_sha256":sha256(png),"image_shape":[4,4,3],
        "processor_class":type(processor).__name__,"processor_config":processor.to_dict(),
        "processed_tensors":{key:tensor_record(value) for key,value in pixels.items()},
        "image_grid_thw":pixels['image_grid_thw'].tolist(),
        "output_kwargs":str(kwargs),"output_kwargs_sha256":sha256(kwargs),
        "preserved_keys":sorted(key for key in original if key not in ('pixel_values','image_grid_thw')),
        "kwargs_tensors":{key:tensor_record(value) for key,value in call.items() if isinstance(value,torch.Tensor)}}
    try:
        Gemma4ImageProcessor(patch_size=4,pooling_kernel_size=2,max_soft_tokens=4)
    except ValueError as exc:
        report['gemma_existing_fixture']={"status":"BLOCKED_PROCESSOR_PROFILE_MISMATCH",
            "error_type":type(exc).__name__,"message":str(exc),
            "existing_fixture_max_soft_tokens":4}
    else:
        raise AssertionError('Expected installed HF Gemma processor to reject max_soft_tokens=4')
    # This processor probe proposes a new profile; it is not applied to a model.
    yy,xx=np.meshgrid(np.arange(16,dtype=np.uint8),np.arange(16,dtype=np.uint8),indexing='ij')
    gemma_png=output/'gemma_16x16_profile_proposal.png'
    Image.fromarray(np.stack((xx*16,yy*16,((xx.astype(np.uint16)+yy)*8).astype(np.uint8)),axis=-1)).save(gemma_png)
    gemma=Gemma4ImageProcessor(patch_size=4,pooling_kernel_size=2,max_soft_tokens=70,do_resize=False)
    gemma_pixels=gemma(images=Image.open(gemma_png).convert('RGB'),return_tensors='pt')
    gemma.save_pretrained(output/'gemma_processor_profile_proposal')
    assert tuple(gemma_pixels['pixel_values'].shape)==(1,280,48)
    assert tuple(gemma_pixels['image_position_ids'].shape)==(1,280,2)
    assert torch.as_tensor(gemma_pixels['num_soft_tokens_per_image']).tolist()==[4]
    torch.save(dict(gemma_pixels),output/'gemma_processor_outputs_profile_proposal.pt')
    report['gemma_new_profile_proposal']={"status":"PROCESSOR_ONLY_COMPLETED_MODEL_NOT_RUN",
        "png":str(gemma_png),"png_sha256":sha256(gemma_png),
        "processor_config":gemma.to_dict(),
        "processed_tensors":{key:tensor_record(value) for key,value in gemma_pixels.items()},
        "num_soft_tokens_per_image":torch.as_tensor(gemma_pixels['num_soft_tokens_per_image']).tolist(),
        "required_new_checkpoint_config":{"vision_config.default_output_length":70,
            "model_args.vision.max_soft_tokens":70,"model_args.vision.num_visual_tokens":4,
            "model_args.vision.patch_grid_height":4,"model_args.vision.patch_grid_width":4},
        "limitation":"A new profile-compatible frozen checkpoint calibrated with these actual processor outputs is required; existing max_soft_tokens=4 checkpoint is not modified."}
    (output/'inputs_manifest.json').write_text(json.dumps(report,indent=2,default=str,allow_nan=False)+'\n')
    print(json.dumps({"manifest":str(output/'inputs_manifest.json'),
        "qwen_kwargs":str(kwargs),"qwen_pixels_shape":list(pixels['pixel_values'].shape),
        "qwen_grid":pixels['image_grid_thw'].tolist(),
        "gemma_existing_status":report['gemma_existing_fixture']['status'],
        "gemma_new_processor_shape":list(gemma_pixels['pixel_values'].shape)},indent=2))


if __name__=='__main__':
    main()
