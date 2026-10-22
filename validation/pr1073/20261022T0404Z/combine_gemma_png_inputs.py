"""Bind normal HF Gemma PNG processor outputs to a matching new fixture."""
import hashlib
import json
from pathlib import Path
import torch


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root=Path(__file__).resolve().parent
    output=root/'png_circle_inputs'
    checkpoint=root/'fixtures/baseline_gemma_image70.pt'
    source=root/'fixtures/baseline_gemma_image70.image_inputs.pt'
    provenance=json.loads(checkpoint.with_suffix('.provenance.json').read_text())
    assert provenance['model_args']['vision']['max_soft_tokens']==70
    assert provenance['hf_config']['vision_config']['default_output_length']==70
    original=torch.load(source,map_location='cpu',weights_only=True)
    calls=original if isinstance(original,list) else [original]
    assert len(calls)==1 and isinstance(calls[0],dict)
    processed=torch.load(output/'gemma_processor_outputs_profile_proposal.pt',map_location='cpu',weights_only=True)
    assert tuple(processed['pixel_values'].shape)==(1,280,48)
    assert tuple(processed['image_position_ids'].shape)==(1,280,2)
    assert tuple(calls[0]['pixel_values'].shape)==tuple(processed['pixel_values'].shape)
    assert tuple(calls[0]['image_position_ids'].shape)==tuple(processed['image_position_ids'].shape)
    call={**calls[0], 'pixel_values':processed['pixel_values'],
                      'image_position_ids':processed['image_position_ids']}
    for key,value in calls[0].items():
        if key not in ('pixel_values','image_position_ids') and isinstance(value,torch.Tensor):
            assert torch.equal(value,call[key]),key
    destination=output/'gemma_actual_png_kwargs.pt'
    assert not destination.exists()
    torch.save([call],destination)
    report={'scope':'tiny frozen Gemma image70 fixture + actual PNG + normal HF processor; one image forward; no metric/pretrained/linked-cache decode claim',
        'checkpoint':str(checkpoint),'checkpoint_sha256':sha256(checkpoint),
        'source_kwargs':str(source),'source_kwargs_sha256':sha256(source),
        'png':str(output/'gemma_16x16_profile_proposal.png'),
        'png_sha256':sha256(output/'gemma_16x16_profile_proposal.png'),
        'processor_config':str(output/'gemma_processor_profile_proposal/preprocessor_config.json'),
        'processor_config_sha256':sha256(output/'gemma_processor_profile_proposal/preprocessor_config.json'),
        'output_kwargs':str(destination),'output_kwargs_sha256':sha256(destination),
        'preserved_keys':sorted(key for key in calls[0] if key not in ('pixel_values','image_position_ids')),
        'pixel_values_shape':list(processed['pixel_values'].shape),
        'image_position_ids_shape':list(processed['image_position_ids'].shape),
        'num_soft_tokens_per_image':processed['num_soft_tokens_per_image'].tolist(),
        'normal_processor_positions_equal_synthetic_template':bool(torch.equal(calls[0]['image_position_ids'],processed['image_position_ids'])),
        'max_soft_tokens':70,'valid_visual_tokens':4}
    (output/'gemma_inputs_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
