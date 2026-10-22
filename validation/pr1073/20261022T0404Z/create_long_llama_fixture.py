"""Create deterministic tiny real-HF PTQ checkpoints for export validation only.

These random fixtures are not pretrained quality measurements. The same saved
checkpoint is supplied to both export workers; no independent recalibration is
performed by either worker.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--family", choices=("llama", "qwen3_vl", "gemma4"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--capacity", type=int, default=64)
    args = parser.parse_args()
    repo, output = Path(args.repo).resolve(), Path(args.output).resolve()
    if output.exists():
        parser.error("Choose a new checkpoint path.")
    sys.path.insert(0, str(repo))
    import torch
    import tico
    import yaml
    from test.quantization.quant_spec_helpers import make_affine_ptq_config
    from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper

    assert Path(tico.__file__).resolve().is_relative_to(repo), tico.__file__
    torch.set_num_threads(1)
    torch.manual_seed(1073)
    image_calls = []
    if args.family == "llama":
        from transformers.models.llama.configuration_llama import LlamaConfig
        from transformers.models.llama.modeling_llama import LlamaForCausalLM
        from tico.quantization.wrapq.wrappers.llama.quant_model_for_causal_lm import QuantLlamaForCausalLM

        config = LlamaConfig(
            vocab_size=64, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
            head_dim=16, max_position_embeddings=args.capacity,
            rope_theta=500000.0,
            rope_scaling={"rope_type": "llama3", "factor": 32.0,
                          "low_freq_factor": 1.0, "high_freq_factor": 4.0,
                          "original_max_position_embeddings": 8192},
            attention_dropout=0.0, use_cache=True,
        )
        config._attn_implementation = "eager"
        model_args = {"profile": "npu_export"}
        model = PTQWrapper(LlamaForCausalLM(config).eval(),
            qcfg=make_affine_ptq_config(model_args=model_args)).eval()
    elif args.family == "qwen3_vl":
        from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
        from tico.quantization.wrapq.wrappers.qwen_vl.quant_for_conditional_generation import QuantQwen3VLForConditionalGeneration

        config = Qwen3VLConfig(
            text_config=dict(vocab_size=64, hidden_size=32, intermediate_size=64,
                num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                head_dim=8, max_position_embeddings=args.capacity,
                rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 2]},
                use_cache=True, attention_dropout=0.0),
            vision_config=dict(hidden_size=32, intermediate_size=64, num_heads=4,
                depth=1, temporal_patch_size=2, patch_size=2, out_hidden_size=32,
                spatial_merge_size=2, deepstack_visual_indexes=[0]),
            image_token_id=62, video_token_id=63, vision_start_token_id=61,
        )
        config.text_config._attn_implementation = "eager"
        config.vision_config._attn_implementation = "eager"
        model_args = {"profile": "npu_export", "vision": {
            "grid_thw": (1, 2, 2), "visual_start_idx": 0,
            "spatial_merge_size": 2}}
        model = PTQWrapper(Qwen3VLForConditionalGeneration(config).eval(),
            qcfg=make_affine_ptq_config(model_args=model_args)).eval()
        ids = torch.tensor([[62, 4, 5, 6, 7, 8, 9, 10]])
        positions = torch.arange(8).view(1, 1, -1).expand(3, 1, -1)
        image_calls = [{"input_ids": ids, "attention_mask": torch.ones_like(ids),
            "pixel_values": torch.randn(1, 4, 24),
            "image_grid_thw": torch.tensor([[1, 2, 2]]),
            "position_ids": positions, "use_cache": False, "return_dict": True}]
    else:
        from transformers.models.gemma4.configuration_gemma4 import Gemma4Config, Gemma4TextConfig, Gemma4VisionConfig
        from transformers.models.gemma4.modeling_gemma4 import Gemma4ForConditionalGeneration
        from tico.quantization.wrapq.wrappers.gemma4.quant_for_conditional_generation import QuantGemma4ForConditionalGeneration

        text = Gemma4TextConfig.from_dict(dict(vocab_size=64, hidden_size=32,
            intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
            num_key_value_heads=1, num_global_key_value_heads=1,
            head_dim=16, global_head_dim=16, max_position_embeddings=args.capacity,
            layer_types=["sliding_attention", "full_attention"], sliding_window=8,
            rope_parameters={"sliding_attention": {"rope_type": "default", "rope_theta":10000.0},
                             "full_attention": {"rope_type": "default", "rope_theta":10000.0}},
            attention_bias=False, attention_dropout=0.0, use_cache=True,
            enable_moe_block=False, hidden_size_per_layer_input=0,
            num_kv_shared_layers=0, attention_k_eq_v=False))
        vision = Gemma4VisionConfig.from_dict(dict(hidden_size=32, intermediate_size=64,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
            head_dim=8, patch_size=4, position_embedding_size=8,
            pooling_kernel_size=2, default_output_length=4,
            attention_dropout=0.0, max_position_embeddings=128,
            rms_norm_eps=1e-6, use_clipped_linears=False,
            rope_parameters={"rope_type":"default", "rope_theta":100.0}, standardize=True))
        text._attn_implementation = vision._attn_implementation = "eager"
        config = Gemma4Config.from_dict(dict(text_config=text, vision_config=vision,
            audio_config=None, image_token_id=62, video_token_id=63, audio_token_id=61))
        model_args = {"text": {"max_seq": 8}, "vision": {
            "visual_start_idx": 0, "num_visual_tokens": 4, "max_soft_tokens":4,
            "patch_grid_height":4, "patch_grid_width":4, "validate_static_layout":False}}
        model = PTQWrapper(Gemma4ForConditionalGeneration(config).eval(),
            qcfg=make_affine_ptq_config(model_args=model_args)).eval()
        ids = torch.tensor([[62,62,62,62,4,5,6,7]])
        yy, xx = torch.meshgrid(torch.arange(4), torch.arange(4), indexing="ij")
        image_calls = [{"input_ids": ids,
            "pixel_values": torch.randn(1,16,48),
            "image_position_ids":torch.stack((xx.flatten(),yy.flatten()),dim=-1).unsqueeze(0),
            "use_cache":False, "return_dict":True}]

    model.enable_calibration()
    calibration = []
    with torch.no_grad():
        for length in (8, 6, 4):
            sample = {"input_ids":torch.randint(1,50,(1,length)),
                      "use_cache":False, "return_dict":True}
            model(**sample)
            calibration.append({"kind":"text_only", "seq_len":length})
        cached = model(input_ids=torch.randint(1,50,(1,4)), use_cache=True, return_dict=True)
        assert cached.past_key_values is not None
        cache = cached.past_key_values
        for _ in range(2):
            cached = model(input_ids=torch.randint(1,50,(1,1)), past_key_values=cache,
                           use_cache=True, return_dict=True)
            assert torch.isfinite(cached.logits).all()
            cache = cached.past_key_values
        calibration.append({"kind":"cached_text", "prefill_tokens":4,"decode_steps":2})
        for sample in image_calls:
            if args.family == "gemma4":
                from tico.quantization.wrapq.wrappers.gemma4.static_vision_profile import build_gemma4_static_vision_profile
                profile = build_gemma4_static_vision_profile(model_args,
                    vision_config=config.vision_config, max_seq_len=8)
                profile.validate_processor_outputs(sample, image_token_id=config.image_token_id)
                profile.validate_wrapped_model(model.wrapped.model.wrapped)
                features = model.wrapped.model.wrapped.get_image_features(
                    sample["pixel_values"], image_position_ids=sample["image_position_ids"])
                assert features.shape[-2] == model_args["vision"]["num_visual_tokens"]
                assert features.shape[-1] == config.text_config.hidden_size
                calibration.append({"kind":"manual_vision_contract", "shape":list(features.shape),
                    "effective_validate_static_layout":False,"profile":profile.to_manifest()})
            model(**sample)
            calibration.append({"kind":"synthetic_image", "seq_len":sample["input_ids"].shape[-1]})
    model.freeze_qparams()
    # Check that quantized execution completes before saving the trusted fixture.
    with torch.no_grad():
        result = model(input_ids=torch.randint(1,50,(1,8)), use_cache=True, return_dict=True)
        assert torch.isfinite(result.logits).all()
        assert result.past_key_values is not None
        for sample in image_calls:
            result = model(**sample)
            assert torch.isfinite(result.logits).all()
    output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(model,output)
    if image_calls:
        torch.save(image_calls,output.with_suffix(".image_inputs.pt"))
    recipe = {"model":{"family":args.family,"name_or_path":"tiny-random-local-validation-fixture"},
        "runtime":{"device":"cpu","dtype":"float32","seed":42,"show_progress":False},
        "model_args":model_args, "calibration":{"seq_len":8}, "pipeline":[],
        "evaluation":{"enabled":False},
        "export":{"enabled":True,"max_seq_len":8,"prefill_decode":True,"strict":False,
                  "artifacts":["circle_bundle" if args.family == "llama" else "circle_per_layer"]}}
    output.with_suffix(".export.yaml").write_text(yaml.safe_dump(recipe,sort_keys=False))
    commit = subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip()
    metadata = {"scope":"tiny random real-HF PTQ fixture; not pretrained validation",
        "family":args.family, "repo":str(repo), "commit":commit,
        "tico_file":tico.__file__, "checkpoint":str(output),
        "checkpoint_sha256":hashlib.sha256(output.read_bytes()).hexdigest(),
        "checkpoint_bytes":output.stat().st_size, "seed":1073,
        "dtype":"float32", "device":"cpu", "native_capacity":args.capacity,
        "calibration":calibration, "model_args":model_args, "hf_config":config.to_dict(),
        "code_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    output.with_suffix(".provenance.json").write_text(json.dumps(metadata,indent=2,default=str)+"\n")
    print(json.dumps({key:value for key,value in metadata.items() if key!="hf_config"},indent=2))


if __name__ == "__main__":
    main()
