"""Execute generated PNGs through local HF image processors and tiny real VLMs."""
import argparse
import copy
import hashlib
import json
import traceback
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from supplemental_eager_state import assert_same, snapshot, tensor_hash
from test.quantization.quant_spec_helpers import make_affine_ptq_config


def setup(family, image):
    if family == "qwen3_vl":
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
        from test.quantization.wrapq.wrappers.qwen_vl.test_quant_model import TestQuantQwen3VLModel
        from tico.quantization.wrapq.wrappers.qwen_vl.quant_model import QuantQwen3VLModel
        TestQuantQwen3VLModel.setUpClass()
        fixture = TestQuantQwen3VLModel()
        processor = Qwen2VLImageProcessor(patch_size=16, temporal_patch_size=2, merge_size=2, size={"shortest_edge": 4096, "longest_edge": 4096})
        processed = processor(images=image, return_tensors="pt")
        grid = tuple(processed["image_grid_thw"][0].tolist())
        count = int(np.prod(grid)) // 4
        ids = torch.arange(1, count + 9).view(1, -1)
        ids[:, :count] = fixture.image_token_id
        positions = fixture._compute_3d_position_ids(ids, grid, 2, fixture.image_token_id)
        sample = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "pixel_values": processed["pixel_values"], "image_grid_thw": processed["image_grid_thw"], "position_ids": positions}
        fp = copy.deepcopy(fixture.fp_model).eval()
        qcfg = make_affine_ptq_config(model_args={"vision": {"grid_thw": grid, "visual_start_idx": 0, "spatial_merge_size": 2}})
        wrapped = QuantQwen3VLModel(copy.deepcopy(fp), qcfg=qcfg).eval()
        return fp, wrapped, sample, processor, processed
    from transformers.models.gemma4.image_processing_gemma4 import Gemma4ImageProcessor
    from test.quantization.wrapq.wrappers.gemma4.test_quant_model import _make_gemma4_model
    from tico.quantization.wrapq.wrappers.gemma4.quant_model import QuantGemma4Model
    processor = Gemma4ImageProcessor(patch_size=4, pooling_kernel_size=2, max_soft_tokens=70, do_resize=False)
    processed = processor(images=image.resize((16, 16)), return_tensors="pt")
    count = int(processed["num_soft_tokens_per_image"][0])
    ids = torch.arange(20, 20 + count + 8).view(1, -1)
    ids[:, :count] = 10
    sample = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "pixel_values": processed["pixel_values"], "image_position_ids": processed["image_position_ids"]}
    fp = _make_gemma4_model()
    qcfg = make_affine_ptq_config(model_args={"vision": {"visual_start_idx": 0, "num_visual_tokens": count}, "text": {"max_seq": 6}})
    wrapped = QuantGemma4Model(copy.deepcopy(fp), qcfg=qcfg).eval()
    return fp, wrapped, sample, processor, processed


def execute(family, image, output):
    torch.manual_seed(1073)
    fp, wrapped, sample, processor, processed = setup(family, image)
    with torch.no_grad():
        fp_output = fp(**sample, use_cache=False).last_hidden_state
        no_quant = wrapped(**sample, use_cache=False).last_hidden_state
        torch.testing.assert_close(no_quant, fp_output, atol=1e-5, rtol=1e-4)
        wrapped.enable_calibration()
        wrapped(**sample, use_cache=False)
        wrapped.freeze_qparams()
        before = snapshot(wrapped)
        quant_output = wrapped(**sample, use_cache=False).last_hidden_state
        assert torch.isfinite(quant_output).all()
        prefill = wrapped(**sample, use_cache=True)
        cache = prefill.past_key_values
        assert cache is not None
        steps = []
        original_length = sample["input_ids"].shape[1]
        for index in range(4):
            kwargs = {"input_ids": torch.tensor([[40 + index]]), "attention_mask": torch.ones(1, original_length + index + 1, dtype=torch.long), "past_key_values": cache, "use_cache": True}
            if family == "qwen3_vl":
                kwargs["position_ids"] = torch.full((3, 1, 1), int(sample["position_ids"].max()) + index + 1, dtype=torch.long)
            decoded = wrapped(**kwargs)
            cache = decoded.past_key_values
            assert cache is not None
            assert cache.get_seq_length() == original_length + index + 1
            assert torch.isfinite(decoded.last_hidden_state).all()
            steps.append(tensor_hash(decoded.last_hidden_state))
        after = snapshot(wrapped)
        assert_same(before, after)
    (output / f"{family}_state.json").write_text(json.dumps({"before": before, "after": after}, indent=2) + "\n")
    torch.save({"sample": sample, "fp": fp_output, "no_quant": no_quant, "quant": quant_output}, output / f"{family}_tensors.pt")
    return {"status": "PASS", "level": "tiny-real-HF-generated-PNG-real-processor-PTQ", "pretrained": False, "benchmark": False, "circle": False, "processor": type(processor).__name__, "processor_config": processor.to_dict(), "inputs": {name: tensor_hash(value) for name, value in sample.items()}, "processor_outputs": {name: list(value.shape) if isinstance(value, torch.Tensor) else value for name, value in processed.items()}, "no_quant_hf_max_abs_error": float((no_quant - fp_output).abs().max()), "quant_fp_max_abs_delta": float((quant_output - fp_output).abs().max()), "quant_finite": True, "connected_cached_decode_steps": steps, "frozen_state": "PASS"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    yy, xx = np.mgrid[:64, :64]
    rgb = np.stack(((xx * 4) % 256, (yy * 4) % 256, ((xx // 8 + yy // 8) % 2) * 255), axis=-1).astype(np.uint8)
    image = Image.fromarray(rgb)
    image_path = args.output / "generated_input.png"
    image.save(image_path)
    result = {"scope": "Generated PNG with real HF image processors, tiny randomly initialized models, actual vision+text eager and connected cached decode. No pretrained task score or Circle isolation claim.", "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(), "models": {}}
    for family in ("qwen3_vl", "gemma4"):
        try:
            result["models"][family] = execute(family, image, args.output)
        except Exception as error:
            traceback.print_exc()
            result["models"][family] = {"status": "FAIL", "error": repr(error), "traceback": traceback.format_exc()}
        print(family, result["models"][family]["status"], flush=True)
        (args.output / "summary.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    raise SystemExit(0 if all(value["status"] == "PASS" for value in result["models"].values()) else 1)


if __name__ == "__main__":
    main()
