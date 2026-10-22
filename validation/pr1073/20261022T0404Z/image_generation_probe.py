"""Probe normal tiny VLM image generation under identical FP/Quant conditions."""
import argparse
import copy
import hashlib
import json
import traceback
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import tico

from generated_image_eager import setup
from supplemental_eager_state import snapshot, tensor_hash
from tico.quantization.recipes.context import RecipeContext
from tico.quantization.recipes.evaluation.eager import prepare_eager_evaluation
from tico.quantization.recipes.adapters.qwen3_vl import Qwen3VLAdapter
from tico.quantization.recipes.adapters.gemma4 import Gemma4Adapter


def state_diff(before, after):
    differences = {}
    for group in before:
        changed = [name for name in set(before[group]) | set(after[group]) if before[group].get(name) != after[group].get(name)]
        if changed:
            differences[group] = sorted(changed)
    # Qwen stores per-request mRoPE deltas as a nonpersistent inference buffer.
    frozen_differences = {group: [name for name in names if not (group == "buffers" and name.endswith("rope_deltas"))] for group, names in differences.items()}
    frozen_differences = {group: names for group, names in frozen_differences.items() if names}
    return {"all_changes": differences, "frozen_changes": frozen_differences, "frozen_state": "PASS" if not frozen_differences else "FAIL", "allowed_request_buffer": "rope_deltas only; all raw changes remain recorded"}


def attempt(model, sample, settings, output, label):
    torch.manual_seed(1073)
    try:
        with torch.no_grad():
            tokens = model.generate(**copy.deepcopy(sample), **settings)
        torch.save(tokens, output / f"{label}_tokens.pt")
        return {"status": "PASS", "tokens": tokens.tolist(), "token_tensor": tensor_hash(tokens), "new_tokens": tokens.shape[1] - sample["input_ids"].shape[1]}
    except Exception as error:
        traceback.print_exc()
        return {"status": "FAIL", "error": repr(error), "traceback": traceback.format_exc()}


def run(family, image, output):
    torch.manual_seed(1073)
    base, base_wrapper, sample, processor, _ = setup(family, image)
    if family == "qwen3_vl":
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
        from tico.quantization.wrapq.wrappers.qwen_vl.quant_for_conditional_generation import QuantQwen3VLForConditionalGeneration
        fp = Qwen3VLForConditionalGeneration(copy.deepcopy(base.config)).eval()
        wrapper_class = QuantQwen3VLForConditionalGeneration
        adapter = Qwen3VLAdapter()
        image_id = base.config.image_token_id
    else:
        from transformers.models.gemma4.modeling_gemma4 import Gemma4ForConditionalGeneration
        from tico.quantization.wrapq.wrappers.gemma4.quant_for_conditional_generation import QuantGemma4ForConditionalGeneration
        fp = Gemma4ForConditionalGeneration(copy.deepcopy(base.config)).eval()
        wrapper_class = QuantGemma4ForConditionalGeneration
        adapter = Gemma4Adapter()
        image_id = base.config.image_token_id
    fp.model.load_state_dict(base.state_dict())
    wrapped = wrapper_class(copy.deepcopy(fp), qcfg=base_wrapper.qcfg).eval()
    sample["mm_token_type_ids"] = (sample["input_ids"] == image_id).long()
    # Native generation computes positions and connects its own KV cache.
    sample.pop("position_ids", None)
    wrapped.enable_calibration()
    with torch.no_grad():
        wrapped(**sample, use_cache=False)
    wrapped.freeze_qparams()
    for label, model in (("fp", fp), ("quant", wrapped)):
        ctx = RecipeContext(cfg={"model": {"family": family, "name_or_path": "tiny-random-image-generation"}, "runtime": {"seed": 1073}, "evaluation": {"max_seq_len": 64, "use_cache": True, "output_dir": str(output / label)}, "calibration": {"seq_len": int(sample["input_ids"].shape[1])}, "export": {"max_seq_len": 6}}, adapter=adapter, model=model)
        prepare_eager_evaluation(ctx)
    settings = {"max_new_tokens": 2, "min_new_tokens": 2, "do_sample": False, "num_beams": 1, "use_cache": True, "pad_token_id": 0, "eos_token_id": None}
    checkpoint = output / "frozen_full_module.pt"
    torch.save(wrapped, checkpoint)
    torch.save({"sample": sample, "settings": settings}, output / "generation_inputs.pt")
    (output / "model_config.json").write_text(json.dumps(fp.config.to_dict(), indent=2, default=str) + "\n")
    before = snapshot(wrapped)
    fp_result = attempt(fp, sample, settings, output, "fp")
    quant_result = attempt(wrapped, sample, settings, output, "quant")
    after = snapshot(wrapped)
    state = state_diff(before, after)
    (output / "state.json").write_text(json.dumps({"before": before, "after": after, "comparison": state}, indent=2) + "\n")
    return {"status": "PASS" if fp_result["status"] == quant_result["status"] == state["frozen_state"] == "PASS" else "FAIL", "scope": "tiny random real-HF image generation; not pretrained metric", "seed": 1073, "dtype": "torch.float32", "processor": type(processor).__name__, "settings": settings, "input_hashes": {name: tensor_hash(value) for name, value in sample.items()}, "fp": fp_result, "quant": quant_result, "state": state, "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(), "tico_file": tico.__file__, "quant_class": f"{wrapper_class.__module__}.{wrapper_class.__name__}"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    image = Image.open(Path(__file__).parent / "generated_image_eager/generated_input.png").convert("RGB")
    result = {"scope": "Actual normal generate calls, local HF image processors, no monkeypatches or product changes", "models": {}}
    for family in ("qwen3_vl", "gemma4"):
        directory = args.output / family
        directory.mkdir(exist_ok=True)
        try:
            result["models"][family] = run(family, image, directory)
        except Exception as error:
            traceback.print_exc()
            result["models"][family] = {"status": "FAIL", "setup_error": repr(error), "traceback": traceback.format_exc()}
        print(family, result["models"][family]["status"], flush=True)
        (args.output / "summary.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    raise SystemExit(0 if all(item["status"] == "PASS" for item in result["models"].values()) else 1)


if __name__ == "__main__":
    main()
