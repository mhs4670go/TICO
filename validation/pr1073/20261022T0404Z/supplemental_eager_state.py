"""Validate tiny real-HF eager state preservation without network or pretrained claims."""
import argparse
import copy
import hashlib
import json
import traceback
from pathlib import Path

import torch

from tico.quantization.recipes.adapters.gemma4 import Gemma4Adapter
from tico.quantization.recipes.adapters.llama import LlamaAdapter
from tico.quantization.recipes.adapters.qwen3_vl import Qwen3VLAdapter
from tico.quantization.recipes.context import RecipeContext
from tico.quantization.recipes.evaluation.eager import prepare_eager_evaluation
from tico.quantization.recipes.sequence_length import resolve_evaluation_max_seq_len, resolve_export_max_seq_len
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.utils.static_export import prepared_static_export
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from test.quantization.quant_spec_helpers import make_affine_ptq_config


def tensor_hash(tensor):
    value = tensor.detach().cpu().contiguous()
    return {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()}


def snapshot(model):
    """Include weights, all buffers, observer flags, lifecycle modes and profiles."""
    return {
        "parameters": {name: tensor_hash(value) for name, value in model.named_parameters()},
        "buffers": {name: tensor_hash(value) for name, value in model.named_buffers()},
        "observers": {name: {"enabled": observer.enabled, "state": {key: tensor_hash(value) for key, value in observer.state_dict().items()}} for name, observer in model.named_observers()},
        "quant_modules": {name: {"mode": str(module._mode), "profile": repr(getattr(getattr(module, "qcfg", None), "model_args", None)), "return_type": getattr(module, "return_type", None), "force_export": getattr(module, "force_export", None)} for name, module in model.named_modules() if isinstance(module, QuantModuleBase)},
    }


def assert_same(before, after):
    """Keep any mismatch localized in the JSON failure evidence."""
    differences = []
    for group in before:
        for key in set(before[group]) | set(after[group]):
            if before[group].get(key) != after[group].get(key):
                differences.append(f"{group}.{key}")
    if differences:
        raise AssertionError("Frozen state changed: " + ", ".join(differences))


def positional_config(model):
    config = getattr(model.config, "text_config", model.config)
    return {name: copy.deepcopy(getattr(config, name, None)) for name in ("max_position_embeddings", "rope_scaling", "rope_parameters")}


def context(family, model, directory):
    adapter = {"llama": LlamaAdapter, "qwen3_vl": Qwen3VLAdapter, "gemma4": Gemma4Adapter}[family]()
    return RecipeContext(cfg={"model": {"family": family, "name_or_path": "random-local-config-no-pretrained"}, "calibration": {"seq_len": 3}, "export": {"max_seq_len": 6}, "runtime": {"seed": 1073, "device": "cpu"}, "evaluation": {"max_seq_len": None, "output_dir": str(directory)}}, adapter=adapter, model=model, artifacts={"pretrained_context_length": model.config.max_position_embeddings})


def make_model(family):
    if family == "llama":
        from test.quantization.wrapq.wrappers.llama.test_eager_context import TestLlamaEagerContext
        fixture = TestLlamaEagerContext()
        return fixture._wrapper(fixture._model(32, scaled_rope=True), "npu_export")
    if family == "qwen3_vl":
        from test.quantization.wrapq.wrappers.qwen_vl.test_eager_context import TestQwen3VLEagerContext
        from tico.quantization.wrapq.wrappers.qwen_vl.quant_text_model import QuantQwen3VLTextModel
        return QuantQwen3VLTextModel(TestQwen3VLEagerContext._make_model(32), qcfg=make_affine_ptq_config()).eval()
    from test.quantization.wrapq.wrappers.gemma4.test_quant_text_model import _make_text_config, _make_text_model
    from tico.quantization.wrapq.wrappers.gemma4.quant_text_model import QuantGemma4TextModel
    cfg = _make_text_config(layer_types=["sliding_attention", "full_attention"], sliding_window=4, max_position_embeddings=32)
    return QuantGemma4TextModel(_make_text_model(cfg), qcfg=make_affine_ptq_config(model_args={"text": {"max_seq": 6}})).eval()


def run_family(family, directory):
    torch.manual_seed(1073)
    model = make_model(family)
    config_before = positional_config(model)
    ids = torch.arange(1, 17).view(1, -1) % model.config.vocab_size
    model.enable_calibration()
    with torch.no_grad():
        model(input_ids=ids[:, :3], use_cache=False)
    ctx = context(family, model, directory / family)
    try:
        prepare_eager_evaluation(ctx)
    except ValueError as error:
        assert "calibration mode" in str(error), str(error)
    else:
        raise AssertionError("Calibration-mode evaluation was accepted")
    model.freeze_qparams()
    before = snapshot(model)
    assert before["observers"], "Frozen-state check requires actual quantization observers"
    assert all(not value["enabled"] for value in before["observers"].values())
    assert prepare_eager_evaluation(ctx) == 32
    assert resolve_export_max_seq_len(ctx) == 6
    assert positional_config(model) == config_before
    assert_same(before, snapshot(model))
    outputs = []
    with torch.no_grad():
        score = model(input_ids=ids[:, :12], use_cache=False)
        outputs.append(tensor_hash(score.last_hidden_state))
        assert score.past_key_values is None
        prefill = model(input_ids=ids[:, :12], use_cache=True)
        assert prefill.past_key_values is not None
        cache = prefill.past_key_values
        for offset in range(12, 16):
            decoded = model(input_ids=ids[:, offset:offset+1], attention_mask=torch.ones_like(ids[:, :offset+1]), past_key_values=cache, use_cache=True)
            cache = decoded.past_key_values
            assert cache is not None
            assert cache.get_seq_length() == offset + 1
            assert torch.isfinite(decoded.last_hidden_state).all()
            outputs.append(tensor_hash(decoded.last_hidden_state))
        assert torch.isfinite(score.last_hidden_state).all()
    after = snapshot(model)
    assert_same(before, after)
    ctx.cfg["evaluation"]["max_seq_len"] = 20
    ctx.cfg["evaluation"]["use_cache"] = False
    assert prepare_eager_evaluation(ctx) == 20
    assert resolve_export_max_seq_len(ctx) == 6
    assert positional_config(model) == config_before
    assert model.config.use_cache is False
    assert_same(before, snapshot(model))
    ctx.cfg["evaluation"]["max_seq_len"] = 33
    try:
        resolve_evaluation_max_seq_len(ctx)
    except ValueError as error:
        assert "exceeds" in str(error)
    else:
        raise AssertionError("Evaluation beyond recorded capacity was accepted")
    ctx.cfg["evaluation"]["max_seq_len"] = 20
    model.force_export = True
    try:
        prepare_eager_evaluation(ctx)
    except ValueError as error:
        assert "force_export" in str(error)
    else:
        raise AssertionError("Forced static evaluation was accepted")
    finally:
        model.force_export = False
    if family in ("llama", "qwen3_vl"):
        # A deliberately raised trace error exercises actual-owner cleanup.
        before_static = snapshot(model)
        try:
            with prepared_static_export(model, 6):
                assert any(hasattr(module, "_static_export_inputs") for module in model.modules())
                raise RuntimeError("deliberate validation trace error")
        except RuntimeError as error:
            assert "deliberate validation trace error" in str(error)
        assert not any(hasattr(module, "_static_export_inputs") for module in model.modules())
        assert_same(before_static, snapshot(model))
    (directory / f"{family}_state_before.json").write_text(json.dumps(before, indent=2) + "\n")
    (directory / f"{family}_state_after.json").write_text(json.dumps(after, indent=2) + "\n")
    return {"status": "PASS", "level": "tiny-real-HF-random-text-PTQ", "pretrained": False, "actual_image": False, "calibration_seq_len": 3, "export_seq_len": 6, "eager_prefill_seq_len": 12, "decode_steps": 4, "native_capacity": 32, "checked_parameters": len(before["parameters"]), "checked_buffers": len(before["buffers"]), "checked_observers": len(before["observers"]), "checked_quant_modules": len(before["quant_modules"]), "positional_config": config_before, "outputs": outputs, "static_exception_cleanup": "PASS" if family in ("llama", "qwen3_vl") else "not checked"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    result = {"scope": "Supplemental actual tiny HF text models and real PTQ; no pretrained benchmark, image evaluation or Circle execution", "seed": 1073, "models": {}}
    for family in ("llama", "qwen3_vl", "gemma4"):
        try:
            result["models"][family] = run_family(family, args.output)
        except Exception as error:
            traceback.print_exc()
            result["models"][family] = {"status": "FAIL", "error": repr(error), "traceback": traceback.format_exc()}
        print(family, result["models"][family]["status"], flush=True)
        (args.output / "summary.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    raise SystemExit(0 if all(value["status"] == "PASS" for value in result["models"].values()) else 1)


if __name__ == "__main__":
    main()
