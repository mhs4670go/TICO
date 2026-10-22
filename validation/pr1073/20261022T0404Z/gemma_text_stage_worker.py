"""Export only explicit Gemma text decoder stages using existing public stage adapters.

Vision, fusion, embeddings and LM-head conversion are outside this independent
probe. The original multimodal runtime failure remains a separate failure.
"""
import argparse
import hashlib
import json
import subprocess
import traceback
from pathlib import Path

import numpy as np
import torch
import tico
from tico.quantization.recipes.export.gemma4 import (
    _attention_contract, _make_prefill_attention_mask, _make_decode_attention_mask,
    _unwrap_gemma4_components, _convert_and_save,
)
from gemma_text_export_recording import ExportRecorder, _state_contract, _host_contract


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def tensors(items):
    return {name: {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": hashlib.sha256(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()} for name, value in items}


def frozen(model):
    return {"parameters": tensors(model.named_parameters()), "buffers": tensors(model.named_buffers()), "observers": {name: {"enabled": observer.enabled, "state": tensors(observer.state_dict().items())} for name, observer in model.named_observers()}, "modules": {name: {"mode": str(module._mode), "profile": repr(module.qcfg.model_args), "return_type": getattr(module, "return_type", None), "force_export": getattr(module, "force_export", None)} for name, module in model.named_modules() if hasattr(module, "_mode") and hasattr(module, "qcfg")}}


def changes(before, after):
    return {group: [name for name in set(before[group]) | set(after[group]) if before[group].get(name) != after[group].get(name)] for group in before}


def export_phase(model, qtext, root, initial, initial_full):
    root.mkdir(exist_ok=False)
    artifacts = root / "artifacts"
    artifacts.mkdir()
    torch.manual_seed(1073)
    prefill_hidden = torch.randn(1, 8, qtext.config.hidden_size)
    decode_hidden = torch.randn(1, 1, qtext.config.hidden_size)
    errors, manifest, host = {}, {}, {}
    recorder = ExportRecorder(root, atol=1e-5, rtol=1e-4)
    try:
        with recorder:
            for layer_index, layer in enumerate(qtext.layers):
                attention, kv_heads, head_dim, sliding, shared = _attention_contract(layer, max_seq_len=8)
                if shared:
                    raise ValueError("This bounded checkpoint probe requires shared-KV disabled")
                layer_type = qtext.config.layer_types[layer_index]
                cos = getattr(qtext, qtext._cos_template_name(layer_type)).detach().clone()
                sin = getattr(qtext, qtext._sin_template_name(layer_type)).detach().clone()
                if cos.shape[1] < 8 or sin.shape != cos.shape:
                    raise ValueError("Frozen host RoPE does not cover the requested static profile")
                prefill_mask = _make_prefill_attention_mask(max_seq_len=8, sliding_window=sliding)
                decode_mask = _make_decode_attention_mask(max_seq_len=8, sliding_window=sliding)
                host.update({f"layer{layer_index}_positions": torch.arange(8).numpy(), f"layer{layer_index}_prefill_mask": prefill_mask.numpy(), f"layer{layer_index}_decode_mask": decode_mask.numpy(), f"layer{layer_index}_cos": cos[:, :8].numpy(), f"layer{layer_index}_sin": sin[:, :8].numpy()})
                past_k = torch.randn(1, kv_heads, 7, head_dim)
                past_v = torch.randn_like(past_k)
                for mode in ("prefill", "decode"):
                    stage = f"decoder_layer_{mode}_{layer_index}.q.circle"
                    kwargs = {"attention_mask": prefill_mask if mode == "prefill" else decode_mask, "position_embeddings": (cos[:, :8], sin[:, :8]) if mode == "prefill" else (cos[:, 7:8], sin[:, 7:8])}
                    if mode == "decode":
                        kwargs["past_key_value"] = (past_k, past_v)
                    manifest[stage] = {"kind": mode, "layer_index": layer_index, "layer_type": layer_type, "sliding_window": sliding, "kv_heads": kv_heads, "head_dim": head_dim, "expected_outputs": 3, "expected_delta_kv_length": 8 if mode == "prefill" else 1, "host_rope": "frozen checkpoint static templates, including its recorded sine convention"}
                    try:
                        module = layer.wrapped.as_export_module(mode, return_kv=True)
                        _convert_and_save(module, (prefill_hidden if mode == "prefill" else decode_hidden,), artifacts / stage, kwargs=kwargs, strict=False)
                        print("COMPLETED", stage, flush=True)
                    except Exception as error:
                        traceback.print_exc()
                        errors[stage] = {"error": repr(error), "traceback": traceback.format_exc()}
    except Exception as error:
        traceback.print_exc()
        errors["recorder"] = {"error": repr(error), "traceback": traceback.format_exc()}
    np.savez(artifacts / "gemma_text_host.npz", **host)
    after_full = frozen(model)
    differences = changes(initial_full, after_full)
    dump(root / "full_state_after.json", after_full)
    snapshot = {"schema_version": 1, "stages": recorder.stages, "state_preserved": _state_contract(model) == initial and not any(differences.values()), "state_differences": differences, "host": _host_contract(artifacts)}
    dump(root / "snapshot.json", snapshot)
    dump(root / "stage_manifest.json", {"scope": "Explicit text decoder stages only; original multimodal vision failure not skipped as success", "expected_stages": manifest, "errors": errors, "actual_stages": sorted(recorder.stages)})
    return {"complete": len(recorder.stages) == 4 and not errors, "state_preserved": snapshot["state_preserved"], "errors": errors}


def eager_text(qtext):
    torch.manual_seed(1074)
    ids = torch.arange(1, 17).reshape(1, -1)
    with torch.no_grad():
        output = qtext(input_ids=ids[:, :12], use_cache=True, return_dict=True)
        cache = output.past_key_values
        if cache is None:
            raise RuntimeError("Text eager prefill returned no cache")
        finite = bool(torch.isfinite(output.last_hidden_state).all())
        hashes = [tensors([("hidden", output.last_hidden_state)])["hidden"]]
        for index in range(12, 16):
            output = qtext(input_ids=ids[:, index:index+1], attention_mask=torch.ones_like(ids[:, :index+1]), past_key_values=cache, use_cache=True, return_dict=True)
            cache = output.past_key_values
            if cache is None or cache.get_seq_length() != index + 1:
                raise RuntimeError("Text cached decode did not retain its growing cache")
            finite &= bool(torch.isfinite(output.last_hidden_state).all())
            hashes.append(tensors([("hidden", output.last_hidden_state)])["hidden"])
    return {"kind": "actual Gemma text wrapper only; no image/video or conditional generation", "prefill_tokens": 12, "decode_steps": 4, "native_capacity": qtext.config.max_position_embeddings, "static_capacity": qtext.static_max_seq, "all_hidden_outputs_finite": finite, "output_hashes": hashes}


parser = argparse.ArgumentParser()
parser.add_argument("--repo", type=Path, required=True)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--stress", action="store_true")
args = parser.parse_args()
args.output.mkdir(exist_ok=False)
if not Path(tico.__file__).resolve().is_relative_to(args.repo.resolve()):
    raise RuntimeError("Unexpected TICO source import")
torch.set_num_threads(1)
model = torch.load(args.checkpoint, map_location="cpu", weights_only=False).eval()
if any(value.dtype != torch.float32 or value.device.type != "cpu" for value in model.parameters()):
    raise ValueError("Expected unchanged CPU float32 checkpoint")
qmodel, gemma, qvision, qtext = _unwrap_gemma4_components(model)
if int(qtext.config.hidden_size_per_layer_input) or int(qtext.config.num_kv_shared_layers):
    raise ValueError("This explicit text-only fixture must have PLE and shared-KV disabled")
initial, initial_full = _state_contract(model), frozen(model)
dump(args.output / "full_state_initial.json", initial_full)
dump(args.output / "environment.json", {"tico_file": tico.__file__, "commit": subprocess.check_output(["git", "-C", str(args.repo), "rev-parse", "HEAD"], text=True).strip(), "torch": torch.__version__, "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(), "static_capacity": 8, "native_capacity": qtext.config.max_position_embeddings, "rope_convention": qtext.rope_convention, "PLE": 0, "shared_KV": 0})
result = {"scope": "independent Gemma text decoder stage validation, not multimodal export", "before": export_phase(model, qtext, args.output / "before", initial, initial_full)}
if args.stress:
    try:
        exercise = eager_text(qtext)
        after = frozen(model)
        exercise["state_changes"] = changes(initial_full, after)
        exercise["state_preserved"] = not any(exercise["state_changes"].values())
        result["eager"] = exercise
    except Exception as error:
        traceback.print_exc()
        result["eager"] = {"error": repr(error), "traceback": traceback.format_exc(), "state_preserved": False}
    dump(args.output / "eager.json", result["eager"])
    result["after_eager"] = export_phase(model, qtext, args.output / "after_eager", initial, initial_full)
dump(args.output / "worker_result.json", result)
raise SystemExit(0 if all(value["complete"] and value["state_preserved"] for key, value in result.items() if key in ("before", "after_eager")) and result.get("eager", {}).get("state_preserved", True) else 1)
