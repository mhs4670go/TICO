"""Build a genuine baseline wrapper from the same source HF weights and image inputs."""
import argparse
import copy
import hashlib
import json
import traceback
from pathlib import Path

import torch
import tico
from tico.quantization.wrapq.wrappers.qwen_vl.quant_for_conditional_generation import QuantQwen3VLForConditionalGeneration


def tensors_digest(values):
    result = {}
    for name, value in values:
        tensor = value.detach().cpu().contiguous()
        result[name] = {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "sha256": hashlib.sha256(tensor.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()}
    return result


def state(model):
    return {"parameters": tensors_digest(model.named_parameters()), "buffers": tensors_digest(model.named_buffers()), "observers": {name: {"enabled": observer.enabled, "state": tensors_digest(observer.state_dict().items())} for name, observer in model.named_observers()}, "modes_profiles": {name: {"mode": str(module._mode), "profile": repr(module.qcfg.model_args)} for name, module in model.named_modules() if hasattr(module, "_mode") and hasattr(module, "qcfg")}}


def attempt(model, payload):
    torch.manual_seed(1073)
    try:
        with torch.no_grad():
            tokens = model.generate(**copy.deepcopy(payload["sample"]), **payload["settings"])
        return {"status": "PASS", "tokens": tokens.tolist()}
    except Exception as error:
        traceback.print_exc()
        return {"status": "FAIL", "error": repr(error), "traceback": traceback.format_exc()}


parser = argparse.ArgumentParser()
parser.add_argument("--source-checkpoint", type=Path, required=True)
parser.add_argument("--inputs", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
result = {"scope": "Fresh baseline wrapper and tiny independent calibration from identical source HF weights and PNG processor tensors. Not same-quant-checkpoint export-only equivalence.", "tico_file": tico.__file__, "source_checkpoint": str(args.source_checkpoint), "source_checkpoint_sha256": hashlib.sha256(args.source_checkpoint.read_bytes()).hexdigest()}
try:
    trusted_candidate = torch.load(args.source_checkpoint, map_location="cpu", weights_only=False)
    fp = copy.deepcopy(trusted_candidate.module).eval()
    qcfg = trusted_candidate.qcfg
    del trusted_candidate
    payload = torch.load(args.inputs, map_location="cpu", weights_only=False)
    result["source_hf_parameter_hashes"] = tensors_digest(fp.named_parameters())
    quant = QuantQwen3VLForConditionalGeneration(copy.deepcopy(fp), qcfg=qcfg).eval()
    result["fresh_legacy_templates"] = [name for name, _ in quant.named_buffers() if name.endswith("causal_mask_template")]
    quant.enable_calibration()
    with torch.no_grad():
        quant(**payload["sample"], use_cache=False)
    quant.freeze_qparams()
    before = state(quant)
    result["fp"] = attempt(fp, payload)
    result["quant"] = attempt(quant, payload)
    after = state(quant)
    result["state_changes"] = {group: [name for name in set(before[group]) | set(after[group]) if before[group].get(name) != after[group].get(name)] for group in before}
    result["frozen_state"] = "PASS" if not any(result["state_changes"].values()) else "FAIL"
    (args.output / "state.json").write_text(json.dumps({"before": before, "after": after}, indent=2) + "\n")
    result["status"] = "PASS" if result["fp"]["status"] == result["quant"]["status"] == result["frozen_state"] == "PASS" else "FAIL"
except Exception as error:
    traceback.print_exc()
    result.update(status="FAIL", setup_error=repr(error), traceback=traceback.format_exc())
(args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
raise SystemExit(0 if result["status"] == "PASS" else 1)
