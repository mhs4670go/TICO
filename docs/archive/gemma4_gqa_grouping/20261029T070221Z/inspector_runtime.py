# Copyright (c) 2026 Samsung Electronics Co., Ltd. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Recreate default inspector calibration/evaluation; execute its saved Circle."""

import argparse, json, pathlib, subprocess, traceback
from typing import Any, cast

import numpy as np
import tico, torch
from export_check import digest, epgraph, metrics, qdesc
from tico.circle import CircleDocument
from tico.circle.runtime import CircleReferenceRuntime, ExecutionMode
from tico.circle.runtime.quantization import fake_quantize_array
from tico.quantization.recipes.config import load_recipe_config
from tico.quantization.recipes.debug.wrapper_smoke import get_case
from tico.quantization.recipes.debug.wrapper_smoke.cases.gemma4 import (
    Gemma4TextAttentionBaseCase,
)


def run(case_name, circle):
    cfg = load_recipe_config("tico/quantization/examples/configs/wrapper_smoke.yaml")
    torch.manual_seed(cfg["runtime"]["seed"])
    case = cast(Gemma4TextAttentionBaseCase, get_case(case_name))
    with torch.no_grad():
        model, _ = case.build(cfg)
        prepared = case.prepare_model(model, cfg)
        case.after_prepare(prepared, cfg)
        cal = case.calibration_inputs(prepared, cfg)[:3]
        for s in cal:
            case.forward(prepared, s)
        sample = case.eval_input(prepared, cfg)
        if case.compare_reference_source == "prepared":
            case.forward(prepared, sample)
        quant = case.convert_model(prepared, cfg)
        case.forward(quant, sample)
        with case.export_context(quant, sample, cfg):
            inv = case.export_invocation(quant, sample, cfg)
            mod = inv.module.eval()
            args = tuple(inv.inputs.args)
            kwargs = dict(inv.inputs.kwargs)
            eager = mod(*args, **kwargs)
            ep = torch.export.export(mod, args, kwargs, strict=False)
            epout = ep.module()(*args, **kwargs)
            ef, _ = torch.utils._pytree.tree_flatten(eager)
            xf, _ = torch.utils._pytree.tree_flatten(epout)
            doc = CircleDocument.load(circle)
            doc.verify()
            rt = CircleReferenceRuntime(doc)
            leaves, spec = torch.utils._pytree.tree_flatten((args, kwargs))
            encoded = []
            for val, i in zip(leaves, rt.program.inputs):
                arr = val.numpy()
                t = rt.program.tensor(i)
                if t.is_quantized_integer:
                    arr = fake_quantize_array(arr, t)
                encoded.append(np.array(arr, copy=True))
            ea, ek = torch.utils._pytree.tree_unflatten(
                [torch.from_numpy(x.copy()) for x in encoded], spec
            )
            abi, _ = torch.utils._pytree.tree_flatten(mod(*ea, **ek))
            result = {
                "case": case_name,
                "Hq": case.text_cfg.num_attention_heads,
                "Hkv": case.text_cfg.num_key_value_heads,
                "D": case.attention_head_dim,
                "torch_export": {
                    "status": "PASS",
                    "errors": [metrics(x.numpy(), e.numpy()) for x, e in zip(xf, ef)],
                },
                "circle_verify": "PASS",
                "evaluation_sha256": digest(leaves),
            }
            try:
                actual = rt.run(encoded, mode=ExecutionMode.FAKE_QUANTIZE).outputs
                result["runtime"] = {
                    "status": "EXECUTED",
                    "mode": "FAKE_QUANTIZE",
                    "errors": [metrics(x, y.numpy()) for x, y in zip(actual, abi)],
                }
            except Exception as e:
                traceback.print_exc()
                result["runtime"] = {"status": "FAIL", "error": str(e)}
            return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact-dir", type=pathlib.Path, required=True)
    ns = ap.parse_args()
    torch.set_num_threads(1)
    output: dict[str, Any] = {
        "implementation_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "tico_import": tico.__file__,
        "cases": [],
    }
    for mode in ["prefill", "decode"]:
        name = "gemma4_text_attention_" + mode
        try:
            output["cases"].append(
                run(
                    name, ns.artifact_dir / ("inspector-" + mode) / (name + ".q.circle")
                )
            )
        except Exception as e:
            traceback.print_exc()
            output["cases"].append({"case": name, "status": "FAIL", "error": str(e)})
    (ns.artifact_dir / "inspector_runtime.json").write_text(
        json.dumps(output, indent=2)
    )
    print(json.dumps(output, indent=2))
