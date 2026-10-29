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

"""Run from an explicit worktree with PYTHONPATH set to that worktree.
Small real Gemma4 wrapper, same seeded weights/data in every process.
Artifacts stay under --output; JSON records validation levels separately.
"""

import argparse, collections, hashlib, json, pathlib, subprocess, traceback
from typing import Any

import numpy as np
import tico
import torch
from tico.circle import CircleDocument
from tico.circle.runtime import CircleReferenceRuntime, ExecutionMode
from tico.circle.runtime.quantization import fake_quantize_array
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.wrappers.gemma4.quant_text_attention import (
    QuantGemma4TextAttention,
)
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextAttention


def config(hq=24, hkv=8):
    return Gemma4TextConfig.from_dict(
        dict(
            vocab_size=32,
            hidden_size=hq * 4,
            intermediate_size=hq * 8,
            num_hidden_layers=1,
            num_attention_heads=hq,
            num_key_value_heads=hkv,
            num_global_key_value_heads=hkv,
            head_dim=4,
            global_head_dim=4,
            attention_bias=False,
            attention_dropout=0.0,
            max_position_embeddings=32,
            layer_types=["full_attention"],
            sliding_window=8,
            hidden_size_per_layer_input=0,
            attention_k_eq_v=False,
            num_kv_shared_layers=0,
            enable_moe_block=False,
            allow_global_per_layer_attribute_access=True,
        )
    )


def sample(cfg, t, s):
    args = (
        torch.randn(1, t, cfg.hidden_size),
        torch.zeros(1, 1, t, s),
        (torch.ones(1, t, 4), torch.zeros(1, t, 4)),
    )
    # supplied offset causal mask (query-row positions refer to S-T..S-1)
    mask = args[1]
    for i in range(t):
        mask[0, 0, i, s - t + i + 1 :] = -120
    kw = (
        {}
        if s == t
        else {
            "past_key_value": (
                torch.randn(1, cfg.num_key_value_heads, s - t, 4),
                torch.randn(1, cfg.num_key_value_heads, s - t, 4),
            )
        }
    )
    return args, kw


def metrics(a, b):
    e = np.abs(np.asarray(a) - np.asarray(b))
    return {
        "mean_abs": float(e.mean()),
        "max_abs": float(e.max()),
        "relative_l2": float(np.linalg.norm(e) / (np.linalg.norm(b) + 1e-30)),
        "differing": int(np.count_nonzero(e)),
    }


def digest(values):
    h = hashlib.sha256()
    for x in values:
        if isinstance(x, torch.Tensor):
            h.update(x.detach().contiguous().numpy().tobytes())
    return h.hexdigest()


def qdesc(t):
    q = t.quantization
    return {
        "index": t.index,
        "name": t.name,
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "scale": list(q.scale) if q else [],
        "zero_point": list(q.zero_point) if q else [],
        "axis": q.quantized_dimension if q else None,
    }


def epgraph(ep, s):
    ops: collections.Counter[str] = collections.Counter()
    mm = []
    for n in ep.graph.nodes:
        if n.op != "call_function":
            continue
        name = str(n.target)
        ops[name] += 1
        if "matmul" in name or name in ("aten.bmm.default", "aten.mm.default"):
            shapes = [
                list(a.meta["val"].shape)
                for a in n.args[:2]
                if isinstance(a, torch.fx.Node) and hasattr(a.meta.get("val"), "shape")
            ]
            out = list(n.meta["val"].shape)
            mm.append(
                {
                    "name": n.name,
                    "operands": shapes,
                    "output": out,
                    "kind": "QK" if out[-1] == s else "PV",
                }
            )
    return {"ops": dict(ops), "attention_matmul": mm}


def run_case(out, quant, mode, hq, hkv, optimize=True):
    torch.manual_seed(20261029)
    cfg = config(hq, hkv)
    cfg._attn_implementation = "eager"
    attn = QuantGemma4TextAttention(
        Gemma4TextAttention(cfg, layer_idx=0).eval(),
        qcfg=PTQConfig(model_args={"text": {"max_seq": 32}}),
    ).eval()
    weights_sha = digest(list(attn.parameters()))
    calibration_sha = []
    if quant:
        attn.enable_calibration()
        for j in range(3):
            ca, ck = sample(cfg, 3, 6)
            calibration_sha.append(
                digest(torch.utils._pytree.tree_flatten((ca, ck))[0])
            )
            with torch.no_grad():
                attn(ca[0], ca[2], attention_mask=ca[1], use_cache=True, **ck)
        attn.freeze_qparams()
    t, s = (3, 3) if mode == "prefill" else (1, 8)
    args, kwargs = sample(cfg, t, s)
    module = attn.as_export_module(mode).eval()
    result = {
        "fixture": {
            "Hq": hq,
            "Hkv": hkv,
            "G": hq // hkv,
            "B": 1,
            "T": t,
            "S": s,
            "D": 4,
        },
        "mode": mode,
        "quant": quant,
        "weights_sha256": weights_sha,
        "evaluation_sha256": digest(
            torch.utils._pytree.tree_flatten((args, kwargs))[0]
        ),
        "calibration_sha256": calibration_sha,
    }
    with torch.no_grad():
        eager = module(*args, **kwargs)
        ep = torch.export.export(module, args, kwargs, strict=False)
        exported = ep.module()(*args, **kwargs)
    el, _ = torch.utils._pytree.tree_flatten(eager)
    xl, _ = torch.utils._pytree.tree_flatten(exported)
    result["torch_export"] = {
        "status": "PASS",
        "errors": [metrics(x.numpy(), e.numpy()) for x, e in zip(xl, el)],
        **epgraph(ep, s),
    }
    with torch.no_grad():
        converted = tico.convert_from_exported_program(
            ep, config=tico.CompileConfigV1(circle_optimize=optimize)
        )
    path = out / f'h{hq}_{mode}_{"quant" if quant else "fp"}.circle'
    converted.save(path)
    data = path.read_bytes()
    doc = CircleDocument.from_bytes(data)
    doc.verify()
    rt = CircleReferenceRuntime(doc)
    p = rt.program
    mm = []
    counts = collections.Counter(op.name for op in p.operators)
    for op in p.operators:
        if op.name == "BATCH_MATMUL":
            inputs = [p.tensor(i) for i in op.inputs]
            outputs = [p.tensor(i) for i in op.outputs]
            mm.append(
                {
                    "index": op.index,
                    "kind": "QK" if outputs[0].shape[-1] == s else "PV",
                    "operands": [qdesc(x) for x in inputs],
                    "outputs": [qdesc(x) for x in outputs],
                }
            )
    floats = [qdesc(x) for x in p.tensors if not x.is_constant and x.dtype.kind == "f"]
    movement = []
    for op in p.operators:
        if op.name in {
            "CONCATENATION",
            "RESHAPE",
            "TRANSPOSE",
            "SLICE",
            "STRIDED_SLICE",
            "SPLIT",
            "SPLIT_V",
            "GATHER",
        }:
            movement.append(
                {
                    "index": op.index,
                    "name": op.name,
                    "inputs": [qdesc(p.tensor(i)) for i in op.inputs if i >= 0],
                    "outputs": [qdesc(p.tensor(i)) for i in op.outputs],
                }
            )
    result["circle"] = {
        "status": "PASS",
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "ops": dict(counts),
        "attention_matmul": mm,
        "float_activations": floats,
        "movement": movement,
        "input": [qdesc(p.tensor(i)) for i in p.inputs],
        "output": [qdesc(p.tensor(i)) for i in p.outputs],
    }
    leaves, spec = torch.utils._pytree.tree_flatten((args, kwargs))
    enc = []
    for v, i in zip(leaves, p.inputs):
        a = v.detach().numpy()
        if p.tensor(i).is_quantized_integer:
            a = fake_quantize_array(a, p.tensor(i))
        enc.append(np.array(a, copy=True))
    ea, ek = torch.utils._pytree.tree_unflatten(
        [torch.from_numpy(x.copy()) for x in enc], spec
    )
    with torch.no_grad():
        abi_eager = module(*ea, **ek)
    abi_eager, _ = torch.utils._pytree.tree_flatten(abi_eager)
    try:
        actual = rt.run(
            enc, mode=ExecutionMode.FAKE_QUANTIZE if quant else ExecutionMode.NATIVE
        ).outputs
        errors = [metrics(x, y.numpy()) for x, y in zip(actual, abi_eager)]
        result["circle_runtime"] = {
            "status": "EXECUTED",
            "execution_mode": "FAKE_QUANTIZE" if quant else "NATIVE",
            "errors": errors,
            "note": "Quantized runtime is semantic fake-quant reference, not bit-exact NPU.",
        }
    except Exception as e:
        traceback.print_exc()
        result["circle_runtime"] = {"status": "FAIL", "error": str(e)}
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=pathlib.Path, required=True)
    ap.add_argument("--hq", type=int, default=24)
    ap.add_argument("--hkv", type=int, default=8)
    ap.add_argument("--quant", choices=["all", "fp", "quant"], default="all")
    ap.add_argument("--mode", choices=["all", "prefill", "decode"], default="all")
    ap.add_argument("--no-optimize", action="store_true")
    ns = ap.parse_args()
    ns.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    report: dict[str, Any] = {
        "implementation_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "tico_import": tico.__file__,
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "seed": 20261029,
        "optimize": not ns.no_optimize,
        "product_source_sha256": hashlib.sha256(
            pathlib.Path(
                QuantGemma4TextAttention.__module__.replace(".", "/") + ".py"
            ).read_bytes()
        ).hexdigest(),
        "npu_compile": "NOT_RUN: triv24-compile absent",
        "cases": [],
    }
    for q in ([False, True] if ns.quant == "all" else [ns.quant == "quant"]):
        for m in (["prefill", "decode"] if ns.mode == "all" else [ns.mode]):
            try:
                r = run_case(ns.output, q, m, ns.hq, ns.hkv, not ns.no_optimize)
            except Exception as e:
                traceback.print_exc()
                r = {"mode": m, "quant": q, "status": "FAIL", "error": str(e)}
            report["cases"].append(r)
            (ns.output / "export_metrics.json").write_text(json.dumps(report, indent=2))
            print(m, q, r.get("status", r.get("circle", {}).get("status")), flush=True)
