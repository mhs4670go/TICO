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

"""Compact actual stage/op/qparam/runtime measurements from export_check outputs."""

import argparse, collections, json, pathlib
from typing import Any


def movement_contracts(c):
    result: collections.Counter[tuple[str, str, bool]] = collections.Counter()
    for op in c["circle"]["movement"]:
        # data operands only; shape/axis/slice control operands intentionally excluded
        inputs = (
            op["inputs"]
            if op["name"] == "CONCATENATION"
            else op["inputs"][1:2] if op["name"] == "SPLIT" else op["inputs"][:1]
        )
        for t in inputs + op["outputs"]:
            result[(op["name"], t["dtype"], bool(t["scale"]))] += 1
    return [
        {"op": k[0], "dtype": k[1], "has_qparam": k[2], "operand_occurrences": v}
        for k, v in result.items()
    ]


def compact(r):
    out = {k: v for k, v in r.items() if k != "cases"}
    out["cases"] = []
    for c in r["cases"]:
        if "circle" not in c:
            out["cases"].append(c)
            continue
        d = {k: v for k, v in c.items() if k not in {"circle", "torch_export"}}
        ep = c["torch_export"]
        ci = c["circle"]
        d["torch_export"] = {
            "status": ep["status"],
            "errors": ep["errors"],
            "op_counts": ep["ops"],
            "attention_matmul_count": dict(
                collections.Counter(m["kind"] for m in ep["attention_matmul"])
            ),
            "unique_attention_shapes": list(
                {json.dumps(m["operands"]) for m in ep["attention_matmul"]}
            ),
        }
        d["circle"] = {
            "status": ci["status"],
            "sha256": ci["sha256"],
            "bytes": ci["bytes"],
            "ops": ci["ops"],
            "attention_matmul_count": dict(
                collections.Counter(m["kind"] for m in ci["attention_matmul"])
            ),
            "unique_attention_contracts": list(
                {
                    json.dumps(
                        {
                            "kind": m["kind"],
                            "operands": [
                                {
                                    k: t[k]
                                    for k in [
                                        "shape",
                                        "dtype",
                                        "scale",
                                        "zero_point",
                                        "axis",
                                    ]
                                }
                                for t in m["operands"]
                            ],
                            "outputs": [
                                {
                                    k: t[k]
                                    for k in [
                                        "shape",
                                        "dtype",
                                        "scale",
                                        "zero_point",
                                        "axis",
                                    ]
                                }
                                for t in m["outputs"]
                            ],
                        },
                        sort_keys=True,
                    )
                    for m in ci["attention_matmul"]
                }
            ),
            "float_activation_count": len(ci["float_activations"]),
            "movement_data_contracts": movement_contracts(c),
        }
        out["cases"].append(d)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact-root", type=pathlib.Path, required=True)
    ap.add_argument("--output", type=pathlib.Path, required=True)
    ns = ap.parse_args()
    summary: dict[str, Any] = {
        "implementations": {},
        "npu_compile": "NOT_RUN: no triv24-compile found at known path or PATH",
        "notes": [
            "Default O1 retained. FP BASE prefill/decode and B decode Circle reference runtime fail serialized rank contract: Softmax is rank4 but PV BMM output contract rank3. Eager/torch.export are valid.",
            "Quantized runtime uses explicit FAKE_QUANTIZE semantic reference, not bit-exact NPU execution. ABI inputs are aligned to serialized grids before eager comparison.",
            "The initial per-head PV-output marker left grouped MatMul outputs FLOAT32; final wrapper-local scalar-affine group marker removes all such float islands without extra Q/DQ.",
        ],
    }
    rows = [
        "|Fixture|Impl|Mode|Quant|EP QK/PV|Circle QK/PV|Concat|Reshape|Transpose|Slice|Softmax|Q/DQ|Float activations|Runtime|max abs output0|",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for v in ["base", "a", "b"]:
        imp = {}
        for fixture in ["custom", "tiny"]:
            r = json.loads(
                (ns.artifact_root / v / fixture / "export_metrics.json").read_text()
            )
            imp[fixture] = compact(r)
            for c in imp[fixture]["cases"]:
                ci = c["circle"]
                ops = ci["ops"]
                e = c["torch_export"]["attention_matmul_count"]
                m = ci["attention_matmul_count"]
                rt = c["circle_runtime"]
                err = rt.get("errors", [{}])[0].get("max_abs", "FAIL")
                rows.append(
                    f"|{fixture}|{v}|{c['mode']}|{c['quant']}|{e.get('QK',0)}/{e.get('PV',0)}|{m.get('QK',0)}/{m.get('PV',0)}|{ops.get('CONCATENATION',0)}|{ops.get('RESHAPE',0)}|{ops.get('TRANSPOSE',0)}|{ops.get('SLICE',0)+ops.get('STRIDED_SLICE',0)}|{ops.get('SOFTMAX',0)}|{ops.get('QUANTIZE',0)}/{ops.get('DEQUANTIZE',0)}|{ci['float_activation_count']}|{rt['status']}|{err}|"
                )
        for label, file in [
            ("inspector_runtime", "inspector_runtime.json"),
            ("identity", "identity.json"),
        ]:
            p = ns.artifact_root / v / file
            if p.exists():
                imp[label] = json.loads(p.read_text())
        summary["implementations"][v] = imp
    ns.output.mkdir(parents=True, exist_ok=True)
    (ns.output / "export_summary_detailed.json").write_text(
        json.dumps(summary, indent=2)
    )
    small: dict[str, Any] = {
        "npu_compile": summary["npu_compile"],
        "notes": summary["notes"],
        "implementations": {},
    }
    for v, imp in summary["implementations"].items():
        sd = {k: val for k, val in imp.items() if k not in {"custom", "tiny"}}
        for fixture in ["custom", "tiny"]:
            sd[fixture] = []
            for c in imp[fixture]["cases"]:
                ci = c["circle"]
                ep = c["torch_export"]
                rt = dict(c["circle_runtime"])
                if "error" in rt:
                    rt["error"] = rt["error"].split("  at ")[0]
                sd[fixture].append(
                    {
                        "fixture": c["fixture"],
                        "mode": c["mode"],
                        "quant": c["quant"],
                        "torch_export_status": ep["status"],
                        "torch_export_errors": ep["errors"],
                        "torch_export_qk_pv": ep["attention_matmul_count"],
                        "circle_verify": ci["status"],
                        "circle_sha256": ci["sha256"],
                        "circle_bytes": ci["bytes"],
                        "circle_qk_pv": ci["attention_matmul_count"],
                        "circle_ops": ci["ops"],
                        "float_activation_count": ci["float_activation_count"],
                        "attention_contracts": (
                            [json.loads(x) for x in ci["unique_attention_contracts"]]
                            if fixture == "custom" and c["quant"]
                            else []
                        ),
                        "runtime": rt,
                    }
                )
        small["implementations"][v] = sd
    ids = [d["identity"]["fixtures"] for d in small["implementations"].values()]
    small["fixture_fingerprints_equal"] = ids[0] == ids[1] == ids[2]
    assert small["fixture_fingerprints_equal"]
    (ns.output / "export_summary.json").write_text(
        json.dumps(small, separators=(",", ":"))
    )
    (ns.output / "export_summary.md").write_text(
        "\n".join(rows) + "\n\n" + "\n\n".join(summary["notes"]) + "\n"
    )
