"""Independently audit complete full-Circle evidence across long eager stress."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from tico.circle import CircleDocument


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def phase(root):
    snapshot = json.loads((root / "snapshot.json").read_text())
    actual_stages = {str(path.relative_to(root / "artifacts")) for path in (root / "artifacts").rglob("*.circle")}
    result = {"artifact_stage_sets_equal": actual_stages == set(snapshot["stages"]), "state_preserved": snapshot["state_preserved"], "host": snapshot["host"], "stages": {}}
    for name, record_name in snapshot["stages"].items():
        circle_path = root / "artifacts" / name
        metadata = json.loads((root / "records" / (record_name + ".json")).read_text())
        with CircleDocument.load_mapped(circle_path) as document:
            document.verify(raise_on_error=True)
            graph = document.model.subgraphs[0]
            boundaries = {kind: [{"tensor_index": int(index), "shape": list(map(int, graph.tensors[int(index)].shape)), "type": int(graph.tensors[int(index)].type), "name": str(graph.tensors[int(index)].name)} for index in indices] for kind, indices in (("inputs", graph.inputs), ("outputs", graph.outputs))}
            quantized = [tensor for tensor in graph.tensors if tensor.quantization is not None]
            scales = [np.asarray(tensor.quantization.scale, dtype=np.float64) for tensor in quantized if tensor.quantization.scale is not None]
            qparams_finite = all(bool(np.isfinite(value).all()) for value in scales)
            qparam_scales_positive = all(bool((value > 0).all()) for value in scales)
            operator_count = len(graph.operators)
        probes = []
        with np.load(root / "records" / (record_name + ".npz"), allow_pickle=False) as arrays:
            for index, probe in enumerate(metadata["probes"]):
                keys = {group: sorted(key for key in arrays.files if key.startswith(f"p{index}_{group}_")) for group in ("input", "circle", "torch")}
                complete = set(keys["input"]) == {f"p{index}_input_{i}" for i in range(len(boundaries["inputs"]))} and all(set(keys[group]) == {f"p{index}_{group}_{i}" for i in range(len(boundaries["outputs"]))} for group in ("circle", "torch"))
                complete &= len(probe["torch_parity"]) == len(boundaries["outputs"])
                finite = all(bool(np.isfinite(arrays[key]).all()) for group in keys.values() for key in group)
                numeric = {key: {"shape": list(arrays[key].shape), "dtype": str(arrays[key].dtype), "min": float(arrays[key].min()) if arrays[key].size else None, "max": float(arrays[key].max()) if arrays[key].size else None, "sha256": hashlib.sha256(arrays[key].tobytes()).hexdigest()} for group in keys.values() for key in group}
                torch_parity = bool(probe["torch_parity"]) and all(item["ok"] for item in probe["torch_parity"])
                probes.append({"name": probe["probe"], "coverage_complete": bool(complete), "finite": finite, "torch_parity_ok": bool(torch_parity), "torch_parity": probe["torch_parity"], "arrays": numeric})
        result["stages"][name] = {"circle_sha256": digest(circle_path), "io": boundaries, "operator_count": operator_count, "qparam_tensor_count": len(quantized), "qparams_finite": qparams_finite, "qparam_scales_positive": qparam_scales_positive, "execution_mode": metadata["mode"], "contract": metadata["contract"], "fx": metadata["fx"], "probes": probes}
    return result


parser = argparse.ArgumentParser()
parser.add_argument("--run", type=Path, required=True)
args = parser.parse_args()
candidate = args.run / "candidate"
result = {"scope": "Independent actual full Circle I/O/numerical evidence audit, not pretrained or NPU validation", "run": str(args.run), "eager": json.loads((candidate / "eager.json").read_text()), "phases": {name: phase(candidate / name) for name in ("before", "after_eager")}}
before, after = result["phases"]["before"], result["phases"]["after_eager"]
comparisons = {}
for name in set(before["stages"]) | set(after["stages"]):
    a, b = before["stages"].get(name), after["stages"].get(name)
    comparisons[name] = {"present_both": a is not None and b is not None}
    if a and b:
        comparisons[name].update(circle_binary_identical=a["circle_sha256"] == b["circle_sha256"], complete_contract_identical=a["contract"] == b["contract"], io_identical=a["io"] == b["io"], torch_fx_identical=a["fx"] == b["fx"], probes_identical=a["probes"] == b["probes"])
result["pre_post"] = comparisons
result["host_identical"] = before["host"] == after["host"]
result["expected_stage_set"] = ["model.q.circle"]
result["evidence_complete"] = all(value["artifact_stage_sets_equal"] and value["state_preserved"] and set(value["stages"]) == set(result["expected_stage_set"]) and all(stage["qparams_finite"] and stage["qparam_scales_positive"] and stage["probes"] and all(probe["coverage_complete"] and probe["finite"] for probe in stage["probes"]) for stage in value["stages"].values()) for value in result["phases"].values())
result["pre_post_component_identity"] = result["host_identical"] and all(all(value.values()) for value in comparisons.values())
result["torch_circle_parity"] = all(probe["torch_parity_ok"] for value in result["phases"].values() for stage in value["stages"].values() for probe in stage["probes"])
result["strict_ok"] = bool(result["evidence_complete"] and result["pre_post_component_identity"] and result["torch_circle_parity"] and result["eager"]["state_preserved"])
destination = args.run / "independent_full_evidence.json"
destination.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({key: result[key] for key in ("eager", "evidence_complete", "pre_post_component_identity", "torch_circle_parity", "strict_ok")}, indent=2))
raise SystemExit(0 if result["strict_ok"] else 1)
