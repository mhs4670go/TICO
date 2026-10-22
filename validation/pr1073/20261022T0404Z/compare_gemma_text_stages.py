"""Require complete actual Circle text-stage execution and all delta-KV outputs."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from tico.circle import CircleDocument
from gemma_text_export_recording import compare_snapshots


def audit(phase):
    snapshot = json.loads((phase / "snapshot.json").read_text())
    manifest = json.loads((phase / "stage_manifest.json").read_text())
    expected = {f"decoder_layer_{mode}_{index}.q.circle" for index in (0, 1) for mode in ("prefill", "decode")}
    files = {str(path.relative_to(phase / "artifacts")) for path in (phase / "artifacts").rglob("*.circle")}
    result = {"stage_sets_complete": expected == set(snapshot["stages"]) == files == set(manifest["expected_stages"]), "errors": manifest["errors"], "state_preserved": snapshot["state_preserved"], "stages": {}}
    for name, record_name in snapshot["stages"].items():
        metadata = json.loads((phase / "records" / (record_name + ".json")).read_text())
        with CircleDocument.load_mapped(phase / "artifacts" / name) as document:
            document.verify(raise_on_error=True)
            graph = document.model.subgraphs[0]
            input_count, output_count = len(graph.inputs), len(graph.outputs)
            io = {kind: [{"index": int(index), "shape": list(map(int, graph.tensors[int(index)].shape)), "type": int(graph.tensors[int(index)].type), "name": str(graph.tensors[int(index)].name)} for index in indices] for kind, indices in (("inputs", graph.inputs), ("outputs", graph.outputs))}
            qparams = [np.asarray(tensor.quantization.scale, dtype=np.float64) for tensor in graph.tensors if tensor.quantization is not None and tensor.quantization.scale is not None]
            qparam_scales_valid = bool(qparams) and all(bool(np.isfinite(value).all() and (value > 0).all()) for value in qparams)
        probes = []
        expected_kv = manifest["expected_stages"][name]["expected_delta_kv_length"]
        with np.load(phase / "records" / (record_name + ".npz"), allow_pickle=False) as arrays:
            for index, probe in enumerate(metadata["probes"]):
                keys = {group: {key for key in arrays.files if key.startswith(f"p{index}_{group}_")} for group in ("input", "circle", "torch")}
                complete = keys["input"] == {f"p{index}_input_{i}" for i in range(input_count)} and all(keys[group] == {f"p{index}_{group}_{i}" for i in range(output_count)} for group in ("circle", "torch")) and output_count == 3 and len(probe["torch_parity"]) == 3
                finite = all(bool(np.isfinite(arrays[key]).all()) for group in keys.values() for key in group)
                delta_complete = all(arrays[f"p{index}_{group}_{i}"].shape == (1, 1, expected_kv, 16) for group in ("circle", "torch") for i in (1, 2)) and all(arrays[f"p{index}_{group}_0"].shape == (1, expected_kv, 32) for group in ("circle", "torch")) if complete else False
                probes.append({"probe": probe["probe"], "complete_all_inputs_and_3_outputs": complete, "finite": finite, "delta_kv_shape_correct": delta_complete, "torch_parity": probe["torch_parity"], "numeric": {key: {"shape": list(arrays[key].shape), "dtype": str(arrays[key].dtype), "min": float(arrays[key].min()), "max": float(arrays[key].max())} for group in keys.values() for key in sorted(group)}})
        result["stages"][name] = {"io": io, "qparam_tensor_count": len(qparams), "qparam_scales_valid": qparam_scales_valid, "execution_mode": metadata["mode"], "probes": probes}
    result["complete_execution"] = bool(result["stage_sets_complete"] and not result["errors"] and result["state_preserved"] and all(stage["qparam_scales_valid"] and stage["probes"] and all(probe["complete_all_inputs_and_3_outputs"] and probe["finite"] and probe["delta_kv_shape_correct"] for probe in stage["probes"]) for stage in result["stages"].values()))
    result["torch_circle_parity"] = result["complete_execution"] and all(item["ok"] for stage in result["stages"].values() for probe in stage["probes"] for item in probe["torch_parity"])
    return result


def exact_components(left, right):
    a = json.loads((left / "snapshot.json").read_text())
    b = json.loads((right / "snapshot.json").read_text())
    result = {"stages_equal": set(a["stages"]) == set(b["stages"]), "host_arrays_identical": a["host"] == b["host"], "stages": {}}
    for name in set(a["stages"]) & set(b["stages"]):
        ap, bp = left / "records" / a["stages"][name], right / "records" / b["stages"][name]
        am, bm = json.loads(Path(str(ap) + ".json").read_text()), json.loads(Path(str(bp) + ".json").read_text())
        with np.load(str(ap) + ".npz", allow_pickle=False) as av, np.load(str(bp) + ".npz", allow_pickle=False) as bv:
            arrays_identical = set(av.files) == set(bv.files) and all(np.array_equal(av[key], bv[key]) for key in set(av.files) & set(bv.files))
        result["stages"][name] = {"circle_binary_identical": hashlib.sha256((left / "artifacts" / name).read_bytes()).digest() == hashlib.sha256((right / "artifacts" / name).read_bytes()).digest(), "complete_contract_identical": am["contract"] == bm["contract"], "torch_fx_identical": am["fx"] == bm["fx"], "all_inputs_torch_circle_arrays_identical": bool(arrays_identical)}
    result["all_exact"] = bool(result["stages_equal"] and result["host_arrays_identical"] and len(result["stages"]) == 4 and all(all(value.values()) for value in result["stages"].values()))
    return result


parser = argparse.ArgumentParser()
parser.add_argument("--baseline", type=Path, required=True)
parser.add_argument("--candidate", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
left = args.baseline / "before"
right = args.candidate / "before"
after = args.candidate / "after_eager"
result = {"scope": "explicit text decoder stages only, not multimodal export or omitted vision PASS", "atol": 1e-5, "rtol": 1e-4, "original_multimodal_result": "ERROR: RMS_NORM rank2 vision op102; preserved separately", "baseline_environment": json.loads((args.baseline / "environment.json").read_text()), "candidate_environment": json.loads((args.candidate / "environment.json").read_text()), "audits": {"baseline_before": audit(left), "candidate_before": audit(right), "candidate_after_eager": audit(after)}, "old_new_strict": compare_snapshots(left, right, atol=1e-5, rtol=1e-4, require_torch_parity=True), "candidate_isolation_strict": compare_snapshots(right, after, atol=1e-5, rtol=1e-4, require_torch_parity=True), "eager": json.loads((args.candidate / "eager.json").read_text())}
result["same_checkpoint"] = result["baseline_environment"]["checkpoint_sha256"] == result["candidate_environment"]["checkpoint_sha256"]
result["component_equality_without_overriding_strict_result"] = {"old_new": compare_snapshots(left, right, atol=1e-5, rtol=1e-4, require_torch_parity=False), "candidate_isolation": compare_snapshots(right, after, atol=1e-5, rtol=1e-4, require_torch_parity=False)}
result["exact_component_evidence"] = {"old_new": exact_components(left, right), "candidate_isolation": exact_components(right, after)}
result["strict_ok"] = bool(result["same_checkpoint"] and all(item["complete_execution"] and item["torch_circle_parity"] for item in result["audits"].values()) and result["old_new_strict"]["ok"] and result["candidate_isolation_strict"]["ok"] and result["eager"].get("state_preserved") and result["eager"].get("all_hidden_outputs_finite"))
args.output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({"strict_ok": result["strict_ok"], "same_checkpoint": result["same_checkpoint"], "audits": {key: {field: value[field] for field in ("complete_execution", "torch_circle_parity")} for key, value in result["audits"].items()}, "old_new_strict": result["old_new_strict"]["ok"], "isolation_strict": result["candidate_isolation_strict"]["ok"], "components": {key: value["ok"] for key, value in result["component_equality_without_overriding_strict_result"].items()}, "eager": result["eager"]}, indent=2))
raise SystemExit(0 if result["strict_ok"] else 1)
