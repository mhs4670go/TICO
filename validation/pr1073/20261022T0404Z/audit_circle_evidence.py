"""Independently inventory complete real Circle stage inputs and outputs."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    sys.path.insert(0, "/workspace/TICO")
    from tico.circle import CircleDocument
    run = Path(args.run).resolve()
    report = {"run":str(run),"scope":"actual serialized Circle stage evidence inventory", "phases":{}}
    overall = True
    for snapshot_path in sorted(run.rglob("snapshot.json")):
        phase = snapshot_path.parent
        snapshot = json.loads(snapshot_path.read_text())
        existing = {str(path.relative_to(phase/"artifacts")) for path in (phase/"artifacts").rglob("*.circle")}
        complete = existing == set(snapshot["stages"])
        stages = {}
        for stage, record_name in snapshot["stages"].items():
            metadata = json.loads((phase/"records"/(record_name+".json")).read_text())
            with CircleDocument.load_mapped(phase/"artifacts"/stage) as document:
                document.verify(raise_on_error=True)
                graph = document.model.subgraphs[0]
                input_count, output_count = len(graph.inputs), len(graph.outputs)
                boundary = {kind:[{"shape":list(map(int,graph.tensors[int(i)].shape)),
                    "type":int(graph.tensors[int(i)].type),
                    "name":str(graph.tensors[int(i)].name)} for i in indices]
                    for kind,indices in (("inputs",graph.inputs),("outputs",graph.outputs))}
                qparams = sum(tensor.quantization is not None for tensor in graph.tensors)
            probes = []
            with np.load(phase/"records"/(record_name+".npz")) as arrays:
                for index, probe in enumerate(metadata["probes"]):
                    groups = {kind:sorted(key for key in arrays.files if key.startswith(f"p{index}_{kind}_"))
                              for kind in ("input","circle","torch")}
                    finite = all(np.isfinite(arrays[key]).all() for keys in groups.values() for key in keys)
                    coverage = (len(groups["input"])==input_count and len(groups["circle"])==output_count
                                and len(groups["torch"])==output_count and len(probe["torch_parity"])==output_count)
                    parity = bool(probe["torch_parity"]) and all(item["ok"] for item in probe["torch_parity"])
                    ok = bool(finite and coverage and parity)
                    complete &= ok
                    probes.append({"probe":probe["probe"],"ok":ok,
                        "expected_inputs":input_count,"expected_outputs":output_count,
                        "array_keys":groups,"finite":bool(finite),
                        "coverage_complete":coverage,"torch_parity":probe["torch_parity"],
                        "circle_shapes":{key:list(arrays[key].shape) for key in groups["circle"]}})
            stages[stage] = {"boundary":boundary,"qparam_tensor_count":qparams,"probes":probes,
                             "execution_mode":metadata["mode"]}
        phase_ok = bool(complete and stages and snapshot["state_preserved"])
        overall &= phase_ok
        report["phases"][str(phase.relative_to(run))] = {"ok":phase_ok,
            "artifact_stage_sets_equal":existing==set(snapshot["stages"]),
            "stage_count":len(stages),"state_preserved":snapshot["state_preserved"],"stages":stages}
    report["ok"] = bool(overall and report["phases"])
    destination = run/"independent_evidence_inventory.json"
    destination.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"ok":report["ok"],"report":str(destination),
        "phases":{key:{k:v for k,v in value.items() if k!="stages"} for key,value in report["phases"].items()}},indent=2))
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
