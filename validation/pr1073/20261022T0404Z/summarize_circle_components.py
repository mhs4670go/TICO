"""Summarize comparison components without overriding the original strict result."""
import argparse
import json
from pathlib import Path


def component(value):
    stages = value.get("stages", {})
    return {"stage_count":len(stages),
        "structure_equal":bool(stages) and all(stage["circle_structure_equal"] for stage in stages.values()),
        "all_input_and_output_arrays_match":bool(stages) and all(all(item["ok"] for item in stage["arrays"].values()) for stage in stages.values()),
        "host_equal":not any(item.startswith("Host") for item in value.get("failures",[])),
        "strict_comparison_ok":value.get("ok"),
        "stage_failures":value.get("failures",[]),
        "actual_numerical_differences":{name:{key:item for key,item in stage["arrays"].items() if not item["ok"]} for name,stage in stages.items() if any(not item["ok"] for item in stage["arrays"].values())}}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",required=True)
    args=p.parse_args()
    root=Path(args.run)
    summary=json.loads((root/"summary.json").read_text())
    report={"raw_status":summary["status"],"raw_torch_parity_required":summary.get("torch_parity_required"),
        "comparisons_complete":summary["comparisons_complete"],
        "same_checkpoint_and_dependencies":summary.get("same_checkpoint_and_dependencies"),
        "same_initial_quantization_state":summary.get("same_initial_quantization_state"),
        "frozen_state_preserved":summary.get("eager",{}).get("state_preserved"),
        "baseline_candidate_components":component(summary.get("baseline_vs_candidate",{})),
        "candidate_before_after_components":component(summary.get("candidate_before_vs_after_eager",{}))}
    inventory=root/"independent_evidence_inventory.json"
    if inventory.exists():
        phases=json.loads(inventory.read_text())["phases"]
        probes=[probe for phase in phases.values() for stage in phase["stages"].values() for probe in stage["probes"]]
        report.update(actual_stage_count_per_phase={key:value["stage_count"] for key,value in phases.items()},
            independent_complete_array_coverage=bool(probes) and all(probe["coverage_complete"] for probe in probes),
            independent_finite_arrays=bool(probes) and all(probe["finite"] for probe in probes))
    diagnostic=root/"llama_rope_diagnosis.json"
    if diagnostic.exists():
        diag=json.loads(diagnostic.read_text())
        report.update(old_new_same_external_input_replay_exact=all(item["max_abs_error"]==0 for stage in diag["stages"].values() for probe in stage["same_external_inputs_replay"] for item in probe["comparisons"]),host_diagnosis=diag["host"])
    (root/"component_summary.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({key:value for key,value in report.items() if key!="host_diagnosis"},indent=2))


if __name__=="__main__":
    main()
