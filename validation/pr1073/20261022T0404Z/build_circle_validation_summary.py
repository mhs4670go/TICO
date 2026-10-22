"""Aggregate recorded tiny-model Circle evidence without changing raw verdicts."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def read(relative):
    return json.loads((ROOT / relative).read_text())


def regression(relative, scope, infrastructure_copy=False):
    summary = read(f"{relative}/summary.json")
    entry = {
        "level": "tiny random real-HF frozen PTQ + actual Torch/Circle",
        "scope": scope,
        "output": relative,
        "raw_status": summary["status"],
        "comparisons_complete": summary["comparisons_complete"],
        "torch_parity_required": summary.get("torch_parity_required"),
        "atol": summary.get("atol"),
        "rtol": summary.get("rtol"),
        "infrastructure_copy": infrastructure_copy,
        "same_checkpoint_and_dependencies": summary.get("same_checkpoint_and_dependencies"),
        "same_initial_quantization_state": summary.get("same_initial_quantization_state"),
        "eager": summary.get("eager"),
    }
    component = ROOT / relative / "component_summary.json"
    if component.exists():
        entry["components_without_overriding_raw_status"] = json.loads(component.read_text())
    return entry


def main():
    cases = {
        "llama_bundle": regression("tiny-llama-old-new-v2", "native64, export8, eager12+4; baseline/candidate/candidate after", False),
        "llama_full": regression("tiny-llama-full-shape-retry", "native64, export8, eager12+4; full model", True),
        "qwen_per_layer": regression("tiny-qwen-old-new-cached", "native64, export8, eager12+4; vision/text per-layer", False),
        "qwen_full_text_png": regression("tiny-qwen-full-text-png-dirs-retry", "text static graph + one whole-VLM real-PNG eager call", True),
        "gemma_multimodal": regression("tiny-gemma-old-new-image70", "main baseline vision conversion reached unsupported rank2 RMS_NORM", False),
        "gemma_candidate_vision": regression("tiny-gemma-candidate-vision-isolation", "candidate independently reproduced same unsupported vision RMS_NORM", False),
        "gemma_full_text_png": regression("tiny-gemma-full-text-png-masks-retry", "text static graph + one whole-VLM real-PNG eager call; recorded static masks", True),
        "gemma_text_independent": read("gemma_text_analysis.json"),
        "llama_long_eager_full": read("llama_long_eager_analysis.json"),
    }
    observed = {
        name: value["components_without_overriding_raw_status"].get("actual_stage_count_per_phase", {})
        for name, value in cases.items()
        if "components_without_overriding_raw_status" in value
    }
    completed = sum(sum(phases.values()) for phases in observed.values()) + cases["gemma_text_independent"]["actual_total_stages"] + 2
    report = {
        "candidate_sha": "6b2f6a0511458d6a750e2a260d706080a78149dd",
        "baseline_sha": "bb09ca852b0ca1b1a6271cfe1e10c162104c997e",
        "scope": "tiny fixture Circle validation only; excludes separately reported pretrained runs, NPU, static2048, leaderboard",
        "cases": cases,
        "completed_stage_counts": observed,
        "completed_actual_circle_stages_including_independent_gemma_text_and_long_llama": completed,
        "expected_numerical_change": {
            "scope": "Llama bundle host rotary tables and prefill inputs only",
            "evidence": "tiny-llama-old-new-v2/llama_rope_diagnosis.json",
            "candidate_vs_independent_hf_max_abs": 0.0,
            "baseline_cos_max_abs": 4.4465065002441406e-05,
            "baseline_sin_max_abs": 0.006892439676448703,
            "same_external_input_actual_circle_replay_exact": True,
            "raw_strict_fail_retained": True,
        },
        "known_failures": [
            "Baseline-existing strict Torch/Circle mismatches in Llama/Qwen/Gemma decoders/full paths; raw FAIL retained.",
            "Gemma rank2 vision RMS_NORM unsupported in actual Circle reference runtime; raw ERROR retained.",
            "Gemma validate_static_layout=True token-count bug reads hidden width of 2D features; separate default-False profile does not erase it.",
            "Comparator false-accepts both missing Torch arrays/empty parity and both missing input arrays; actual runs independently audited.",
        ],
        "diagnostics": [
            "COMPARATOR_GAPS.md", "FIXTURE_INFRASTRUCTURE.md", "PARITY_DIAGNOSIS.md",
            "ROUNDING_AND_PAST_DIAGNOSIS.json", "QWEN_PARITY_DIAGNOSTIC.md",
            "GEMMA_TEXT_PARITY_DIAGNOSIS.md", "GEMMA_TEXT_STAGE_VALIDATION.md",
            "GEMMA_FULL_TEXT_DIAGNOSTIC.md",
            "tiny-llama-full-shape-retry/README.md",
        ],
        "limits": [
            "Random tiny PTQ fixtures do not establish pretrained-model quality or pre-PR pretrained equivalence.",
            "All static profiles used sequence8; native16384 Llama actual eager4096+4 completed independently.",
            "Full-text probes are not full multimodal static exports.",
            "Custom image kwargs replace automatic text stress; one image forward is not linked-cache multimodal decode.",
            "No tolerance increase, product source edit, baseline source edit, checkpoint recast/recalibration in workers, commit/push/merge/PR metadata edit.",
        ],
    }
    (ROOT / "circle_validation_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"case_count": len(cases), "completed_actual_circle_stages": completed, "output": "circle_validation_summary.json"}))


if __name__ == "__main__":
    main()
