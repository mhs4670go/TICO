"""Diagnose raw Llama old/new failures without changing comparator outcomes."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np


def difference(left, right):
    if left.shape != right.shape:
        return {"same_shape":False,"left_shape":list(left.shape),"right_shape":list(right.shape)}
    delta = np.abs(left.astype(np.float64)-right.astype(np.float64))
    unequal = np.argwhere(left != right)
    return {"same_shape":True,"max_abs_error":float(delta.max()) if delta.size else 0.0,
            "different_elements":int(len(unequal)), "first_positions":unequal[:8].tolist()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",required=True)
    parser.add_argument("--provenance",required=True)
    args=parser.parse_args()
    sys.path.insert(0,"/workspace/TICO")
    import torch
    from transformers.models.llama.configuration_llama import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
    from tico.circle.runtime import CircleReferenceRuntime, ExecutionMode
    from tico.quantization.recipes.debug.export_regression import compare_arrays
    torch.set_num_threads(1)
    run=Path(args.run)
    provenance=json.loads(Path(args.provenance).read_text())
    before=run/"baseline/before"
    after=run/"candidate/before"
    report={"scope":"additional diagnosis; raw original old/new comparator FAIL is preserved",
            "host":{},"stages":{}}
    baseline_host=np.load(before/"artifacts/llama_host.npz")
    candidate_host=np.load(after/"artifacts/llama_host.npz")
    length=next(candidate_host[key].shape[1] for key in candidate_host.files if key.startswith("rope_cos"))
    rotary=LlamaRotaryEmbedding(config=LlamaConfig.from_dict(provenance["hf_config"]))
    with torch.no_grad():
        expected_cos,expected_sin=rotary(torch.empty((),dtype=torch.float32),torch.arange(length).unsqueeze(0))
    half=expected_sin.shape[-1]//2
    expected_sin=torch.cat((-expected_sin[...,:half],expected_sin[...,half:]),dim=-1)
    oracle={"rope_cos":expected_cos.numpy(),"rope_sin":expected_sin.numpy()}
    for key in candidate_host.files:
        item={"old_new":difference(baseline_host[key],candidate_host[key])}
        for prefix,target in oracle.items():
            if key.startswith(prefix):
                item["candidate_vs_independent_HF"]=difference(candidate_host[key],target)
                item["baseline_vs_independent_HF"]=difference(baseline_host[key],target)
        report["host"][key]=item
    old=json.loads((before/"snapshot.json").read_text())
    new=json.loads((after/"snapshot.json").read_text())
    for stage in sorted(old["stages"].keys() & new["stages"].keys()):
        old_record=old["stages"][stage]
        new_record=new["stages"][stage]
        with np.load(before/"records"/(old_record+".npz")) as av, np.load(after/"records"/(new_record+".npz")) as bv:
            deltas={key:difference(av[key],bv[key]) for key in av.files if key in bv.files and not np.array_equal(av[key],bv[key])}
            ar=CircleReferenceRuntime((before/"artifacts"/stage).read_bytes())
            br=CircleReferenceRuntime((after/"artifacts"/stage).read_bytes())
            mode=ExecutionMode.FAKE_QUANTIZE if ar.has_quantized_activations() else ExecutionMode.NATIVE
            probe_meta=json.loads((before/"records"/(old_record+".json")).read_text())
            same_inputs=[]
            for index,probe in enumerate(probe_meta["probes"]):
                keys=sorted((key for key in av.files if key.startswith(f"p{index}_input_")),key=lambda key:int(key.rsplit("_",1)[1]))
                inputs=[av[key].copy() for key in keys]
                ao,bo=ar.run(inputs,mode=mode).outputs,br.run(inputs,mode=mode).outputs
                comparisons=[compare_arrays(x,y,atol=1e-5,rtol=1e-4) for x,y in zip(ao,bo)]
                same_inputs.append({"probe":probe["probe"],"output_count_equal":len(ao)==len(bo),"comparisons":comparisons})
        report["stages"][stage]={"raw_array_differences":deltas,"same_external_inputs_replay":same_inputs}
    report["same_initial_frozen_state"] = json.loads((run/"baseline/quantization_state.json").read_text()) == json.loads((run/"candidate/quantization_state.json").read_text())
    (run/"llama_rope_diagnosis.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
