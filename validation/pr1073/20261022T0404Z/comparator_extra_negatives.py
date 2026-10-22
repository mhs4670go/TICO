"""Check comparator rejection of incomplete evidence using unit fixtures only."""

import json
from pathlib import Path
import tempfile

import numpy as np

from test.quantization.recipes.test_export_regression import _phase
from tico.quantization.recipes.debug.export_regression import compare_snapshots


def main():
    report = {"scope":"comparator unit negative tests; schema-shaped fixture, no compiler/runtime", "cases":{}}
    for case in ("missing_torch_arrays_and_parity", "missing_input_arrays"):
        with tempfile.TemporaryDirectory() as directory:
            left, right = Path(directory)/"left", Path(directory)/"right"
            for root in (left, right):
                _phase(root)
                arrays = root/"records/decoder.npz"
                with np.load(arrays) as loaded:
                    values = {key:loaded[key].copy() for key in loaded.files}
                if case == "missing_torch_arrays_and_parity":
                    values = {key:value for key,value in values.items() if "_torch_" not in key}
                    record_path = root/"records/decoder.json"
                    record = json.loads(record_path.read_text())
                    for probe in record["probes"]:
                        probe["torch_parity"] = []
                    record_path.write_text(json.dumps(record))
                else:
                    values = {key:value for key,value in values.items() if "_input_" not in key}
                np.savez(arrays, **values)
            actual = compare_snapshots(left,right,atol=1e-5,rtol=1e-4,require_torch_parity=True)
            report["cases"][case] = {"expected_comparator_ok":False,
                "actual_comparator_ok":actual["ok"],
                "negative_test_status":"FAIL" if actual["ok"] else "PASS",
                "comparison":actual}
    destination = Path(__file__).with_name("comparator_extra_negatives.json")
    destination.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
    raise SystemExit(1 if any(item["actual_comparator_ok"] for item in report["cases"].values()) else 0)


if __name__ == "__main__":
    main()
