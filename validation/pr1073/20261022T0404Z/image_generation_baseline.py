"""Load the exact trusted full module in the selected checkout and call generate."""
import argparse
import hashlib
import json
import traceback
from pathlib import Path
import torch
import tico

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--inputs", type=Path, required=True)
parser.add_argument("--report", type=Path, required=True)
args = parser.parse_args()
result = {"tico_file": tico.__file__, "checkpoint": str(args.checkpoint), "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(), "torch": torch.__version__}
try:
    model = torch.load(args.checkpoint, map_location="cpu", weights_only=False).eval()
    payload = torch.load(args.inputs, map_location="cpu", weights_only=False)
    result["class"] = type(model).__module__ + "." + type(model).__name__
    result["parameter_dtypes"] = sorted({str(value.dtype) for value in model.parameters()})
    torch.manual_seed(1073)
    with torch.no_grad():
        tokens = model.generate(**payload["sample"], **payload["settings"])
    result.update(status="PASS", tokens=tokens.tolist())
except Exception as error:
    traceback.print_exc()
    result.update(status="FAIL", error=repr(error), traceback=traceback.format_exc())
args.report.write_text(json.dumps(result, indent=2) + "\n")
raise SystemExit(0 if result["status"] == "PASS" else 1)
