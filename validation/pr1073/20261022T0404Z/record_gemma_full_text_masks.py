"""Record the immutable checkpoint templates used by the corrected full-text probe."""
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,"/workspace/TICO")
import numpy as np
import torch

root=Path(__file__).resolve().parent
checkpoint=root/"fixtures/baseline_gemma_image70.pt"
model=torch.load(checkpoint,map_location="cpu",weights_only=False)
owner=next(module for module in model.modules() if type(module).__name__=="QuantGemma4TextModel")
values={layer_type:getattr(owner,owner._mask_template_name(layer_type))[...,:8,:8].detach().numpy().copy() for layer_type in owner.unique_layer_types}
destination=root/"tiny-gemma-full-text-png-masks-retry"
np.savez(destination/"recorded_host_masks.npz",**values)
report={"scope":"provenance for recorded host templates; not an additional export/metric result",
    "checkpoint":str(checkpoint),"checkpoint_sha256":hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    "text_export_capacity":8,"sliding_window":owner.config.sliding_window,
    "capture":"same recorded checkpoint template slices captured in TextProbe.__init__ before tracing, unchanged for both worker checkouts",
    "masks":{name:{"shape":list(value.shape),"dtype":str(value.dtype),"sha256":hashlib.sha256(value.tobytes()).hexdigest(),"min":float(value.min()),"max":float(value.max())} for name,value in values.items()}}
(destination/"recorded_host_masks.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2))
