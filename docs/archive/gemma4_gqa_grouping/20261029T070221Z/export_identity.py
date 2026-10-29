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

"""Recreate deterministic export fixtures and verify committed product identity."""

import argparse, hashlib, json, pathlib, subprocess
from typing import Any

import torch
from export_check import (
    config,
    digest,
    Gemma4TextAttention,
    PTQConfig,
    QuantGemma4TextAttention,
    sample,
)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=pathlib.Path, required=True)
    ns = ap.parse_args()
    torch.set_num_threads(1)
    source = "tico/quantization/wrapq/wrappers/gemma4/quant_text_attention.py"
    committed = subprocess.check_output(["git", "show", "HEAD:" + source])
    actual = pathlib.Path(source).read_bytes()
    assert committed == actual
    out: dict[str, Any] = {
        "implementation_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "product_source_sha256": hashlib.sha256(actual).hexdigest(),
        "committed_source_verified": True,
        "note": "Measured worktree product was committed unchanged; identity is verified after export. Fixture fingerprints are regenerated deterministically.",
        "fixtures": [],
    }
    for hq, hkv in [(24, 8), (4, 2)]:
        for quant in [False, True]:
            for mode in ["prefill", "decode"]:
                torch.manual_seed(20261029)
                cfg = config(hq, hkv)
                cfg._attn_implementation = "eager"
                attn = QuantGemma4TextAttention(
                    Gemma4TextAttention(cfg, layer_idx=0).eval(),
                    qcfg=PTQConfig(model_args={"text": {"max_seq": 32}}),
                ).eval()
                weights = digest(list(attn.parameters()))
                cal = []
                if quant:
                    attn.enable_calibration()
                    for j in range(3):
                        a, k = sample(cfg, 3, 6)
                        cal.append(digest(torch.utils._pytree.tree_flatten((a, k))[0]))
                        with torch.no_grad():
                            attn(a[0], a[2], attention_mask=a[1], use_cache=True, **k)
                    attn.freeze_qparams()
                t, s = (3, 3) if mode == "prefill" else (1, 8)
                a, k = sample(cfg, t, s)
                out["fixtures"].append(
                    {
                        "Hq": hq,
                        "Hkv": hkv,
                        "quant": quant,
                        "mode": mode,
                        "weights_sha256": weights,
                        "calibration_sha256": cal,
                        "evaluation_sha256": digest(
                            torch.utils._pytree.tree_flatten((a, k))[0]
                        ),
                    }
                )
    ns.output.write_text(json.dumps(out, indent=2))
