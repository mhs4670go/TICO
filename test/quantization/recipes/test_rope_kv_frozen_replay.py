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

"""Small fixed-grid replay distinguishes both members of the KV policy bundle."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from tico.quantization.recipes.debug.rope_kv_experiment.fixture import make_fp
from tico.quantization.recipes.debug.rope_kv_experiment.runner import (
    fp_trace,
    frozen_replay,
)


class TestRoPEKVFrozenReplay(unittest.TestCase):
    def test_six_paths_share_storage_grids_bytes_and_preserve_frozen_state(self):
        with tempfile.TemporaryDirectory() as directory, torch.no_grad():
            root = Path(directory)
            fp, _ = make_fp(123, 16)
            rows, _ = fp_trace(fp, "u8s16c", 16, 123, 1)
            results = frozen_replay(fp, "u8s16c", 16, 123, rows, root)
            payload = json.loads((root / "frozen-replay.json").read_text())
            self.assertEqual(len(results), 6)
            self.assertTrue(payload["source_frozen_unchanged"])
            self.assertEqual(
                len({row["past_S_bytes_sha256"] for row in results.values()}), 1
            )
            for row in results.values():
                self.assertTrue(row["frozen_unchanged"])
                self.assertIn("obs_q_cat", row["boundaries_vs_local_fp"])
                self.assertIn("obs_k_cat", row["boundaries_vs_local_fp"])
                self.assertIn("obs_present_key.current", row["boundaries_vs_local_fp"])
                self.assertIn("obs_present_value.past", row["boundaries_vs_local_fp"])
            for pair in payload["paired_delta_codes"].values():
                for tensor in pair.values():
                    self.assertTrue(tensor["numeric_S_grid_identical"])
                    self.assertIsNotNone(tensor["integer_code_mismatch_rate"])
            for key in ("K", "V"):
                # The K axis changes attention compute paths, never the S delta.
                self.assertEqual(
                    results["R0K1"]["delta_codes_vs_R0K0"][key][
                        "integer_code_mismatch_rate"
                    ],
                    0.0,
                )
            self.assertEqual(
                results["R0K0"]["clipping"]["obs_experiment_past_key"][
                    "fake_quant_calls"
                ],
                1,
            )
            self.assertEqual(
                results["R0_pastP_removed"]["clipping"]["obs_experiment_past_key"][
                    "fake_quant_calls"
                ],
                0,
            )
            self.assertEqual(
                results["R0_currentS_removed"]["clipping"]["obs_experiment_past_key"][
                    "fake_quant_calls"
                ],
                1,
            )
            with np.load(root / "frozen-replay.npz", allow_pickle=False) as archive:
                self.assertEqual(archive["past_k"].dtype, np.uint8)
                for variant in ("R0_pastP_removed", "R0_currentS_removed", "R0K1"):
                    self.assertFalse(np.array_equal(archive["R0K0"], archive[variant]))
            metric = {
                name: row["local_fp_output"]["mae"] for name, row in results.items()
            }
            expected = (
                metric["R0K1"]
                - metric["R0_currentS_removed"]
                - metric["R0_pastP_removed"]
                + metric["R0K0"]
            )
            self.assertEqual(payload["kv_decomposition"]["interaction"], expected)


if __name__ == "__main__":
    unittest.main()
