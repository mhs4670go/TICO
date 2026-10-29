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

"""Host cache, independent control, and paired experiment accounting."""

import copy
import tempfile
import unittest
from pathlib import Path

import torch

from tico.quantization.recipes.debug.rope_kv_experiment.circle import export_and_check
from tico.quantization.recipes.debug.rope_kv_experiment.fixture import (
    append,
    call,
    export_args,
    make_fp,
    make_wrapper,
    mask,
    pad_cache,
    sample,
)
from tico.quantization.recipes.debug.rope_kv_experiment.metrics import state_fingerprint
from tico.quantization.recipes.debug.rope_kv_experiment.runner import (
    boundary_metrics,
    calibrate,
    check_noquant,
    contrasts,
    fp_trace,
    qparams,
    semantic_graph,
    validation_status,
)


class TestRopeKVRunnerHost(unittest.TestCase):
    def test_causal_and_sliding_masks_account_for_padding_and_current_slot(self):
        prefill = mask(3, 0)
        self.assertTrue(
            torch.equal(prefill == 0, torch.ones(3, 3).tril().bool()[None, None])
        )
        fixed = mask(1, 3, slots=6)
        self.assertTrue(
            torch.equal(
                fixed.flatten(),
                torch.tensor([0.0, 0.0, 0.0, -120.0, -120.0, -120.0, 0.0]),
            )
        )
        sliding = mask(1, 5, slots=6, window=3)
        self.assertTrue(
            torch.equal(
                sliding.flatten(),
                torch.tensor([-120.0, -120.0, -120.0, 0.0, 0.0, -120.0, 0.0]),
            )
        )
        # A two-query append has a distinct causal current-token boundary.
        append_mask = mask(2, 4, slots=6, window=3)
        expected = torch.tensor(
            [
                [-120.0, -120.0, 0.0, 0.0, -120.0, -120.0, 0.0, -120.0],
                [-120.0, -120.0, -120.0, 0.0, -120.0, -120.0, 0.0, 0.0],
            ]
        )
        self.assertTrue(torch.equal(append_mask[0, 0], expected))

    def test_append_preserves_original_bytes_and_only_evicts_requested_prefix(self):
        old = (
            torch.tensor([[[[9.0, 8.0], [7.0, 6.0]]]]),
            torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]]),
        )
        delta = (torch.tensor([[[[5.0, 4.0]]]]), torch.tensor([[[[5.0, 6.0]]]]))
        snapshots = tuple(value.clone() for value in old)
        combined = append(old, delta)
        for previous, snapshot, actual, new in zip(old, snapshots, combined, delta):
            self.assertTrue(torch.equal(previous, snapshot))
            self.assertTrue(torch.equal(actual[..., :2, :], snapshot))
            self.assertTrue(torch.equal(actual[..., 2:, :], new))
        retained = append(old, delta, retain=2)
        self.assertTrue(torch.equal(retained[0], combined[0][..., -2:, :]))
        padded = pad_cache(old, 5)
        self.assertTrue(torch.equal(padded[0][..., :2, :], old[0]))
        self.assertEqual(torch.count_nonzero(padded[0][..., 2:, :]), 0)
        self.assertIsNot(append(None, delta)[0], delta[0])

    def test_paired_contrasts_use_requested_factorial_signs(self):
        actual = contrasts({"R0K0": 10.0, "R1K0": 7.0, "R0K1": 6.0, "R1K1": 5.0})
        self.assertEqual(
            actual,
            {
                "R_at_K0": -3.0,
                "R_at_K1": -1.0,
                "K_at_R0": -4.0,
                "K_at_R1": -2.0,
                "interaction": 2.0,
            },
        )


class TestRopeKVRunnerEvidence(unittest.TestCase):
    def test_compute_boundaries_keep_past_and_current_errors_separate(self):
        reference = torch.tensor([1.0, 1.0, 1.0, 1.0, 4.0]).reshape(1, 1, 5, 1)
        actual = [
            reference[..., :4, :].clone(),
            torch.tensor([8.0]).reshape(1, 1, 1, 1),
            reference.clone(),
        ]
        actual[2][..., -1, :] = 8.0
        record = boundary_metrics(
            {"obs_present_key": [reference]}, {"obs_present_key": actual}
        )
        self.assertEqual(record["obs_present_key.past"][0]["mae"], 0.0)
        self.assertEqual(record["obs_present_key.current"][0]["mae"], 4.0)
        self.assertEqual(record["obs_present_key.concat"][0]["mae"], 0.8)
        self.assertEqual(record["obs_present_key.current"][0]["elements"], 1)
        with self.assertRaisesRegex(ValueError, "call counts"):
            boundary_metrics({"obs_q_rot": [reference]}, {"obs_q_rot": actual})
        with self.assertRaisesRegex(ValueError, "shape"):
            boundary_metrics(
                {"obs_q_rot": [reference]}, {"obs_q_rot": [reference[..., :1, :]]}
            )

    @staticmethod
    def _executed_evidence():
        circle = {
            "structural_status": "PASS",
            "frozen_state_preserved": True,
            "runtime": {
                "native": {"status": "UNSUPPORTED"},
                "fake_quantize": {
                    "status": "EXECUTED",
                    "strict_fidelity_status": "PASS",
                    "outputs": [{"mae": 0.0}],
                },
            },
        }
        return {"prefill": copy.deepcopy(circle), "decode": copy.deepcopy(circle)}, {
            "status": "EXECUTED",
            "steps": [{"step": 1}],
        }

    def test_missing_runtime_and_empty_measurements_never_pass(self):
        circles, recurrent = self._executed_evidence()
        self.assertEqual(validation_status(circles, recurrent), "PASS")
        self.assertEqual(validation_status({}, {}, expected_circle=False), "NOT_RUN")
        malformed = [
            ({}, recurrent),
            (circles, {"status": "NOT_RUN"}),
            (circles, {"status": "UNSUPPORTED"}),
            (circles, {"status": "EXECUTED", "steps": []}),
        ]
        for section, replacement in [
            ("structural_status", "ERROR"),
            ("frozen_state_preserved", False),
        ]:
            altered = copy.deepcopy(circles)
            altered["decode"][section] = replacement
            malformed.append((altered, recurrent))
        for mode in ("native", "fake_quantize"):
            altered = copy.deepcopy(circles)
            altered["decode"]["runtime"][mode]["status"] = "ERROR"
            malformed.append((altered, recurrent))
        for field in ("outputs", "strict_fidelity_status"):
            altered = copy.deepcopy(circles)
            del altered["decode"]["runtime"]["fake_quantize"][field]
            malformed.append((altered, recurrent))
        for artifact, rollout in malformed:
            with self.subTest(artifact=artifact, rollout=rollout):
                self.assertEqual(validation_status(artifact, rollout), "ERROR")
        mismatch = copy.deepcopy(circles)
        mismatch["decode"]["runtime"]["fake_quantize"][
            "strict_fidelity_status"
        ] = "FAIL"
        self.assertEqual(validation_status(mismatch, recurrent), "FIDELITY_DIFFERENCE")


class TestRopeKVRunnerControl(unittest.TestCase):
    def test_no_quant_wrapper_matches_independent_hf_prefill_and_decode(self):
        with torch.no_grad():
            fp, _ = make_fp(123, 20)
            rows, _ = fp_trace(fp, "affine", 20, 123, 3)
            result = check_noquant(fp, "affine", 20, rows)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(len(result["comparisons"]), 4)
        for step in result["comparisons"]:
            self.assertEqual(len(step), 5)
            self.assertTrue(all(record["max_abs"] <= result["atol"] for record in step))

    def test_r0k1_control_has_same_calibration_eager_and_saved_graph_contract(self):
        with torch.no_grad(), tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fp, _ = make_fp(456, 20)
            production = make_wrapper(fp, "u8s16c", 20)
            experiment = make_wrapper(fp, "u8s16c", 20, "R0K1")
            for model in (production, experiment):
                calibrate(model, 456, samples=1, decode_steps=2)
            self.assertEqual(qparams(experiment), qparams(production))
            frozen = [state_fingerprint(model) for model in (production, experiment)]
            caches = [None, None]
            decode = None
            for step in range(4):
                hidden, rope = sample(
                    900 + step, 8 if step == 0 else 1, 0 if step == 0 else 7 + step
                )
                outputs = []
                for index, model in enumerate((production, experiment)):
                    output = call(model, hidden, rope, caches[index])
                    if step == 1 and index == 1:
                        decode = export_args(
                            model, hidden, rope, caches[index], slots=19
                        )
                    caches[index] = append(caches[index], output[2])
                    outputs.append(output)
                for left, right in zip(outputs[0][:2], outputs[1][:2]):
                    self.assertTrue(torch.equal(left, right))
                for left, right in zip(outputs[0][2], outputs[1][2]):
                    self.assertTrue(torch.equal(left, right))
            assert decode is not None
            records = [
                export_and_check(
                    model.as_export_module("decode").eval(),
                    decode,
                    root / name,
                    "decode",
                )
                for model, name in zip(
                    (production, experiment), ("production", "experiment")
                )
            ]
            self.assertEqual(semantic_graph(records[0]), semantic_graph(records[1]))
            # Known class names are normalized, but actual qparam changes must fail.
            changed = semantic_graph(records[1])
            changed["inputs"][0]["qparams"]["scale"][0] *= 2
            self.assertNotEqual(semantic_graph(records[0]), changed)
            self.assertEqual(
                frozen, [state_fingerprint(model) for model in (production, experiment)]
            )
            self.assertTrue(
                all(
                    record["runtime"]["fake_quantize"]["status"] == "EXECUTED"
                    for record in records
                )
            )


if __name__ == "__main__":
    unittest.main()
