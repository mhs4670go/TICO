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

"""Independent metric accounting and observational trace contracts."""

import math
import unittest

import torch

from tico.quantization.recipes.debug.rope_kv_experiment.metrics import (
    BoundaryTrace,
    cached_nll,
    observer_records,
    ObserverTrace,
    shifted_cached_nll,
    state_fingerprint,
    tensor_metrics,
)
from tico.quantization.wrapq.observers.minmax import MinMaxObserver
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase


class TestRoPEKVMetrics(unittest.TestCase):
    def test_metric_values_and_zero_norm_convention(self):
        result = tensor_metrics(torch.tensor([0.0, 2.0]), torch.tensor([1.0, 1.0]))
        self.assertEqual(result["mae"], 1.0)
        self.assertEqual(result["rmse"], 1.0)
        self.assertEqual(result["max_abs"], 1.0)
        self.assertAlmostEqual(result["normalized_l2"], math.sqrt(0.5))
        self.assertAlmostEqual(result["cosine"], math.sqrt(0.5))
        self.assertEqual(result["peir"], 0.5)
        self.assertEqual(tensor_metrics(torch.zeros(2), torch.zeros(2))["cosine"], 1.0)
        zero = tensor_metrics(torch.zeros(2), torch.ones(2))
        self.assertEqual(zero["cosine"], 0.0)
        self.assertEqual(zero["peir_denominator"], 1.0)
        self.assertTrue(math.isfinite(zero["normalized_l2"]))

    def test_missing_nonfinite_or_misaligned_metric_evidence_rejects(self):
        for left, right in (
            (torch.empty(0), torch.empty(0)),
            (torch.ones(1), torch.ones(2)),
            (torch.tensor([math.nan]), torch.ones(1)),
            (torch.ones(1), torch.tensor([math.inf])),
        ):
            with self.subTest(left=left, right=right), self.assertRaises(ValueError):
                tensor_metrics(left, right)

    def test_trace_counts_actual_collection_and_restores_on_exception(self):
        observer = MinMaxObserver(name="sample")
        model = torch.nn.Module()
        model.add_module("observer", observer)
        self.assertNotIn("collect", observer.__dict__)
        with ObserverTrace(model) as trace:
            observer.collect(torch.tensor([-2.0, 3.0]))
            observer.enabled = False
            observer.collect(torch.tensor([-9.0, 9.0]))
            observer.collect(torch.empty(0))
        record = trace.summary()["observer"]
        self.assertEqual(record["collect_calls"], 3)
        self.assertEqual(record["effective_collect_calls"], 1)
        self.assertEqual(record["collect_min"], -2.0)
        self.assertEqual(record["collect_max"], 3.0)
        self.assertEqual(float(observer.min_val), -2.0)
        self.assertNotIn("collect", observer.__dict__)
        self.assertNotIn("fake_quant", observer.__dict__)
        with self.assertRaisesRegex(RuntimeError, "intentional"):
            with ObserverTrace(model):
                raise RuntimeError("intentional")
        self.assertNotIn("collect", observer.__dict__)

    def test_clipping_is_distinct_from_endpoint_occupancy_and_state_preserved(self):
        observer = MinMaxObserver(name="sample")
        observer.load_qparams(torch.tensor(1.0), torch.tensor(0), lock=True)
        before = state_fingerprint(observer)
        values = torch.tensor([-0.1, 0.0, 0.2, 254.8, 255.0, 255.1])
        with ObserverTrace(observer) as trace:
            actual = observer.fake_quant(values)
        record = trace.summary()[""]
        self.assertEqual(record["clipped_elements"], 2)
        self.assertEqual(record["endpoint_elements"], 6)
        self.assertEqual(record["clipping_rate"], 2 / 6)
        self.assertEqual(record["endpoint_occupancy"], 1.0)
        self.assertEqual(state_fingerprint(observer), before)
        self.assertTrue(
            torch.equal(actual, torch.tensor([0.0, 0.0, 0.0, 255.0, 255.0, 255.0]))
        )
        record = observer_records(observer)[0]
        self.assertEqual(record["origin"], "imported")
        self.assertEqual(record["scale"], 1.0)
        self.assertEqual(record["dtype"], "uint8")
        observer._cached_scale.mul_(2)
        self.assertNotEqual(state_fingerprint(observer), before)

    def test_boundary_trace_no_quant_and_reorder_have_one_semantic_record(self):
        class Fixture(QuantModuleBase):
            def __init__(self, inherited):
                super().__init__()
                self.inherited = inherited
                self.obs_in = MinMaxObserver(name="input")
                self.obs_cat = MinMaxObserver(name="cat")

            def _all_observers(self):
                return self.obs_in, self.obs_cat

            def _rot(self, value, obs_x1, obs_x2, obs_neg, obs_cat):
                first, second = value.chunk(2, dim=-1)
                result = torch.cat((second, first), dim=-1)
                return result if self.inherited else self._fq(result, obs_cat)

            def forward(self, value):
                value = self._fq(value, self.obs_in)
                return self._rot(value, None, None, None, self.obs_cat)

        for inherited in (False, True):
            with self.subTest(inherited=inherited):
                model = Fixture(inherited)
                value = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
                before = state_fingerprint(model)
                with BoundaryTrace(model) as trace:
                    actual = model(value)
                self.assertTrue(
                    torch.equal(actual, torch.tensor([[3.0, 4.0, 1.0, 2.0]]))
                )
                self.assertEqual(len(trace.values["obs_cat"]), 1)
                self.assertEqual(len(trace.values["obs_in"]), 1)
                self.assertTrue(torch.equal(trace.values["obs_cat"][0], actual))
                self.assertEqual(before, state_fingerprint(model))
                self.assertNotIn("_fq", model.__dict__)
                self.assertNotIn("_rot", model.__dict__)

    def test_cached_target_shift_and_exact_token_sum(self):
        # Uniform vocab-2 logits contribute log(2) each. Nonuniform logits
        # expose an incorrect target shift: first target is 1, next is 0.
        logits = torch.tensor([[[0.0, math.log(3)], [math.log(3), 0.0]]])
        sequence = torch.tensor([[0, 0, 1, 0, 1]])
        scored = shifted_cached_nll(logits, sequence, prefix_tokens=2)
        expected = -math.log(0.75)
        self.assertEqual(scored["target_tokens"], [1, 0])
        self.assertEqual(scored["scored_tokens"], 2)
        self.assertAlmostEqual(scored["nll_sum"], expected * 2, places=7)
        self.assertAlmostEqual(scored["mean_nll"], expected, places=7)
        self.assertAlmostEqual(scored["ppl"], 4 / 3, places=7)
        self.assertEqual(scored["target_start"], 2)
        masked = cached_nll(logits, sequence[:, 2:4], torch.tensor([[False, True]]))
        self.assertEqual(masked["scored_tokens"], 1)
        self.assertAlmostEqual(masked["nll_sum"], expected, places=7)

    def test_cached_nll_rejects_empty_scores_bad_alignment_and_nonfinite(self):
        logits = torch.zeros(1, 2, 3)
        targets = torch.zeros(1, 2, dtype=torch.long)
        for call in (
            lambda: cached_nll(
                logits, targets, torch.zeros_like(targets, dtype=torch.bool)
            ),
            lambda: cached_nll(logits, targets[:, :1]),
            lambda: cached_nll(logits + math.nan, targets),
            lambda: shifted_cached_nll(logits, targets, 0),
            lambda: shifted_cached_nll(logits, targets, 1),
            lambda: shifted_cached_nll(logits, targets, True),
        ):
            with self.assertRaises(ValueError):
                call()


if __name__ == "__main__":
    unittest.main()
