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

"""Saved-byte execution and evidence validation for the policy experiment."""

import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from tico.circle.errors import CircleIOError
from tico.quantization import convert, prepare, QuantStub
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.config.specs import affine
from tico.quantization.recipes.debug.rope_kv_experiment.circle import (
    assert_rope_kv_domains,
    error_metrics,
    export_and_check,
    SavedCircle,
)
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.qscheme import QScheme
from torch import nn


class SavedPolicyCircleTest(unittest.TestCase):
    """Exercise real conversion and reload, without a model download."""

    def test_saved_bytes_execute_and_native_is_unsupported(self):
        torch.manual_seed(123)
        model = nn.Sequential(QuantStub(), nn.Linear(4, 4)).eval()
        spec = affine(DType.uint(8), qscheme=QScheme.PER_TENSOR_ASYMM)
        prepared = prepare(model, PTQConfig(activation=spec, weight=spec), inplace=True)
        sample = torch.randn(1, 2, 4)
        with torch.no_grad():
            prepared(sample)
            quantized = convert(prepared, inplace=True)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = export_and_check(quantized, (sample,), root, "linear")
            path = root / "linear.circle"
            self.assertEqual(
                evidence["sha256"], hashlib.sha256(path.read_bytes()).hexdigest()
            )
            self.assertTrue(evidence["frozen_state_preserved"])
            self.assertEqual(evidence["structural_status"], "PASS")
            self.assertEqual(evidence["runtime"]["native"]["status"], "UNSUPPORTED")
            self.assertEqual(evidence["runtime"]["fake_quantize"]["status"], "EXECUTED")
            saved = SavedCircle(path)
            encoded = saved.encoded_inputs((sample,))
            self.assertEqual(encoded[0].dtype, np.uint8)
            actual = saved.run((sample,))
            self.assertEqual(actual[0].dtype, np.float32)
            self.assertEqual(tuple(actual[0].shape), tuple(sample.shape))
            self.assertTrue(np.isfinite(actual[0]).all())
            path.write_bytes(b"invalid saved bytes")
            with self.assertRaises(CircleIOError):
                SavedCircle(path)

    def test_serialized_concat_legality_is_required(self):
        graph = {
            "concats": [
                {"operator": 4, "region": "rope", "same_serialized_domain": True},
                {
                    "operator": 9,
                    "region": "kv_compute_or_storage",
                    "same_serialized_domain": True,
                },
            ]
        }
        self.assertEqual(
            assert_rope_kv_domains(graph)["checked_concat_counts"],
            {"rope": 1, "kv_compute_or_storage": 1},
        )
        graph["concats"][1]["same_serialized_domain"] = False
        with self.assertRaisesRegex(AssertionError, "Concat operator 9"):
            assert_rope_kv_domains(graph)

    def test_nonfinite_or_empty_metrics_are_errors(self):
        for value in (np.array([np.nan]), np.array([np.inf]), np.array([])):
            with self.assertRaises(ValueError):
                error_metrics(value, value)
        self.assertEqual(error_metrics(np.zeros(4), np.zeros(4))["cosine"], 1.0)
        self.assertEqual(error_metrics(np.zeros(4), np.ones(4))["cosine"], 0.0)


if __name__ == "__main__":
    unittest.main()
