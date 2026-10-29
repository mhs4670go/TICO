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

"""Focused regressions for the opt-in boundary experiment."""

import copy
import unittest
from typing import Any
from unittest.mock import patch

import torch

from tico.quantization.config.ptq import PTQConfig
from tico.quantization.config.specs import affine
from tico.quantization.recipes.debug.rope_kv_experiment.policy import (
    ExperimentGemma4TextAttention,
)
from tico.quantization.wrapq.dtypes import INT16, UINT8
from tico.quantization.wrapq.observers.ema import EMAObserver
from tico.quantization.wrapq.wrappers.gemma4.quant_text_attention import (
    QuantGemma4TextAttention,
)

from test.quantization.wrapq.wrappers.gemma4.test_quant_text_attention import (
    _has_gemma4,
    _make_text_config,
)


@unittest.skipUnless(_has_gemma4(), "Transformers Gemma4 is unavailable")
class TestRopeKVPolicy(unittest.TestCase):
    def setUp(self):
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextAttention

        torch.manual_seed(34)
        self.fp = Gemma4TextAttention(_make_text_config(), layer_idx=0).eval()
        self.hidden = torch.randn(1, 3, 16)
        self.rope = (torch.ones(1, 3, 4), torch.zeros(1, 3, 4))
        self.qcfg = PTQConfig(
            overrides={"new_k": affine(UINT8), "new_v": affine(UINT8)},
            activation=affine(INT16),
            model_args={"text": {"max_seq": 32}},
        )

    def _make(self, r=False, k=True, qcfg=None):
        return ExperimentGemma4TextAttention(
            copy.deepcopy(self.fp),
            qcfg=copy.deepcopy(qcfg or self.qcfg),
            rope_unified=r,
            kv_simplified=k,
        ).eval()

    def _calibrate(self, attention):
        attention.enable_calibration()
        with torch.no_grad():
            past = attention(self.hidden, self.rope, use_cache=True)[2]
            attention(
                self.hidden[:, :1],
                tuple(table[:, :1] for table in self.rope),
                past_key_value=past,
                use_cache=True,
            )
        attention.freeze_qparams()

    def test_r0k1_matches_independent_production_across_decode(self):
        production = QuantGemma4TextAttention(
            copy.deepcopy(self.fp), qcfg=copy.deepcopy(self.qcfg)
        ).eval()
        experiment = self._make()
        for attention in (production, experiment):
            self._calibrate(attention)
        for (name, actual), (control_name, control) in zip(
            experiment.named_observers(), production.named_observers()
        ):
            self.assertEqual(name, control_name)
            for actual_grid, expected_grid in zip(
                actual.require_frozen().clone().__dict__.values(),
                control.require_frozen().clone().__dict__.values(),
            ):
                if isinstance(actual_grid, torch.Tensor):
                    self.assertTrue(torch.equal(actual_grid, expected_grid), name)
        caches: list[Any] = [None, None]
        for step in range(6):
            length = 3 if step == 0 else 1
            results = []
            for index, attention in enumerate((production, experiment)):
                result = attention(
                    self.hidden[:, :length],
                    tuple(table[:, :length] for table in self.rope),
                    past_key_value=caches[index],
                    use_cache=True,
                )
                delta = result[2]
                caches[index] = (
                    delta
                    if caches[index] is None
                    else tuple(
                        torch.cat((old, new), dim=-2)
                        for old, new in zip(caches[index], delta)
                    )
                )
                results.append(result)
            for actual, expected in zip(results[0][:2], results[1][:2]):
                self.assertTrue(torch.equal(actual, expected))
            for actual, expected in zip(results[0][2], results[1][2]):
                self.assertTrue(torch.equal(actual, expected))

    def test_r1_norm_ema_collects_once_and_dormant_sites_are_excluded(self):
        cfg = copy.deepcopy(self.qcfg)
        cfg.set_override("q_norm.act_out", {"observer": EMAObserver})
        attention = self._make(r=True, qcfg=cfg)
        observer = attention.q_norm.wrapped.obs_act_out
        attention.enable_calibration()
        with patch.object(observer, "collect", wraps=observer.collect) as collect:
            attention(self.hidden, self.rope, use_cache=True)
        self.assertEqual(collect.call_count, 1)
        active = dict(attention.named_observers())
        for name in ("q_x1", "q_x2", "q_cat", "k_x1", "k_x2", "k_cat"):
            self.assertNotIn(name, active)
        self._calibrate(attention)
        tensor = torch.tensor([0.27, -0.19, 0.41, 0.12]).reshape(1, 1, 1, 4)
        quantized = observer.fake_quant(tensor)
        actual = attention._rot(
            quantized,
            attention.obs_q_x1,
            attention.obs_q_x2,
            attention.obs_q_neg,
            attention.obs_q_cat,
        )
        self.assertTrue(
            torch.equal(actual, torch.cat(quantized.chunk(2, -1)[::-1], -1))
        )

    def test_hf_convention_keeps_original_observers(self):
        cfg = copy.deepcopy(self.qcfg)
        cfg.model_args = {**cfg.model_args, "attention": {"rope": "hf"}}
        attention = self._make(r=True, qcfg=cfg)
        self.assertIn("q_cat", dict(attention.named_observers()))
        self.assertIn("q_neg", dict(attention.named_observers()))

    def test_k0_p_requires_real_read_and_is_independent(self):
        attention = self._make(k=False)
        attention.enable_calibration()
        attention(self.hidden, self.rope, use_cache=True)
        with self.assertRaisesRegex(RuntimeError, "experiment_past_key"):
            attention.freeze_qparams()
        self._calibrate(attention)
        for storage, past in zip(
            attention.cache_storage_observers(),
            (attention.obs_experiment_past_key, attention.obs_experiment_past_value),
        ):
            self.assertIsNot(storage, past)
            self.assertEqual(storage.policy_signature(), past.policy_signature())
            self.assertIsNotNone(past.frozen_state)

    def test_k0_rounding_and_cache_append_use_separate_branches(self):
        serial, simplified = self._make(k=False), self._make(k=True)
        for attention in (serial, simplified):
            self._calibrate(attention)
            for storage in attention.cache_storage_observers():
                storage.load_qparams(torch.tensor(0.5), torch.tensor(128))
            for compute in (attention.obs_present_key, attention.obs_present_value):
                compute.load_qparams(torch.tensor(0.3), torch.tensor(0))
        for observer in (
            serial.obs_experiment_past_key,
            serial.obs_experiment_past_value,
        ):
            observer.load_qparams(torch.tensor(0.4), torch.tensor(128))
        raw = torch.tensor([0.22, 200.0, -200.0, 0.22]).reshape(1, 1, 1, 4)
        raw = raw.repeat(1, 2, 1, 1)
        past = torch.full_like(raw, 0.5)
        storages, computes = [], []
        for attention in (serial, simplified):
            delta = tuple(
                observer.fake_quant(raw)
                for observer in attention.cache_storage_observers()
            )
            compute, storage, _ = attention._build_present_key_value(
                new_key_value=delta,
                current_key_value=(raw, raw),
                past_key_value=(past, past),
            )
            storages.append(storage)
            computes.append(compute)
            self.assertTrue(torch.equal(storage[0][..., :1, :], past))
            self.assertTrue(torch.equal(storage[0][..., 1:, :], delta[0]))
        self.assertTrue(torch.equal(storages[0][0], storages[1][0]))
        self.assertAlmostEqual(computes[0][0][0, 0, -1, 0].item(), 0.0)
        self.assertAlmostEqual(computes[1][0][0, 0, -1, 0].item(), 0.3, places=6)
        # The intentionally saturating S grid still saturates on the serial branch.
        self.assertLess(computes[0][0][0, 0, -1, 1].item(), 65)
        self.assertGreater(computes[1][0][0, 0, -1, 1].item(), 199)
        self.assertAlmostEqual(computes[0][0][0, 0, 0, 0].item(), 0.3, places=6)
        self.assertAlmostEqual(computes[1][0][0, 0, 0, 0].item(), 0.6, places=6)

    def test_variant_state_cannot_mix(self):
        first, second = self._make(r=False), self._make(r=True)
        self._calibrate(first)
        with self.assertRaisesRegex(RuntimeError, "policy differs"):
            second.load_state_dict(first.state_dict(), strict=False)
        self.assertIsNot(first.obs_new_k, second.obs_new_k)
        self.assertIsNot(
            first.q_proj.wrapped.module.weight, second.q_proj.wrapped.module.weight
        )

    def test_shared_consumer_p_follows_producer_policy_without_sharing_statistics(self):
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextAttention

        config = _make_text_config(
            num_hidden_layers=2,
            layer_types=["full_attention", "full_attention"],
            num_kv_shared_layers=1,
        )
        producer = ExperimentGemma4TextAttention(
            Gemma4TextAttention(config, layer_idx=0).eval(),
            qcfg=copy.deepcopy(self.qcfg),
            kv_simplified=False,
        ).eval()
        consumer = ExperimentGemma4TextAttention(
            Gemma4TextAttention(config, layer_idx=1).eval(),
            qcfg=PTQConfig(activation=affine(INT16)),
            kv_simplified=False,
        ).eval()
        consumer.bind_cache_source(producer)
        self.assertEqual(consumer.obs_experiment_past_key.dtype, UINT8)
        self.assertIsNot(consumer.obs_experiment_past_key, producer.obs_new_k)
        producer.enable_calibration()
        consumer.enable_calibration()
        past = producer(self.hidden, self.rope, use_cache=True)[2]
        producer(
            self.hidden[:, :1],
            tuple(table[:, :1] for table in self.rope),
            past_key_value=past,
            use_cache=True,
        )
        observer = consumer.obs_experiment_past_key
        with patch.object(observer, "collect", wraps=observer.collect) as collect:
            consumer(self.hidden, self.rope, shared_key_value=past)
        self.assertEqual(collect.call_count, 1)
        producer.freeze_qparams()
        consumer.freeze_qparams()
        result = consumer(self.hidden, self.rope, shared_key_value=past, use_cache=True)
        self.assertIsNone(result[2])

    def test_per_channel_rope_boundary_preserves_original_behavior(self):
        from tico.quantization.wrapq.qscheme import QScheme

        cfg = copy.deepcopy(self.qcfg)
        cfg.set_override(
            "q_x1",
            {"qscheme": QScheme.PER_CHANNEL_SYMM, "channel_axis": -1},
        )
        attention = self._make(r=True, qcfg=cfg)
        self.assertIn("q_x1", dict(attention.named_observers()))
        self.assertIn("q_cat", dict(attention.named_observers()))
        self.assertNotIn("k_cat", dict(attention.named_observers()))

    def test_all_variants_cross_sliding_window_with_own_storage_cache(self):
        from transformers.cache_utils import DynamicCache
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextAttention

        config = _make_text_config(
            num_hidden_layers=2,
            layer_types=["sliding_attention", "full_attention"],
            sliding_window=4,
        )
        for r in (False, True):
            for k in (False, True):
                with self.subTest(r=r, k=k):
                    attention = ExperimentGemma4TextAttention(
                        Gemma4TextAttention(config, layer_idx=0).eval(),
                        qcfg=copy.deepcopy(self.qcfg),
                        rope_unified=r,
                        kv_simplified=k,
                    ).eval()
                    self._calibrate(attention)
                    cache: Any = DynamicCache(config=config)
                    for step in range(12):
                        length = 3 if step == 0 else 1
                        before = None if step == 0 else cache.layers[0].keys.clone()
                        output, _weights, delta = attention(
                            self.hidden[:, :length],
                            tuple(table[:, :length] for table in self.rope),
                            past_key_value=cache,
                            use_cache=True,
                        )
                        self.assertTrue(torch.isfinite(output).all())
                        stored = cache.layers[0].keys
                        expected = (
                            delta[0]
                            if before is None
                            else torch.cat((before, delta[0]), dim=-2)
                        )
                        # DynamicSlidingWindowLayer evicts only the old prefix.
                        self.assertTrue(
                            torch.equal(stored, expected[..., -stored.size(-2) :, :])
                        )
                        self.assertLessEqual(stored.size(-2), config.sliding_window)
                        self.assertTrue(
                            torch.equal(stored, attention.obs_new_k.fake_quant(stored))
                        )
                    self.assertGreater(
                        int(cache.get_seq_length(0)), config.sliding_window
                    )

    def test_all_variants_shared_producer_consumer_preserve_storage(self):
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextAttention

        config = _make_text_config(
            num_hidden_layers=2,
            layer_types=["full_attention", "full_attention"],
            num_kv_shared_layers=1,
        )
        for r in (False, True):
            for k in (False, True):
                with self.subTest(r=r, k=k):
                    producer, consumer = (
                        ExperimentGemma4TextAttention(
                            Gemma4TextAttention(config, layer_idx=index).eval(),
                            qcfg=copy.deepcopy(self.qcfg),
                            rope_unified=r,
                            kv_simplified=k,
                        ).eval()
                        for index in (0, 1)
                    )
                    consumer.bind_cache_source(producer)
                    producer.enable_calibration()
                    consumer.enable_calibration()
                    shared: dict[Any, Any] = {}
                    past = None
                    for step in range(3):
                        length = 3 if step == 0 else 1
                        args = self.hidden[:, :length], tuple(
                            table[:, :length] for table in self.rope
                        )
                        _output, _weights, delta = producer(
                            *args,
                            past_key_value=past,
                            shared_kv_states=shared,
                            use_cache=True,
                        )
                        past = (
                            delta
                            if past is None
                            else tuple(
                                torch.cat((old, new), -2)
                                for old, new in zip(past, delta)
                            )
                        )
                        consumer(*args, shared_kv_states=shared)
                    producer.freeze_qparams()
                    consumer.freeze_qparams()
                    past = None
                    for step in range(7):
                        length = 3 if step == 0 else 1
                        args = self.hidden[:, :length], tuple(
                            table[:, :length] for table in self.rope
                        )
                        _output, _weights, delta = producer(
                            *args,
                            past_key_value=past,
                            shared_kv_states=shared,
                            use_cache=True,
                        )
                        past = (
                            delta
                            if past is None
                            else tuple(
                                torch.cat((old, new), -2)
                                for old, new in zip(past, delta)
                            )
                        )
                        before = tuple(
                            tensor.clone() for tensor in shared[producer.layer_type]
                        )
                        output, _weights, absent_delta = consumer(
                            *args, shared_kv_states=shared, use_cache=True
                        )
                        self.assertIsNone(absent_delta)
                        self.assertTrue(torch.isfinite(output).all())
                        for expected, actual, owned in zip(
                            before, shared[producer.layer_type], past
                        ):
                            self.assertTrue(torch.equal(expected, actual))
                            self.assertTrue(torch.equal(owned, actual))
