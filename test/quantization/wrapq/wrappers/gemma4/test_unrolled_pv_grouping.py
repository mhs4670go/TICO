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

"""PV grouping regressions exercise the real Gemma4 wrapper core."""

import unittest
from unittest.mock import patch

import torch
from tico.quantization.wrapq.wrappers.gemma4.quant_text_attention import (
    QuantGemma4TextAttention,
)
from torch import nn

from test.quantization.wrapq.wrappers.gemma4.test_quant_text_attention import (
    _has_gemma4,
    _make_text_config,
)


class _Core(nn.Module):
    """Export the actual wrapper's attention core with explicit Q/K/V."""

    def __init__(self, attention):
        super().__init__()
        self.attention = attention

    def forward(self, query, key, value, mask):
        return self.attention._forward_unrolled(
            query_states=query,
            key_states=key,
            value_states=value,
            attention_mask=mask,
        )


@unittest.skipUnless(_has_gemma4(), "Transformers Gemma4 is required")
class TestGemma4PVGrouping(unittest.TestCase):
    """Protect row ordering, mask broadcasting and the PV-only graph contract."""

    @staticmethod
    def _attention(heads=24, kv_heads=8):
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextAttention

        torch.manual_seed(17)
        config = _make_text_config(
            hidden_size=heads * 4,
            num_attention_heads=heads,
            num_key_value_heads=kv_heads,
            num_global_key_value_heads=kv_heads,
        )
        return QuantGemma4TextAttention(
            Gemma4TextAttention(config, layer_idx=0).eval()
        ).eval()

    @staticmethod
    def _inputs(batch, heads, kv_heads, tokens, *, noncontiguous=False):
        """Use a head/token-dependent pattern rather than exchangeable zeros."""
        key_tokens, dimension = 5, 4
        query = (
            torch.arange(batch * heads * tokens * dimension)
            .reshape(batch, heads, tokens, dimension)
            .float()
        )
        query = query.remainder(19) / 7 - 1
        key = (
            torch.arange(batch * kv_heads * key_tokens * dimension)
            .reshape(batch, kv_heads, key_tokens, dimension)
            .float()
            .remainder(13)
            / 5
            - 1
        )
        value = (
            torch.arange(batch * kv_heads * key_tokens * dimension)
            .reshape(batch, kv_heads, key_tokens, dimension)
            .float()
            / 11
        )
        if noncontiguous:
            query, key, value = (
                tensor.transpose(-1, -2).contiguous().transpose(-1, -2)
                for tensor in (query, key, value)
            )
            assert not query.is_contiguous()
        return query, key, value

    def test_core_matches_original_head_order_and_broadcast_masks(self):
        """T>1 catches head/token swaps; singleton query masks stay valid."""
        cases = (
            (1, 24, 8, 3, 1, 3, False, 1.0),
            (2, 4, 2, 3, 1, 1, True, 0.5),
            (2, 4, 2, 3, 2, 3, True, 1.0),
            (1, 3, 1, 1, 1, 1, False, 0.5),
            (1, 2, 2, 3, 1, 1, True, 1.0),
        )
        for batch, heads, kv_heads, tokens, mask_batch, mask_tokens, nc, scale in cases:
            with self.subTest(case=(batch, heads, kv_heads, tokens, mask_tokens, nc)):
                attention = self._attention(heads, kv_heads)
                attention.scaling = scale
                query, key, value = self._inputs(
                    batch, heads, kv_heads, tokens, noncontiguous=nc
                )
                mask = torch.zeros(mask_batch, 1, mask_tokens, key.size(-2))
                mask[..., -1] = -100.0
                if mask_tokens > 1:
                    mask[..., 0, 1:] = -100.0
                actual, weights = _Core(attention)(query, key, value, mask)
                # Each expected head sees precisely the original T rows and mask.
                expected_weights = []
                expected_heads = []
                for head in range(heads):
                    kv_head = head // (heads // kv_heads)
                    logits = query[:, head] @ key[:, kv_head].transpose(-2, -1)
                    p = torch.softmax(logits * scale + mask.squeeze(1), dim=-1)
                    expected_weights.append(p)
                    expected_heads.append(p @ value[:, kv_head])
                expected_input = torch.stack(expected_heads, dim=2).reshape(
                    batch, tokens, heads * 4
                )
                torch.testing.assert_close(
                    weights, torch.stack(expected_weights, dim=1), atol=1e-6, rtol=1e-6
                )
                torch.testing.assert_close(
                    actual, attention.o_proj(expected_input), atol=1e-6, rtol=1e-6
                )

    def test_observer_boundaries_keep_per_head_shapes_and_order(self):
        """PV grouping retains all existing fake-quant boundary invocations."""
        attention = self._attention(6, 2)
        query, key, value = self._inputs(1, 6, 2, 3)
        mask = torch.zeros(1, 1, 3, 5)
        original = attention._fq
        with patch.object(attention, "_fq", wraps=original) as recording:
            _Core(attention)(query, key, value, mask)
        for name in (
            "obs_logits_raw",
            "obs_logits",
            "obs_mask_add",
            "obs_softmax",
            "obs_attn_out",
        ):
            shapes = [
                tuple(call.args[0].shape)
                for call in recording.call_args_list
                if call.args[1] is getattr(attention, name)
            ]
            self.assertEqual(shapes, [(1, 3, 4 if name == "obs_attn_out" else 5)] * 6)
        for name, shape in (
            ("obs_attn_weights", (1, 6, 3, 5)),
            ("obs_attn_out_h", (1, 3, 6, 4)),
        ):
            self.assertEqual(
                [
                    tuple(call.args[0].shape)
                    for call in recording.call_args_list
                    if call.args[1] is getattr(attention, name)
                ],
                [shape],
            )

    def test_frozen_output_grid_is_applied_once_before_group_split(self):
        """A shared affine grid commutes with slicing and annotates the raw PV."""
        attention = self._attention(6, 2)
        attention.enable_calibration()
        hidden = torch.randn(1, 3, 24)
        rope = (torch.ones(1, 3, 4), torch.zeros(1, 3, 4))
        attention(hidden, rope, attention_mask=torch.zeros(1, 1, 3, 3), use_cache=True)
        attention.freeze_qparams()
        query, key, value = self._inputs(1, 6, 2, 3)
        mask = torch.zeros(1, 1, 3, 5)
        original = attention._fq
        with patch.object(attention, "_fq", wraps=original) as recording:
            actual, weights = _Core(attention)(query, key, value, mask)
        output_calls = [
            call
            for call in recording.call_args_list
            if call.args[1] is attention.obs_attn_out
        ]
        self.assertEqual(
            [tuple(call.args[0].shape) for call in output_calls], [(1, 9, 4)] * 2
        )
        heads = []
        for head in range(6):
            kv_head = head // 3
            logits = attention._apply_attention_scale(
                query[:, head] @ key[:, kv_head].transpose(-2, -1)
            )
            logits = original(logits + mask.squeeze(1), attention.obs_mask_add)
            p = original(torch.softmax(logits, dim=-1), attention.obs_softmax)
            heads.append(original(p @ value[:, kv_head], attention.obs_attn_out))
        expected_h = original(torch.stack(heads, dim=2), attention.obs_attn_out_h)
        expected = attention.o_proj(expected_h.reshape(1, 3, 24))
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
        self.assertEqual(tuple(weights.shape), (1, 6, 3, 5))
        for _, observer in attention.named_observers():
            observer.require_frozen()
            self.assertFalse(observer.enabled)

    def test_export_has_24_qk_8_pv_and_no_kv_replication(self):
        """Count attention MatMuls separately using unambiguous operand shapes."""
        attention = self._attention()
        query, key, value = self._inputs(1, 24, 8, 3, noncontiguous=True)
        mask = torch.zeros(1, 1, 3, 5)
        exported = torch.export.export(_Core(attention), (query, key, value, mask))
        qk: list[tuple] = []
        pv: list[tuple] = []
        for node in exported.graph.nodes:
            if node.op != "call_function":
                continue
            target = str(node.target)
            self.assertNotIn("repeat_interleave", target)
            if "matmul" not in target and "bmm" not in target:
                continue
            lhs, rhs = (arg.meta["val"] for arg in node.args[:2])
            self.assertEqual((lhs.dim(), rhs.dim()), (3, 3))
            (qk if rhs.shape[-2:] == (4, 5) else pv).append(
                (tuple(lhs.shape), tuple(rhs.shape), tuple(node.meta["val"].shape))
            )
        self.assertEqual(qk, [((1, 3, 4), (1, 4, 5), (1, 3, 5))] * 24)
        self.assertEqual(pv, [((1, 9, 5), (1, 5, 4), (1, 9, 4))] * 8)
        expected = _Core(attention)(query, key, value, mask)
        actual = exported.module()(query, key, value, mask)
        for left, right in zip(actual, expected):
            torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-6)

    def test_training_dropout_retains_headwise_random_sampling(self):
        """PV-only grouping leaves seeded per-head dropout calls unchanged."""
        attention = self._attention(4, 2).train()
        attention.attention_dropout = 0.25
        query, key, value = self._inputs(1, 4, 2, 3)
        mask = torch.zeros(1, 1, 3, 5)
        torch.manual_seed(71)
        actual, weights = _Core(attention)(query, key, value, mask)
        torch.manual_seed(71)
        ps, heads = [], []
        for head in range(4):
            logits = query[:, head] @ key[:, head // 2].transpose(-2, -1)
            p = torch.softmax(logits * attention.scaling, dim=-1)
            p = torch.nn.functional.dropout(p, p=0.25, training=True)
            ps.append(p)
            heads.append(p @ value[:, head // 2])
        expected = attention.o_proj(torch.stack(heads, dim=2).reshape(1, 3, 16))
        torch.testing.assert_close(weights, torch.stack(ps, dim=1), atol=0, rtol=0)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

    def test_g1_does_not_add_group_concat(self):
        attention = self._attention(2, 2)
        query, key, value = self._inputs(1, 2, 2, 3)
        exported = torch.export.export(
            _Core(attention), (query, key, value, torch.zeros(1, 1, 3, 5))
        )
        targets = [str(node.target) for node in exported.graph.nodes]
        self.assertFalse(any("cat." in target for target in targets))


if __name__ == "__main__":
    unittest.main()
