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

"""Gamma splitting preserves float semantics and adds one quantization site."""

import copy
import io
import unittest

import torch

from tico.quantization import convert, prepare
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.config.specs import affine, mx, no_quant
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.lifecycle import validate_frozen_model
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.observers.affine_base import AffineObserverBase
from tico.quantization.wrapq.observers.base import ObserverNotReadyError
from tico.quantization.wrapq.observers.mx import MXObserver
from tico.quantization.wrapq.wrappers.gemma4.quant_rmsnorm import QuantGemma4RMSNorm
from tico.quantization.wrapq.wrappers.ops.quant_rmsnorm import QuantRMSNorm
from tico.utils.mx.mx_ops import _quantize_mx
from torch import nn


class _RMSNorm(nn.Module):
    def __init__(self, hidden=8, *, with_gamma=True, gamma_dtype=torch.float32):
        super().__init__()
        self.eps = self.variance_epsilon = 0.13
        if with_gamma:
            gamma = torch.linspace(-3.7, 5.1, hidden, dtype=gamma_dtype)
            gamma[hidden // 2] = 0
            self.weight = nn.Parameter(gamma)


_WRAPPERS = (QuantRMSNorm, QuantGemma4RMSNorm)


def _normalized(x, eps):
    # The existing custom RMSNorm casts BEFORE multiplying the learned gamma.
    xf = x.float()
    return (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)).to(x.dtype)


def _affine_reference(x, observer):
    """Use the frozen grid directly, independently of observer.fake_quant()."""
    state = observer.require_frozen()
    assert state.scale is not None and state.zero_point is not None
    return torch.fake_quantize_per_tensor_affine(
        x,
        scale=state.scale,
        zero_point=state.zero_point,
        quant_min=observer.dtype.qmin,
        quant_max=observer.dtype.qmax,
    )


def _mx_reference(x, axis):
    return _quantize_mx(
        x,
        scale_bits=8,
        elem_format="int8",
        axes=[axis],
        block_size=32,
        shared_exp_method="max",
        round="nearest",
    )


def _calibrate(wrapper, x):
    wrapper.enable_calibration()
    wrapper(x)
    wrapper.freeze_qparams()


class TestRMSNormGammaSplit(unittest.TestCase):
    def setUp(self):
        self.x = torch.randn(2, 3, 8, generator=torch.Generator().manual_seed(92))

    def test_float_semantics_dtype_cast_order_and_parameter_identity(self):
        for wrapper_type in _WRAPPERS:
            for split in (False, True):
                for dtype in (torch.float32, torch.float16, torch.bfloat16):
                    with self.subTest(wrapper=wrapper_type, split=split, dtype=dtype):
                        fp = _RMSNorm()
                        original = fp.weight.detach().clone()
                        alias = nn.Module()
                        alias.weight = fp.weight
                        wrapper = wrapper_type(
                            fp, qcfg=PTQConfig(split_rmsnorm_gamma=split)
                        )
                        x = self.x.to(dtype)
                        expected = _normalized(x, fp.eps) * fp.weight
                        if wrapper_type is QuantGemma4RMSNorm:
                            expected = expected.to(dtype)
                        actual = wrapper(x)
                        self.assertEqual(actual.dtype, expected.dtype)
                        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                        self.assertIs(fp.weight, alias.weight)
                        self.assertIs(wrapper.module.weight, fp.weight)
                        self.assertTrue(torch.equal(fp.weight, original))

    def test_false_preserves_fused_observers_and_numerical_path(self):
        for wrapper_type in _WRAPPERS:
            with self.subTest(wrapper=wrapper_type):
                fp = _RMSNorm()
                wrapper = wrapper_type(fp, qcfg=PTQConfig(split_rmsnorm_gamma=False))
                self.assertEqual(
                    set(dict(wrapper.named_observers())),
                    {"weight", "act_in", "act_out"},
                )
                self.assertIsNone(wrapper.get_observer("act_pre_gamma"))
                _calibrate(wrapper, self.x)
                xq = _affine_reference(self.x, wrapper.obs_act_in)
                gammaq = _affine_reference(fp.weight, wrapper.obs_weight)
                expected = _affine_reference(
                    _normalized(xq, fp.eps) * gammaq, wrapper.obs_act_out
                )
                torch.testing.assert_close(wrapper(self.x), expected, atol=0, rtol=0)

    def test_parameter_of_ones_still_creates_pre_gamma_site(self):
        for wrapper_type in _WRAPPERS:
            with self.subTest(wrapper=wrapper_type):
                fp = _RMSNorm()
                with torch.no_grad():
                    fp.weight.fill_(1)
                wrapper = wrapper_type(fp)
                self.assertIsNotNone(wrapper.get_observer("act_pre_gamma"))
                _calibrate(wrapper, self.x)
                self.assertIsNotNone(wrapper.get_observer("act_pre_gamma").frozen_state)

    def test_no_gamma_does_not_add_intermediate_observer_or_multiply(self):
        for wrapper_type in _WRAPPERS:
            for split in (False, True):
                for parameter_is_none in (False, True):
                    with self.subTest(
                        wrapper=wrapper_type,
                        split=split,
                        parameter_is_none=parameter_is_none,
                    ):
                        fp = _RMSNorm(with_gamma=False)
                        if parameter_is_none:
                            fp.register_parameter("weight", None)
                        wrapper = wrapper_type(
                            fp, qcfg=PTQConfig(split_rmsnorm_gamma=split)
                        )
                        self.assertIsNone(wrapper.get_observer("act_pre_gamma"))
                        torch.testing.assert_close(
                            wrapper(self.x), _normalized(self.x, fp.eps), atol=0, rtol=0
                        )
                        _calibrate(wrapper, self.x)
                        graph = torch.export.export(wrapper, (self.x,)).graph
                        self.assertFalse(
                            any(
                                node.target == torch.ops.aten.mul.Tensor
                                for node in graph.nodes
                            )
                        )
                        self.assertEqual(
                            len(
                                [
                                    node
                                    for node in graph.nodes
                                    if node.target
                                    == torch.ops.circle_custom.rms_norm.default
                                ]
                            ),
                            1,
                        )

    def test_affine_intermediate_collects_independent_calibrated_grid(self):
        for wrapper_type in _WRAPPERS:
            for dtype in (DType.uint(8), DType.int(16)):
                with self.subTest(wrapper=wrapper_type, dtype=dtype):
                    fp = _RMSNorm()
                    wrapper = wrapper_type(
                        fp,
                        qcfg=PTQConfig(
                            activation=affine(dtype), weight=affine(DType.int(8))
                        ),
                    )
                    wrapper.enable_calibration()
                    wrapper(self.x)
                    pre = wrapper.get_observer("act_pre_gamma")
                    self.assertIsInstance(pre, AffineObserverBase)
                    assert isinstance(pre, AffineObserverBase)
                    normalized = _normalized(self.x, fp.eps)
                    torch.testing.assert_close(
                        pre.min_val, normalized.min(), atol=0, rtol=0
                    )
                    torch.testing.assert_close(
                        pre.max_val, normalized.max(), atol=0, rtol=0
                    )
                    output = normalized * fp.weight
                    torch.testing.assert_close(
                        wrapper.obs_act_out.min_val, output.min(), atol=0, rtol=0
                    )
                    torch.testing.assert_close(
                        wrapper.obs_act_out.max_val, output.max(), atol=0, rtol=0
                    )
                    wrapper.freeze_qparams()
                    state = pre.require_frozen()
                    self.assertEqual(state.provenance, "calibrated")
                    self.assertEqual(pre.dtype, dtype)
                    assert state.scale is not None and state.zero_point is not None
                    self.assertTrue(bool((state.scale > 0).all()))
                    self.assertTrue(bool(torch.isfinite(state.scale).all()))
                    self.assertTrue(bool((state.zero_point >= dtype.qmin).all()))
                    self.assertTrue(bool((state.zero_point <= dtype.qmax).all()))
                    self.assertFalse(
                        torch.equal(state.scale, wrapper.obs_act_out._cached_scale)
                    )
                    self.assertNotEqual(
                        state.scale.data_ptr(),
                        wrapper.obs_act_out._cached_scale.data_ptr(),
                    )
                    xq = _affine_reference(self.x, wrapper.obs_act_in)
                    r = _normalized(xq, fp.eps)
                    rq = _affine_reference(r, pre)
                    gammaq = _affine_reference(fp.weight, wrapper.obs_weight)
                    expected = _affine_reference(rq * gammaq, wrapper.obs_act_out)
                    without_mid = _affine_reference(r * gammaq, wrapper.obs_act_out)
                    torch.testing.assert_close(
                        wrapper(self.x), expected, atol=0, rtol=0
                    )
                    self.assertFalse(torch.equal(expected, without_mid))

    def test_pre_gamma_override_uses_activation_role_and_changes_calculation(self):
        for wrapper_type in _WRAPPERS:
            with self.subTest(wrapper=wrapper_type):
                fp = _RMSNorm()
                wrapper = wrapper_type(
                    fp,
                    qcfg=PTQConfig(
                        activation=no_quant(),
                        weight=no_quant(),
                        overrides={"act_pre_gamma": affine(DType.int(16))},
                    ),
                )
                _calibrate(wrapper, self.x)
                pre = wrapper.get_observer("act_pre_gamma")
                self.assertIsInstance(pre, AffineObserverBase)
                assert isinstance(pre, AffineObserverBase)
                self.assertEqual(pre.dtype, DType.int(16))
                normalized = _normalized(self.x, fp.eps)
                expected = _affine_reference(normalized, pre) * fp.weight
                torch.testing.assert_close(wrapper(self.x), expected, atol=0, rtol=0)
                self.assertFalse(torch.equal(expected, normalized * fp.weight))

    def test_mx_dynamic_intermediate_matches_explicit_block_reference(self):
        # Last-axis full/multiple blocks, padded tail, and a distinct override axis.
        for shape, mid_axis in (((2, 3, 64), -1), ((2, 3, 70), -1), ((2, 65, 9), 1)):
            x = torch.randn(shape, generator=torch.Generator().manual_seed(2026))
            for wrapper_type in _WRAPPERS:
                with self.subTest(wrapper=wrapper_type, shape=shape, axis=mid_axis):
                    fp = _RMSNorm(shape[-1])
                    wrapper = wrapper_type(
                        fp,
                        qcfg=PTQConfig(
                            activation=mx("int8", axis=-1),
                            weight=affine(DType.int(8)),
                            overrides={"act_pre_gamma": mx("int8", axis=mid_axis)},
                        ),
                    )
                    _calibrate(wrapper, x)
                    pre = wrapper.get_observer("act_pre_gamma")
                    self.assertIsInstance(pre, MXObserver)
                    assert isinstance(pre, MXObserver)
                    state = pre.require_frozen()
                    self.assertEqual(state.kind, "mx")
                    self.assertEqual(state.provenance, "stateless")
                    self.assertIsNone(state.scale)
                    self.assertIsNone(state.zero_point)
                    self.assertEqual(state.observed_shape, shape)
                    self.assertEqual(pre.axis, mid_axis)
                    xq = _mx_reference(x, -1)
                    r = _normalized(xq, fp.eps)
                    rq = _mx_reference(r, mid_axis)
                    gammaq = _affine_reference(fp.weight, wrapper.obs_weight)
                    expected = _mx_reference(rq * gammaq, -1)
                    without_mid = _mx_reference(r * gammaq, -1)
                    torch.testing.assert_close(wrapper(x), expected, atol=0, rtol=0)
                    self.assertFalse(torch.equal(expected, without_mid))

    def test_state_roundtrip_preserves_structure_frozen_grids_and_output(self):
        for wrapper_type in _WRAPPERS:
            for split in (False, True):
                for activation in (affine(DType.uint(8)), mx("int8", axis=-1)):
                    with self.subTest(
                        wrapper=wrapper_type, split=split, activation=activation
                    ):
                        cfg = PTQConfig(
                            split_rmsnorm_gamma=split, activation=activation
                        )
                        wrapper = wrapper_type(_RMSNorm(), qcfg=cfg)
                        _calibrate(wrapper, self.x)
                        stream = io.BytesIO()
                        torch.save(wrapper.state_dict(), stream)
                        stream.seek(0)
                        state = torch.load(stream, weights_only=True)
                        self.assertIs(
                            state["_extra_state"]["split_rmsnorm_gamma"], split
                        )
                        restored = wrapper_type(
                            _RMSNorm(),
                            qcfg=PTQConfig(
                                split_rmsnorm_gamma=split, activation=activation
                            ),
                        )
                        restored.load_state_dict(state)
                        whole_stream = io.BytesIO()
                        torch.save(wrapper, whole_stream)
                        whole_stream.seek(0)
                        whole_restored = torch.load(whole_stream, weights_only=False)
                        validate_frozen_model(whole_restored)
                        self.assertIs(whole_restored.split_rmsnorm_gamma, split)
                        torch.testing.assert_close(
                            whole_restored(self.x), wrapper(self.x), atol=0, rtol=0
                        )
                        self.assertIs(restored._mode, Mode.QUANT)
                        self.assertEqual(
                            set(dict(wrapper.named_observers())),
                            set(dict(restored.named_observers())),
                        )
                        torch.testing.assert_close(
                            restored(self.x), wrapper(self.x), atol=0, rtol=0
                        )
                        for name, obs in wrapper.named_observers():
                            before = obs.require_frozen()
                            after = restored.get_observer(name).require_frozen()
                            self.assertEqual(before.policy, after.policy)
                            if before.scale is not None:
                                torch.testing.assert_close(
                                    before.scale, after.scale, atol=0, rtol=0
                                )
                                torch.testing.assert_close(
                                    before.zero_point, after.zero_point, atol=0, rtol=0
                                )

    def test_public_prepare_convert_reaches_registered_model_families(self):
        from transformers.models.gemma4.modeling_gemma4 import Gemma4RMSNorm
        from transformers.models.llama.modeling_llama import LlamaRMSNorm
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRMSNorm

        for module_type, wrapper_type in (
            (LlamaRMSNorm, QuantRMSNorm),
            (Qwen3VLTextRMSNorm, QuantRMSNorm),
            (Gemma4RMSNorm, QuantGemma4RMSNorm),
        ):
            for split in (False, True):
                with self.subTest(module=module_type, split=split):
                    fp = module_type(8, eps=0.13)
                    with torch.no_grad():
                        fp.weight.copy_(_RMSNorm().weight)
                    model = nn.Sequential(fp)
                    reference = model(self.x)
                    cfg = PTQConfig(
                        split_rmsnorm_gamma=split,
                        overrides={"0.act_pre_gamma": affine(DType.int(16))},
                    )
                    prepared = prepare(model, cfg)
                    norm = prepared[0].wrapped
                    self.assertIsInstance(norm, wrapper_type)
                    self.assertIs(norm.split_rmsnorm_gamma, split)
                    self.assertIs(norm._mode, Mode.CALIB)
                    if split:
                        self.assertEqual(
                            norm.get_observer("act_pre_gamma").dtype, DType.int(16)
                        )
                    else:
                        self.assertIsNone(norm.get_observer("act_pre_gamma"))
                    torch.testing.assert_close(
                        prepared(self.x), reference, atol=0, rtol=0
                    )
                    cfg.split_rmsnorm_gamma = not split
                    converted = convert(prepared)
                    self.assertIs(norm._mode, Mode.QUANT)
                    self.assertIs(norm.split_rmsnorm_gamma, split)
                    self.assertIs(converted, prepared)
                    state = norm.get_extra_state()
                    self.assertIs(state["split_rmsnorm_gamma"], split)
                    # Conversion consumes the calibrated intermediate when present.
                    for _, observer in norm.named_observers():
                        self.assertIsNotNone(observer.frozen_state)

    def test_legacy_whole_module_retains_fused_execution(self):
        for wrapper_type in _WRAPPERS:
            with self.subTest(wrapper=wrapper_type):
                fused = wrapper_type(
                    _RMSNorm(), qcfg=PTQConfig(split_rmsnorm_gamma=False)
                )
                _calibrate(fused, self.x)
                expected = fused(self.x)
                # Emulate historical whole-module objects without split metadata.
                del fused._split_rmsnorm_gamma
                del fused.obs_act_pre_gamma
                if wrapper_type is QuantRMSNorm:
                    del fused.with_scale
                stream = io.BytesIO()
                torch.save(fused, stream)
                stream.seek(0)
                restored = torch.load(stream, weights_only=False)
                validate_frozen_model(restored)
                self.assertFalse(restored.split_rmsnorm_gamma)
                self.assertIsNone(restored.get_observer("act_pre_gamma"))
                torch.testing.assert_close(restored(self.x), expected, atol=0, rtol=0)

    def test_legacy_fused_state_loads_only_into_false_structure(self):
        for wrapper_type in _WRAPPERS:
            with self.subTest(wrapper=wrapper_type):
                fused = wrapper_type(
                    _RMSNorm(), qcfg=PTQConfig(split_rmsnorm_gamma=False)
                )
                _calibrate(fused, self.x)
                state = copy.deepcopy(fused.state_dict())
                del state["_extra_state"]["split_rmsnorm_gamma"]
                del state["_extra_state"]["rmsnorm_has_gamma"]
                restored = wrapper_type(
                    _RMSNorm(), qcfg=PTQConfig(split_rmsnorm_gamma=False)
                )
                restored.load_state_dict(state)
                torch.testing.assert_close(
                    restored(self.x), fused(self.x), atol=0, rtol=0
                )
                for strict in (False, True):
                    split = wrapper_type(_RMSNorm())
                    with self.assertRaisesRegex(RuntimeError, "reprepare.*recalibrat"):
                        split.load_state_dict(state, strict=strict)

    def test_whole_module_rejects_missing_setting_with_split_observer(self):
        for wrapper_type in _WRAPPERS:
            split = wrapper_type(_RMSNorm())
            _calibrate(split, self.x)
            del split._split_rmsnorm_gamma
            stream = io.BytesIO()
            torch.save(split, stream)
            stream.seek(0)
            with self.assertRaisesRegex(RuntimeError, "reprepare.*recalibrat"):
                torch.load(stream, weights_only=False)

    def test_opposite_structure_state_rejects_even_when_not_strict(self):
        for wrapper_type in _WRAPPERS:
            for saved_split in (False, True):
                with self.subTest(wrapper=wrapper_type, saved_split=saved_split):
                    source = wrapper_type(
                        _RMSNorm(), qcfg=PTQConfig(split_rmsnorm_gamma=saved_split)
                    )
                    _calibrate(source, self.x)
                    target = wrapper_type(
                        _RMSNorm(), qcfg=PTQConfig(split_rmsnorm_gamma=not saved_split)
                    )
                    with self.assertRaisesRegex(RuntimeError, "reprepare.*recalibrat"):
                        target.load_state_dict(source.state_dict(), strict=False)

    def test_config_mutation_after_prepare_does_not_change_boundary(self):
        for wrapper_type in _WRAPPERS:
            for split in (False, True):
                with self.subTest(wrapper=wrapper_type, split=split):
                    cfg = PTQConfig(split_rmsnorm_gamma=split)
                    wrapper = wrapper_type(_RMSNorm(), qcfg=cfg)
                    names = set(dict(wrapper.named_observers()))
                    cfg.split_rmsnorm_gamma = not split
                    _calibrate(wrapper, self.x)
                    before = wrapper(self.x)
                    self.assertEqual(set(dict(wrapper.named_observers())), names)
                    self.assertIs(
                        wrapper.get_extra_state()["split_rmsnorm_gamma"], split
                    )
                    torch.testing.assert_close(wrapper(self.x), before, atol=0, rtol=0)

    def test_incomplete_state_still_rejects_and_missing_mid_calibration_fails(self):
        for wrapper_type in _WRAPPERS:
            with self.subTest(wrapper=wrapper_type):
                wrapper = wrapper_type(_RMSNorm())
                wrapper.enable_calibration()
                # Collect every original site, leaving only the new required site empty.
                wrapper.obs_act_in.collect(self.x)
                wrapper.obs_act_out.collect(self.x)
                with self.assertRaisesRegex(ObserverNotReadyError, "act_pre_gamma"):
                    wrapper.freeze_qparams()
                self.assertIs(wrapper._mode, Mode.CALIB)
                wrapper(self.x)
                wrapper.freeze_qparams()
                state = copy.deepcopy(wrapper.state_dict())
                del state["_extra_state"]
                with self.assertRaisesRegex(RuntimeError, "reprepare.*recalibrat"):
                    wrapper_type(_RMSNorm()).load_state_dict(state, strict=False)


if __name__ == "__main__":
    unittest.main()
