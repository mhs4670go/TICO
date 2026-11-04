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

"""Check gamma boundaries after the complete default Circle export pipeline."""

import unittest

import numpy as np

import tico
import torch
from circle_schema import circle
from tico.circle import CircleDocument
from tico.circle.runtime import CircleReferenceRuntime, ExecutionMode
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.config.specs import affine, mx, no_quant
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.observers.mx import MXObserver
from tico.quantization.wrapq.wrappers.gemma4.quant_rmsnorm import QuantGemma4RMSNorm
from tico.quantization.wrapq.wrappers.ops.quant_rmsnorm import QuantRMSNorm


class _Norm(torch.nn.Module):
    def __init__(self, width, gamma=True, unit_gamma=False):
        super().__init__()
        self.eps = self.variance_epsilon = 0.003
        values = torch.tensor([-2.0, 0.0, 0.25, 3.5, -0.75, 1.1, 8.0, -0.2])
        self.weight = (
            torch.nn.Parameter(
                torch.ones(width)
                if unit_gamma
                else values.repeat((width + 7) // 8)[:width]
            )
            if gamma
            else None
        )


def _operators(model, code):
    return [
        op
        for op in model.subgraphs[0].operators
        if model.operatorCodes[op.opcodeIndex].builtinCode == code
    ]


def _constant(model, tensor):
    dtype = {
        circle.TensorType.TensorType.UINT8: np.uint8,
        circle.TensorType.TensorType.INT8: np.int8,
        circle.TensorType.TensorType.INT16: np.int16,
    }[tensor.type]
    encoded = np.frombuffer(model.buffers[tensor.buffer].data, dtype=dtype)
    quant = tensor.quantization
    return (encoded.astype(np.float32) - np.float32(quant.zeroPoint[0])) * np.float32(
        quant.scale[0]
    )


class RMSNormGammaExportTest(unittest.TestCase):
    def _export(
        self,
        wrapper,
        spec,
        *,
        split=True,
        gamma=True,
        unit_gamma=False,
        width=8,
        pre_spec=None,
        qcfg=None,
        disable_fake_quant=False,
    ):
        module = wrapper(
            _Norm(width, gamma=gamma, unit_gamma=unit_gamma),
            qcfg=(
                qcfg
                if qcfg is not None
                else PTQConfig(
                    activation=spec,
                    weight=affine(DType.uint(8)),
                    split_rmsnorm_gamma=split,
                    overrides=(
                        {"act_pre_gamma": pre_spec} if pre_spec is not None else {}
                    ),
                )
            ),
        ).eval()
        sample = torch.linspace(-2.173, 1.921, 6 * width).reshape(1, 6, width)
        module.enable_calibration()
        with torch.no_grad():
            module(sample)
        module.freeze_qparams()
        if disable_fake_quant:
            for _, observer in module.named_observers():
                observer.disable_fake_quant()
        # No compile-config overrides: this includes the default Circle O1 passes.
        converted = tico.convert(module, (sample,))
        document = CircleDocument.from_bytes(converted.circle_binary)
        document.verify(raise_on_error=True)
        return module, sample, document

    def _assert_floating_export(self, module, sample, document, *, split):
        model = document.model
        graph = model.subgraphs[0]
        for tensor in graph.tensors:
            self.assertEqual(tensor.type, circle.TensorType.TensorType.FLOAT32)
            self.assertIsNone(tensor.quantization)
        self.assertEqual(
            _operators(model, circle.BuiltinOperator.BuiltinOperator.QUANTIZE), []
        )
        self.assertEqual(
            _operators(model, circle.BuiltinOperator.BuiltinOperator.DEQUANTIZE), []
        )
        norms = _operators(model, circle.BuiltinOperator.BuiltinOperator.RMS_NORM)
        self.assertEqual(len(norms), 1)
        muls = _operators(model, circle.BuiltinOperator.BuiltinOperator.MUL)
        self.assertEqual(len(muls), 1 if split else 0)
        if split:
            unit = graph.tensors[norms[0].inputs[1]]
            np.testing.assert_array_equal(
                np.frombuffer(model.buffers[unit.buffer].data, dtype=np.float32),
                np.ones(sample.shape[-1], dtype=np.float32),
            )
            self.assertIn(norms[0].outputs[0], muls[0].inputs)
        # Independent FP32 reference, with the existing cast-before-gamma order.
        value = sample.float()
        normalized = value * torch.rsqrt(
            value.square().mean(-1, keepdim=True) + module.eps
        )
        expected = (normalized * module.module.weight.detach()).numpy()
        runtime = CircleReferenceRuntime(document)
        self.assertFalse(runtime.has_quantized_activations())
        probe = runtime.probe_static_contracts()
        np.testing.assert_array_equal(probe.outputs[0], np.zeros_like(expected))
        # Use the default NATIVE execution mode, including its operand checks.
        actual = runtime.run([sample.numpy()]).outputs[0]
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

    def test_no_quant_policies_export_float_and_run_natively(self):
        for wrapper in (QuantRMSNorm, QuantGemma4RMSNorm):
            for split in (True, False):
                with self.subTest(wrapper=wrapper.__name__, split=split):
                    module, sample, document = self._export(
                        wrapper,
                        no_quant(),
                        qcfg=PTQConfig(
                            activation=no_quant(),
                            weight=no_quant(),
                            split_rmsnorm_gamma=split,
                        ),
                    )
                    self._assert_floating_export(module, sample, document, split=split)

    def test_no_quant_observer_overrides_export_float(self):
        for wrapper in (QuantRMSNorm, QuantGemma4RMSNorm):
            for activation in (affine(DType.uint(8)), mx(axis=-1)):
                with self.subTest(wrapper=wrapper.__name__, activation=activation):
                    qcfg = PTQConfig(
                        activation=activation, weight=affine(DType.uint(8))
                    )
                    for name in ("weight", "act_in", "act_pre_gamma", "act_out"):
                        qcfg.set_override(name, no_quant())
                    module, sample, document = self._export(
                        wrapper, activation, qcfg=qcfg
                    )
                    self._assert_floating_export(module, sample, document, split=True)

    def test_disabled_fake_quant_policies_export_float(self):
        for wrapper in (QuantRMSNorm, QuantGemma4RMSNorm):
            for activation in (affine(DType.uint(8)), mx(axis=-1)):
                with self.subTest(wrapper=wrapper.__name__, activation=activation):
                    module, sample, document = self._export(
                        wrapper,
                        activation,
                        disable_fake_quant=True,
                    )
                    for _, observer in module.named_observers():
                        self.assertFalse(observer.enabled)
                        self.assertFalse(observer.fake_quant_enabled)
                    self._assert_floating_export(module, sample, document, split=True)

    def test_quantized_intermediate_override_survives_no_quant_defaults(self):
        for wrapper in (QuantRMSNorm, QuantGemma4RMSNorm):
            for intermediate in (affine(DType.int(16)), mx(axis=-1)):
                with self.subTest(wrapper=wrapper.__name__, intermediate=intermediate):
                    qcfg = PTQConfig(
                        activation=no_quant(),
                        weight=no_quant(),
                        overrides={"act_pre_gamma": intermediate},
                    )
                    module, _, document = self._export(
                        wrapper,
                        no_quant(),
                        qcfg=qcfg,
                        width=65,
                    )
                    model = document.model
                    graph = model.subgraphs[0]
                    norm = _operators(
                        model, circle.BuiltinOperator.BuiltinOperator.RMS_NORM
                    )[0]
                    muls = _operators(model, circle.BuiltinOperator.BuiltinOperator.MUL)
                    self.assertEqual(len(muls), 1)
                    mul = muls[0]
                    self.assertIn(norm.outputs[0], mul.inputs)
                    self._assert_activation(
                        graph.tensors[norm.outputs[0]], module.obs_act_pre_gamma
                    )
                    self.assertFalse(module.obs_act_pre_gamma.enabled)
                    self.assertTrue(module.obs_act_pre_gamma.fake_quant_enabled)
                    unit = graph.tensors[norm.inputs[1]]
                    self.assertEqual(unit.type, circle.TensorType.TensorType.UINT8)
                    self.assertEqual(list(unit.quantization.scale), [1.0])
                    self.assertEqual(list(unit.quantization.zeroPoint), [0])
                    np.testing.assert_array_equal(_constant(model, unit), np.ones(65))
                    gamma_index = next(
                        index for index in mul.inputs if index != norm.outputs[0]
                    )
                    self.assertNotEqual(gamma_index, norm.inputs[1])
                    gamma = graph.tensors[gamma_index]
                    self.assertEqual(gamma.type, circle.TensorType.TensorType.FLOAT32)
                    self.assertIsNone(gamma.quantization)
                    self.assertEqual(
                        sum(gamma_index in op.inputs for op in graph.operators), 1
                    )
                    self.assertEqual(
                        graph.tensors[mul.outputs[0]].type,
                        circle.TensorType.TensorType.FLOAT32,
                    )
                    self.assertIsNone(graph.tensors[mul.outputs[0]].quantization)

    def _assert_affine(self, tensor, observer):
        quant = tensor.quantization
        self.assertIsNotNone(quant)
        np.testing.assert_allclose(
            quant.scale,
            observer._cached_scale.detach().numpy().reshape(-1),
            rtol=0,
            atol=0,
        )
        np.testing.assert_array_equal(
            quant.zeroPoint, observer._cached_zp.detach().numpy().reshape(-1)
        )

    def _assert_activation(self, tensor, observer):
        if isinstance(observer, MXObserver):
            self.assertEqual(tensor.type, circle.TensorType.TensorType.MXINT8)
            self.assertIsNotNone(tensor.quantization)
            self.assertEqual(
                tensor.quantization.quantizedDimension,
                observer.axis % len(tensor.shape),
            )
            self.assertIsNone(tensor.quantization.scale)
            self.assertIsNone(tensor.quantization.zeroPoint)
        else:
            wanted_type = (
                circle.TensorType.TensorType.UINT8
                if observer.dtype == DType.uint(8)
                else circle.TensorType.TensorType.INT16
            )
            self.assertEqual(tensor.type, wanted_type)
            self._assert_affine(tensor, observer)

    def _assert_split(self, module, document):
        model = document.model
        graph = model.subgraphs[0]
        norms = _operators(model, circle.BuiltinOperator.BuiltinOperator.RMS_NORM)
        multiplies = _operators(model, circle.BuiltinOperator.BuiltinOperator.MUL)
        self.assertEqual(len(norms), 1)
        self.assertEqual(len(multiplies), 1)
        norm, multiply = norms[0], multiplies[0]
        # The same observed intermediate is RMSNorm's result and gamma Mul's input.
        self.assertIn(norm.outputs[0], multiply.inputs)
        intermediate = graph.tensors[norm.outputs[0]]
        unit = graph.tensors[norm.inputs[1]]
        self.assertEqual(unit.type, circle.TensorType.TensorType.UINT8)
        self.assertIsNotNone(unit.quantization)
        self.assertEqual(list(unit.quantization.scale), [1.0])
        self.assertEqual(list(unit.quantization.zeroPoint), [0])
        np.testing.assert_array_equal(
            _constant(model, unit), np.ones(len(module.module.weight))
        )
        gamma_index = next(
            index for index in multiply.inputs if index != norm.outputs[0]
        )
        self.assertNotEqual(gamma_index, norm.inputs[1])
        gamma = graph.tensors[gamma_index]
        self._assert_affine(gamma, module.obs_weight)
        np.testing.assert_array_equal(
            _constant(model, gamma),
            module.obs_weight.fake_quant(module.module.weight).detach().numpy(),
        )
        self.assertEqual(sum(gamma_index in op.inputs for op in graph.operators), 1)
        self._assert_activation(intermediate, module.obs_act_pre_gamma)
        self._assert_activation(graph.tensors[multiply.outputs[0]], module.obs_act_out)

    def test_affine_split_metadata_and_reference_execution(self):
        for wrapper in (QuantRMSNorm, QuantGemma4RMSNorm):
            for dtype in (DType.uint(8), DType.int(16)):
                with self.subTest(wrapper=wrapper.__name__, dtype=dtype):
                    module, sample, document = self._export(wrapper, affine(dtype))
                    self._assert_split(module, document)
                    self.assertFalse(
                        torch.equal(
                            module.obs_act_pre_gamma._cached_scale,
                            module.obs_act_out._cached_scale,
                        )
                    )
                    runtime = CircleReferenceRuntime(document)
                    actual = runtime.run(
                        [sample.numpy()], mode=ExecutionMode.FAKE_QUANTIZE
                    ).outputs[0]
                    with torch.no_grad():
                        expected = module(sample).numpy()
                    # Circle uses float64 normalization and rounds affine ties away
                    # from zero; PyTorch uses float32 and ties-to-even. One output
                    # grid step bounds this reference-runtime rounding difference.
                    output_step = float(module.obs_act_out._cached_scale)
                    np.testing.assert_allclose(
                        actual, expected, rtol=0, atol=output_step
                    )

    def test_mx_split_metadata_multiple_blocks_and_partial_block(self):
        for wrapper in (QuantRMSNorm, QuantGemma4RMSNorm):
            with self.subTest(wrapper=wrapper.__name__):
                module, _, document = self._export(wrapper, mx(axis=-1), width=65)
                self._assert_split(module, document)
                self.assertEqual(
                    list(
                        document.model.subgraphs[0]
                        .tensors[document.model.subgraphs[0].outputs[0]]
                        .shape
                    ),
                    [1, 6, 65],
                )

    def test_intermediate_overrides_keep_distinct_domains(self):
        policies = (
            (affine(DType.uint(8)), affine(DType.int(16))),
            (affine(DType.int(16)), affine(DType.uint(8))),
            (affine(DType.uint(8)), mx(axis=-1)),
            (mx(axis=-1), affine(DType.int(16))),
            (mx(axis=-1), mx(axis=1)),
        )
        for wrapper in (QuantRMSNorm, QuantGemma4RMSNorm):
            for output, intermediate in policies:
                with self.subTest(
                    wrapper=wrapper.__name__, output=output, intermediate=intermediate
                ):
                    module, _, document = self._export(
                        wrapper, output, pre_spec=intermediate, width=65
                    )
                    self._assert_split(module, document)

    def test_false_preserves_fused_gamma_for_affine_and_mx(self):
        for wrapper in (QuantRMSNorm, QuantGemma4RMSNorm):
            for spec in (affine(DType.uint(8)), mx(axis=-1)):
                with self.subTest(wrapper=wrapper.__name__, spec=spec):
                    module, _, document = self._export(
                        wrapper, spec, split=False, width=65
                    )
                    model = document.model
                    norm = _operators(
                        model, circle.BuiltinOperator.BuiltinOperator.RMS_NORM
                    )
                    self.assertEqual(len(norm), 1)
                    self.assertEqual(
                        _operators(model, circle.BuiltinOperator.BuiltinOperator.MUL),
                        [],
                    )
                    weight = model.subgraphs[0].tensors[norm[0].inputs[1]]
                    self._assert_affine(weight, module.obs_weight)
                    np.testing.assert_array_equal(
                        _constant(model, weight),
                        module.obs_weight.fake_quant(module.module.weight)
                        .detach()
                        .numpy(),
                    )
                    self.assertNotIn("act_pre_gamma", dict(module.named_observers()))

    def test_no_gamma_has_no_mul_or_intermediate_observer(self):
        for wrapper in (QuantRMSNorm, QuantGemma4RMSNorm):
            with self.subTest(wrapper=wrapper.__name__):
                module, _, document = self._export(
                    wrapper, affine(DType.uint(8)), gamma=False
                )
                self.assertNotIn("act_pre_gamma", dict(module.named_observers()))
                self.assertEqual(
                    len(
                        _operators(
                            document.model,
                            circle.BuiltinOperator.BuiltinOperator.RMS_NORM,
                        )
                    ),
                    1,
                )
                self.assertEqual(
                    _operators(
                        document.model, circle.BuiltinOperator.BuiltinOperator.MUL
                    ),
                    [],
                )

    def test_learned_all_one_gamma_keeps_declared_split_structure(self):
        for wrapper in (QuantRMSNorm, QuantGemma4RMSNorm):
            with self.subTest(wrapper=wrapper.__name__):
                module, _, document = self._export(
                    wrapper, affine(DType.uint(8)), unit_gamma=True
                )
                self._assert_split(module, document)


if __name__ == "__main__":
    unittest.main()
