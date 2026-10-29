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

"""Saved-byte graph evidence and explicit reference-runtime execution.

FAKE_QUANTIZE exchanges dequantized FLOAT32, but each quantized input is first
encoded with its serialized ABI and decoded again. NATIVE exchanges integer codes
and is reported independently; unsupported integer arithmetic is never a pass.
"""

from __future__ import annotations

import hashlib
import json
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils._pytree import tree_flatten

from tico.circle.document import CircleDocument
from tico.circle.runtime import (
    CircleProgram,
    CircleReferenceRuntime,
    ExecutionMode,
    RuntimeTensor,
    UnsupportedCircleOperatorError,
)
from tico.circle.runtime.program import tensor_type_name
from tico.circle.runtime.quantization import dequantize_array, quantize_array
from tico.utils.signature import ModelInputSpec


def _array(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def error_metrics(reference: Any, actual: Any) -> dict[str, Any]:
    """Use finite FLOAT64 errors and an explicit small-norm convention."""
    left, right = _array(reference).astype(np.float64), _array(actual).astype(
        np.float64
    )
    if left.shape != right.shape or left.size == 0:
        raise ValueError(f"Invalid comparison shapes: {left.shape}, {right.shape}.")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Nonfinite numerical evidence cannot pass.")
    diff = right - left
    norm, actual_norm = float(np.linalg.norm(left)), float(np.linalg.norm(right))
    cosine = (
        float(np.vdot(left, right) / (norm * actual_norm))
        if norm > 1e-12 and actual_norm > 1e-12
        else float(np.array_equal(left, right))
    )
    return {
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "max_abs": float(np.max(np.abs(diff))),
        "normalized_l2": float(np.linalg.norm(diff) / max(norm, 1e-12)),
        "cosine": cosine,
        "reference_norm": norm,
        "epsilon": 1e-12,
        "elements": int(left.size),
        "strict_allclose_atol_1e-5_rtol_1e-4": bool(
            np.allclose(left, right, atol=1e-5, rtol=1e-4)
        ),
    }


def _tensor(tensor: RuntimeTensor) -> dict[str, Any]:
    quantization = tensor.quantization
    return {
        "index": tensor.index,
        "name": tensor.name,
        "dtype": tensor_type_name(tensor.tensor_type),
        "shape": list(tensor.shape),
        "shape_signature": tensor.shape_signature,
        "qparams": (
            None
            if quantization is None
            else {
                "scale": list(quantization.scale),
                "zero_point": list(quantization.zero_point),
                "axis": quantization.quantized_dimension,
            }
        ),
        "bytes": int(np.prod(tensor.shape, dtype=np.int64)) * tensor.dtype.itemsize,
    }


def _option_value(value: Any) -> Any:
    """Represent serialized builtin options without dropping defaults or arrays."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (tuple, list)):
        return [_option_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _option_value(item) for key, item in value.items()}
    return {key: _option_value(item) for key, item in vars(value).items()}


def assert_rope_kv_domains(graph: dict[str, Any]) -> dict[str, Any]:
    """Reject serialized value-preserving concats with incompatible actual grids."""
    checked = {"rope": 0, "kv_compute_or_storage": 0}
    for concat in graph["concats"]:
        if concat["region"] not in checked:
            continue
        checked[concat["region"]] += 1
        if not concat["same_serialized_domain"]:
            raise AssertionError(
                f"Concat operator {concat['operator']} has incompatible serialized domains."
            )
    return {"status": "PASS", "checked_concat_counts": checked}


def graph_evidence(program: CircleProgram) -> dict[str, Any]:
    """Record serialized contracts and disjoint structural Quantize regions.

    Regions describe immediate value-preserving paths around RoPE last-axis and
    KV sequence-axis concats. Operator/tensor records retain the full evidence;
    these structural regions are not inferred from calibrated qparam equality.
    """
    producers = {index: op for op in program.operators for index in op.outputs}
    consumers: dict[int, list[Any]] = {}
    for op in program.operators:
        for index in op.inputs:
            consumers.setdefault(index, []).append(op)
    transparent = {
        "QUANTIZE",
        "RESHAPE",
        "TRANSPOSE",
        "SLICE",
        "STRIDED_SLICE",
        "SPLIT",
        "SPLIT_V",
        "SQUEEZE",
        "EXPAND_DIMS",
        "IDENTITY",
    }
    regions: dict[int, str] = {}

    def upstream(index: int, region: str) -> None:
        op = producers.get(index)
        if op is None or op.name not in transparent:
            return
        if op.name == "QUANTIZE":
            regions.setdefault(op.index, region)
        if op.inputs:
            upstream(op.inputs[0], region)

    def downstream(index: int, region: str, seen: set[int]) -> None:
        if index in seen:
            return
        seen.add(index)
        for op in consumers.get(index, []):
            if op.name not in transparent:
                continue
            if op.name == "QUANTIZE":
                regions.setdefault(op.index, region)
            for out in op.outputs:
                downstream(out, region, seen)

    concats = []
    for op in program.operators:
        if op.name != "CONCATENATION":
            continue
        output = program.tensor(op.outputs[0])
        axis = int(op.options.axis) % len(output.shape)
        region = "other"
        if len(output.shape) == 4 and axis == 3:
            region = "rope"
        elif len(output.shape) == 4 and axis == 2:
            region = "kv_compute_or_storage"
        for position, index in enumerate(op.inputs):
            if region == "rope":
                upstream(index, "rope")
            elif region == "kv_compute_or_storage":
                upstream(index, "past_read" if position == 0 else "current_compute")
        if region == "rope":
            downstream(op.outputs[0], "rope", set())
        contracts = [_tensor(program.tensor(index)) for index in op.inputs + op.outputs]
        qparams = [(value["dtype"], value["qparams"]) for value in contracts]
        concats.append(
            {
                "operator": op.index,
                "axis": axis,
                "region": region,
                "inputs": contracts[:-1],
                "outputs": contracts[-1:],
                "same_serialized_domain": all(value == qparams[0] for value in qparams),
            }
        )
    quantizes = []
    chains = []
    for op in program.operators:
        if op.name != "QUANTIZE":
            continue
        quantizes.append(
            {
                "operator": op.index,
                "region": regions.get(op.index, "other"),
                "input": _tensor(program.tensor(op.inputs[0])),
                "output": _tensor(program.tensor(op.outputs[0])),
            }
        )
        previous = producers.get(op.inputs[0])
        if previous is not None and previous.name == "QUANTIZE":
            chains.append(
                {
                    "operators": [previous.index, op.index],
                    "tensors": [
                        _tensor(program.tensor(index))
                        for index in (
                            previous.inputs[0],
                            previous.outputs[0],
                            op.outputs[0],
                        )
                    ],
                }
            )
    counts = Counter(value["region"] for value in quantizes)
    return {
        "operator_count": len(program.operators),
        "operator_counts": dict(Counter(op.name for op in program.operators)),
        "operators": [
            {
                "index": op.index,
                "name": op.name,
                "builtin_code": op.builtin_code,
                "version": op.version,
                "custom_code": op.custom_code,
                "inputs": list(op.inputs),
                "outputs": list(op.outputs),
                "options_type": op.options_type,
                "options": _option_value(op.options),
            }
            for op in program.operators
        ],
        "quantize_count": len(quantizes),
        "quantize_by_region": {
            key: counts[key]
            for key in ("rope", "past_read", "current_compute", "other")
        },
        "region_definition": "Transparent paths adjacent to rank-4 last-axis RoPE or sequence-axis KV concats; current region includes any storage Quantize on that path.",
        "quantizes": quantizes,
        "consecutive_quantize_pairs": chains,
        "concats": concats,
        "inputs": [_tensor(program.tensor(index)) for index in program.inputs],
        "outputs": [_tensor(program.tensor(index)) for index in program.outputs],
        "tensors": [_tensor(tensor) for tensor in program.tensors],
    }


class SavedCircle:
    """Load the actual file once and provide ABI-correct repeatable execution."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.binary = self.path.read_bytes()
        self.sha256 = hashlib.sha256(self.binary).hexdigest()
        self.document = CircleDocument.from_bytes(self.binary)
        self.document.verify(raise_on_error=True)
        self.program = CircleProgram(self.document)
        self.spec = ModelInputSpec(self.binary)
        self.runtime: CircleReferenceRuntime | None = None

    def _runtime(self) -> CircleReferenceRuntime:
        if self.runtime is None:
            self.runtime = CircleReferenceRuntime(self.program)
        return self.runtime

    def encoded_inputs(self, args: tuple[Any, ...]) -> list[np.ndarray]:
        """Encode each flattened positional input using its serialized qparams."""
        bound = self.spec.bind(args, {}, check=False)
        result = []
        for index, raw in zip(self.program.inputs, bound):
            tensor = self.program.tensor(index)
            value = _array(raw)
            if tensor.is_quantized_integer:
                value = quantize_array(value.astype(np.float32), tensor)
            else:
                value = value.astype(tensor.dtype, copy=False)
            result.append(value)
        return result

    def run_encoded(
        self, values: list[np.ndarray], *, mode: ExecutionMode, trace: bool = False
    ):
        """Execute integer ABI values, decoding only for the fake-quant mode."""
        inputs = []
        for index, value in zip(self.program.inputs, values):
            tensor = self.program.tensor(index)
            inputs.append(
                dequantize_array(value, tensor)
                if mode is ExecutionMode.FAKE_QUANTIZE and tensor.is_quantized_integer
                else value
            )
        return self._runtime().run(inputs, mode=mode, trace=trace)

    def run(
        self,
        args: tuple[Any, ...],
        *,
        mode: ExecutionMode = ExecutionMode.FAKE_QUANTIZE,
        trace: bool = False,
    ) -> tuple[np.ndarray, ...]:
        """Return real-valued outputs, suitable for S-delta cache append."""
        result = self.run_encoded(self.encoded_inputs(args), mode=mode, trace=trace)
        return tuple(
            (
                dequantize_array(value, self.program.tensor(index))
                if mode is ExecutionMode.NATIVE
                and self.program.tensor(index).is_quantized_integer
                else value
            )
            for index, value in zip(self.program.outputs, result.outputs)
        )


def export_and_check(
    module: torch.nn.Module, args: tuple[Any, ...], output_dir: Path, name: str
) -> dict[str, Any]:
    """Export once, reload saved bytes, inspect, and attempt both runtime modes.

    Numerical mismatch is measured with strict allclose reported separately from
    execution success. Unexpected execution/export failures are ERROR, never PASS.
    """
    import tico
    from tico.quantization.recipes.debug.export_regression import _state_contract

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.circle"
    before = _state_contract(module)
    with torch.no_grad():
        reference = [
            _array(value)
            for value in tree_flatten(module(*args))[0]
            if isinstance(value, torch.Tensor)
        ]
        converted = tico.convert(module, args)
        converted.save(path)
    saved = SavedCircle(path)
    evidence: dict[str, Any] = {
        "path": str(path),
        "sha256": saved.sha256,
        "bytes": len(saved.binary),
        "structural_status": "PASS",
        "graph": graph_evidence(saved.program),
        "runtime": {},
    }
    evidence["domain_legality"] = assert_rope_kv_domains(evidence["graph"])
    for mode in (ExecutionMode.NATIVE, ExecutionMode.FAKE_QUANTIZE):
        try:
            actual = saved.run(args, mode=mode)
            if len(actual) != len(reference) or not actual:
                raise ValueError("Saved Circle output count differs from eager.")
            metrics = [
                error_metrics(left, right) for left, right in zip(reference, actual)
            ]
            exact = all(
                value["strict_allclose_atol_1e-5_rtol_1e-4"] for value in metrics
            )
            evidence["runtime"][mode.value] = {
                "status": "EXECUTED",
                "strict_fidelity_status": "PASS" if exact else "FAIL",
                "outputs": metrics,
            }
            payload: dict[str, Any] = {
                f"eager_{index}": value for index, value in enumerate(reference)
            }
            payload.update(
                {f"circle_{index}": value for index, value in enumerate(actual)}
            )
            np.savez(output_dir / f"{name}.{mode.value}.npz", **payload)
        except UnsupportedCircleOperatorError as error:
            evidence["runtime"][mode.value] = {
                "status": "UNSUPPORTED",
                "type": type(error).__name__,
                "message": str(error),
            }
        except Exception as error:
            evidence["runtime"][mode.value] = {
                "status": "ERROR",
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            }
    evidence["frozen_state_preserved"] = before == _state_contract(module)
    if not evidence["frozen_state_preserved"]:
        raise RuntimeError("Export or evaluation mutated frozen weights/observers.")
    (output_dir / f"{name}.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return evidence
