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

"""Numerical and state evidence for the bounded RoPE/KV experiment."""

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F

from tico.quantization.wrapq.observers.base import ObserverBase


def tensor_metrics(reference, candidate, *, epsilon: float = 1e-12) -> dict[str, Any]:
    """Compare nonempty finite tensors in FP64, rejecting absent evidence.

    Normalized L2 uses ``max(norm(reference), epsilon)``. Cosine is one
    for two norms <= epsilon and zero when only one norm is <= epsilon.
    PEIR is max error divided by reference max-min, with denominator 1.0
    for a constant reference, matching the existing recipe metric. PEIR is
    dimensionless (a ratio, not a percentage); no epsilon enters PEIR.
    """
    left = torch.as_tensor(reference).detach().to(device="cpu", dtype=torch.float64)
    right = torch.as_tensor(candidate).detach().to(device="cpu", dtype=torch.float64)
    if left.shape != right.shape:
        raise ValueError(f"Metric shape mismatch: {left.shape} versus {right.shape}.")
    if left.numel() == 0:
        raise ValueError("Metrics require nonempty tensor evidence.")
    if epsilon <= 0 or not math.isfinite(epsilon):
        raise ValueError("Metric epsilon must be finite and positive.")
    if not bool(torch.isfinite(left).all() and torch.isfinite(right).all()):
        raise ValueError("Metrics reject NaN or infinite tensor evidence.")
    left, right = left.reshape(-1), right.reshape(-1)
    difference = right - left
    norm_left, norm_right = float(left.norm()), float(right.norm())
    if norm_left <= epsilon or norm_right <= epsilon:
        cosine = 1.0 if norm_left <= epsilon and norm_right <= epsilon else 0.0
    else:
        cosine = float(torch.dot(left / norm_left, right / norm_right).clamp(-1, 1))
    peak = float(difference.abs().max())
    interval = float(left.max() - left.min())
    return {
        "elements": left.numel(),
        "mae": float(difference.abs().mean()),
        "rmse": float(difference.square().mean().sqrt()),
        "max_abs": peak,
        "normalized_l2": float(difference.norm()) / max(norm_left, epsilon),
        "cosine": cosine,
        "reference_norm": norm_left,
        "epsilon": epsilon,
        "peir": peak / (interval if interval else 1.0),
        "peir_denominator": interval if interval else 1.0,
    }


def _state_value(value: Any) -> Any:
    """Represent nested state deterministically, including tensor dtype/bytes."""
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        return {
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "bytes": tensor.reshape(-1).view(torch.uint8).numpy().tobytes().hex(),
        }
    if isinstance(value, Mapping):
        return {str(key): _state_value(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_state_value(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"Unsupported state fingerprint payload: {type(value)!r}.")


def _fingerprint(value: Any) -> str:
    payload = json.dumps(_state_value(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def state_fingerprint(module: torch.nn.Module) -> str:
    """Hash parameters, buffers (also nonpersistent), modes and extra state.

    This reads state without collecting, freezing or casting any source grid.
    Dormant observers are included, so accidental mutation cannot be hidden by
    removing a site from the active-observer inventory.
    """
    return _fingerprint(
        {
            "state_dict": module.state_dict(),
            "buffers": dict(module.named_buffers()),
            "modules": {
                name: {
                    "training": child.training,
                    "mode": getattr(getattr(child, "_mode", None), "name", None),
                }
                for name, child in module.named_modules()
            },
        }
    )


def observer_records(module: torch.nn.Module) -> list[dict[str, Any]]:
    """Read every observer's exact policy, origin and frozen-state digest."""
    records = []
    for path, observer in module.named_modules():
        if not isinstance(observer, ObserverBase):
            continue
        state = observer.frozen_state
        record: dict[str, Any] = {
            "path": path,
            "name": observer.name,
            "observer": type(observer).__module__ + "." + type(observer).__qualname__,
            "dtype": str(observer.dtype),
            "qmin": observer.dtype.qmin,
            "qmax": observer.dtype.qmax,
            "qscheme": observer.qscheme.name,
            "channel_axis": observer.channel_axis,
            "frozen": state is not None,
            "enabled": observer.enabled,
            "fake_quant_enabled": observer.fake_quant_enabled,
            "state_sha256": state_fingerprint(observer),
            "origin": None if state is None else state.provenance,
        }
        if state is not None:
            observer.require_frozen()
            record.update(
                {
                    "kind": state.kind,
                    "locked": state.locked,
                    "scale": (
                        None if state.scale is None else state.scale.cpu().tolist()
                    ),
                    "scale_dtype": (
                        None if state.scale is None else str(state.scale.dtype)
                    ),
                    "zero_point": (
                        None
                        if state.zero_point is None
                        else state.zero_point.cpu().tolist()
                    ),
                    "zero_point_dtype": (
                        None
                        if state.zero_point is None
                        else str(state.zero_point.dtype)
                    ),
                    "observed_shape": state.observed_shape,
                }
            )
        records.append(record)
    return records


class ObserverTrace:
    """Temporarily record actual observer calls without issuing extra calls.

    ``records`` contains JSON-friendly counters; ``values`` contains CPU clones
    under each observer path, with ``collect``, ``before`` and ``after`` lists.
    Effective collection means enabled, unfrozen, nonempty collection. Imported
    locked observers therefore have zero effective updates. Restore hooks before
    export; this trace is for eager evidence only, never graph annotation.
    """

    def __init__(self, module: torch.nn.Module):
        self.module = module
        self.records: dict[str, dict[str, Any]] = {}
        self.values: dict[str, dict[str, list[torch.Tensor]]] = {}
        self._restores: list[tuple[ObserverBase, str, bool, Any]] = []

    def __enter__(self):
        if self._restores:
            raise RuntimeError("ObserverTrace context cannot be entered twice.")
        for path, observer in self.module.named_modules():
            if not isinstance(observer, ObserverBase):
                continue
            record = self.records.setdefault(
                path,
                {
                    "collect_calls": 0,
                    "effective_collect_calls": 0,
                    "collect_elements": 0,
                    "collect_min": None,
                    "collect_max": None,
                    "fake_quant_calls": 0,
                    "quantized_elements": 0,
                    "clipped_elements": 0,
                    "endpoint_elements": 0,
                },
            )
            self.values.setdefault(path, {"collect": [], "before": [], "after": []})
            self._install(observer, path, record)
        return self

    def _install(self, observer: ObserverBase, path: str, record: dict[str, Any]):
        original_collect, original_fq = observer.collect, observer.fake_quant
        for method in ("collect", "fake_quant"):
            self._restores.append(
                (
                    observer,
                    method,
                    method in observer.__dict__,
                    observer.__dict__.get(method),
                )
            )

        def collect(value):
            record["collect_calls"] += 1
            effective = (
                observer.enabled and observer.frozen_state is None and value.numel() > 0
            )
            if effective:
                detached = value.detach().cpu().clone()
                if not bool(torch.isfinite(detached).all()):
                    raise ValueError(f"Nonfinite calibration evidence at {path}.")
                self.values[path]["collect"].append(detached)
                record["effective_collect_calls"] += 1
                record["collect_elements"] += detached.numel()
                low, high = float(detached.min()), float(detached.max())
                record["collect_min"] = (
                    low
                    if record["collect_min"] is None
                    else min(record["collect_min"], low)
                )
                record["collect_max"] = (
                    high
                    if record["collect_max"] is None
                    else max(record["collect_max"], high)
                )
            return original_collect(value)

        def fake_quant(value):
            before = value.detach().cpu().clone()
            if not bool(torch.isfinite(before).all()):
                raise ValueError(f"Nonfinite prequantization evidence at {path}.")
            result = original_fq(value)
            after = result.detach().cpu().clone()
            if not bool(torch.isfinite(after).all()):
                raise ValueError(f"Nonfinite postquantization evidence at {path}.")
            self.values[path]["before"].append(before)
            self.values[path]["after"].append(after)
            record["fake_quant_calls"] += 1
            state = observer.frozen_state
            if (
                observer.fake_quant_enabled
                and state is not None
                and state.scale is not None
                and state.zero_point is not None
            ):
                scale = state.scale.detach().cpu().double()
                zero_point = state.zero_point.detach().cpu().double()
                if observer.channel_axis is not None:
                    shape = [1] * before.ndim
                    shape[observer.channel_axis] = scale.numel()
                    scale, zero_point = scale.reshape(shape), zero_point.reshape(shape)
                codes = before.double() / scale + zero_point
                low, high = observer.dtype.qmin, observer.dtype.qmax
                clipped = (codes < low) | (codes > high)
                rounded = codes.round().clamp(low, high)
                record["quantized_elements"] += codes.numel()
                record["clipped_elements"] += int(clipped.sum())
                record["endpoint_elements"] += int(
                    ((rounded == low) | (rounded == high)).sum()
                )
            return result

        observer.collect = collect  # type: ignore[method-assign]
        observer.fake_quant = fake_quant  # type: ignore[method-assign]

    def __exit__(self, exc_type, exc_value, traceback):
        for observer, method, present, value in reversed(self._restores):
            if present:
                setattr(observer, method, value)
            else:
                delattr(observer, method)
        self._restores.clear()
        return False

    def summary(self) -> dict[str, dict[str, Any]]:
        """Return counters with clipping and endpoint occupancy kept distinct."""
        result = {}
        for name, record in self.records.items():
            total = record["quantized_elements"]
            result[name] = dict(
                record,
                clipping_rate=record["clipped_elements"] / total if total else None,
                endpoint_occupancy=(
                    record["endpoint_elements"] / total if total else None
                ),
                clipping_rule="unrounded x/scale+zero_point outside [qmin,qmax]",
                endpoint_rule="rounded and clamped integer code equals qmin or qmax",
            )
        return result


def cached_nll(
    logits: torch.Tensor, target_tokens: torch.Tensor, score_mask=None
) -> dict[str, Any]:
    """Score already-aligned next-token targets; never silently shift targets.

    Inputs are logits [B,S,V], targets [B,S] and an optional Boolean [B,S]
    mask. Every true position contributes once to the NLL sum. At least one
    target is required. Keep per-token losses for paired sample/chunk analysis.
    """
    if logits.ndim != 3 or target_tokens.shape != logits.shape[:2]:
        raise ValueError("Cached NLL needs [B,S,V] logits and aligned [B,S] targets.")
    if target_tokens.dtype not in (torch.int32, torch.int64):
        raise ValueError("Cached NLL targets must be integer token IDs.")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("Cached NLL rejects nonfinite logits.")
    if score_mask is None:
        score_mask = torch.ones_like(target_tokens, dtype=torch.bool)
    if score_mask.shape != target_tokens.shape or score_mask.dtype != torch.bool:
        raise ValueError("Cached NLL score mask must be Boolean and match targets.")
    count = int(score_mask.sum())
    if count == 0:
        raise ValueError("Cached NLL requires at least one scored token.")
    selected_logits = logits[score_mask].double()
    selected_targets = target_tokens[score_mask].long()
    if bool(((selected_targets < 0) | (selected_targets >= logits.shape[-1])).any()):
        raise ValueError("Cached NLL scored target is outside the vocabulary.")
    loss = F.cross_entropy(selected_logits, selected_targets, reduction="none")
    total, mean = float(loss.sum()), float(loss.mean())
    ppl = math.exp(mean)
    if not math.isfinite(ppl):
        raise ValueError("Cached NLL perplexity overflow.")
    return {
        "nll_sum": total,
        "scored_tokens": count,
        "mean_nll": mean,
        "ppl": ppl,
        "token_nll": loss.detach().cpu().tolist(),
        "target_tokens": selected_targets.detach().cpu().tolist(),
    }


def shifted_cached_nll(logits, sequence, prefix_tokens: int, score_mask=None):
    """Align cached logits to original sequence target positions explicitly.

    ``prefix_tokens`` is the original sequence index of the first scored
    target. The first logit must come from processing the preceding token
    (position ``prefix_tokens-1``), e.g. the last prefill output. Later logits
    come from teacher-forced decode of each immediately preceding token.
    Prefix/warm-up tokens before that target index are excluded.
    """
    if logits.ndim != 3 or sequence.ndim != 2 or sequence.shape[0] != logits.shape[0]:
        raise ValueError("Cached alignment requires [B,S,V] logits and [B,T] sequence.")
    if (
        isinstance(prefix_tokens, bool)
        or not isinstance(prefix_tokens, int)
        or prefix_tokens < 1
    ):
        raise ValueError("Cached target prefix must be a positive integer.")
    stop = prefix_tokens + logits.shape[1]
    if stop > sequence.shape[1]:
        raise ValueError("Cached logits exceed available next-token targets.")
    result = cached_nll(logits, sequence[:, prefix_tokens:stop], score_mask)
    result["target_start"] = prefix_tokens
    result["target_stop_exclusive"] = stop
    return result


class BoundaryTrace:
    """Record wrapper boundaries in every mode, including NO_QUANT.

    Keys are observer module paths. ``values[path]`` and ``before[path]``
    retain ordered CPU tensor clones per real _fq invocation. The reordered
    RoPE result uses its original cat observer's path in every R policy;
    _rot instrumentation only adds it when _fq did not already record it.
    This avoids double-counting R0 while exposing the inherited R1 domain.
    """

    def __init__(self, module: torch.nn.Module):
        self.module = module
        self.values: dict[str, list[torch.Tensor]] = {}
        self.before: dict[str, list[torch.Tensor]] = {}
        self._restores: list[tuple[torch.nn.Module, str, bool, Any]] = []
        self._paths = {
            id(observer): path
            for path, observer in module.named_modules()
            if isinstance(observer, ObserverBase)
        }

    def __enter__(self):
        from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase

        if self._restores:
            raise RuntimeError("BoundaryTrace context cannot be entered twice.")
        for path, child in self.module.named_modules():
            if isinstance(child, QuantModuleBase):
                self._install(child, path)
        return self

    def _save(self, child: torch.nn.Module, method: str):
        self._restores.append(
            (child, method, method in child.__dict__, child.__dict__.get(method))
        )

    def _install(self, child, owner_path):
        original_fq = child._fq
        self._save(child, "_fq")

        def fake_quant(value, observer):
            path = self._paths.get(id(observer), owner_path + "." + observer.name)
            self.before.setdefault(path, []).append(value.detach().cpu().clone())
            result = original_fq(value, observer)
            self.values.setdefault(path, []).append(result.detach().cpu().clone())
            return result

        child._fq = fake_quant
        original_rot = getattr(child, "_rot", None)
        if not callable(original_rot):
            return
        self._save(child, "_rot")

        def reordered(value, obs_x1, obs_x2, obs_neg, obs_cat):
            path = self._paths.get(id(obs_cat), owner_path + "." + obs_cat.name)
            count = len(self.values.get(path, ()))
            result = original_rot(value, obs_x1, obs_x2, obs_neg, obs_cat)
            if len(self.values.get(path, ())) == count:
                self.values.setdefault(path, []).append(result.detach().cpu().clone())
                # Value-preserving pre-negated-sine reorder before inherited FQ.
                first, second = value.chunk(2, dim=-1)
                raw = torch.cat((second, first), dim=-1)
                self.before.setdefault(path, []).append(raw.detach().cpu().clone())
            return result

        child._rot = reordered

    def __exit__(self, exc_type, exc_value, traceback):
        for child, method, present, value in reversed(self._restores):
            if present:
                setattr(child, method, value)
            else:
                delattr(child, method)
        self._restores.clear()
        return False
