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

"""Global end-to-end scale and AdaRound refinement for affine Conv weights."""

from __future__ import annotations

import math

from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn

from tico.quantization.algorithm.adaround.joint import (
    JointAdaRoundWeightGroup,
    JointAdaRoundWeightStatistics,
    LearnableScaleAdaRoundWeightQuantizer,
)
from tico.quantization.algorithm.adaround.joint_runner import JointAdaRoundObjective
from tico.quantization.algorithm.block_reconstruction.selection import (
    copy_outputs,
    metric_value,
    OutputMetrics,
)
from tico.quantization.analysis import OutputAdapter
from tico.quantization.wrapq.control import (
    FakeQuantState,
    iter_quantization_sites,
    QuantizationSite,
    SiteRole,
)
from tico.quantization.wrapq.observers.affine_base import AffineObserverBase
from tico.quantization.wrapq.observers.base import ObserverBase


MetricsEvaluator = Callable[[], OutputMetrics]
ProgressCallback = Callable[["GlobalWeightRefinementCheckpoint"], None]


@dataclass(frozen=True)
class GlobalRefinementWeightStatistics:
    """Summarize one globally refined Conv weight tensor."""

    base: JointAdaRoundWeightStatistics
    changed_from_checkpoint_count: int

    @property
    def changed_from_checkpoint_ratio(self) -> float:
        return self.changed_from_checkpoint_count / max(self.base.element_count, 1)

    def to_dict(self) -> dict[str, float | int | str]:
        value = self.base.to_dict()
        value.update(
            {
                "changed_from_checkpoint_count": self.changed_from_checkpoint_count,
                "changed_from_checkpoint_ratio": (self.changed_from_checkpoint_ratio),
            }
        )
        return value


class CheckpointInitializedScaleAdaRoundQuantizer(
    LearnableScaleAdaRoundWeightQuantizer
):
    """Initialize global scale/AdaRound from an existing hard W8 checkpoint.

    The reference tensor is always the original FP32 weight. The initial scale
    and zero-point come from the checkpoint-replayed affine observer, while the
    initial hard code comes from the candidate's current effective W8 weight.
    This preserves the checkpoint exactly without discarding the FP fractional
    information needed for a second optimization pass.
    """

    def __init__(
        self,
        original: AffineObserverBase,
        reference_weight: torch.Tensor,
        initial_hard_codes: torch.Tensor,
        checkpoint_effective_weight: torch.Tensor | None = None,
        *,
        gamma: float,
        zeta: float,
        initialization_epsilon: float,
        max_scale_ratio: float,
    ) -> None:
        super().__init__(
            original,
            reference_weight,
            gamma=gamma,
            zeta=zeta,
            initialization_epsilon=initialization_epsilon,
            max_scale_ratio=max_scale_ratio,
        )
        codes = initial_hard_codes.detach().to(
            device=self.alpha.device,
            dtype=self.alpha.dtype,
        )
        if codes.shape != self.alpha.shape:
            raise ValueError(
                "Global refinement initial-code shape differs from the FP "
                f"weight shape: {tuple(codes.shape)} != {tuple(self.alpha.shape)}."
            )
        if not torch.all((codes >= self.dtype.qmin) & (codes <= self.dtype.qmax)):
            raise ValueError("Global refinement initial codes exceed dtype bounds.")
        self.register_buffer(
            "_checkpoint_initial_codes",
            codes.clone(),
            persistent=False,
        )
        effective = checkpoint_effective_weight
        if effective is None:
            scale, zero_point = self.compute_qparams()
            scale_broadcast, zero_point_broadcast = _broadcast_qparams(
                self._reference_weight,
                scale,
                zero_point,
                channel_axis=self.channel_axis,
            )
            effective = (codes - zero_point_broadcast) * scale_broadcast
        effective = effective.detach().to(
            device=self.alpha.device,
            dtype=self.alpha.dtype,
        )
        if effective.shape != self.alpha.shape:
            raise ValueError(
                "Global refinement checkpoint-effective weight shape differs "
                f"from the FP weight shape: {tuple(effective.shape)} != "
                f"{tuple(self.alpha.shape)}."
            )
        self.register_buffer(
            "_checkpoint_effective_weight",
            effective.clone(),
            persistent=False,
        )
        self._nearest_override = False
        self._initialize_alpha_from_codes(codes)
        reproduced = self.quantized_codes(hard=True).detach()
        if not torch.equal(reproduced, codes):
            mismatch = int((reproduced != codes).sum().cpu().item())
            raise RuntimeError(
                "Global refinement could not reproduce checkpoint hard codes; "
                f"mismatched elements={mismatch}."
            )

    def set_nearest_override(self, enabled: bool) -> None:
        """Use round-to-nearest at the current learned scale for diagnostics."""
        self._nearest_override = bool(enabled)

    def hard_weight(self) -> torch.Tensor:
        """Return the exact checkpoint tensor while the hard state is unchanged."""
        codes = self.quantized_codes(hard=True).detach()
        if self._matches_checkpoint_state(codes):
            return self._checkpoint_effective_weight
        return super().hard_weight()

    def fake_quant(self, x: torch.Tensor) -> torch.Tensor:
        """Preserve checkpoint execution exactly before global updates begin."""
        if not self.fake_quant_enabled:
            return x
        if x.shape != self._reference_weight.shape:
            raise ValueError(
                "Global refinement weight shape changed: "
                f"{tuple(x.shape)} != {tuple(self._reference_weight.shape)}."
            )
        if self.hard and not self._nearest_override:
            codes = self.quantized_codes(hard=True).detach()
            if self._matches_checkpoint_state(codes):
                return self._checkpoint_effective_weight
        return super().fake_quant(x)

    def quantized_codes(self, *, hard: bool | None = None) -> torch.Tensor:
        if self._nearest_override:
            scale, zero_point = self.compute_qparams()
            scale_broadcast, zero_point_broadcast = _broadcast_qparams(
                self._reference_weight,
                scale,
                zero_point,
                channel_axis=self.channel_axis,
            )
            normalized = self._reference_weight / scale_broadcast
            normalized = normalized + zero_point_broadcast
            return torch.round(normalized).clamp(
                self.dtype.qmin,
                self.dtype.qmax,
            )
        return super().quantized_codes(hard=hard)

    def _matches_checkpoint_state(self, codes: torch.Tensor) -> bool:
        if not torch.equal(codes, self._checkpoint_initial_codes):
            return False
        return bool(torch.count_nonzero(self.raw_log_scale_delta.detach()) == 0)

    def global_statistics(
        self,
        site_path: str,
        family: str,
    ) -> GlobalRefinementWeightStatistics:
        base = super().statistics(site_path, family)
        final_codes = super().quantized_codes(hard=True).detach()
        changed = int(
            (final_codes != self._checkpoint_initial_codes).sum().cpu().item()
        )
        return GlobalRefinementWeightStatistics(
            base=base,
            changed_from_checkpoint_count=changed,
        )

    def _initialize_alpha_from_codes(self, codes: torch.Tensor) -> None:
        scale, zero_point = self.compute_qparams()
        scale_broadcast, zero_point_broadcast = _broadcast_qparams(
            self._reference_weight,
            scale,
            zero_point,
            channel_axis=self.channel_axis,
        )
        normalized = self._reference_weight / scale_broadcast
        normalized = normalized + zero_point_broadcast
        floor_codes = torch.floor(normalized)
        floor_clipped = floor_codes.clamp(self.dtype.qmin, self.dtype.qmax)
        ceil_clipped = (floor_codes + 1.0).clamp(
            self.dtype.qmin,
            self.dtype.qmax,
        )
        reachable = (codes == floor_clipped) | (codes == ceil_clipped)
        if not torch.all(reachable):
            count = int((~reachable).sum().cpu().item())
            raise ValueError(
                "Checkpoint codes are not reachable by floor/ceil decisions "
                f"at the checkpoint scale; unreachable elements={count}."
            )
        round_up = (codes == ceil_clipped) & (codes != floor_clipped)
        round_down = (codes == floor_clipped) & (codes != ceil_clipped)
        epsilon = self.initialization_epsilon
        with torch.no_grad():
            alpha = self.alpha
            alpha.copy_(
                torch.where(
                    round_up,
                    alpha.abs() + epsilon,
                    torch.where(
                        round_down,
                        -alpha.abs() - epsilon,
                        alpha,
                    ),
                )
            )


@dataclass
class _GlobalBinding:
    group: JointAdaRoundWeightGroup
    owner: nn.Module
    attribute_names: tuple[str, ...]
    weight_module: nn.Conv2d
    original_observer: AffineObserverBase
    entry_weight: torch.Tensor
    entry_scale: torch.Tensor
    entry_zero_point: torch.Tensor
    entry_qparams_locked: bool
    entry_enabled: bool
    entry_fake_quant_enabled: bool
    proxy: CheckpointInitializedScaleAdaRoundQuantizer


class GlobalAdaRoundWeightSet:
    """Install all-Conv global proxies and commit or restore atomically."""

    def __init__(
        self,
        model: nn.Module,
        groups: Sequence[JointAdaRoundWeightGroup],
        source_weights: Mapping[str, torch.Tensor],
        *,
        gamma: float,
        zeta: float,
        initialization_epsilon: float,
        max_scale_ratio: float,
    ) -> None:
        definitions = tuple(groups)
        _validate_groups(definitions, source_weights)
        sites = {site.path: site for site in iter_quantization_sites(model)}
        bindings: list[_GlobalBinding] = []
        try:
            for group in definitions:
                site = sites.get(group.site_path)
                if site is None:
                    raise KeyError(
                        f"Unknown global-refinement weight site {group.site_path!r}."
                    )
                binding = _build_binding(
                    group,
                    site,
                    source_weights[group.site_path],
                    gamma=gamma,
                    zeta=zeta,
                    initialization_epsilon=initialization_epsilon,
                    max_scale_ratio=max_scale_ratio,
                )
                _replace_observer_attributes(
                    binding.owner,
                    binding.attribute_names,
                    expected=binding.original_observer,
                    replacement=binding.proxy,
                    site_path=group.site_path,
                )
                bindings.append(binding)
        except Exception:
            for binding in reversed(bindings):
                _replace_observer_attributes(
                    binding.owner,
                    binding.attribute_names,
                    expected=binding.proxy,
                    replacement=binding.original_observer,
                    site_path=binding.group.site_path,
                )
            raise
        self.bindings = tuple(bindings)
        self._closed = False

    def alpha_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(binding.proxy.alpha for binding in self.bindings)

    def scale_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(binding.proxy.raw_log_scale_delta for binding in self.bindings)

    def trainable_parameters(self) -> tuple[nn.Parameter, ...]:
        return (*self.alpha_parameters(), *self.scale_parameters())

    def set_hard(self, hard: bool) -> None:
        for binding in self.bindings:
            binding.proxy.set_hard(hard)

    def set_nearest_override(self, enabled: bool) -> None:
        for binding in self.bindings:
            binding.proxy.set_nearest_override(enabled)

    def state_snapshot(self) -> dict[str, dict[str, torch.Tensor]]:
        return {
            binding.group.name: {
                "alpha": binding.proxy.alpha.detach().cpu().clone(),
                "raw_log_scale_delta": (
                    binding.proxy.raw_log_scale_delta.detach().cpu().clone()
                ),
            }
            for binding in self.bindings
        }

    def load_state_snapshot(
        self,
        state: Mapping[str, Mapping[str, torch.Tensor]],
    ) -> None:
        expected = tuple(binding.group.name for binding in self.bindings)
        if tuple(state) != expected:
            raise ValueError(
                f"Global-refinement state keys differ: {tuple(state)} != {expected}."
            )
        with torch.no_grad():
            for binding in self.bindings:
                values = state[binding.group.name]
                for name, parameter in (
                    ("alpha", binding.proxy.alpha),
                    (
                        "raw_log_scale_delta",
                        binding.proxy.raw_log_scale_delta,
                    ),
                ):
                    value = values.get(name)
                    if not isinstance(value, torch.Tensor):
                        raise TypeError(
                            f"Global-refinement state {name!r} for "
                            f"{binding.group.name!r} is not a Tensor."
                        )
                    if value.shape != parameter.shape:
                        raise ValueError(
                            f"Global-refinement {name} shape mismatch for "
                            f"{binding.group.name!r}."
                        )
                    parameter.copy_(
                        value.to(
                            device=parameter.device,
                            dtype=parameter.dtype,
                        )
                    )

    def rounding_regularizer(self, beta: float) -> torch.Tensor:
        weighted = [
            binding.proxy.rounding_regularizer(beta) * binding.proxy.alpha.numel()
            for binding in self.bindings
        ]
        total = sum(binding.proxy.alpha.numel() for binding in self.bindings)
        return torch.stack(weighted).sum() / max(total, 1)

    def scale_regularizer(self) -> torch.Tensor:
        weighted = [
            binding.proxy.scale_regularizer()
            * binding.proxy.raw_log_scale_delta.numel()
            for binding in self.bindings
        ]
        total = sum(
            binding.proxy.raw_log_scale_delta.numel() for binding in self.bindings
        )
        return torch.stack(weighted).sum() / max(total, 1)

    def statistics(self) -> tuple[GlobalRefinementWeightStatistics, ...]:
        return tuple(
            binding.proxy.global_statistics(
                binding.group.site_path,
                binding.group.family,
            )
            for binding in self.bindings
        )

    def finalize(self) -> tuple[GlobalRefinementWeightStatistics, ...]:
        if self._closed:
            raise RuntimeError("Global refinement weight set is already closed.")
        self.set_nearest_override(False)
        self.set_hard(True)
        statistics = self.statistics()
        committed = tuple(
            (
                binding,
                binding.proxy.hard_weight().detach().clone(),
                binding.proxy.learned_scale().detach().clone(),
                binding.proxy._fixed_zero_point.detach().clone(),
            )
            for binding in self.bindings
        )
        try:
            with torch.no_grad():
                for binding, weight, _, _ in committed:
                    binding.weight_module.weight.copy_(weight)
            for binding, _, scale, zero_point in committed:
                device = binding.original_observer.min_val.device
                binding.original_observer.load_qparams(
                    scale.to(device=device),
                    zero_point.to(device=device),
                    lock=True,
                )
                binding.original_observer.fake_quant_enabled = (
                    binding.entry_fake_quant_enabled
                )
            self._restore_observers()
        except Exception:
            self._restore_entry_state()
            raise
        self._closed = True
        return statistics

    def restore(self) -> None:
        if self._closed:
            return
        self._restore_entry_state()
        self._closed = True

    def _restore_entry_state(self) -> None:
        with torch.no_grad():
            for binding in self.bindings:
                binding.weight_module.weight.copy_(binding.entry_weight)
        for binding in self.bindings:
            device = binding.original_observer.min_val.device
            binding.original_observer.load_qparams(
                binding.entry_scale.to(device=device),
                binding.entry_zero_point.to(device=device),
                lock=binding.entry_qparams_locked,
            )
            binding.original_observer.enabled = binding.entry_enabled
            binding.original_observer.fake_quant_enabled = (
                binding.entry_fake_quant_enabled
            )
        self._restore_observers()

    def _restore_observers(self) -> None:
        for binding in self.bindings:
            current = tuple(
                getattr(binding.owner, name, None) for name in binding.attribute_names
            )
            if all(value is binding.original_observer for value in current):
                continue
            if not all(value is binding.proxy for value in current):
                raise RuntimeError(
                    f"Observer aliases for {binding.group.site_path!r} are in "
                    "a mixed state and cannot be restored transactionally."
                )
            _replace_observer_attributes(
                binding.owner,
                binding.attribute_names,
                expected=binding.proxy,
                replacement=binding.original_observer,
                site_path=binding.group.site_path,
            )


@dataclass(frozen=True)
class GlobalWeightRefinementConfig:
    """Configure end-to-end all-Conv scale and rounding refinement."""

    steps: int = 3_000
    gradient_accumulation_steps: int = 4
    evaluation_interval: int = 100
    alpha_learning_rate: float = 1.0e-4
    scale_learning_rate: float = 3.0e-5
    primary_output: str = "regressors"
    auxiliary_output: str = "classifiers"
    auxiliary_loss_weight: float = 0.25
    loss_epsilon: float = 1.0e-8
    rounding_loss_weight: float = 1.0e-3
    scale_loss_weight: float = 1.0e-4
    warmup_fraction: float = 0.2
    beta_start: float = 20.0
    beta_end: float = 2.0
    gamma: float = -0.1
    zeta: float = 1.1
    initialization_epsilon: float = 1.0e-6
    max_scale_ratio: float = 1.25
    gradient_clip_norm: float | None = 1.0
    initialization_metric_tolerance: float = 1.0e-4
    initialization_metric_relative_tolerance: float = 1.0e-3
    seed: int = 20260831

    def validate(self) -> None:
        if self.steps < 0:
            raise ValueError("Global refinement steps must be nonnegative.")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive.")
        if self.evaluation_interval <= 0:
            raise ValueError("evaluation_interval must be positive.")
        if not self.primary_output or not self.auxiliary_output:
            raise ValueError("Global refinement output names must be non-empty.")
        if self.primary_output == self.auxiliary_output:
            raise ValueError("Primary and auxiliary outputs must differ.")
        for name, value in (
            ("alpha_learning_rate", self.alpha_learning_rate),
            ("scale_learning_rate", self.scale_learning_rate),
            ("loss_epsilon", self.loss_epsilon),
            ("beta_start", self.beta_start),
            ("beta_end", self.beta_end),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        for name, value in (
            ("auxiliary_loss_weight", self.auxiliary_loss_weight),
            ("rounding_loss_weight", self.rounding_loss_weight),
            ("scale_loss_weight", self.scale_loss_weight),
            (
                "initialization_metric_tolerance",
                self.initialization_metric_tolerance,
            ),
            (
                "initialization_metric_relative_tolerance",
                self.initialization_metric_relative_tolerance,
            ),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        if not 0.0 <= self.warmup_fraction < 1.0:
            raise ValueError("warmup_fraction must be in [0, 1).")
        if self.beta_start < self.beta_end:
            raise ValueError("beta_start must be at least beta_end.")
        if not math.isfinite(self.gamma) or self.gamma >= 0.0:
            raise ValueError("gamma must be finite and negative.")
        if not math.isfinite(self.zeta) or self.zeta <= 1.0:
            raise ValueError("zeta must be finite and greater than one.")
        if not 0.0 < self.initialization_epsilon < 0.5:
            raise ValueError("initialization_epsilon must be in (0, 0.5).")
        if not math.isfinite(self.max_scale_ratio) or self.max_scale_ratio <= 1.0:
            raise ValueError("max_scale_ratio must be greater than one.")
        if self.gradient_clip_norm is not None and (
            not math.isfinite(self.gradient_clip_norm) or self.gradient_clip_norm <= 0.0
        ):
            raise ValueError("gradient_clip_norm must be positive or None.")
        if not isinstance(self.seed, int):
            raise TypeError("Global refinement seed must be an integer.")


@dataclass(frozen=True)
class GlobalWeightRefinementCheckpoint:
    """Record one hard end-to-end validation checkpoint."""

    step: int
    train_primary_loss: float | None
    train_auxiliary_loss: float | None
    train_rounding_loss: float | None
    train_scale_loss: float | None
    train_total_loss: float | None
    selection_outputs: OutputMetrics
    primary_score: float
    beta: float | None
    selected_as_best: bool
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "step": self.step,
            "train_primary_loss": self.train_primary_loss,
            "train_auxiliary_loss": self.train_auxiliary_loss,
            "train_rounding_loss": self.train_rounding_loss,
            "train_scale_loss": self.train_scale_loss,
            "train_total_loss": self.train_total_loss,
            "selection_outputs": copy_outputs(self.selection_outputs),
            "primary_score": self.primary_score,
            "beta": self.beta,
            "selected_as_best": self.selected_as_best,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class GlobalWeightRefinementResult:
    """Summarize global optimization, diagnostics, and transactional decision."""

    steps: int
    training_sample_count: int
    gradient_accumulation_steps: int
    weight_groups: tuple[str, ...]
    weight_families: tuple[str, ...]
    entry_selection_outputs: OutputMetrics
    initialized_selection_outputs: OutputMetrics
    selected_outputs: OutputMetrics
    entry_acceptance_outputs: OutputMetrics
    acceptance_outputs: OutputMetrics
    entry_evaluation_outputs: OutputMetrics
    selected_evaluation_outputs: OutputMetrics
    scale_only_selection_outputs: OutputMetrics
    scale_only_acceptance_outputs: OutputMetrics
    scale_only_evaluation_outputs: OutputMetrics
    final_evaluation_outputs: OutputMetrics
    best_step: int
    accepted: bool
    acceptance_reason: str
    weight_statistics: tuple[GlobalRefinementWeightStatistics, ...]
    checkpoint_history: tuple[GlobalWeightRefinementCheckpoint, ...]
    training_primary_history: tuple[float, ...] = ()
    training_auxiliary_history: tuple[float, ...] = ()
    training_rounding_history: tuple[float, ...] = ()
    training_scale_history: tuple[float, ...] = ()
    training_total_history: tuple[float, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "steps": self.steps,
            "training_sample_count": self.training_sample_count,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "weight_group_count": len(self.weight_groups),
            "weight_groups": list(self.weight_groups),
            "weight_families": list(self.weight_families),
            "entry_selection_outputs": copy_outputs(self.entry_selection_outputs),
            "initialized_selection_outputs": copy_outputs(
                self.initialized_selection_outputs
            ),
            "selected_outputs": copy_outputs(self.selected_outputs),
            "entry_acceptance_outputs": copy_outputs(self.entry_acceptance_outputs),
            "acceptance_outputs": copy_outputs(self.acceptance_outputs),
            "entry_evaluation_outputs": copy_outputs(self.entry_evaluation_outputs),
            "selected_evaluation_outputs": copy_outputs(
                self.selected_evaluation_outputs
            ),
            "scale_only_selection_outputs": copy_outputs(
                self.scale_only_selection_outputs
            ),
            "scale_only_acceptance_outputs": copy_outputs(
                self.scale_only_acceptance_outputs
            ),
            "scale_only_evaluation_outputs": copy_outputs(
                self.scale_only_evaluation_outputs
            ),
            "final_evaluation_outputs": copy_outputs(self.final_evaluation_outputs),
            "best_step": self.best_step,
            "accepted": self.accepted,
            "acceptance_reason": self.acceptance_reason,
            "weight_statistics": [value.to_dict() for value in self.weight_statistics],
            "checkpoint_history": [
                value.to_dict() for value in self.checkpoint_history
            ],
            "training_primary_history": list(self.training_primary_history),
            "training_auxiliary_history": list(self.training_auxiliary_history),
            "training_rounding_history": list(self.training_rounding_history),
            "training_scale_history": list(self.training_scale_history),
            "training_total_history": list(self.training_total_history),
        }


class GlobalWeightRefinementRunner:
    """Optimize every selected Conv scale and rounding decision end to end."""

    def __init__(
        self,
        config: GlobalWeightRefinementConfig | None = None,
    ) -> None:
        self.config = config or GlobalWeightRefinementConfig()
        self.config.validate()

    def refine(
        self,
        *,
        reference_model: nn.Module,
        candidate_model: nn.Module,
        training_samples: Sequence[torch.Tensor],
        weight_groups: Sequence[JointAdaRoundWeightGroup],
        source_weights: Mapping[str, torch.Tensor],
        output_adapter: OutputAdapter,
        selection_evaluator: MetricsEvaluator,
        selection_objective: JointAdaRoundObjective,
        acceptance_evaluator: MetricsEvaluator,
        acceptance_objective: JointAdaRoundObjective,
        evaluation_evaluator: MetricsEvaluator,
        progress_callback: ProgressCallback | None = None,
        device: torch.device | str | None = None,
    ) -> GlobalWeightRefinementResult:
        groups = tuple(weight_groups)
        if not groups:
            raise ValueError("Global refinement requires Conv weight groups.")
        if not training_samples:
            raise ValueError("Global refinement requires training samples.")
        optimization_device = torch.device(device or _module_device(candidate_model))
        _validate_batch_one(training_samples)
        teacher_cache = _TeacherOutputCache(
            reference_model,
            training_samples,
            output_adapter=output_adapter,
            device=optimization_device,
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.config.seed)
        entry_selection = copy_outputs(selection_evaluator())
        entry_acceptance = copy_outputs(acceptance_evaluator())
        entry_evaluation = copy_outputs(evaluation_evaluator())

        with (
            _RequiresGradState(candidate_model),
            FakeQuantState(candidate_model) as fake_quant_state,
        ):
            fake_quant_state.set_all(True)
            weights = GlobalAdaRoundWeightSet(
                candidate_model,
                groups,
                source_weights,
                gamma=self.config.gamma,
                zeta=self.config.zeta,
                initialization_epsilon=self.config.initialization_epsilon,
                max_scale_ratio=self.config.max_scale_ratio,
            )
            try:
                weights.set_hard(True)
                initialized_selection = copy_outputs(selection_evaluator())
                self._validate_initialization(
                    entry_selection,
                    initialized_selection,
                )
                best_state = weights.state_snapshot()
                best_step = 0
                best_outputs = initialized_selection
                entry_checkpoint = GlobalWeightRefinementCheckpoint(
                    step=0,
                    train_primary_loss=None,
                    train_auxiliary_loss=None,
                    train_rounding_loss=None,
                    train_scale_loss=None,
                    train_total_loss=None,
                    selection_outputs=initialized_selection,
                    primary_score=selection_objective.score(initialized_selection),
                    beta=None,
                    selected_as_best=True,
                    reason="checkpoint hard-code and scale entry state",
                )
                checkpoints = [entry_checkpoint]
                if progress_callback is not None:
                    progress_callback(entry_checkpoint)
                optimizer = torch.optim.Adam(
                    (
                        {
                            "params": weights.alpha_parameters(),
                            "lr": self.config.alpha_learning_rate,
                        },
                        {
                            "params": weights.scale_parameters(),
                            "lr": self.config.scale_learning_rate,
                        },
                    )
                )
                parameters = weights.trainable_parameters()
                primary_history: list[float] = []
                auxiliary_history: list[float] = []
                rounding_history: list[float] = []
                scale_history: list[float] = []
                total_history: list[float] = []

                weights.set_hard(False)
                for step in range(1, self.config.steps + 1):
                    optimizer.zero_grad(set_to_none=True)
                    primary_value = 0.0
                    auxiliary_value = 0.0
                    for _ in range(self.config.gradient_accumulation_steps):
                        index = int(
                            torch.randint(
                                len(teacher_cache),
                                (1,),
                                generator=generator,
                            ).item()
                        )
                        sample, target = teacher_cache.get(
                            index,
                            device=optimization_device,
                        )
                        outputs = output_adapter(candidate_model(sample))
                        primary = _normalized_l1(
                            outputs[self.config.primary_output],
                            target[self.config.primary_output],
                            epsilon=self.config.loss_epsilon,
                        )
                        auxiliary = _normalized_l1(
                            outputs[self.config.auxiliary_output],
                            target[self.config.auxiliary_output],
                            epsilon=self.config.loss_epsilon,
                        )
                        data_loss = (
                            primary + self.config.auxiliary_loss_weight * auxiliary
                        )
                        data_loss = data_loss / (
                            self.config.gradient_accumulation_steps
                        )
                        data_loss.backward()
                        primary_value += float(primary.detach().cpu().item())
                        auxiliary_value += float(auxiliary.detach().cpu().item())
                    primary_value /= self.config.gradient_accumulation_steps
                    auxiliary_value /= self.config.gradient_accumulation_steps

                    beta = self._beta(step)
                    if beta is None:
                        rounding = next(iter(parameters)).new_zeros(())
                    else:
                        rounding = weights.rounding_regularizer(beta)
                    scale = weights.scale_regularizer()
                    regularization = (
                        self.config.rounding_loss_weight * rounding
                        + self.config.scale_loss_weight * scale
                    )
                    regularization.backward()
                    if self.config.gradient_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            parameters,
                            self.config.gradient_clip_norm,
                        )
                    optimizer.step()

                    rounding_value = float(rounding.detach().cpu().item())
                    scale_value = float(scale.detach().cpu().item())
                    total_value = (
                        primary_value
                        + self.config.auxiliary_loss_weight * auxiliary_value
                        + self.config.rounding_loss_weight * rounding_value
                        + self.config.scale_loss_weight * scale_value
                    )
                    if not math.isfinite(total_value):
                        raise FloatingPointError(
                            "Global refinement produced a non-finite loss."
                        )
                    primary_history.append(primary_value)
                    auxiliary_history.append(auxiliary_value)
                    rounding_history.append(rounding_value)
                    scale_history.append(scale_value)
                    total_history.append(total_value)

                    should_evaluate = (
                        step % self.config.evaluation_interval == 0
                        or step == self.config.steps
                    )
                    if not should_evaluate:
                        continue
                    weights.set_hard(True)
                    candidate_outputs = copy_outputs(selection_evaluator())
                    better, reason = selection_objective.better(
                        candidate_outputs,
                        best_outputs,
                        entry_selection,
                    )
                    if better:
                        best_state = weights.state_snapshot()
                        best_step = step
                        best_outputs = candidate_outputs
                    checkpoint = GlobalWeightRefinementCheckpoint(
                        step=step,
                        train_primary_loss=primary_value,
                        train_auxiliary_loss=auxiliary_value,
                        train_rounding_loss=rounding_value,
                        train_scale_loss=scale_value,
                        train_total_loss=total_value,
                        selection_outputs=candidate_outputs,
                        primary_score=selection_objective.score(candidate_outputs),
                        beta=beta,
                        selected_as_best=better,
                        reason=reason,
                    )
                    checkpoints.append(checkpoint)
                    if progress_callback is not None:
                        progress_callback(checkpoint)
                    weights.set_hard(False)

                weights.load_state_snapshot(best_state)
                weights.set_hard(True)
                selected_outputs = copy_outputs(selection_evaluator())
                acceptance_outputs = copy_outputs(acceptance_evaluator())
                selected_evaluation = copy_outputs(evaluation_evaluator())
                weights.set_nearest_override(True)
                scale_only_selection = copy_outputs(selection_evaluator())
                scale_only_acceptance = copy_outputs(acceptance_evaluator())
                scale_only_evaluation = copy_outputs(evaluation_evaluator())
                weights.set_nearest_override(False)
                accepted, acceptance_reason = acceptance_objective.accepted(
                    acceptance_outputs,
                    entry_acceptance,
                )
                selected_statistics = weights.statistics()
                if accepted:
                    weight_statistics = weights.finalize()
                else:
                    weights.restore()
                    weight_statistics = selected_statistics
            except Exception:
                weights.restore()
                raise

        final_evaluation = copy_outputs(evaluation_evaluator())
        return GlobalWeightRefinementResult(
            steps=self.config.steps,
            training_sample_count=len(training_samples),
            gradient_accumulation_steps=(self.config.gradient_accumulation_steps),
            weight_groups=tuple(group.name for group in groups),
            weight_families=tuple(group.family for group in groups),
            entry_selection_outputs=entry_selection,
            initialized_selection_outputs=initialized_selection,
            selected_outputs=selected_outputs,
            entry_acceptance_outputs=entry_acceptance,
            acceptance_outputs=acceptance_outputs,
            entry_evaluation_outputs=entry_evaluation,
            selected_evaluation_outputs=selected_evaluation,
            scale_only_selection_outputs=scale_only_selection,
            scale_only_acceptance_outputs=scale_only_acceptance,
            scale_only_evaluation_outputs=scale_only_evaluation,
            final_evaluation_outputs=final_evaluation,
            best_step=best_step,
            accepted=accepted,
            acceptance_reason=acceptance_reason,
            weight_statistics=weight_statistics,
            checkpoint_history=tuple(checkpoints),
            training_primary_history=tuple(primary_history),
            training_auxiliary_history=tuple(auxiliary_history),
            training_rounding_history=tuple(rounding_history),
            training_scale_history=tuple(scale_history),
            training_total_history=tuple(total_history),
        )

    def _validate_initialization(
        self,
        entry: OutputMetrics,
        initialized: OutputMetrics,
    ) -> None:
        for output_name in (
            self.config.primary_output,
            self.config.auxiliary_output,
        ):
            left = metric_value(entry, output_name, "mae")
            right = metric_value(initialized, output_name, "mae")
            difference = abs(left - right)
            allowed = (
                self.config.initialization_metric_tolerance
                + self.config.initialization_metric_relative_tolerance
                * max(abs(left), abs(right))
            )
            if difference > allowed:
                raise RuntimeError(
                    "Global refinement did not reproduce the loaded checkpoint "
                    f"for {output_name}.mae: {left:.9e} != {right:.9e}; "
                    f"difference={difference:.3e}, allowed={allowed:.3e} "
                    "(absolute + relative tolerance)."
                )

    def _beta(self, step: int) -> float | None:
        warmup_steps = int(round(self.config.steps * self.config.warmup_fraction))
        if step <= warmup_steps:
            return None
        remaining = max(self.config.steps - warmup_steps, 1)
        progress = min(max((step - warmup_steps) / remaining, 0.0), 1.0)
        return (
            self.config.beta_start
            + (self.config.beta_end - self.config.beta_start) * progress
        )


class _TeacherOutputCache:
    def __init__(
        self,
        reference_model: nn.Module,
        samples: Sequence[torch.Tensor],
        *,
        output_adapter: OutputAdapter,
        device: torch.device,
    ) -> None:
        values: list[tuple[torch.Tensor, dict[str, torch.Tensor]]] = []
        reference_model.eval()
        with torch.no_grad():
            for sample in samples:
                input_cpu = sample.detach().to(device="cpu").clone()
                outputs = output_adapter(reference_model(sample.to(device=device)))
                targets = {
                    name: tensor.detach().to(device="cpu").clone()
                    for name, tensor in outputs.items()
                }
                values.append((input_cpu, targets))
        self._values = tuple(values)

    def __len__(self) -> int:
        return len(self._values)

    def get(
        self,
        index: int,
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        sample, outputs = self._values[index]
        return (
            sample.to(device=device),
            {name: tensor.to(device=device) for name, tensor in outputs.items()},
        )


class _RequiresGradState(AbstractContextManager["_RequiresGradState"]):
    def __init__(self, module: nn.Module) -> None:
        self._states = tuple(
            (parameter, parameter.requires_grad) for parameter in module.parameters()
        )

    def __enter__(self) -> "_RequiresGradState":
        for parameter, _ in self._states:
            parameter.requires_grad_(False)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        for parameter, requires_grad in self._states:
            parameter.requires_grad_(requires_grad)
        return None


def _build_binding(
    group: JointAdaRoundWeightGroup,
    site: QuantizationSite,
    source_weight: torch.Tensor,
    *,
    gamma: float,
    zeta: float,
    initialization_epsilon: float,
    max_scale_ratio: float,
) -> _GlobalBinding:
    if site.role is not SiteRole.PARAMETER or site.observer_name != "weight":
        raise ValueError(f"Global refinement site {site.path!r} must be a weight site.")
    if not isinstance(site.observer, AffineObserverBase):
        raise TypeError(f"Global refinement site {site.path!r} is not affine.")
    weight_module = getattr(site.module, "module", None)
    if not isinstance(weight_module, nn.Conv2d):
        raise TypeError(
            f"Global refinement supports Conv2d, got "
            f"{type(weight_module).__name__} at {site.path!r}."
        )
    if site.observer.channel_axis != 0:
        raise ValueError(
            f"Global Conv refinement expects channel_axis=0 at {site.path!r}."
        )
    source = source_weight.detach().to(
        device=weight_module.weight.device,
        dtype=weight_module.weight.dtype,
    )
    if source.shape != weight_module.weight.shape:
        raise ValueError(
            f"FP source shape mismatch at {site.path!r}: "
            f"{tuple(source.shape)} != {tuple(weight_module.weight.shape)}."
        )
    scale, zero_point = site.observer.compute_qparams()
    entry_weight = weight_module.weight.detach().clone()
    with torch.no_grad():
        checkpoint_effective_weight = site.observer.fake_quant(entry_weight)
        checkpoint_effective_weight = checkpoint_effective_weight.detach().clone()
    initial_codes = _effective_hard_codes(
        checkpoint_effective_weight,
        scale,
        zero_point,
        channel_axis=site.observer.channel_axis,
        qmin=site.observer.dtype.qmin,
        qmax=site.observer.dtype.qmax,
    )
    attributes = _observer_attribute_names(site.module, site.observer)
    if not attributes:
        raise RuntimeError(f"No registered observer alias exists for {site.path!r}.")
    proxy = CheckpointInitializedScaleAdaRoundQuantizer(
        site.observer,
        source,
        initial_codes,
        checkpoint_effective_weight,
        gamma=gamma,
        zeta=zeta,
        initialization_epsilon=initialization_epsilon,
        max_scale_ratio=max_scale_ratio,
    )
    return _GlobalBinding(
        group=group,
        owner=site.module,
        attribute_names=attributes,
        weight_module=weight_module,
        original_observer=site.observer,
        entry_weight=entry_weight,
        entry_scale=scale.detach().clone(),
        entry_zero_point=zero_point.detach().clone(),
        entry_qparams_locked=bool(site.observer._qparams_locked),
        entry_enabled=bool(site.observer.enabled),
        entry_fake_quant_enabled=bool(site.observer.fake_quant_enabled),
        proxy=proxy,
    )


def _effective_hard_codes(
    weight: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    *,
    channel_axis: int | None,
    qmin: int,
    qmax: int,
) -> torch.Tensor:
    scale_broadcast, zero_point_broadcast = _broadcast_qparams(
        weight,
        scale,
        zero_point,
        channel_axis=channel_axis,
    )
    return torch.round(weight / scale_broadcast + zero_point_broadcast).clamp(
        qmin,
        qmax,
    )


def _normalized_l1(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    *,
    epsilon: float,
) -> torch.Tensor:
    if candidate.shape != reference.shape:
        raise ValueError(
            "Global output shape mismatch: "
            f"{tuple(candidate.shape)} != {tuple(reference.shape)}."
        )
    numerator = (candidate - reference).abs().mean()
    denominator = reference.abs().mean().clamp_min(epsilon)
    return numerator / denominator


def _validate_batch_one(samples: Sequence[torch.Tensor]) -> None:
    for index, sample in enumerate(samples):
        if sample.ndim == 0 or int(sample.shape[0]) != 1:
            raise ValueError(
                "Global refinement uses B=1 forwards with gradient "
                f"accumulation; sample {index} has shape {tuple(sample.shape)}."
            )


def _validate_groups(
    groups: Sequence[JointAdaRoundWeightGroup],
    source_weights: Mapping[str, torch.Tensor],
) -> None:
    if not groups:
        raise ValueError("Global refinement requires at least one Conv group.")
    names = tuple(group.name for group in groups)
    paths = tuple(group.site_path for group in groups)
    if len(set(names)) != len(names):
        raise ValueError("Global refinement group names must be unique.")
    if len(set(paths)) != len(paths):
        raise ValueError("Global refinement site paths must be unique.")
    missing = tuple(path for path in paths if path not in source_weights)
    extra = tuple(path for path in source_weights if path not in set(paths))
    if missing or extra:
        raise ValueError(
            "Global refinement source-weight coverage mismatch: "
            f"missing={missing}, extra={extra}."
        )


def _broadcast_qparams(
    weight: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    *,
    channel_axis: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if channel_axis is None:
        return (
            scale.to(device=weight.device, dtype=weight.dtype),
            zero_point.to(device=weight.device, dtype=weight.dtype),
        )
    axis = channel_axis % weight.ndim
    shape = [1] * weight.ndim
    shape[axis] = -1
    return (
        scale.reshape(shape).to(device=weight.device, dtype=weight.dtype),
        zero_point.reshape(shape).to(device=weight.device, dtype=weight.dtype),
    )


def _observer_attribute_names(
    owner: nn.Module,
    observer: ObserverBase,
) -> tuple[str, ...]:
    return tuple(name for name, child in owner._modules.items() if child is observer)


def _replace_observer_attributes(
    owner: nn.Module,
    attribute_names: tuple[str, ...],
    *,
    expected: ObserverBase,
    replacement: ObserverBase,
    site_path: str,
) -> None:
    mismatched = tuple(
        name for name in attribute_names if getattr(owner, name, None) is not expected
    )
    if mismatched:
        raise RuntimeError(
            f"Observer aliases {mismatched} for {site_path!r} no longer "
            "reference the expected object."
        )
    replaced: list[str] = []
    try:
        for name in attribute_names:
            setattr(owner, name, replacement)
            replaced.append(name)
    except Exception:
        for name in replaced:
            setattr(owner, name, expected)
        raise


def _module_device(module: nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")
