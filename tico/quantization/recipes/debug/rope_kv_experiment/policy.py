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

"""Opt-in Gemma4 boundary policies; production registration stays untouched."""

import copy
from typing import Any, Iterable, Optional

import torch
import torch.nn as nn

from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.observers.affine_base import AffineObserverBase
from tico.quantization.wrapq.observers.base import ObserverBase
from tico.quantization.wrapq.wrappers.gemma4.quant_text_attention import (
    LayerKV,
    QuantGemma4TextAttention,
)


class ExperimentGemma4TextAttention(QuantGemma4TextAttention):
    """Run R0/R1 and K0/K1 on the same production wrapper and exporter.

    R1 replaces the *choice* of half/reordered grids with the upstream norm
    grid. During calibration these value-preserving sites do not collect.
    During QUANT the canonical frozen fake quantizer declares their inherited
    domain, which the normal exporter legalizes/folds. No grid is overwritten.

    K0 restores two changes as a bundle: current ``r -> S -> C`` and past
    ``p_S -> P -> C``. P is independently calibrated at the actual cache read;
    its policy is cloned from S before preparation, including a shared producer.
    Persistent storage still receives only the original S delta. K1 delegates
    unchanged to production. Old production checkpoint validation remains active.
    """

    obs_experiment_past_key: ObserverBase
    obs_experiment_past_value: ObserverBase

    def __init__(
        self,
        fp_attn: nn.Module,
        *,
        rope_unified: bool = False,
        kv_simplified: bool = True,
        qcfg: Optional[PTQConfig] = None,
        fp_name: Optional[str] = None,
    ):
        super().__init__(fp_attn, qcfg=qcfg, fp_name=fp_name)
        self.rope_unified = bool(rope_unified)
        self.kv_simplified = bool(kv_simplified)
        if not self.kv_simplified:
            self._make_past_observers((self.obs_new_k, self.obs_new_v))

    def _make_past_observers(self, storage: tuple[ObserverBase, ObserverBase]) -> None:
        for suffix, source in zip(("key", "value"), storage):
            observer = copy.deepcopy(source)
            observer.reset()
            observer.name = "experiment_past_" + suffix
            setattr(self, "obs_experiment_past_" + suffix, observer)

    def bind_cache_source(self, source: QuantGemma4TextAttention) -> None:
        previous = getattr(self, "_cache_source", None)
        super().bind_cache_source(source)
        if not self.kv_simplified and previous is not source:
            self._make_past_observers(self.cache_storage_observers())

    def bind_external_float_cache(self) -> None:
        was_bound = getattr(self, "_external_float_cache", False)
        super().bind_external_float_cache()
        if not self.kv_simplified and not was_bound:
            self._make_past_observers(self.cache_storage_observers())

    @staticmethod
    def _per_tensor_affine(observer: ObserverBase) -> bool:
        return (
            isinstance(observer, AffineObserverBase)
            and observer.channel_axis is None
            and not observer.qscheme.is_per_channel()
            and observer.fake_quant_enabled
        )

    def _rope_norm_observer(self, obs_x1: ObserverBase) -> Optional[ObserverBase]:
        if not self.rope_unified or self.attn_options.rope != "pre_negated_sin":
            return None
        if obs_x1 is self.obs_q_x1:
            norm = self.q_norm.wrapped
            boundaries = (self.obs_q_x1, self.obs_q_x2, self.obs_q_cat)
        elif not self.is_kv_shared_layer and obs_x1 is self.obs_k_x1:
            norm = self.k_norm.wrapped
            boundaries = (self.obs_k_x1, self.obs_k_x2, self.obs_k_cat)
        else:
            return None
        observer = getattr(norm, "obs_act_out", None)
        if not isinstance(observer, ObserverBase) or not all(
            self._per_tensor_affine(site) for site in (*boundaries, observer)
        ):
            return None
        return observer

    def _rot(self, tensor, obs_x1, obs_x2, obs_neg, obs_cat):
        canonical = self._rope_norm_observer(obs_x1)
        if canonical is None:
            return super()._rot(tensor, obs_x1, obs_x2, obs_neg, obs_cat)
        x1, x2 = torch.chunk(tensor, 2, dim=-1)
        # Never call _fq here: it would recollect the norm's EMA statistics.
        if self._mode is Mode.QUANT:
            x1, x2 = canonical.fake_quant(x1), canonical.fake_quant(x2)
        reordered = torch.cat((x2, x1), dim=-1)
        return (
            canonical.fake_quant(reordered) if self._mode is Mode.QUANT else reordered
        )

    def _read_cache_key_value(self, key_value: LayerKV) -> LayerKV:
        stored = super()._read_cache_key_value(key_value)
        # Production invokes _cache_to_compute only in QUANT for regular layers.
        # Collect P exactly once here in CALIB, including shared consumers.
        if not self.kv_simplified and self._mode is Mode.CALIB:
            self.obs_experiment_past_key.collect(stored[0].detach())
            self.obs_experiment_past_value.collect(stored[1].detach())
        return stored

    def _cache_to_compute(self, key_value: LayerKV) -> LayerKV:
        if not self.kv_simplified and self._mode is Mode.QUANT:
            key_value = (
                self.obs_experiment_past_key.fake_quant(key_value[0]),
                self.obs_experiment_past_value.fake_quant(key_value[1]),
            )
        return super()._cache_to_compute(key_value)

    def _build_present_key_value(
        self,
        *,
        new_key_value: LayerKV,
        current_key_value: LayerKV,
        past_key_value: Optional[Any],
        use_cache: bool = True,
    ):
        return super()._build_present_key_value(
            new_key_value=new_key_value,
            current_key_value=(
                current_key_value if self.kv_simplified else new_key_value
            ),
            past_key_value=past_key_value,
            use_cache=use_cache,
        )

    def _all_observers(self) -> Iterable[ObserverBase]:
        active = list(super()._all_observers())
        if getattr(self, "rope_unified", False):
            groups = [(self.obs_q_x1, self.obs_q_x2, self.obs_q_cat)]
            if not self.is_kv_shared_layer:
                groups.append((self.obs_k_x1, self.obs_k_x2, self.obs_k_cat))
            for group in groups:
                if self._rope_norm_observer(group[0]) is not None:
                    active = [site for site in active if site not in group]
        if not getattr(self, "kv_simplified", True):
            active.extend(
                (self.obs_experiment_past_key, self.obs_experiment_past_value)
            )
        return tuple(active)

    def get_extra_state(self):
        state = super().get_extra_state()
        state["rope_kv_experiment"] = (self.rope_unified, self.kv_simplified)
        return state

    def set_extra_state(self, state):
        if state.get("rope_kv_experiment") != (
            self.rope_unified,
            self.kv_simplified,
        ):
            raise RuntimeError(
                "Experimental R/K policy differs; prepare the same variant."
            )
        super().set_extra_state(state)
