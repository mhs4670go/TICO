# Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

from typing import cast, Iterable, Optional, TYPE_CHECKING

import torch
import torch.nn as nn

from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.observers.identity import IdentityObserver
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from tico.quantization.wrapq.wrappers.registry import try_register

if TYPE_CHECKING:
    from transformers.models.llama.modeling_llama import LlamaRMSNorm
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRMSNorm


class _QuantRMSNormBase(QuantModuleBase):
    """Shared structural state for RMSNorm's optional gamma boundary."""

    with_scale: bool

    def __setstate__(self, state):
        super().__setstate__(state)
        # Existing whole-module checkpoints predate the split and retain their
        # fused execution. Observer/frozen-schema checks still run on use/load.
        if not hasattr(self, "_split_rmsnorm_gamma"):
            if getattr(self, "obs_act_pre_gamma", None) is not None:
                raise RuntimeError(
                    "RMSNorm act_pre_gamma state lacks its split setting; "
                    "reprepare the original float model and recalibrate."
                )
            self._split_rmsnorm_gamma = False
            self.obs_act_pre_gamma = None
            self.with_scale = getattr(self.module, "weight", None) is not None
        self._validate_loaded_state()

    def _init_gamma_split(self) -> None:
        # Snapshot the prepare-time choice; later config edits cannot change sites.
        self._split_rmsnorm_gamma = self.qcfg.split_rmsnorm_gamma
        self.obs_act_pre_gamma = (
            self._make_obs("act_pre_gamma")
            if self._split_rmsnorm_gamma and self.with_scale
            else None
        )

    @property
    def split_rmsnorm_gamma(self) -> bool:
        return self._split_rmsnorm_gamma

    def _unit_weight(self, x: torch.Tensor) -> torch.Tensor:
        unit = torch.ones((x.shape[-1],), dtype=x.dtype, device=x.device)
        if self._mode is Mode.QUANT and any(
            observer.fake_quant_enabled and not isinstance(observer, IdentityObserver)
            for observer in self._all_observers()
        ):
            # This exact constant grid is independent of the learned gamma policy.
            # It also gives the Circle normalization weight valid integer metadata.
            # Identity and bypassed observers must keep a fully floating export.
            unit = torch.fake_quantize_per_tensor_affine(unit, 1.0, 0, 0, 255)
        return unit

    def get_extra_state(self):
        state = super().get_extra_state()
        state["split_rmsnorm_gamma"] = self._split_rmsnorm_gamma
        state["rmsnorm_has_gamma"] = self.with_scale
        return state

    def _check_split_state(self, state) -> None:
        # Missing option metadata denotes the historical gamma-in-RMSNorm path.
        saved_split = state.get("split_rmsnorm_gamma", False)
        if (
            not isinstance(saved_split, bool)
            or saved_split != self._split_rmsnorm_gamma
        ):
            raise RuntimeError(
                "Incompatible RMSNorm split_rmsnorm_gamma state; reprepare the "
                "original float model with the saved option (legacy states use "
                "False), or reprepare and recalibrate for the new boundary."
            )
        if "rmsnorm_has_gamma" in state and (
            not isinstance(state["rmsnorm_has_gamma"], bool)
            or state["rmsnorm_has_gamma"] != self.with_scale
        ):
            raise RuntimeError(
                "Incompatible RMSNorm gamma parameter structure; reprepare "
                "the original float model and recalibrate."
            )

    def set_extra_state(self, state):
        if isinstance(state, dict):
            self._check_split_state(state)
        super().set_extra_state(state)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        state = state_dict.get(prefix + "_extra_state", {})
        if isinstance(state, dict):
            self._check_split_state(state)
        has_pre_gamma = any(
            key.startswith(prefix + "obs_act_pre_gamma.") for key in state_dict
        )
        if has_pre_gamma != (self.obs_act_pre_gamma is not None):
            raise RuntimeError(
                "Incompatible RMSNorm act_pre_gamma observer state; reprepare "
                "with the saved split_rmsnorm_gamma option and recalibrate "
                "when changing the quantization boundary."
            )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _validate_loaded_state(self) -> None:
        if not isinstance(getattr(self, "_split_rmsnorm_gamma", None), bool):
            raise RuntimeError(
                "Legacy RMSNorm whole-module state lacks its prepare-time "
                "boundary; reprepare with split_rmsnorm_gamma=False and recalibrate."
            )
        if not isinstance(getattr(self, "with_scale", None), bool) or (
            self.with_scale != (getattr(self.module, "weight", None) is not None)
        ):
            raise RuntimeError(
                "Incompatible RMSNorm gamma parameter structure; reprepare "
                "the original float model and recalibrate."
            )
        if (getattr(self, "obs_act_pre_gamma", None) is not None) != (
            self._split_rmsnorm_gamma and self.with_scale
        ):
            raise RuntimeError(
                "Incompatible RMSNorm observer structure; reprepare and recalibrate."
            )


@try_register(
    "transformers.models.llama.modeling_llama.LlamaRMSNorm",
    "transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLTextRMSNorm",
)
class QuantRMSNorm(_QuantRMSNormBase):
    def __init__(
        self,
        fp: nn.Module,
        *,
        qcfg: Optional[PTQConfig] = None,
        fp_name: Optional[str] = None,
    ):
        super().__init__(qcfg, fp_name=fp_name)
        self.module = cast("LlamaRMSNorm | Qwen3VLTextRMSNorm", fp)
        self.eps = float(self.module.variance_epsilon)
        self.with_scale = getattr(self.module, "weight", None) is not None

        self.obs_weight = self._make_per_tensor_obs("weight")
        self.obs_act_in = self._make_obs("act_in")
        self.obs_act_out = self._make_obs("act_out")
        self._init_gamma_split()

    def _prepare_calibration(self) -> None:
        # immediately capture the fixed weight range
        if self.with_scale:
            self.obs_weight.collect(self.module.weight)

    def forward(self, x: torch.Tensor):
        # 1) quantize input once
        x_q = self._fq(x, self.obs_act_in)

        # 2) quantize weights
        w: torch.Tensor = (
            self.module.weight
            if self.with_scale
            else torch.ones((x.shape[-1],), dtype=x.dtype, device=x.device)
        )
        if self._mode is Mode.CALIB and not self.with_scale:
            self.obs_weight.collect(w.detach())
        if self._mode is Mode.QUANT:
            w = self.obs_weight.fake_quant(w)

        # 3) rms
        if self.obs_act_pre_gamma is not None:
            rms = torch.ops.circle_custom.rms_norm(
                x_q,
                weight=self._unit_weight(x_q),
                eps=self.eps,
                preserve_gamma_boundary=True,
            )
            rms = self._fq(rms, self.obs_act_pre_gamma) * w
        else:
            rms = torch.ops.circle_custom.rms_norm(x_q, weight=w, eps=self.eps)
        rms_q = self._fq(rms, self.obs_act_out)

        return rms_q

    def _all_observers(self) -> Iterable:
        observers = (self.obs_weight, self.obs_act_in, self.obs_act_out)
        if self.obs_act_pre_gamma is not None:
            return (*observers, self.obs_act_pre_gamma)
        return observers
