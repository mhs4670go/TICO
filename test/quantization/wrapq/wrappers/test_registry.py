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

"""
Unit-tests for the lightweight wrapper registry.

What is verified
----------------
1. `register` decorator adds the mapping and lookup returns it.
2. `try_register` succeeds when the target class exists.
3. `try_register` is a NO-OP when the module / class is absent.
4. Variant-specific lookup returns the correct wrapper when registered.
5. Variant fallback order is respected:
   - exact variant match
   - "prefill" fallback
   - any registered variant (last resort)
"""

import sys
import types
import unittest

import torch.nn as nn

from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from tico.quantization.wrapq.wrappers.registry import lookup, register, try_register


# Dummy fp32 & quant modules for tests
class DummyFP(nn.Linear):  # inherit nn.Module for type-compat
    def __init__(self):
        super().__init__(4, 4)


class DummyQuant(QuantModuleBase):
    def forward(self, x):
        return x

    def _all_observers(self):
        return ()


class TestRegistry(unittest.TestCase):

    # 1) plain @register ------------------------------------------------
    def test_register_and_lookup(self):
        @register(DummyFP)  # defaults to variant="prefill"
        class _Q(DummyQuant):
            ...

        # Default lookup() should return the "prefill" wrapper.
        self.assertIs(lookup(DummyFP), _Q)
        # Explicit variant also returns it because only "prefill" exists.
        self.assertIs(lookup(DummyFP, variant="prefill"), _Q)
        self.assertIs(lookup(DummyFP, variant="decode"), _Q)  # falls back to "prefill"

    # 2) try_register when path exists ---------------------------------
    def test_try_register_success(self):
        # create a throw-away module with a class inside
        mod = types.ModuleType("tmp_mod")

        class TmpFP(nn.Linear):  # noqa: D401
            def __init__(self):
                super().__init__(2, 2)

        mod.TmpFP = TmpFP  # type: ignore[attr-defined]
        sys.modules["tmp_mod"] = mod  # inject into sys.modules

        @try_register("tmp_mod.TmpFP")  # defaults to variant="prefill"
        class TmpQuant(DummyQuant):
            ...

        self.assertIs(lookup(TmpFP), TmpQuant)
        self.assertIs(lookup(TmpFP, variant="prefill"), TmpQuant)
        self.assertIs(
            lookup(TmpFP, variant="decode"), TmpQuant
        )  # falls back to "prefill"

        del sys.modules["tmp_mod"]  # clean up

    # 3) try_register when target missing --------------------------------
    def test_try_register_graceful_skip(self):
        path = "nonexistent.module.Foo"

        @try_register(path)
        class SkipQuant(DummyQuant):
            ...

        # lookup should fail (module missing) without raising
        self.assertIsNone(lookup(type("Fake", (), {})))

    # 4) variant-specific registration -----------------------------------
    def test_variant_exact_match(self):
        class VFP(nn.Linear):
            def __init__(self):
                super().__init__(3, 3)

        @register(VFP, variant="prefill")
        class QPrefill(DummyQuant):
            ...

        @register(VFP, variant="decode")
        class QDecode(DummyQuant):
            ...

        self.assertIs(lookup(VFP, variant="prefill"), QPrefill)
        self.assertIs(lookup(VFP, variant="decode"), QDecode)

        # Default variant is "prefill"
        self.assertIs(lookup(VFP), QPrefill)

    # 5) fallback order ---------------------------------------------------
    def test_variant_fallback_order(self):
        class FFP(nn.Linear):
            def __init__(self):
                super().__init__(5, 5)

        # Case A: only prefill exists -> any other variant falls back to prefill.
        @register(FFP, variant="prefill")
        class OnlyPrefill(DummyQuant):
            ...

        self.assertIs(lookup(FFP, variant="decode"), OnlyPrefill)
        self.assertIs(lookup(FFP, variant="whatever"), OnlyPrefill)

        # Case B: no prefill, but some other variant exists -> last-resort "any variant".
        class AFP(nn.Linear):
            def __init__(self):
                super().__init__(6, 6)

        @register(AFP, variant="backendX")
        class BackendX(DummyQuant):
            ...

        # No exact match for "decode", no "prefill" either -> return any registered variant.
        self.assertIs(lookup(AFP, variant="decode"), BackendX)
        # Default lookup asks for "prefill" first, but since it's missing, should still return "any".
        self.assertIs(lookup(AFP), BackendX)

    # 6) try_register variant support ------------------------------------
    def test_try_register_variant(self):
        # create a throw-away module with a class inside
        mod = types.ModuleType("tmp_mod_v")

        class VTmpFP(nn.Linear):
            def __init__(self):
                super().__init__(7, 7)

        mod.VTmpFP = VTmpFP  # type: ignore[attr-defined]
        sys.modules["tmp_mod_v"] = mod

        @try_register("tmp_mod_v.VTmpFP", variant="decode")
        class VTmpQuantDecode(DummyQuant):
            ...

        # Exact decode match should resolve
        self.assertIs(lookup(VTmpFP, variant="decode"), VTmpQuantDecode)

        # If only "decode" exists, default ("prefill") falls back to "any variant".
        self.assertIs(lookup(VTmpFP), VTmpQuantDecode)

        del sys.modules["tmp_mod_v"]
