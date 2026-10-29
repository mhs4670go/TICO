# Copyright (c) 2026 Samsung Electronics Co., Ltd. All Rights Reserved
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
"""Deterministic real Gemma4 modules and host cache operations."""

import copy

import torch

from tico.quantization.config.ptq import PTQConfig
from tico.quantization.config.specs import affine
from tico.quantization.recipes.debug.wrapper_smoke.cases.gemma4 import (
    _build_text_config,
)
from tico.quantization.wrapq.dtypes import INT16, UINT8
from tico.quantization.wrapq.wrappers.gemma4.quant_text_attention import (
    QuantGemma4TextAttention,
)
from tico.quantization.wrapq.wrappers.gemma4.rope import prepare_gemma4_rope_sin

from .policy import ExperimentGemma4TextAttention

VARIANTS = {
    "R0K0": (False, False),
    "R1K0": (True, False),
    "R0K1": (False, True),
    "R1K1": (True, True),
}
PROFILES = ("affine", "u8s16c")


def make_fp(seed, capacity=136, *, sliding=False, shared=False):
    """Use smoke's tiny dimensions, extending only legal context capacity."""
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextAttention

    torch.manual_seed(seed)
    layer = "sliding_attention" if sliding else "full_attention"
    config = _build_text_config(
        size_profile="tiny",
        layer_types=(layer, layer) if shared else (layer,),
        num_kv_shared_layers=1 if shared else 0,
    )
    config.max_position_embeddings = max(256, capacity)
    config.sliding_window = 8
    first = Gemma4TextAttention(config, layer_idx=0).eval()
    second = Gemma4TextAttention(config, layer_idx=1).eval() if shared else None
    return first, second


def make_qcfg(profile, capacity):
    """Keep smoke weights UINT8 in both profiles; change activation policy only."""
    return PTQConfig(
        activation=affine(INT16 if profile == "u8s16c" else UINT8),
        overrides={"new_k": affine(UINT8), "new_v": affine(UINT8)},
        model_args={"max_seq": capacity},
    )


def make_wrapper(fp, profile, capacity, variant=None):
    """Preserve an independently instantiated, unmodified production control."""
    config = make_qcfg(profile, capacity)
    if variant is None:
        return QuantGemma4TextAttention(copy.deepcopy(fp), qcfg=config).eval()
    rope, kv = VARIANTS[variant]
    return ExperimentGemma4TextAttention(
        copy.deepcopy(fp), qcfg=config, rope_unified=rope, kv_simplified=kv
    ).eval()


def sample(seed, length, offset=0):
    """Disjoint PRNG streams identify calibration/evaluation hidden traces."""
    generator = torch.Generator().manual_seed(seed)
    hidden = torch.randn(1, length, 64, generator=generator)
    position = torch.arange(offset, offset + length).float().view(1, length, 1)
    inv = torch.pow(10000.0, -torch.arange(16).float() / 16).view(1, 1, 16)
    phase = (position * inv).repeat(1, 1, 2)
    return hidden, (phase.cos(), phase.sin())


def mask(q_len, valid_past, *, slots=None, window=None):
    """Mask padding independently of logical positions and fixed current slot."""
    past_slots = valid_past if slots is None else slots
    keys = past_slots + q_len
    result = torch.full((1, 1, q_len, keys), -120.0)
    for query in range(q_len):
        start = max(0, valid_past + query - window + 1) if window else 0
        result[..., query, start:valid_past] = 0
        current_start = max(0, start - valid_past)
        result[..., query, past_slots + current_start : past_slots + query + 1] = 0
    return result


def call(module, hidden, rope, past=None, *, attention_mask=None):
    """Run tuple-cache wrapper with a fresh caller-owned cache and explicit mask."""
    past_length = 0 if past is None else past[0].size(-2)
    return module(
        hidden,
        (rope[0], prepare_gemma4_rope_sin(rope[1], module.attn_options.rope)),
        attention_mask=(
            attention_mask
            if attention_mask is not None
            else mask(hidden.size(1), past_length, window=module.sliding_window)
        ),
        past_key_value=past,
        use_cache=True,
    )


def append(past, delta, retain=None):
    """Append only producer S deltas; existing prefix bytes stay untouched."""
    values = (
        tuple(value.clone() for value in delta)
        if past is None
        else tuple(torch.cat((old, new), dim=-2) for old, new in zip(past, delta))
    )
    if retain is not None:
        values = tuple(value[..., -retain:, :].clone() for value in values)
    return values


def pad_cache(past, slots):
    """Static cache has a valid prefix and ignored right padding."""
    return tuple(
        torch.nn.functional.pad(value, (0, 0, 0, slots - value.size(-2)))
        for value in past
    )


def export_args(module, hidden, rope, past=None, *, slots=None):
    valid = 0 if past is None else past[0].size(-2)
    common = (
        hidden,
        mask(hidden.size(1), valid, slots=slots, window=module.sliding_window),
        (rope[0], prepare_gemma4_rope_sin(rope[1], module.attn_options.rope)),
    )
    return (
        common
        if past is None
        else (*common, pad_cache(past, slots) if slots is not None else past)
    )
