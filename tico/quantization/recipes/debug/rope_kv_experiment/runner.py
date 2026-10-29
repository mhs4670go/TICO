# Copyright (c) 2026 Samsung Electronics Co., Ltd. All Rights Reserved
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
"""Run the bounded fresh-calibration 2x2 and fixed-grid attention replay."""

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .circle import export_and_check, SavedCircle
from .fixture import (
    append,
    call,
    export_args,
    make_fp,
    make_wrapper,
    PROFILES,
    sample,
    VARIANTS,
)
from .metrics import (
    BoundaryTrace,
    observer_records,
    ObserverTrace,
    state_fingerprint,
    tensor_metrics,
)

BASE_SHA = "54411f0ebf4782988193f1b40beab9d82c0f830d"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def calibrate(model, seed, *, samples=3, decode_steps=4, imported=None):
    """Fresh lifecycle; every past read comes from a preceding real prefill."""
    model.enable_calibration()
    with ObserverTrace(model) as trace, torch.no_grad():
        for index in range(samples):
            stream = seed + 10000 + index * 100
            hidden, rope = sample(stream, 8)
            output = call(model, hidden, rope)
            past = append(None, output[2])
            for step in range(decode_steps):
                hidden, rope = sample(stream + step + 1, 1, 8 + step)
                output = call(model, hidden, rope, past)
                past = append(past, output[2])
    if imported is not None:
        # Public pre-freeze import, never mutating an already frozen snapshot.
        for name, observer in model.named_modules():
            record = imported.get(name)
            if record is not None and hasattr(observer, "load_qparams"):
                observer.load_qparams(
                    record.scale.clone(), record.zero_point.clone(), lock=True
                )
    model.freeze_qparams()
    return trace.summary()


def qparams(model):
    return {
        record["path"]: {
            key: record.get(key) for key in ("dtype", "qscheme", "scale", "zero_point")
        }
        for record in observer_records(model)
        if record["frozen"]
    }


def fp_trace(fp, profile, capacity, seed, steps):
    """Fully FP local wrapper has its own FP cache; inputs are fixed hidden traces."""
    model = make_wrapper(fp, profile, capacity)
    rows, past, boundary_values = [], None, []
    for step in range(steps + 1):
        hidden, rope = sample(
            seed + 20000 + step, 8 if step == 0 else 1, 0 if step == 0 else 7 + step
        )
        with BoundaryTrace(model) as trace:
            output = call(model, hidden, rope, past)
        rows.append((hidden, rope, past, output))
        boundary_values.append(trace.values)
        past = append(past, output[2])
    return rows, boundary_values


def check_noquant(fp, profile, capacity, rows):
    """Original HF code independently checks mask, rotation, layout, and deltas."""
    from transformers.cache_utils import DynamicCache

    model = make_wrapper(fp, profile, capacity)
    records = []
    for hidden, rope, past, expected in rows[:5]:
        from .fixture import mask

        cache = DynamicCache(config=fp.config)
        if past is not None:
            cache.update(past[0].clone(), past[1].clone(), 0)
        actual = fp(
            hidden,
            rope,
            mask(hidden.size(1), 0 if past is None else past[0].size(-2)),
            shared_kv_states={},
            past_key_values=cache,
        )
        wrapped = call(model, hidden, rope, past)
        tensors = [
            (expected[0], actual[0]),
            (expected[1], actual[1]),
            (wrapped[0], actual[0]),
        ]
        # HF cache contains the newly generated K/V in its last q_len slots.
        cache_layer: Any = cache.layers[0]
        tensors.extend(
            zip(
                expected[2],
                (
                    cache_layer.keys[..., -hidden.size(1) :, :],
                    cache_layer.values[..., -hidden.size(1) :, :],
                ),
            )
        )
        comparisons = [tensor_metrics(left, right) for left, right in tensors]
        if any(item["max_abs"] > 2e-6 for item in comparisons):
            raise AssertionError(
                "NO_QUANT differs from original HF attention beyond FP32 numerical error."
            )
        records.append(comparisons)
    return {
        "status": "PASS",
        "comparisons": records,
        "atol": 2e-6,
        "reference": "original transformers Gemma4TextAttention with DynamicCache and raw HF sine",
    }


def boundary_metrics(reference, actual):
    result = {}
    for name in sorted(set(reference) & set(actual)):
        # Repeated per-head sites pair corresponding calls, not just last head.
        left, right = reference[name], actual[name]
        if (
            name in ("obs_present_key", "obs_present_value")
            and len(left) == 1
            and len(right) == 3
        ):
            current_length = right[1].size(-2)
            result[name + ".past"] = [
                tensor_metrics(left[0][..., :-current_length, :], right[0])
            ]
            result[name + ".current"] = [
                tensor_metrics(left[0][..., -current_length:, :], right[1])
            ]
            result[name + ".concat"] = [tensor_metrics(left[0], right[2])]
            continue
        if len(left) != len(right):
            raise ValueError(
                f"Unaccounted boundary call counts at {name}: {len(left)} vs {len(right)}"
            )
        values = [tensor_metrics(a, b) for a, b in zip(left, right)]
        if values:
            result[name] = values
    return result


def frozen_replay(fp, profile, capacity, seed, rows, output):
    """Replay common frozen grids and bytes, then split the two KV changes.

    All six policies import grids before conversion from the same FP calibration
    trace, including R0 half/cat and K0 P. Auxiliaries keep R0 fixed and remove
    exactly one K0 route; they do not create additional full-model benchmarks.
    """
    import copy

    from tico.quantization.wrapq.wrappers.gemma4.quant_text_attention import (
        QuantGemma4TextAttention,
    )

    from .fixture import make_qcfg
    from .policy import ExperimentGemma4TextAttention

    class PastPRemoved(ExperimentGemma4TextAttention):
        """Past F_C(p_S); current F_C(F_S(r)); persistent storage F_S(r)."""

        def _cache_to_compute(self, key_value):
            return QuantGemma4TextAttention._cache_to_compute(self, key_value)

    class CurrentSRemoved(ExperimentGemma4TextAttention):
        """Past F_C(F_P(p_S)); current F_C(r); persistent storage F_S(r)."""

        def _build_present_key_value(self, **kwargs):
            return QuantGemma4TextAttention._build_present_key_value(self, **kwargs)

    source = make_wrapper(fp, profile, capacity, "R0K0")
    source_calibration = calibrate(source, seed)
    source_frozen = state_fingerprint(source)
    imported = {
        name: obs.require_frozen()
        for name, obs in source.named_modules()
        if hasattr(obs, "require_frozen")
        and obs.frozen_state is not None
        and obs.frozen_state.scale is not None
    }
    hidden, rope, past, full_fp = rows[1]
    storage_observers = source.cache_storage_observers()
    stored = tuple(obs.fake_quant(value) for obs, value in zip(storage_observers, past))

    def encode(value, observer):
        state = observer.require_frozen()
        dtype = torch.int16 if observer.dtype.signed else torch.uint8
        return (
            torch.round(value / state.scale + state.zero_point)
            .clamp(observer.dtype.qmin, observer.dtype.qmax)
            .to(dtype)
        )

    codes = [encode(value, obs) for obs, value in zip(storage_observers, stored)]
    past_hash = hashlib.sha256(
        b"".join(value.numpy().tobytes() for value in codes)
    ).hexdigest()
    local_fp = make_wrapper(fp, profile, capacity)
    with BoundaryTrace(local_fp) as local_trace:
        local_reference = call(local_fp, hidden, rope, stored)
    results, arrays, deltas, storage_grids = {}, {}, {}, {}
    policy_names = (*VARIANTS, "R0_pastP_removed", "R0_currentS_removed")
    for variant in policy_names:
        if variant in VARIANTS:
            model = make_wrapper(fp, profile, capacity, variant)
        else:
            policy_class = (
                PastPRemoved if variant == "R0_pastP_removed" else CurrentSRemoved
            )
            model = policy_class(
                copy.deepcopy(fp),
                qcfg=make_qcfg(profile, capacity),
                rope_unified=False,
                kv_simplified=False,
            ).eval()
        collection = calibrate(model, seed, imported=imported)
        initial = state_fingerprint(model)
        with BoundaryTrace(model) as trace, ObserverTrace(model) as clipping:
            actual = call(model, hidden, rope, stored)
        storage = model.cache_storage_observers()
        storage_grids[variant] = [
            {
                "dtype": str(obs.dtype),
                "qscheme": obs.qscheme.name,
                "scale": obs.require_frozen().scale.clone(),
                "zero_point": obs.require_frozen().zero_point.clone(),
            }
            for obs in storage
        ]
        deltas[variant] = [encode(value, obs) for obs, value in zip(storage, actual[2])]
        results[variant] = {
            "local_fp_output": tensor_metrics(local_reference[0], actual[0]),
            "full_fp_output": tensor_metrics(full_fp[0], actual[0]),
            "past_S_bytes_sha256": past_hash,
            "qparams": qparams(model),
            "calibration": collection,
            "boundaries_vs_local_fp": boundary_metrics(
                local_trace.values, trace.values
            ),
            "delta_vs_local_fp": {
                key: tensor_metrics(left, right)
                for key, left, right in zip(("K", "V"), local_reference[2], actual[2])
            },
            "clipping": clipping.summary(),
            "frozen_state_sha256": initial,
            "frozen_unchanged": initial == state_fingerprint(model),
        }
        arrays[variant] = actual[0].detach().numpy()
        for key, delta in zip(("K", "V"), deltas[variant]):
            arrays[variant + "_delta_" + key] = delta.numpy()
        if not results[variant]["frozen_unchanged"]:
            raise AssertionError("Frozen replay mutated state.")
    paired_codes: dict[str, Any] = {}
    for index, left in enumerate(policy_names):
        for right in policy_names[index + 1 :]:
            paired_codes[left + "__" + right] = {}
            for slot, key in enumerate(("K", "V")):
                first, second = storage_grids[left][slot], storage_grids[right][slot]
                same_grid = (
                    first["dtype"] == second["dtype"]
                    and first["qscheme"] == second["qscheme"]
                    and torch.equal(first["scale"], second["scale"])
                    and torch.equal(first["zero_point"], second["zero_point"])
                )
                paired_codes[left + "__" + right][key] = {
                    "numeric_S_grid_identical": same_grid,
                    "integer_code_mismatch_rate": (
                        float(
                            (deltas[left][slot] != deltas[right][slot]).double().mean()
                        )
                        if same_grid
                        else None
                    ),
                    "elements": deltas[left][slot].numel(),
                }
    for variant in policy_names:
        if variant == "R0K0":
            continue
        results[variant]["delta_codes_vs_R0K0"] = paired_codes["R0K0__" + variant]
    metric = {
        name: record["local_fp_output"]["mae"] for name, record in results.items()
    }
    base = metric["R0K0"]
    remove_p = metric["R0_pastP_removed"]
    remove_s = metric["R0_currentS_removed"]
    both = metric["R0K1"]
    decomposition = {
        "metric": "attention output MAE versus local FP on identical dequantized S past",
        "baseline": "R0K0: past F_C(F_P(p_S)), current F_C(F_S(r))",
        "remove_past_P_only": "R0_pastP_removed: past F_C(p_S), current F_C(F_S(r))",
        "remove_current_S_only": "R0_currentS_removed: past F_C(F_P(p_S)), current F_C(r)",
        "both_removed": "R0K1: past F_C(p_S), current F_C(r)",
        "past_P_removal_effect_at_serial_current": remove_p - base,
        "current_S_removal_effect_at_serial_past": remove_s - base,
        "past_P_removal_effect_at_direct_current": both - remove_s,
        "current_S_removal_effect_at_direct_past": both - remove_p,
        "combined_effect": both - base,
        "interaction": both - remove_s - remove_p + base,
        "scope": "single attention replay; common weights/qparams/hidden/mask/RoPE and exact S cache bytes",
    }
    if source_frozen != state_fingerprint(source):
        raise AssertionError("Frozen replay grid source mutated.")
    np.savez(
        output / "frozen-replay.npz",
        **arrays,
        past_k=codes[0].numpy(),
        past_v=codes[1].numpy(),
    )
    write(
        output / "frozen-replay.json",
        {
            "reference": "local FP receives identical dequantized S past; full FP uses original FP cache; both receive original FP hidden and RoPE",
            "grid_source": "fresh R0K0 common FP calibration, public load_qparams before freeze; R0 half/cat and K0 P from same trace",
            "source_calibration": source_calibration,
            "source_frozen_state_sha256": source_frozen,
            "source_frozen_unchanged": True,
            "variants": results,
            "contrasts": contrasts(metric),
            "kv_decomposition": decomposition,
            "paired_delta_codes": paired_codes,
        },
    )
    return results


def semantic_graph(record):
    graph = record["graph"]
    result = {
        key: graph[key]
        for key in (
            "operators",
            "operator_counts",
            "quantize_count",
            "inputs",
            "outputs",
            "tensors",
            "concats",
            "quantizes",
        )
    }
    # Normalize only the declared class scope introduced by this adapter. Keep
    # actual input/output names, operator IDs, edges, dtypes and qparams exact.
    return json.loads(
        json.dumps(result).replace(
            "tico.quantization.recipes.debug.rope_kv_experiment.policy.ExperimentGemma4TextAttention::",
            "tico.quantization.wrapq.wrappers.gemma4.quant_text_attention.QuantGemma4TextAttention::",
        )
    )


def run_variant(
    fp,
    profile,
    capacity,
    seed,
    variant,
    rows,
    fp_boundaries,
    output,
    control,
    *,
    circles=True,
):
    model = make_wrapper(fp, profile, capacity, variant)
    collection = calibrate(model, seed)
    write(output / "calibration.json", collection)
    write(output / "observers.json", observer_records(model))
    frozen = state_fingerprint(model)
    records, arrays = [], {}
    past: tuple[torch.Tensor, torch.Tensor] | None = None
    production_match = True
    decode_args = None
    for step, (hidden, rope, fp_past, reference) in enumerate(rows):
        # Controlled replay reuses the fully FP past trace. Each quantized read
        # declares S, while own-cache rollout uses only its own emitted S deltas.
        with BoundaryTrace(model) as trace, ObserverTrace(model) as clipping:
            actual = call(model, hidden, rope, past)
        replay = call(model, hidden, rope, fp_past)
        record = {
            "step": step,
            "context_length": 8 + step,
            "own_cache_output": tensor_metrics(reference[0], actual[0]),
            "fp_cache_replay_output": tensor_metrics(reference[0], replay[0]),
            "cache_delta": {
                key: tensor_metrics(a, b)
                for key, a, b in zip(("K", "V"), reference[2], actual[2])
            },
            "boundaries": boundary_metrics(fp_boundaries[step], trace.values),
            "clipping": clipping.summary(),
        }
        if variant == "R0K1":
            baseline = call(control, hidden, rope, past)
            production_match &= all(
                torch.equal(a, b)
                for a, b in [
                    (actual[0], baseline[0]),
                    (actual[1], baseline[1]),
                    *zip(actual[2], baseline[2]),
                ]
            )
        previous = past
        past = append(past, actual[2])
        if previous is not None and not all(
            torch.equal(new[..., : old.size(-2), :], old)
            for old, new in zip(previous, past)
        ):
            raise AssertionError("Persistent cache valid prefix was rewritten.")
        arrays[f"output_{step}"] = actual[0].detach().numpy()
        records.append(record)
        if step == 1:
            decode_args = export_args(model, hidden, rope, previous, slots=capacity - 1)
    if decode_args is None:
        raise ValueError("No decode example was executed.")
    args = export_args(model, rows[0][0], rows[0][1])
    circles_record = {}
    if circles:
        for mode, operands in (("prefill", args), ("decode", decode_args)):
            circles_record[mode] = export_and_check(
                model.as_export_module(mode).eval(), operands, output, mode
            )
            if not all(
                item["same_serialized_domain"]
                for item in circles_record[mode]["graph"]["concats"]
                if item["region"] != "other"
            ):
                raise AssertionError(
                    "Serialized RoPE/KV concat violates domain contract."
                )
        # Independent production graph uses its own class and own calibration.
        if variant == "R0K1":
            base = {}
            for mode, operands in (("prefill", args), ("decode", decode_args)):
                base[mode] = export_and_check(
                    control.as_export_module(mode).eval(),
                    operands,
                    output / "production-control",
                    mode,
                )
            production_match &= all(
                semantic_graph(base[mode]) == semantic_graph(circles_record[mode])
                for mode in base
            )
        recurrent = recurrent_circle(model, rows, output)
    else:
        recurrent = {"status": "NOT_RUN"}
    if variant == "R0K1":
        production_match &= qparams(model) == qparams(control)
        if not production_match:
            raise AssertionError("R0K1 failed independent production equivalence.")
    preserved = state_fingerprint(model) == frozen
    if not preserved:
        raise AssertionError("Export/evaluation changed frozen source state.")
    np.savez(output / "eager-outputs.npz", **arrays)
    summary = {
        "variant": variant,
        "profile": profile,
        "seed": seed,
        "status": "COMPLETED",
        "validation_status": validation_status(
            circles_record, recurrent, expected_circle=circles
        ),
        "frozen_state_sha256": frozen,
        "frozen_unchanged": preserved,
        "production_control_exact": production_match if variant == "R0K1" else None,
        "mean_decode_mae": float(
            np.mean([row["own_cache_output"]["mae"] for row in records[1:]])
        ),
        "prefill": records[0]["own_cache_output"],
        "steps": records,
        "circles": circles_record,
        "circle_recurrent": recurrent,
        "cached_nll": {
            "status": "ASSET_UNAVAILABLE",
            "reason": "attention fixture has no pretrained language model or vocabulary logits",
        },
    }
    write(output / "result.json", summary)
    return summary


def recurrent_circle(model, rows, output):
    """Actual saved prefill->decode linkage; only Circle's S delta is appended."""
    from tico.circle.runtime import ExecutionMode
    from tico.circle.runtime.quantization import dequantize_array, quantize_array

    prefill, decode = SavedCircle(output / "prefill.circle"), SavedCircle(
        output / "decode.circle"
    )
    try:
        for i, pos in ((1, -2), (2, -1)):
            source = prefill.program.tensor(prefill.program.outputs[i])
            target = decode.program.tensor(decode.program.inputs[pos])
            delta_tensor = decode.program.tensor(decode.program.outputs[i])
            if not (
                source.tensor_type == target.tensor_type == delta_tensor.tensor_type
                and source.quantization
                == target.quantization
                == delta_tensor.quantization
            ):
                raise AssertionError("Saved prefill/decode S ABI mismatch.")
        initial = prefill.run(export_args(model, rows[0][0], rows[0][1]))
        # Fake-quantize runtime exposes virtual S values: encode output ABI once
        # and preserve integer codes through the host cache append operation.
        cache_codes = [
            quantize_array(
                initial[i], prefill.program.tensor(prefill.program.outputs[i])
            )
            for i in (1, 2)
        ]
        records, arrays = [], {}
        for step, (hidden, rope, _, reference) in enumerate(rows[1:], 1):
            past = tuple(
                torch.from_numpy(
                    dequantize_array(
                        value, prefill.program.tensor(prefill.program.outputs[i])
                    )
                )
                for i, value in zip((1, 2), cache_codes)
            )
            args = export_args(model, hidden, rope, past, slots=model.max_seq - 1)
            encoded = decode.encoded_inputs(args)
            # Bind exactly the saved S integer codes, including zp padding.
            for pos in (-2, -1):
                tensor = decode.program.tensor(decode.program.inputs[pos])
                code = cache_codes[pos]
                if tensor.quantization is None:
                    raise AssertionError("Cache S input has no qparams.")
                padding = np.full(
                    encoded[pos].shape,
                    tensor.quantization.zero_point[0],
                    dtype=code.dtype,
                )
                padding[..., : code.shape[-2], :] = code
                encoded[pos] = padding
            result = decode.run_encoded(encoded, mode=ExecutionMode.FAKE_QUANTIZE)
            eager = model.as_export_module("decode")(*args)
            records.append(
                {
                    "step": step,
                    "context_length": past[0].size(-2) + 1,
                    "fp_vs_circle": tensor_metrics(reference[0], result.outputs[0]),
                    "same_circle_cache_eager_vs_circle": tensor_metrics(
                        eager[0], result.outputs[0]
                    ),
                }
            )
            arrays[f"output_{step}"] = result.outputs[0]
            for pos, i in enumerate((1, 2)):
                delta = quantize_array(
                    result.outputs[i], decode.program.tensor(decode.program.outputs[i])
                )
                cache_codes[pos] = np.concatenate((cache_codes[pos], delta), axis=-2)
        np.savez(
            output / "circle-rollout.npz",
            **arrays,
            cache_k=cache_codes[0],
            cache_v=cache_codes[1],
        )
        return {
            "status": "EXECUTED",
            "mode": "fake_quantize",
            "cache_contract": "saved S output ABI encoded to persistent bytes, exact original bytes rebound to decode input S ABI; no C writeback",
            "steps": records,
        }
    except Exception as error:
        from tico.circle.runtime import UnsupportedCircleOperatorError

        return {
            "status": (
                "UNSUPPORTED"
                if isinstance(error, UnsupportedCircleOperatorError)
                else "ERROR"
            ),
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }


def validation_status(circles, recurrent, *, expected_circle=True):
    """Missing execution and unexpected errors can never become PASS."""
    if not expected_circle:
        return "NOT_RUN"
    if (
        set(circles) != {"prefill", "decode"}
        or recurrent.get("status") != "EXECUTED"
        or not recurrent.get("steps")
    ):
        return "ERROR"
    for record in circles.values():
        if record.get("structural_status") != "PASS" or not record.get(
            "frozen_state_preserved"
        ):
            return "ERROR"
        runtime = record.get("runtime", {})
        if runtime.get("fake_quantize", {}).get(
            "status"
        ) != "EXECUTED" or not runtime.get("fake_quantize", {}).get("outputs"):
            return "ERROR"
        if runtime["fake_quantize"].get("strict_fidelity_status") not in (
            "PASS",
            "FAIL",
        ):
            return "ERROR"
        if runtime.get("native", {}).get("status") not in ("EXECUTED", "UNSUPPORTED"):
            return "ERROR"
    if any(
        record["runtime"]["fake_quantize"]["strict_fidelity_status"] != "PASS"
        for record in circles.values()
    ):
        return "FIDELITY_DIFFERENCE"
    return "PASS"


def contrasts(values):
    e00, e10, e01, e11 = (values[key] for key in VARIANTS)
    return {
        "R_at_K0": e10 - e00,
        "R_at_K1": e11 - e01,
        "K_at_R0": e01 - e00,
        "K_at_R1": e11 - e10,
        "interaction": e11 - e10 - e01 + e00,
    }


def source_identity():
    def git(*args):
        return subprocess.check_output(["git", *args], text=True).strip()

    files = subprocess.check_output(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
            "--",
            "*.py",
        ]
    ).split(b"\0")
    digest = hashlib.sha256()
    for raw in sorted(set(files) - {b""}):
        path = Path(raw.decode())
        digest.update(raw + b"\0" + path.read_bytes())
    return {
        "base_sha": BASE_SHA,
        "head": git("rev-parse", "HEAD"),
        "dirty": git("status", "--porcelain"),
        "python_source_sha256": digest.hexdigest(),
        "python": sys.version,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("tico", "torch", "transformers", "numpy", "circle-schema")
        },
        "runtime": "tico.circle.runtime built-in reference; CPU; native and fake_quantize attempted independently",
        "command": sys.argv,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=list(VARIANTS), action="append")
    parser.add_argument("--profile", choices=PROFILES, action="append")
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--no-circle", action="store_true")
    options = parser.parse_args(argv)
    if options.steps < 1:
        parser.error("Positive decode rollout required.")
    options.output.mkdir(parents=True, exist_ok=False)
    write(options.output / "environment.json", source_identity())
    torch.set_num_threads(1)
    all_results = []
    with torch.no_grad():
        for seed in options.seed or (123, 456, 789):
            capacity = 8 + options.steps + 1
            fp, _ = make_fp(123, capacity)
            for profile in options.profile or PROFILES:
                root = options.output / f"{profile}-{seed}"
                root.mkdir()
                write(
                    root / "config.json",
                    {
                        "model": fp.config.to_dict(),
                        "fp_state_sha256": state_fingerprint(fp),
                        "fp_weight_seed": 123,
                        "profile": profile,
                        "seed": seed,
                        "capacity": capacity,
                        "prefill_tokens": 8,
                        "decode_steps": options.steps,
                        "calibration_samples": 3,
                        "calibration_decode_steps": 4,
                        "calibration_tokens": 36,
                        "evaluation_tokens": 8 + options.steps,
                        "calibration_sample_ids": [
                            seed + 10000 + i * 100 + j
                            for i in range(3)
                            for j in range(5)
                        ],
                        "evaluation_sample_ids": [
                            seed + 20000 + i for i in range(options.steps + 1)
                        ],
                        "data_revision": "synthetic-hidden-v1; Gaussian hidden, analytic repeated sine/cosine, no token embedding or vocabulary logits",
                        "rotation_gptq": "none",
                        "quantization": repr(make_wrapper(fp, profile, capacity).qcfg),
                    },
                )
                rows, boundaries = fp_trace(fp, profile, capacity, seed, options.steps)
                write(
                    root / "noquant-reference.json",
                    check_noquant(fp, profile, capacity, rows),
                )
                control = make_wrapper(fp, profile, capacity)
                calibrate(control, seed)
                control_frozen = state_fingerprint(control)
                group = {}
                for variant in options.variant or VARIANTS:
                    print(f"Running {profile} seed={seed} {variant}", flush=True)
                    path = root / variant
                    path.mkdir()
                    try:
                        result = run_variant(
                            fp,
                            profile,
                            capacity,
                            seed,
                            variant,
                            rows,
                            boundaries,
                            path,
                            control,
                            circles=not options.no_circle,
                        )
                    except Exception as error:
                        result = {
                            "variant": variant,
                            "profile": profile,
                            "seed": seed,
                            "status": "ERROR",
                            "type": type(error).__name__,
                            "message": str(error),
                            "traceback": traceback.format_exc(),
                        }
                        write(path / "result.json", result)
                        traceback.print_exc()
                    group[variant] = result
                    all_results.append(result)
                if control_frozen != state_fingerprint(control):
                    raise AssertionError("Independent control mutated.")
                if all(
                    key in group and group[key]["status"] == "COMPLETED"
                    for key in VARIANTS
                ):
                    write(
                        root / "contrasts.json",
                        {
                            "metric": "mean own-cache decode attention output MAE, not NLL",
                            "fresh": contrasts(
                                {key: group[key]["mean_decode_mae"] for key in VARIANTS}
                            ),
                        },
                    )
                frozen_replay(fp, profile, capacity, seed, rows, root)
    completed = all(record["status"] == "COMPLETED" for record in all_results)
    write(
        options.output / "summary.json",
        {
            "status": "COMPLETED" if completed else "ERROR",
            "results": [
                {
                    key: value
                    for key, value in record.items()
                    if key not in ("steps", "circles", "circle_recurrent")
                }
                for record in all_results
            ],
            "runtime_validation": "See per-variant result; COMPLETED means measurements produced, not fidelity PASS",
            "pretrained_quality": "ASSET_UNAVAILABLE",
        },
    )
    return (
        0
        if completed
        and all(
            record.get("validation_status") in ("PASS", "NOT_RUN")
            for record in all_results
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
