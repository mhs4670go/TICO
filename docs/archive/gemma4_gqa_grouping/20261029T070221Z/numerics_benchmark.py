#!/usr/bin/env python3
"""Reproduce Gemma4 grouping comparisons; run each worker in its own worktree.

Artifacts/checkpoints live in --out, outside git. Requires CPU Torch and Gemma4
Transformers. Benchmarks must run serially while no other workload is active.
"""

import argparse
import copy
import hashlib
import json
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import tico

import torch
from test.quantization.wrapq.wrappers.gemma4.test_quant_text_attention import (
    _make_text_config,
)
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.config.specs import affine
from tico.quantization.evaluation.metric import compute_peir
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.lifecycle import validate_frozen_model
from tico.quantization.wrapq.observers.ema import EMAObserver
from tico.quantization.wrapq.observers.minmax import MinMaxObserver
from tico.quantization.wrapq.quantizer import PTQQuantizer
from tico.quantization.wrapq.wrappers.gemma4.quant_text_attention import (
    QuantGemma4TextAttention,
)
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextAttention

SEED = 4670
BOUNDARIES = [
    "logits_raw",
    "logits",
    "mask_add",
    "softmax",
    "attn_out",
    "attn_weights",
    "attn_out_h",
]


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def policy(name):
    if name == "default":
        return PTQConfig()
    cls = MinMaxObserver if name == "minmax" else EMAObserver
    return PTQConfig(
        activation=affine(DType.uint(8), observer=cls),
        weight=affine(DType.uint(8), observer=cls),
    )


def make_fp(dim=8, hidden=32, capacity=64):
    torch.manual_seed(SEED)
    cfg = _make_text_config(
        hidden_size=hidden,
        intermediate_size=hidden * 2,
        num_attention_heads=24,
        num_key_value_heads=8,
        num_global_key_value_heads=8,
        head_dim=dim,
        global_head_dim=dim,
        max_position_embeddings=capacity,
        sliding_window=min(capacity, 8),
    )
    return Gemma4TextAttention(cfg, layer_idx=0).eval()


def full_inputs(seed, *, b=1, t=4, s=7, dim=8, hidden=32):
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(b, t, hidden, generator=gen)
    phase = torch.randn(b, t, dim, generator=gen)
    # Synthetic valid RoPE pair; pre-negation belongs to wrapper's configured ABI.
    cos, sin = phase.cos(), phase.sin()
    from tico.quantization.wrapq.wrappers.gemma4.rope import prepare_gemma4_rope_sin

    sin = prepare_gemma4_rope_sin(sin, "pre_negated_sin")
    past = (
        None
        if s == t
        else tuple(torch.randn(b, 8, s - t, dim, generator=gen) for _ in range(2))
    )
    mask = torch.zeros(b, 1, t, s)
    # Preserve append offset, with an actual padding key in addition to causality.
    mask[..., -1] = -120
    return (x, (cos, sin)), dict(
        attention_mask=mask,
        past_key_value=past,
        use_cache=True,
        cache_output_mode="present",
    )


def core_inputs(seed, *, b=1, t=4, s=7, dim=8):
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(b, 24, t, dim, generator=gen)
    k = torch.randn(b, 8, s, dim, generator=gen)
    v = torch.randn(b, 8, s, dim, generator=gen)
    mask = torch.zeros(b, 1, t, s)
    for i in range(t):
        mask[:, :, i, s - t + i + 1 :] = -120
    mask[..., 0] = -120
    return dict(query_states=q, key_states=k, value_states=v, attention_mask=mask)


def snapshots(model):
    result = {}
    for name, obs in model.named_observers():
        snap = obs.require_frozen()
        assert not obs.enabled and obs._frozen, name
        result[name] = dict(
            kind=snap.kind,
            policy=list(snap.policy),
            locked=snap.locked,
            enabled=obs.enabled,
            frozen=obs._frozen,
            fake_quant_enabled=obs.fake_quant_enabled,
            scale=None if snap.scale is None else snap.scale.tolist(),
            zero_point=None if snap.zero_point is None else snap.zero_point.tolist(),
            observed_shape=snap.observed_shape,
            provenance=snap.provenance,
        )
    return result


def tensors(value, prefix=""):
    if isinstance(value, torch.Tensor):
        return {prefix: value.detach().cpu()}
    result = {}
    if isinstance(value, (tuple, list)):
        for i, item in enumerate(value):
            result.update(tensors(item, prefix + "/" + str(i)))
    return result


def compare(reference, output):
    result = {}
    assert reference.keys() == output.keys()
    for key, ref in reference.items():
        value = output[key]
        assert ref.shape == value.shape
        delta = (ref.double() - value.double()).abs()
        result[key] = dict(
            mean_abs=delta.mean().item(),
            max_abs=delta.max().item(),
            relative_l2=(
                (ref.double() - value.double()).norm()
                / ref.double().norm().clamp_min(1e-15)
            ).item(),
            peir_fraction=compute_peir(ref, value),
            peir_percent=compute_peir(ref, value) * 100,
        )
    return result


def collect_trace(model, call):
    trace: dict[str, list[torch.Tensor]] = {}
    orig = model._fq

    def fq(value, observer):
        out = orig(value, observer)
        if observer.name in BOUNDARIES:
            trace.setdefault(observer.name, []).append(out.detach().cpu())
        return out

    model._fq = fq
    pre = []
    handle = model.o_proj.register_forward_pre_hook(
        lambda m, a: pre.append(a[0].detach().cpu())
    )
    try:
        output = call()
    finally:
        del model._fq
        handle.remove()
    result = tensors(output)
    # Each group carries head-major T rows. Flatten batch/head/token identically
    # to the original per-head observer calls, avoiding shape-only divergences.
    for name, values in trace.items():
        if name in ("attn_weights", "attn_out_h"):
            result["trace/" + name] = values[0]
        else:
            result["trace/" + name] = torch.cat(
                [v.reshape(v.shape[0], -1, v.shape[-1]) for v in values], dim=1
            )
    result["o_proj_input"] = pre[0]
    return result


def evaluate(model):
    result = {}
    # Distinct evaluation samples, including T>1 append and B=2.
    for tag, b, t, s in [
        ("decode", 1, 1, 17),
        ("append", 1, 4, 9),
        ("batch2", 2, 3, 7),
    ]:
        c = core_inputs(9000 + t + b, b=b, t=t, s=s)
        f, kw = full_inputs(9500 + t + b, b=b, t=t, s=s)
        for scope, call in [
            ("core", lambda: model._forward_unrolled(**c)),
            ("full", lambda: model(*f, **kw)),
        ]:
            result.update(
                {f"{tag}/{scope}/{k}": v for k, v in collect_trace(model, call).items()}
            )
    return result


def calibrate(fp, cfg):
    quantizer = PTQQuantizer(cfg)
    model = quantizer.prepare(copy.deepcopy(fp)).wrapped.eval()
    counts: dict[str, dict[str, Any]] = {}
    originals = []
    # Record actual collection calls and tensor shapes, then restore before pickle.
    for name, obs in model.named_observers():
        original = obs.collect

        def record(x, *a, _name=name, _original=original, **kw):
            slot = counts.setdefault(_name, dict(calls=0, shapes=[]))
            slot["calls"] += 1
            shape = list(x.shape)
            if shape not in slot["shapes"]:
                slot["shapes"].append(shape)
            return _original(x, *a, **kw)

        obs.collect = record
        originals.append(obs)
    try:
        for i in range(6):
            f, kw = full_inputs(6000 + i, t=4, s=7)
            model(*f, **kw)
            model._forward_unrolled(**core_inputs(6500 + i))
    finally:
        for obs in originals:
            del obs.collect
    model = quantizer.convert(model)
    validate_frozen_model(model)
    return model, counts


def data_digest():
    digest = hashlib.sha256()

    def add(value):
        if isinstance(value, torch.Tensor):
            digest.update(str(tuple(value.shape)).encode())
            digest.update(value.numpy().tobytes())
        elif isinstance(value, dict):
            for key in sorted(value):
                add(value[key])
        elif isinstance(value, (tuple, list)):
            for item in value:
                add(item)

    for i in range(6):
        add(full_inputs(6000 + i, t=4, s=7))
        add(core_inputs(6500 + i))
    calibration = digest.hexdigest()
    digest = hashlib.sha256()
    for b, t, s in [(1, 1, 17), (1, 4, 9), (2, 3, 7)]:
        add(full_inputs(9500 + t + b, b=b, t=t, s=s))
        add(core_inputs(9000 + t + b, b=b, t=t, s=s))
    return dict(calibration_sha256=calibration, evaluation_sha256=digest.hexdigest())


def numerics(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fp = make_fp()
    model = QuantGemma4TextAttention(copy.deepcopy(fp)).eval()
    no_quant = evaluate(model)
    torch.save(no_quant, out / "fp_outputs.pt")
    base = Path(args.base_out or args.out)
    ref_fp = torch.load(base / "fp_outputs.pt", weights_only=True)
    result = {"no_quant_vs_base": compare(ref_fp, no_quant), "policies": {}}
    for name in ["default", "minmax", "ema"]:
        independent, counts = calibrate(fp, policy(name))
        qparams = snapshots(independent)
        indep_outputs = evaluate(independent)
        dump(out / f"{name}_independent_qparams.json", qparams)
        dump(out / f"{name}_calibration_calls.json", counts)
        torch.save(indep_outputs, out / f"{name}_independent_outputs.pt")
        if args.impl == "base":
            torch.save(independent, out / f"{name}_frozen_module.pt")
        # Whole-module checkpoint preserves config, nonpersistent RoPE/mask buffers,
        # observers, capability flags and nonregistered cache links. No recalibration.
        fixed = torch.load(
            base / f"{name}_frozen_module.pt", weights_only=False, map_location="cpu"
        ).eval()
        validate_frozen_model(fixed)
        fixed_before = snapshots(fixed)
        expected = json.loads((base / f"{name}_independent_qparams.json").read_text())
        # Normalize tuple/list representation before exact comparison.
        assert json.loads(json.dumps(fixed_before)) == expected
        fixed_outputs = evaluate(fixed)
        assert snapshots(fixed) == fixed_before, "Frozen evaluation mutated grids/flags"
        dump(out / f"{name}_fixed_qparams.json", fixed_before)
        torch.save(fixed_outputs, out / f"{name}_fixed_outputs.pt")
        base_fixed = torch.load(base / f"{name}_fixed_outputs.pt", weights_only=True)
        base_indep = torch.load(
            base / f"{name}_independent_outputs.pt", weights_only=True
        )
        result["policies"][name] = dict(
            fixed_qparams_identical=True,
            frozen_unchanged_after_evaluation=True,
            observer_count=len(fixed_before),
            fixed_qparam_snapshot_sha256=hashlib.sha256(
                json.dumps(fixed_before, sort_keys=True).encode()
            ).hexdigest(),
            independent_qparam_snapshot_sha256=hashlib.sha256(
                json.dumps(qparams, sort_keys=True).encode()
            ).hexdigest(),
            fixed_vs_base=compare(base_fixed, fixed_outputs),
            independent_vs_base=compare(base_indep, indep_outputs),
            fixed_vs_fp=compare(ref_fp, fixed_outputs),
            independent_vs_fp=compare(ref_fp, indep_outputs),
            qparam_differences_vs_base=[
                k
                for k in expected
                if expected[k]["scale"] != qparams[k]["scale"]
                or expected[k]["zero_point"] != qparams[k]["zero_point"]
            ],
        )
    result["fixture_weights_sha256"] = hashlib.sha256(
        b"".join(
            v.detach().numpy().tobytes()
            for v in fp.state_dict().values()
            if isinstance(v, torch.Tensor)
        )
    ).hexdigest()
    result["dataset_digests"] = data_digest()
    result["environment"] = environment()
    dump(out / "numerics.json", result)
    print(
        json.dumps(
            {"output": str(out / "numerics.json"), "environment": result["environment"]}
        )
    )


def environment():
    return dict(
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        dirty=subprocess.check_output(
            ["git", "status", "--porcelain"], text=True
        ).strip(),
        import_path=tico.__file__,
        cwd=os.getcwd(),
        torch=torch.__version__,
        python=platform.python_version(),
        threads=torch.get_num_threads(),
        interop_threads=torch.get_num_interop_threads(),
        seed=SEED,
        dtype="float32",
        platform=platform.platform(),
    )


def measure(fn, repeats):
    for _ in range(10):
        fn()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        fn()
        samples.append((time.perf_counter_ns() - start) / 1000)
    samples.sort()
    return dict(
        median_us=statistics.median(samples),
        p10_us=samples[int(0.1 * (len(samples) - 1))],
        p90_us=samples[int(0.9 * (len(samples) - 1))],
        min_us=samples[0],
        max_us=samples[-1],
        repeats=repeats,
        warmup=10,
        samples_us=samples,
    )


def benchmark(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fp = make_fp(dim=64, hidden=64, capacity=2200)
    plain = QuantGemma4TextAttention(copy.deepcopy(fp)).eval()
    # Freeze from shared BASE checkpoint generated once for benchmark dimensions.
    checkpoint = Path(args.base_out or args.out) / "benchmark_frozen_module.pt"
    if args.impl == "base" and not checkpoint.exists():
        q = PTQQuantizer(PTQConfig())
        calib = q.prepare(copy.deepcopy(fp)).wrapped.eval()
        for t, s in [(1, 128), (1, 2048), (32, 128), (128, 128)]:
            f, kw = full_inputs(7000 + t + s, t=t, s=s, dim=64, hidden=64)
            calib(*f, **kw)
            calib._forward_unrolled(**core_inputs(7100 + t + s, t=t, s=s, dim=64))
        q.convert(calib)
        torch.save(calib, checkpoint)
    quant = torch.load(checkpoint, weights_only=False, map_location="cpu").eval()
    validate_frozen_model(quant)
    frozen_before = snapshots(quant)
    result = dict(environment=environment(), cases={}, numerical_checks={})
    reference_path = Path(args.base_out or args.out) / "benchmark_reference_outputs.pt"
    reference = (
        None if args.impl == "base" else torch.load(reference_path, weights_only=True)
    )
    outputs = {}
    # The coordinator reverses implementation order in its second round.
    # Case order is shared; setup/calibration/reference checks/I/O are not timed.
    for t, s in [(1, 128), (1, 2048), (32, 128), (128, 128)]:
        c = core_inputs(7100 + t + s, t=t, s=s, dim=64)
        f, kw = full_inputs(7000 + t + s, t=t, s=s, dim=64, hidden=64)
        for mode, model in [("NO_QUANT", plain), ("QUANT", quant)]:
            core = copy.deepcopy(model)
            core.o_proj = torch.nn.Identity()
            label = f"T{t}_S{s}/{mode}"
            outputs[label] = {
                "core": tensors(core._forward_unrolled(**c)),
                "full": tensors(model(*f, **kw)),
            }
            baseline = outputs if reference is None else reference
            fp_label = f"T{t}_S{s}/NO_QUANT"
            result["numerical_checks"][label] = {
                scope: {
                    "vs_base_same_mode": compare(baseline[label][scope], value),
                    "vs_base_fp": compare(baseline[fp_label][scope], value),
                }
                for scope, value in outputs[label].items()
            }
            result["cases"][label] = dict(
                core_projection_excluded=measure(
                    lambda: core._forward_unrolled(**c), args.repeats
                ),
                full_wrapper=measure(lambda: model(*f, **kw), args.repeats),
            )
    if args.impl == "base":
        torch.save(outputs, reference_path)
    assert snapshots(quant) == frozen_before
    result["frozen_qparams_unchanged"] = True
    result["fixture_weights_sha256"] = hashlib.sha256(
        b"".join(
            v.detach().numpy().tobytes()
            for v in fp.state_dict().values()
            if isinstance(v, torch.Tensor)
        )
    ).hexdigest()
    result["benchmark_grid_sha256"] = hashlib.sha256(
        json.dumps(frozen_before, sort_keys=True).encode()
    ).hexdigest()
    dump(out / f"benchmark{args.round}.json", result)
    print(
        json.dumps(
            {
                "output": str(out / f"benchmark{args.round}.json"),
                "environment": result["environment"],
            }
        )
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["numerics", "benchmark"], required=True)
    p.add_argument("--impl", choices=["base", "a", "b"], required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--base-out")
    p.add_argument("--repeats", type=int, default=60)
    p.add_argument("--round", default="")
    a = p.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    assert (
        Path(tico.__file__).resolve().is_relative_to(Path.cwd().resolve())
    ), tico.__file__
    with torch.inference_mode():
        (numerics if a.mode == "numerics" else benchmark)(a)


if __name__ == "__main__":
    main()
