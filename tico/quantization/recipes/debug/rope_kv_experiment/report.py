# Copyright (c) 2026 Samsung Electronics Co., Ltd. All Rights Reserved
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
"""Summarize completed measurements; retain failed/unsupported validation levels."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

VARIANTS = ("R0K0", "R1K0", "R0K1", "R1K1")
METRICS = ("mae", "rmse", "max_abs", "normalized_l2", "cosine")


def _read(path: Path) -> Any:
    def reject(value):
        raise ValueError(f"Nonfinite JSON evidence {value} in {path}.")

    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Missing or empty evidence: {path}.")
    return json.loads(path.read_text(), parse_constant=reject)


def _write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _number(value: Any) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"Expected finite numerical evidence, received {value!r}.")
    return float(value)


def _mean(values) -> float:
    numbers = [_number(value) for value in values]
    if not numbers:
        raise ValueError("Missing numerical measurements.")
    return statistics.mean(numbers)


def _stats(values) -> dict[str, Any]:
    values = [_number(value) for value in values]
    return {
        "mean": _mean(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
        "n": len(values),
        "min": min(values),
        "max": max(values),
    }


def _metrics(rows) -> dict[str, float]:
    rows = list(rows)
    return {key: _mean(row[key] for row in rows) for key in METRICS}


def _contrasts(values: dict[str, float]) -> dict[str, float]:
    e00, e10, e01, e11 = (values[key] for key in VARIANTS)
    return {
        "R_at_K0": e10 - e00,
        "R_at_K1": e11 - e01,
        "K_at_R0": e01 - e00,
        "K_at_R1": e11 - e10,
        "interaction": e11 - e10 - e01 + e00,
    }


def _runtime(record: dict[str, Any], steps: int) -> dict[str, Any]:
    circuits = record["circles"]
    if set(circuits) != {"prefill", "decode"}:
        raise ValueError("Missing saved prefill/decode measurements.")
    result: dict[str, Any] = {}
    for phase, circle in circuits.items():
        if (
            circle["structural_status"] != "PASS"
            or not circle["frozen_state_preserved"]
        ):
            raise ValueError("Circle structural/frozen-state validation failed.")
        for concat in circle["graph"]["concats"]:
            if concat["region"] != "other" and not concat["same_serialized_domain"]:
                raise ValueError("Serialized concat domains differ.")
        modes = circle["runtime"]
        fake = modes["fake_quantize"]
        if fake["status"] != "EXECUTED" or not fake.get("outputs"):
            raise ValueError("FAKE_QUANTIZE has no executed numerical evidence.")
        for output in fake["outputs"]:
            _metrics([output])
        if modes["native"]["status"] not in ("UNSUPPORTED", "EXECUTED"):
            raise ValueError("NATIVE runtime failed unexpectedly.")
        result[phase] = {
            "fake_quantize": fake["strict_fidelity_status"],
            "fake_quantize_max_abs": max(
                output["max_abs"] for output in fake["outputs"]
            ),
            "fake_quantize_attention": {
                key: fake["outputs"][0][key] for key in METRICS
            },
            "native": modes["native"]["status"],
        }
    recurrent = record["circle_recurrent"]
    if recurrent["status"] != "EXECUTED" or len(recurrent.get("steps", [])) != steps:
        raise ValueError("Saved Circle recurrent execution is missing or incomplete.")
    result["recurrent"] = {
        "status": "EXECUTED",
        "mode": recurrent["mode"],
        "steps": steps,
        "same_circle_cache_eager_vs_circle": _metrics(
            row["same_circle_cache_eager_vs_circle"] for row in recurrent["steps"]
        ),
        "max_abs_over_steps": max(
            row["same_circle_cache_eager_vs_circle"]["max_abs"]
            for row in recurrent["steps"]
        ),
        "fp_vs_circle": _metrics(row["fp_vs_circle"] for row in recurrent["steps"]),
    }
    return result


def _summarize_record(record: dict[str, Any], path: Path) -> dict[str, Any]:
    if record["status"] != "COMPLETED" or not record["frozen_unchanged"]:
        raise ValueError(f"Incomplete or mutated variant: {path}.")
    steps = record["steps"]
    if len(steps) < 2 or [row["step"] for row in steps] != list(range(len(steps))):
        raise ValueError("Missing or misordered rollout steps.")
    if record["variant"] == "R0K1" and record["production_control_exact"] is not True:
        raise ValueError("Independent production control did not match.")
    for filename in (
        "calibration.json",
        "observers.json",
        "prefill.circle",
        "decode.circle",
        "eager-outputs.npz",
        "circle-rollout.npz",
        "prefill.fake_quantize.npz",
        "decode.fake_quantize.npz",
    ):
        file = path / filename
        if not file.is_file() or file.stat().st_size == 0:
            raise ValueError(f"Missing nonempty artifact {file}.")
    observers = _read(path / "observers.json")
    calibrated = _read(path / "calibration.json")
    storage = []
    for phase, circle in record["circles"].items():
        if _hash(path / f"{phase}.circle") != circle["sha256"]:
            raise ValueError("Saved Circle hash differs from measured file.")
        for index in (1, 2):
            output = circle["graph"]["outputs"][index]
            storage.append(
                {"phase": phase, "tensor": "K" if index == 1 else "V", **output}
            )
    boundary_names = sorted({key for row in steps for key in row["boundaries"]})
    boundaries = {}
    for name in boundary_names:
        boundaries[name] = {
            phase: _metrics(
                metric for row in rows for metric in row["boundaries"].get(name, [])
            )
            for phase, rows in (("prefill", steps[:1]), ("decode", steps[1:]))
            if any(name in row["boundaries"] for row in rows)
        }
    clipping = {}
    for name in sorted({key for row in steps for key in row["clipping"]}):
        values = [row["clipping"][name] for row in steps if name in row["clipping"]]
        elements = sum(row["quantized_elements"] for row in values)
        if elements:
            clipping[name] = {
                "elements": elements,
                "clipping_rate": sum(row["clipped_elements"] for row in values)
                / elements,
                "endpoint_occupancy": sum(row["endpoint_elements"] for row in values)
                / elements,
            }
    return {
        "profile": record["profile"],
        "seed": record["seed"],
        "variant": record["variant"],
        "steps": len(steps) - 1,
        "context_range": [steps[0]["context_length"], steps[-1]["context_length"]],
        "prefill": _metrics([steps[0]["own_cache_output"]]),
        "decode_own_cache": _metrics(row["own_cache_output"] for row in steps[1:]),
        "decode_fp_cache_replay": _metrics(
            row["fp_cache_replay_output"] for row in steps[1:]
        ),
        "checkpoints": {
            str(row["step"]): {
                "context_length": row["context_length"],
                "own": row["own_cache_output"],
                "replay": row["fp_cache_replay_output"],
            }
            for row in steps
            if row["step"] in (1, 32, 128, len(steps) - 1)
        },
        "cache_delta": {
            key: _metrics(row["cache_delta"][key] for row in steps)
            for key in ("K", "V")
        },
        "boundaries": boundaries,
        "clipping": clipping,
        "runtime": _runtime(record, len(steps) - 1),
        "quantize": {
            phase: {
                "total": value["graph"]["quantize_count"],
                "regions": value["graph"]["quantize_by_region"],
            }
            for phase, value in record["circles"].items()
        },
        "storage": storage,
        "frozen_state_sha256": record["frozen_state_sha256"],
        "observer_file_sha256": _hash(path / "observers.json"),
        "observer_fingerprints": {
            value["path"]: value["state_sha256"]
            for value in observers
            if value["frozen"]
        },
        "calibration_collection": {
            name: {
                key: value[key]
                for key in (
                    "collect_calls",
                    "effective_collect_calls",
                    "collect_elements",
                    "collect_min",
                    "collect_max",
                )
            }
            for name, value in calibrated.items()
        },
        "cached_nll": record["cached_nll"],
    }


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for profile in sorted({record["profile"] for record in records}):
        result[profile] = {}
        for variant in VARIANTS:
            group = [
                record
                for record in records
                if record["profile"] == profile and record["variant"] == variant
            ]
            result[profile][variant] = {
                phase: {
                    metric: _stats(record[phase][metric] for record in group)
                    for metric in METRICS
                }
                for phase in ("prefill", "decode_own_cache", "decode_fp_cache_replay")
            }
            result[profile][variant]["checkpoints"] = {
                step: {
                    mode: {
                        metric: _stats(
                            record["checkpoints"][step][mode][metric]
                            for record in group
                        )
                        for metric in METRICS
                    }
                    for mode in ("own", "replay")
                }
                for step in group[0]["checkpoints"]
            }
    return result


def _paired_selection(records: list[dict[str, Any]], command: list[str]):
    """Require every selected profile/seed and all four equally budgeted cells."""
    profiles = sorted({record["profile"] for record in records})
    seeds = sorted({record["seed"] for record in records})
    selected_profiles = sorted(
        command[index + 1]
        for index, value in enumerate(command[:-1])
        if value == "--profile"
    ) or ["affine", "u8s16c"]
    selected_seeds = sorted(
        int(command[index + 1])
        for index, value in enumerate(command[:-1])
        if value == "--seed"
    ) or [123, 456, 789]
    if profiles != selected_profiles or seeds != selected_seeds:
        raise ValueError(
            "Missing selected profile/seed evidence from recorded command."
        )
    identities = {
        (record["profile"], record["seed"], record["variant"]) for record in records
    }
    if len(identities) != len(records) or identities != {
        (profile, seed, variant)
        for profile in profiles
        for seed in seeds
        for variant in VARIANTS
    }:
        raise ValueError("Missing 2x2 variant/profile/seed combination.")
    if len({record["steps"] for record in records}) != 1:
        raise ValueError("Unequal rollout budgets.")
    return profiles, seeds


def build_summary(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Require full paired 2x2 evidence before calculating effects."""
    environment = _read(root / "environment.json")
    overall = _read(root / "summary.json")
    if overall["status"] != "COMPLETED" or not overall.get("results"):
        raise ValueError("Experiment summary is incomplete.")
    records, groups, frozen, decompositions = [], {}, [], []
    for shallow in overall["results"]:
        identity = (shallow["profile"], shallow["seed"], shallow["variant"])
        if identity in groups:
            raise ValueError(f"Duplicate variant identity {identity}.")
        path = root / f"{identity[0]}-{identity[1]}" / identity[2]
        raw = _read(path / "result.json")
        if identity != (raw["profile"], raw["seed"], raw["variant"]):
            raise ValueError("Summary/detail variant identity mismatch.")
        record = _summarize_record(raw, path)
        groups[identity] = record
        records.append(record)
    profiles, seeds = _paired_selection(records, environment["command"])
    effects = []
    common_grids = []
    configurations = []
    for profile in profiles:
        for seed in seeds:
            group_root = root / f"{profile}-{seed}"
            config = _read(group_root / "config.json")
            configurations.append(
                {
                    "profile": profile,
                    "seed": seed,
                    "fp_state_sha256": config["fp_state_sha256"],
                    "config_file_sha256": _hash(group_root / "config.json"),
                    "calibration_tokens": config["calibration_tokens"],
                    "evaluation_tokens": config["evaluation_tokens"],
                    "fp_weight_seed": config["fp_weight_seed"],
                }
            )
            if _read(group_root / "noquant-reference.json")["status"] != "PASS":
                raise ValueError("Independent NO_QUANT reference did not pass.")
            values = _read(group_root / "frozen-replay.json")
            if not set(VARIANTS).issubset(values["variants"]):
                raise ValueError("Frozen replay misses variant evidence.")
            if values.get("source_frozen_unchanged") is False:
                raise ValueError("Frozen replay source changed during evaluation.")
            if not (group_root / "frozen-replay.npz").is_file():
                raise ValueError("Missing frozen replay tensor evidence.")
            hashes = {
                value["past_S_bytes_sha256"] for value in values["variants"].values()
            }
            if len(hashes) != 1 or not all(
                value["frozen_unchanged"] for value in values["variants"].values()
            ):
                raise ValueError(
                    "Frozen replay cache bytes differ or frozen state mutated."
                )
            frozen.append(
                {
                    "profile": profile,
                    "seed": seed,
                    "reference": values["reference"],
                    "grid_source": values["grid_source"],
                    "past_S_bytes_sha256": hashes.pop(),
                    "variants": {
                        variant: {
                            key: value[key]
                            for key in ("local_fp_output", "full_fp_output")
                        }
                        for variant, value in values["variants"].items()
                    },
                    "contrasts": _contrasts(
                        {
                            key: value["local_fp_output"]["mae"]
                            for key, value in values["variants"].items()
                            if key in VARIANTS
                        }
                    ),
                }
            )
            effects.append(
                {
                    "profile": profile,
                    "seed": seed,
                    "metric": "mean_decode_own_cache_mae",
                    "effects": _contrasts(
                        {
                            variant: groups[(profile, seed, variant)][
                                "decode_own_cache"
                            ]["mae"]
                            for variant in VARIANTS
                        }
                    ),
                }
            )
            qparams = {
                variant: {
                    row["path"]: {
                        key: row[key]
                        for key in ("dtype", "qscheme", "scale", "zero_point")
                    }
                    for row in _read(group_root / variant / "observers.json")
                    if row["frozen"]
                }
                for variant in VARIANTS
            }
            common = set.intersection(*(set(value) for value in qparams.values()))
            different = sorted(
                name
                for name in common
                if any(
                    qparams[variant][name] != qparams["R0K0"][name]
                    for variant in VARIANTS
                )
            )
            common_grids.append(
                {
                    "profile": profile,
                    "seed": seed,
                    "common_boundaries": len(common),
                    "numerically_different_boundaries": different,
                    "all_common_numeric_qparams_equal": not different,
                    "storage_numeric_qparams_equal": all(
                        qparams[variant][name] == qparams["R0K0"][name]
                        for variant in VARIANTS
                        for name in ("obs_new_k", "obs_new_v")
                    ),
                }
            )
            decomposition = values.get("kv_decomposition", {"status": "NOT_RUN"})
            if "kv_decomposition" in values:
                decomposition = {
                    "status": "MEASURED",
                    **decomposition,
                    "intermediate_outputs": {
                        key: value["local_fp_output"]
                        for key, value in values["variants"].items()
                        if key not in VARIANTS
                    },
                }
            decompositions.append(
                {"profile": profile, "seed": seed, "evidence": decomposition}
            )
    if len({row["fp_state_sha256"] for row in configurations}) != 1:
        raise ValueError(
            "FP weights/state differ across the paired experimental groups."
        )
    aggregate_effects = {
        profile: {
            key: _stats(
                value["effects"][key]
                for value in effects
                if value["profile"] == profile
            )
            for key in effects[0]["effects"]
        }
        for profile in profiles
    }
    manifest = {
        path.relative_to(root).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": _hash(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    detail_keys = (
        "boundaries",
        "clipping",
        "observer_fingerprints",
        "calibration_collection",
    )
    detail = {
        "fresh": [
            {
                "profile": record["profile"],
                "seed": record["seed"],
                "variant": record["variant"],
                **{key: record.pop(key) for key in detail_keys},
            }
            for record in records
        ]
    }
    return (
        {
            "status": "MEASURED",
            "source": environment,
            "selected_profiles": profiles,
            "selected_seeds": seeds,
            "decode_steps": records[0]["steps"],
            "replication": {
                "seed_count": len(seeds),
                "population": "one fixed seed-123 random-weight Gemma4 attention fixture; independent calibration/evaluation data seeds; no language-model hidden feedback",
                "statistical_claim": "descriptive paired seed results; no significance claim",
            },
            "pretrained_cached_nll_ppl": {
                "status": "ASSET_UNAVAILABLE",
                "nll": None,
                "ppl": None,
                "reason": "No pretrained language model and held-out text assets; random attention fixtures do not measure language quality.",
            },
            "fresh": {
                "aggregate": _aggregate(records),
                "records": records,
                "common_qparams": common_grids,
                "configurations": configurations,
                "paired_effects": effects,
                "aggregate_effects": aggregate_effects,
            },
            "frozen_replay": frozen,
            "kv_decomposition": decompositions,
            "artifact_manifest_entries": len(manifest),
        },
        manifest,
        detail,
    )


def _format(value: dict[str, Any]) -> str:
    mean = f"{value['mean']:.6g}"
    return mean + (
        f" ± {value['sample_sd']:.3g}" if value["sample_sd"] is not None else " (n=1)"
    )


def render_markdown(summary: dict[str, Any]) -> str:
    """Keep numerical quality, exporter fidelity, and unsupported backends distinct."""
    source = summary["source"]
    records = summary["fresh"]["records"]
    lines = [
        "**Gemma4 RoPE/KV domain 2×2 measurements**",
        "",
        f"Base `{source['base_sha']}`; measured HEAD `{source['head']}`; Python source SHA256 `{source['python_source_sha256']}`.",
        "",
        f"Profiles: {', '.join(summary['selected_profiles'])}; seeds: {summary['selected_seeds']}; 8-token prefill + {summary['decode_steps']} fixed-hidden decode steps. Values below are means ± sample standard deviation across seeds. These are random-initialized attention fixtures, not pretrained language-model accuracy.",
        "",
        "Fresh calibration runs each variant independently from common FP weights and disjoint fixed calibration/evaluation PRNG streams. All use three calibration prefills and four real-cache decode calibration reads per prefill. Compute/storage format equality does not imply equal numerical qparams. Full qparams, statistics, clipping, endpoint occupancy, and per-step boundary metrics remain in the hashed artifacts.",
        "",
        "| Profile | Variant | Prefill MAE | Own-cache decode MAE | FP-cache replay MAE | Decode RMSE | Mean max error | Mean normalized L2 | Mean cosine |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for profile, variants in summary["fresh"]["aggregate"].items():
        for variant, value in variants.items():
            own = value["decode_own_cache"]
            lines.append(
                f"| {profile} | {variant} | {_format(value['prefill']['mae'])} | {_format(own['mae'])} | {_format(value['decode_fp_cache_replay']['mae'])} | {_format(own['rmse'])} | {_format(own['max_abs'])} | {_format(own['normalized_l2'])} | {_format(own['cosine'])} |"
            )
    lines += [
        "",
        "Own-cache rollout appends only each variant's emitted S delta. Replay reinjects the common FP past values at every step and declares each variant's S input domain. Both compare against the original FP wrapper/cache trace; the fixed-qparam table below instead uses a local FP reference receiving the same dequantized S past.",
        "",
        "| Profile | Variant | Decode step | Own-cache MAE | Replay MAE |",
        "|---|---|---:|---:|---:|",
    ]
    for profile, variants in summary["fresh"]["aggregate"].items():
        for variant, value in variants.items():
            for step, metric in value["checkpoints"].items():
                lines.append(
                    f"| {profile} | {variant} | {step} | {_format(metric['own']['mae'])} | {_format(metric['replay']['mae'])} |"
                )
    lines += [
        "",
        "| Profile | Variant | Quantize prefill/decode | Regions decode: RoPE / past / current / other | S dtype; K+V delta bytes | FAKE single prefill/decode | Recurrent max error | NATIVE |",
        "|---|---|---|---|---|---|---:|---|",
    ]
    for profile in summary["selected_profiles"]:
        for variant in VARIANTS:
            group = [
                record
                for record in records
                if record["profile"] == profile and record["variant"] == variant
            ]
            counts = sorted(
                {
                    (
                        record["quantize"]["prefill"]["total"],
                        record["quantize"]["decode"]["total"],
                    )
                    for record in group
                }
            )
            regions = sorted(
                {
                    tuple(
                        record["quantize"]["decode"]["regions"][name]
                        for name in ("rope", "past_read", "current_compute", "other")
                    )
                    for record in group
                }
            )
            storage = group[0]["storage"]
            delta = [value for value in storage if value["phase"] == "decode"]
            dtype = "/".join(sorted({value["dtype"] for value in delta}))
            fidelity = sorted(
                {
                    f"{record['runtime']['prefill']['fake_quantize']}/{record['runtime']['decode']['fake_quantize']}"
                    for record in group
                }
            )
            native = sorted({record["runtime"]["decode"]["native"] for record in group})
            maximum = max(
                record["runtime"]["recurrent"]["max_abs_over_steps"] for record in group
            )
            lines.append(
                f"| {profile} | {variant} | {counts} | {regions} | {dtype}; {sum(value['bytes'] for value in delta)} | {', '.join(fidelity)} | {maximum:.6g} | {', '.join(native)} |"
            )
    lines += [
        "",
        "Graph counts come from saved/reloaded Circle bytes. Region counts use transparent paths adjacent to serialized RoPE/KV concats; the current region includes a storage Quantize when it lies on that path. NATIVE quantized arithmetic is backend-defined and unsupported by the reference runtime. FAKE_QUANTIZE executes dequantized arithmetic with serialized grids; it does not establish bit-exact NPU behavior. Node-count reductions are not latency measurements.",
        "",
        "| Profile | R effect at K0 | R effect at K1 | K effect at R0 | K effect at R1 | Interaction |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for profile, value in summary["fresh"]["aggregate_effects"].items():
        lines.append(
            f"| {profile} | "
            + " | ".join(
                _format(value[key])
                for key in ("R_at_K0", "R_at_K1", "K_at_R0", "K_at_R1", "interaction")
            )
            + " |"
        )
    lines += [
        "",
        "Effects use paired mean decode MAE differences, with negative values indicating lower error. Interaction is E11−E10−E01+E00. K changes both past P removal and current S bypass; these main-table effects do not isolate the two causes.",
        "",
        "| Fixed common qparams: profile | Variant | Local FP output MAE | Fully FP output MAE |",
        "|---|---|---:|---:|",
    ]
    for profile in summary["selected_profiles"]:
        group = [
            value for value in summary["frozen_replay"] if value["profile"] == profile
        ]
        for variant in VARIANTS:
            lines.append(
                f"| {profile} | {variant} | {_format(_stats(value['variants'][variant]['local_fp_output']['mae'] for value in group))} | {_format(_stats(value['variants'][variant]['full_fp_output']['mae'] for value in group))} |"
            )
    decomposition_status = sorted(
        {
            value["evidence"].get("status", "MEASURED")
            for value in summary["kv_decomposition"]
        }
    )
    lines += [
        "",
        "| Frozen KV decomposition | Past P removal at serial current | Current S removal at serial past | Both removed | Interaction |",
        "|---|---:|---:|---:|---:|",
    ]
    for profile in summary["selected_profiles"]:
        measured = [
            value["evidence"]
            for value in summary["kv_decomposition"]
            if value["profile"] == profile
            and value["evidence"].get("status") == "MEASURED"
        ]
        if measured:
            keys = (
                "past_P_removal_effect_at_serial_current",
                "current_S_removal_effect_at_serial_past",
                "combined_effect",
                "interaction",
            )
            lines.append(
                f"| {profile} | "
                + " | ".join(
                    _format(_stats(value[key] for value in measured)) for key in keys
                )
                + " |"
            )
    common_equal = all(
        value["all_common_numeric_qparams_equal"]
        for value in summary["fresh"]["common_qparams"]
    )
    storage_equal = all(
        value["storage_numeric_qparams_equal"]
        for value in summary["fresh"]["common_qparams"]
    )
    lines += [
        "",
        f"KV auxiliary decomposition status: {', '.join(decomposition_status)}. These are common-qparam single-step differences, not fresh-calibration model NLL. Common fresh boundary dtype/qscheme/scale/zero-point equality across all variants: {common_equal}; numeric K/V S equality: {storage_equal}. Dedicated R0 half/cat and K0 P are excluded from the common-boundary intersection.",
        "",
        "Pretrained cached NLL/PPL: **ASSET_UNAVAILABLE**. No random-init PPL is reported. Native integer/NPU execution and latency are unverified. Descriptive seed variation provides no significance claim.",
        "",
        f"Reproduce measurements using the recorded command in `summary.json` source metadata; regenerate this report with `python -m tico.quantization.recipes.debug.rope_kv_experiment.report --input <artifacts> --output <new-report-dir>`. `artifact-manifest.json` records {summary['artifact_manifest_entries']} source artifact sizes/hashes. Raw logs, Circle binaries and tensor arrays stay outside Git.",
        "",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args(argv)
    options.output.mkdir(parents=True, exist_ok=False)
    try:
        summary, manifest, detail = build_summary(options.input)
        _write(options.output / "detail.json", detail)
        _write(options.output / "artifact-manifest.json", manifest)
        summary["detail_sha256"] = _hash(options.output / "detail.json")
        summary["artifact_manifest_sha256"] = _hash(
            options.output / "artifact-manifest.json"
        )
        (options.output / "summary.json").write_text(
            json.dumps(summary, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        )
        (options.output / "report.md").write_text(render_markdown(summary))
        return 0
    except (ValueError, KeyError, TypeError, OSError) as error:
        _write(
            options.output / "error.json",
            {"status": "ERROR", "type": type(error).__name__, "message": str(error)},
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
