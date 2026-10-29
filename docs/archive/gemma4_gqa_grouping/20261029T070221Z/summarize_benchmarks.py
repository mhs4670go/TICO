#!/usr/bin/env python3
"""Pool serial reverse-order benchmark rounds without retaining raw samples."""

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--output", required=True, type=Path, help="Output stem")
    a = p.parse_args()
    result: dict[str, Any] = {
        "timing_unit": "microseconds",
        "threads": 1,
        "method": "Pool 60 timed calls per implementation per round; order BASE,A,B,B,A,BASE; warmup10 before each scope/mode/case in each process. Excludes setup/calibration/reference checks/I/O. Core replaces o_proj by Identity and executes production _forward_unrolled. Full includes projections/cache append/o_proj. CPU only; no inference of NPU speed.",
        "implementations": {},
    }
    rows = []
    weight_hashes: list[str] = []
    grid_hashes: list[str] = []
    for impl in ["base", "a", "b"]:
        rounds = [
            json.loads((a.root / impl / f"benchmark{i}.json").read_text())
            for i in [1, 2]
        ]
        weight_hashes.extend(r["fixture_weights_sha256"] for r in rounds)
        grid_hashes.extend(r["benchmark_grid_sha256"] for r in rounds)
        assert all(r["frozen_qparams_unchanged"] for r in rounds)
        row = {
            "environment": [r["environment"] for r in rounds],
            "cases": {},
            "numerical_checks": rounds[0]["numerical_checks"],
        }
        assert rounds[0]["numerical_checks"] == rounds[1]["numerical_checks"]
        for label in rounds[0]["cases"]:
            row["cases"][label] = {}
            for scope in ["core_projection_excluded", "full_wrapper"]:
                samples = sorted(
                    sum([r["cases"][label][scope]["samples_us"] for r in rounds], [])
                )
                measurement = dict(
                    median_us=statistics.median(samples),
                    p10_us=samples[int(0.1 * (len(samples) - 1))],
                    p90_us=samples[int(0.9 * (len(samples) - 1))],
                    sample_count=len(samples),
                    warmup_per_round=10,
                    round_medians_us=[
                        r["cases"][label][scope]["median_us"] for r in rounds
                    ],
                )
                row["cases"][label][scope] = measurement
                rows.append(
                    dict(implementation=impl, case=label, scope=scope, **measurement)
                )
        result["implementations"][impl] = row
    assert len(set(weight_hashes)) == 1, "Initial weights differ"
    assert len(set(grid_hashes)) == 1, "Frozen qparams differ"
    result["identical_weights_and_frozen_qparams_verified"] = True
    result["weights_sha256"] = weight_hashes[0]
    result["qparam_snapshot_sha256"] = grid_hashes[0]
    for impl, row in result["implementations"].items():
        for case, scopes in row["cases"].items():
            for scope, measurement in scopes.items():
                base = result["implementations"]["base"]["cases"][case][scope]
                measurement["cpu_speed_ratio_base_over_candidate"] = (
                    base["median_us"] / measurement["median_us"]
                )
    # Final-output numerical checks remain small; cache comparisons are retained
    # as actual metrics but raw reference tensors and timing samples are excluded.
    a.output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    with a.output.with_suffix(".csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    lines = [
        "CPU microseconds; pooled median [p10,p90] from120 calls (2 rounds ×60), warmup10 per round. One thread, B1 Hq24/Hkv8 D64, full-wrapper hidden_size64. Core excludes o_proj. No concurrent benchmark workloads.",
        "",
        "|Case/mode|Scope|BASE µs|A µs|B µs|A BASE/A ratio|B BASE/B ratio|",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    b = result["implementations"]["base"]["cases"]
    for case, scopes in b.items():
        for scope in scopes:
            vals = [
                result["implementations"][i]["cases"][case][scope]
                for i in ["base", "a", "b"]
            ]
            cells = [
                f"{v['median_us']:.2f} [{v['p10_us']:.2f},{v['p90_us']:.2f}]"
                for v in vals
            ]
            lines.append(
                f"|{case}|{scope}|"
                + "|".join(cells)
                + f"|{vals[1]['cpu_speed_ratio_base_over_candidate']:.3f}|{vals[2]['cpu_speed_ratio_base_over_candidate']:.3f}|"
            )
    a.output.with_suffix(".md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
