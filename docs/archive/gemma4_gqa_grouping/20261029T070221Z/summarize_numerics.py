#!/usr/bin/env python3
"""Reduce comparison worker results to compact JSON, CSV and Markdown tables."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any

BOUNDARIES = [
    "logits_raw",
    "logits",
    "mask_add",
    "softmax",
    "attn_out",
    "attn_weights",
    "attn_out_h",
]


def short_metric(table, suffix):
    values = {k: v for k, v in table.items() if k.endswith(suffix)}
    return {
        "case_metrics": values,
        "worst_case_max_abs": max(v["max_abs"] for v in values.values()),
        "worst_case_mean_abs": max(v["mean_abs"] for v in values.values()),
        "worst_case_relative_l2": max(v["relative_l2"] for v in values.values()),
        "worst_case_peir_percent": max(v["peir_percent"] for v in values.values()),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument(
        "--output", required=True, type=Path, help="Output stem, without suffix"
    )
    a = p.parse_args()
    result: dict[str, Any] = {
        "peir_definition": "max_abs_error / (reference.max-reference.min), denominator=1 when range=0",
        "peir_units": "fraction and percent (=fraction*100)",
        "summary_aggregation": "Each worst_case field is maximum across decode/append/batch2 x core/full. case_metrics retain actual per-tensor means, maxima, relative L2, PEIR.",
        "implementations": {},
    }
    hashes = []
    datasets = []
    csv_rows = []
    for impl in ["base", "a", "b"]:
        folder = a.root / impl
        x = json.loads((folder / "numerics.json").read_text())
        hashes.append(x["fixture_weights_sha256"])
        datasets.append(x["dataset_digests"])
        row = {
            "environment": x["environment"],
            "fixture_weights_sha256": x["fixture_weights_sha256"],
            "dataset_digests": x["dataset_digests"],
            "no_quant_vs_base": {
                kind: short_metric(x["no_quant_vs_base"], suffix)
                for kind, suffix in [("output", "//0"), ("weights", "//1")]
            },
            "policies": {},
        }
        for kind, metrics in row["no_quant_vs_base"].items():
            for case, m in metrics["case_metrics"].items():
                csv_rows.append(
                    dict(
                        implementation=impl,
                        commit=x["environment"]["commit"],
                        policy="none",
                        experiment="no_quant_vs_base",
                        tensor=kind,
                        case=case,
                        **m,
                    )
                )
        for pol, entry in x["policies"].items():
            d = {k: v for k, v in entry.items() if not isinstance(v, dict)}
            for experiment in [
                "fixed_vs_base",
                "independent_vs_base",
                "fixed_vs_fp",
                "independent_vs_fp",
            ]:
                d[experiment] = {
                    kind: short_metric(entry[experiment], suffix)
                    for kind, suffix in [("output", "//0"), ("weights", "//1")]
                }
                for kind, metrics in d[experiment].items():
                    for case, m in metrics["case_metrics"].items():
                        csv_rows.append(
                            dict(
                                implementation=impl,
                                commit=x["environment"]["commit"],
                                policy=pol,
                                experiment=experiment,
                                tensor=kind,
                                case=case,
                                **m,
                            )
                        )
            first: dict[str, dict[str, str | None]] = {}
            for experiment in ["fixed_vs_base", "independent_vs_base"]:
                first[experiment] = {}
                for case in ["decode", "append", "batch2"]:
                    for scope in ["core", "full"]:
                        first[experiment][case + "/" + scope] = next(
                            (
                                name
                                for name in BOUNDARIES
                                if entry[experiment][
                                    case + "/" + scope + "/trace/" + name
                                ]["max_abs"]
                                > 0
                            ),
                            None,
                        )
            d["first_divergence_boundary"] = first
            for state in ["fixed", "independent"]:
                qs = json.loads((folder / f"{pol}_{state}_qparams.json").read_text())
                d[state + "_boundary_qparams"] = {
                    k: {
                        field: qs[k][field]
                        for field in [
                            "scale",
                            "zero_point",
                            "enabled",
                            "frozen",
                            "fake_quant_enabled",
                        ]
                    }
                    | {"policy": qs[k]["policy"][2:]}
                    for k in BOUNDARIES
                }
                d["observer_class"] = qs["logits"]["policy"][:2]
            calls = json.loads((folder / f"{pol}_calibration_calls.json").read_text())
            d["calibration_calls"] = {k: calls[k] for k in BOUNDARIES}
            row["policies"][pol] = d
        result["implementations"][impl] = row
    assert len(set(hashes)) == 1, "Initial weights differ"
    assert all(
        ds == datasets[0] for ds in datasets
    ), "Calibration/evaluation inputs differ"
    result["identical_initial_weights_verified"] = True
    result["identical_calibration_evaluation_data_verified"] = True
    # Complete per-case metrics live in CSV, so avoid repeating them in JSON.
    for row in result["implementations"].values():
        for value in row["no_quant_vs_base"].values():
            value.pop("case_metrics")
        for d in row["policies"].values():
            for experiment in [
                "fixed_vs_base",
                "independent_vs_base",
                "fixed_vs_fp",
                "independent_vs_fp",
            ]:
                for value in d[experiment].values():
                    value.pop("case_metrics")
    a.output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    with a.output.with_suffix(".csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
        w.writeheader()
        w.writerows(csv_rows)
    lines = [
        "All table values below are worst-case across six cases (decode/append/B2 × core/full). PEIR is percent. Default is UINT8 MinMax; explicit MinMax is repeated as a control. FP32; eval/dropout=0; one CPU thread. Actual per-case mean/max/relative L2 and PEIR are retained in JSON/CSV.",
        "",
        "|Implementation|Experiment/policy|Output max abs vs BASE|Output PEIR % vs BASE|Weights max abs vs BASE|",
        "|---|---|---:|---:|---:|",
    ]
    for impl, row in result["implementations"].items():
        q = row["no_quant_vs_base"]
        lines.append(
            f"|{impl}|NO_QUANT|{q['output']['worst_case_max_abs']:.8g}|{q['output']['worst_case_peir_percent']:.8g}|{q['weights']['worst_case_max_abs']:.8g}|"
        )
        for experiment in ["fixed_vs_base", "independent_vs_base"]:
            for pol, d in row["policies"].items():
                q = d[experiment]
                lines.append(
                    f"|{impl}|{experiment}/{pol}|{q['output']['worst_case_max_abs']:.8g}|{q['output']['worst_case_peir_percent']:.8g}|{q['weights']['worst_case_max_abs']:.8g}|"
                )
    lines.extend(
        [
            "",
            "|Implementation|Policy|Fixed output max abs vs FP|Fixed PEIR % vs FP|Independent output max abs vs FP|Independent PEIR % vs FP|",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for impl, row in result["implementations"].items():
        for pol, d in row["policies"].items():
            f = d["fixed_vs_fp"]["output"]
            i = d["independent_vs_fp"]["output"]
            lines.append(
                f"|{impl}|{pol}|{f['worst_case_max_abs']:.8g}|{f['worst_case_peir_percent']:.8g}|{i['worst_case_max_abs']:.8g}|{i['worst_case_peir_percent']:.8g}|"
            )
    a.output.with_suffix(".md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
