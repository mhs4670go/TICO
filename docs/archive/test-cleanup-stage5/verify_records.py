"""Verify archived evidence without importing TICO or executing tests."""

import collections
import hashlib
import json
import math
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "measurements" / "canonical"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def check_stats(actual, values):
    expected = {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }
    for key, value in expected.items():
        assert math.isclose(actual[key], value, rel_tol=1e-12, abs_tol=1e-12)


def main():
    checksums = (ROOT / "SHA256SUMS").read_text().splitlines()
    paths = set()
    for line in checksums:
        expected, relative = line.split("  ", 1)
        path = ROOT / relative
        assert path.is_relative_to(ROOT)
        assert relative not in paths
        paths.add(relative)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, relative
    actual_paths = {
        str(path.relative_to(ROOT))
        for path in ROOT.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    }
    assert paths == actual_paths, "Missing or extra archive files"
    provenance = read(ROOT / "provenance.json")
    for item in provenance["files"]:
        content = (ROOT / item["path"]).read_bytes()
        assert hashlib.sha256(content).hexdigest() == item["archived_sha256"]
        assert len(content) == item["archived_bytes"]
    for path in ROOT.rglob("*.json"):
        read(path)

    summary = read(ROOT / "metadata" / "summary.json")
    for name in ("discovery", "single", "op", "net", "conversion"):
        manifests = {}
        for phase, label in (("baseline", "<BASELINE>"), ("after", "<CHECKOUT>")):
            samples = [read(DATA / f"{phase}-{name}-{i}.json") for i in range(3)]
            assert all(sample["exit"] == 0 for sample in samples)
            assert all(sample["loader_errors"] == [] for sample in samples)
            assert all(
                sample["tico"] == label + "/tico/__init__.py" for sample in samples
            )
            manifest = samples[0]["manifest"]
            assert all(sample["manifest"] == manifest for sample in samples)
            assert not any("_FailedTest" in test[0] for test in manifest)
            assert len({test[0] for test in manifest}) == len(manifest)
            manifests[phase] = manifest
            report = summary["cases"][name][phase]
            for metric in ("collection", "process_wall", "suite"):
                if samples[0][metric] is not None:
                    check_stats(report[metric], [sample[metric] for sample in samples])
            count = read(DATA / f"{phase}-{name}-count.json")
            assert report["counts_discovery"] == count["counts_discovery"]
            assert report["counts_total"] == count["counts_total"]
            assert report["test_count"] == len(manifest)
            if name != "discovery":
                rows = [
                    row
                    for row in read(DATA / f"{phase}-ccex.json")
                    if row["name"] == name
                ]
                assert len(rows) == 3
                assert all(row["exit"] == 0 for row in rows)
                check_stats(report["ccex_wall"], [row["wall"] for row in rows])
                assert all(sample["testsRun"] == len(manifest) for sample in samples)
                assert all(sample["successful"] for sample in samples)
        before, after = manifests["baseline"], manifests["after"]
        before_counts = collections.Counter(map(tuple, before))
        after_counts = collections.Counter(map(tuple, after))
        assert not before_counts - after_counts
        assert before == [test for test in after if tuple(test) in before_counts]
        preserved = summary["preservation"][name]
        assert preserved["original_count"] == len(before)
        assert preserved["after_count"] == len(after)
        assert preserved["added"] == sum((after_counts - before_counts).values())
        assert preserved["removed"] == 0
        assert sum(test[1] for test in before) == preserved["declarative_skips_before"]
        assert sum(test[1] for test in after) == preserved["declarative_skips_after"]
        if name == "discovery":
            assert (len(before), len(after)) == (4996, 5018)
            assert sum(test[1] for test in before) == 48
        else:
            assert before == after

    for phase in ("baseline", "after"):
        preparation = [read(DATA / f"{phase}-preparation-{i}.json") for i in range(3)]
        times = [sum(item["preparation_discovery"].values()) for item in preparation]
        report = summary["preparation_diagnostic"][phase]
        assert math.isclose(report["median_seconds"], statistics.median(times))
        assert report["range_seconds"] == [min(times), max(times)]
        individual = [read(DATA / f"{phase}-individual-{i}.json") for i in range(3)]
        assert all(item["testsRun"] == 1 and item["successful"] for item in individual)
        durations = [item["durations"][0][1] for item in individual]
        check_stats(summary["individual_diagnostic"][phase], durations)

    full = (ROOT / "validation/local/full.log").read_text()
    assert re.findall(r"^Ran (\d+) tests? in", full, re.MULTILINE)[-1] == "5018"
    assert "OK (skipped=49)" in full
    focused = (ROOT / "validation/local/focused.log").read_text()
    assert re.search(r"^Ran 71 tests in", focused, re.MULTILINE)
    assert re.search(r"^OK$", focused, re.MULTILINE)
    fixed = (ROOT / "validation/ci-fix-local/tests.log").read_text()
    assert re.search(r"^Ran 7 tests in", fixed, re.MULTILINE)
    assert re.search(r"^OK$", fixed, re.MULTILINE)
    for name in ("static.log", "ufmt.log"):
        assert (
            "No lint issues." in (ROOT / "validation/ci-fix-local" / name).read_text()
        )

    comparison = read(ROOT / "metadata/ci-local-header-comparison.json")
    assert comparison["local_main_count"] == 5018
    assert comparison["ci_main_count"] == 5013
    assert comparison["exact_full_id_common"] == 4094
    assert len(comparison["namespace_pairs"]) == 916
    assert len(comparison["local_suffix_only"]) == 8
    assert len(comparison["ci_suffix_only"]) == 3
    assert "Ran 5013 tests in" in (ROOT / "ci/initial-full.log").read_text()
    final_ci = read(ROOT / "ci/fixed-jobs-final.json")
    assert all(
        job["status"] == "completed" and job["conclusion"] == "success"
        for job in final_ci["data"]["jobs"]
    )
    assert len(final_ci["data"]["jobs"]) == 9
    final_log = (ROOT / "ci/fixed-full.log").read_text()
    assert "Ran 5013 tests in 1793.892s" in final_log
    assert "OK (skipped=49)" in final_log
    assert "wheel smoke test passed" in final_log
    for path in (ROOT / "scripts").glob("*.py.txt"):
        compile(path.read_text(), path.name, "exec")
    print(
        f"Verified {len(paths)} archive files, {len(provenance['files'])} source "
        "copies, repeated statistics, discovery preservation and recorded outcomes. "
        "No tests or benchmarks were executed."
    )


if __name__ == "__main__":
    main()
