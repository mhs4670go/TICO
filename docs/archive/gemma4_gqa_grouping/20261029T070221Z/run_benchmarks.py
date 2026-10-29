#!/usr/bin/env python3
"""Run benchmark workers serially in reverse-order rounds; do not parallelize."""

import argparse
import json
import os
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--python", required=True)
    p.add_argument("--worktrees-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=60)
    a = p.parse_args()
    script = Path(__file__).with_name("numerics_benchmark.py")
    a.output.mkdir(parents=True, exist_ok=True)
    order = [
        ("base", "1"),
        ("a", "1"),
        ("b", "1"),
        ("b", "2"),
        ("a", "2"),
        ("base", "2"),
    ]
    results = []
    for impl, round_id in order:
        worktree = a.worktrees_root / ("gemma4-" + impl)
        env = os.environ.copy()
        env.update(
            PYTHONPATH=str(worktree),
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            PYTHONHASHSEED="0",
        )
        cmd = [
            a.python,
            str(script),
            "--mode",
            "benchmark",
            "--impl",
            impl,
            "--out",
            str(a.output / impl),
            "--base-out",
            str(a.output / "base"),
            "--repeats",
            str(a.repeats),
            "--round",
            round_id,
        ]
        with (a.output / f"{impl}-{round_id}.log").open("w") as log:
            completed = subprocess.run(
                cmd, cwd=worktree, env=env, stdout=log, stderr=subprocess.STDOUT
            )
        results.append(
            dict(impl=impl, round=round_id, exit_code=completed.returncode, command=cmd)
        )
        (a.output / "commands.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(results[-1]), flush=True)
        if completed.returncode:
            raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
