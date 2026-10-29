# Copyright (c) 2026 Samsung Electronics Co., Ltd. All Rights Reserved
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
"""Missing or nonfinite report evidence cannot be represented as successful work."""

import json
import tempfile
import unittest
from pathlib import Path

from tico.quantization.recipes.debug.rope_kv_experiment.report import (
    _paired_selection,
    _read,
    _stats,
    main,
    VARIANTS,
)


class RopeKVReportTest(unittest.TestCase):
    def test_nonfinite_empty_and_missing_evidence_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence.json"
            for contents in ('{"metric": NaN}', '{"metric": Infinity}', ""):
                path.write_text(contents)
                with self.assertRaises(ValueError):
                    _read(path)
            with self.assertRaises(ValueError):
                _read(Path(directory) / "absent.json")
        for values in ([], [float("nan")], [float("inf")]):
            with self.assertRaises(ValueError):
                _stats(values)

    def test_pairing_requires_selected_cells_and_equal_budgets(self):
        command = ["runner", "--profile", "affine", "--seed", "123"]
        records = [
            {"profile": "affine", "seed": 123, "variant": variant, "steps": 128}
            for variant in VARIANTS
        ]
        self.assertEqual(_paired_selection(records, command), (["affine"], [123]))
        with self.assertRaisesRegex(ValueError, "Missing 2x2"):
            _paired_selection(records[:-1], command)
        with self.assertRaisesRegex(ValueError, "selected profile/seed"):
            _paired_selection(records, command + ["--seed", "456"])
        records[0]["steps"] = 32
        with self.assertRaisesRegex(ValueError, "Unequal rollout budgets"):
            _paired_selection(records, command)

    def test_missing_runtime_artifacts_cannot_write_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "input", root / "report"
            source.mkdir()
            (source / "environment.json").write_text("{}")
            (source / "summary.json").write_text(
                json.dumps(
                    {
                        "status": "COMPLETED",
                        "results": [
                            {"profile": "affine", "seed": 123, "variant": "R0K0"}
                        ],
                    }
                )
            )
            self.assertEqual(main(["--input", str(source), "--output", str(output)]), 2)
            self.assertFalse((output / "summary.json").exists())
            error = json.loads((output / "error.json").read_text())
            self.assertEqual(error["status"], "ERROR")
            self.assertIn("Missing or empty evidence", error["message"])


if __name__ == "__main__":
    unittest.main()
