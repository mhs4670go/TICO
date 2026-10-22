"""Summarize fresh unittest console evidence without treating skips as passes."""
import argparse
import json
from pathlib import Path
import re

parser = argparse.ArgumentParser()
parser.add_argument("logs", nargs="+")
args = parser.parse_args()
results = []
for name in args.logs:
    path = Path(name)
    content = path.read_text(errors="replace")
    counts = re.findall(r"Ran (\d+) tests? in ([\d.]+)s", content)
    outcome = re.findall(r"^(OK(?: \(.*\))?|FAILED \(.*\))$", content, re.M)
    count = int(counts[-1][0]) if counts else None
    details = outcome[-1] if outcome else None
    count_fields = dict((key, int(value)) for key, value in re.findall(r"(failures|errors|skipped|expected failures|unexpected successes)=(\d+)", details or ""))
    failed = re.findall(r"^(?:FAIL|ERROR): (.+)$", content, re.M)
    skips = []
    current_test = None
    partial_error_tests = []
    partial_failure_tests = []
    for line in content.splitlines():
        if re.match(r"^test\S* .*\)", line):
            current_test = line.split(" ... ")[0]
        if line.endswith(" ... ERROR"):
            partial_error_tests.append(current_test)
        if line.endswith(" ... FAIL"):
            partial_failure_tests.append(current_test)
        if " ... skipped " in line:
            skips.append({"test": current_test, "reason": line.split(" ... skipped ", 1)[1]})
    cmd_path = path.with_suffix(".command.json")
    command = json.loads(cmd_path.read_text()) if cmd_path.exists() else {}
    result = {
        "log": str(path),
        "status": "INCOMPLETE" if count is None else "FAIL" if details and details.startswith("FAILED") else "COMPLETED_WITH_SKIPS" if count_fields.get("skipped") else "PASS",
        "tests_run": count,
        "partial_console_status_counts": {status: len(re.findall(r"\.\.\. " + re.escape(status) + r"(?: \([^\n]*\))?$", content, re.M)) for status in ["ok", "FAIL", "ERROR"]},
        "elapsed_seconds": float(counts[-1][1]) if counts else None,
        "console_outcome": details,
        "counts": count_fields,
        "failures": failed,
        "partial_error_tests": partial_error_tests,
        "partial_failure_tests": partial_failure_tests,
        "skips": skips,
        "exit_code": command.get("exit_code"),
    }
    if result["exit_code"] not in [0, None] and result["status"] == "PASS":
        result["status"] = "ERROR"
    if count is not None:
        result["passed_excluding_skips_expected_failures"] = count - sum(count_fields.values())
    results.append(result)
print(json.dumps(results, indent=2))
