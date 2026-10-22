"""Build strict static-check evidence summaries from preserved command logs."""
import collections
import json
import pathlib
import re
import subprocess

root = pathlib.Path(__file__).resolve().parent
repo = pathlib.Path('/workspace/TICO')
baseline = pathlib.Path('/workspace/TICO-export-baseline')
head = '6b2f6a0511458d6a750e2a260d706080a78149dd'
base = 'bb09ca852b0ca1b1a6271cfe1e10c162104c997e'
raw = (root / 'format_all_retry.log').read_text()
plain = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', raw)
(root / 'format_all_retry.plain.log').write_text(plain)
changed = set(subprocess.check_output(['git', 'diff', '--name-only', base, head], cwd=repo, text=True).splitlines())
parts = re.split(r'^>>> Lint for ([^\n]+):\n', plain, flags=re.M)
diagnostics = []
for path, block in zip(parts[1::2], parts[2::2]):
    messages = list(re.finditer(r'^  (Error|Warning|Advice) \(([^)]+)\) ([^\n]+)\n', block, re.M))
    for index, match in enumerate(messages):
        body = block[match.end():messages[index + 1].start() if index + 1 < len(messages) else len(block)]
        line = re.search(r'^\s*>>>\s*(\d+)\s*\|', body, re.M)
        reformat = re.search(r'^\s*(\d+)\s*\|-', body, re.M)
        existing = (baseline / path).exists()
        classification = 'NEW_FILE_DIAGNOSTIC' if not existing else 'PR_CHANGE_BASELINE_COUNTERPART_VERIFIED_PASS'
        diagnostics.append({'path': path, 'tool': match.group(2), 'severity': match.group(1), 'name': match.group(3), 'line': int(line.group(1)) if line else int(reformat.group(1)) if reformat else None, 'description': body.strip().split('\n\n', 1)[0].strip(), 'changed_in_pr': path in changed, 'exists_in_baseline': existing, 'classification': classification})
assert all(item['changed_in_pr'] for item in diagnostics)
assert not any('command-failed' in item['name'] for item in diagnostics)
counts = collections.Counter((item['tool'], item['severity']) for item in diagnostics)
assert counts == {('UFMT', 'Warning'): 19, ('PYLINT', 'Error'): 5, ('MYPY', 'Error'): 59, ('MYPY', 'Advice'): 3}, counts

def command(name):
    return json.loads((root / (name + '.command.json')).read_text())

verified = {}
for name, tool, expected in [('format_baseline_ufmt_counterparts_verified', 'UFMT', 11), ('format_baseline_mypy_counterparts_verified', 'MYPY', 2)]:
    log = (root / (name + '.log')).read_text()
    match = re.search(r"Linter '" + tool + r"' matched files: \[([^\n]*)\]", log)
    files = [item.strip() for item in match.group(1).split(',') if item.strip()] if match else []
    record = command(name)
    assert len(files) == expected and record['exit_code'] == 0
    assert all(item.startswith(str(baseline) + '/') for item in files)
    assert 'Current rev: ' + base in log and 'Running linter ' + tool + ':' in log
    verified[tool] = {'status': 'PASS', 'exit_code': 0, 'matched_file_count': len(files), 'matched_files': files, 'actual_adapter_invocation_verified': True, 'baseline_sha_verified': base, 'log': name + '.log', 'command_record': name + '.command.json'}
(root / 'static_diagnostics.json').write_text(json.dumps(diagnostics, indent=2) + '\n')
ufmt = [{'path': item['path'], 'first_reformat_line': item['line'], 'changed_in_pr': True, 'exists_in_baseline': item['exists_in_baseline'], 'classification': item['classification']} for item in diagnostics if item['tool'] == 'UFMT']
(root / 'ufmt_findings.json').write_text(json.dumps(ufmt, indent=2) + '\n')
full = command('format_all_retry')
assert full['exit_code'] == 1 and not full.get('timeout', False)
summary = {'status': 'FAIL', 'source_head': head, 'baseline': base, 'full_standard_completed': True, 'source_modified': False, 'config_modified': False, 'patches_applied': False, 'full_check': {'status': 'FAIL', 'command': full['command'], 'exit_code': 1, 'elapsed_seconds': full['elapsed_seconds'], 'log': 'format_all_retry.log', 'plain_log': 'format_all_retry.plain.log', 'command_record': 'format_all_retry.command.json'}, 'results_by_tool': {'UFMT': {'status': 'FAIL', 'warnings': 19, 'files': 19}, 'PYLINT': {'status': 'FAIL', 'errors': 5, 'files': 1}, 'MYPY': {'status': 'FAIL', 'errors': 59, 'advisory_notes': 3, 'files': len({item['path'] for item in diagnostics if item['tool'] == 'MYPY'})}}, 'initial_full_check': {'status': 'TIMEOUT_INCOMPLETE', 'exit_code': 124, 'timeout_seconds': 900, 'command_record': 'format_all.command.json'}, 'changed_check': {'status': 'INCORRECT_SELECTION_AND_TIMEOUT_INCOMPLETE', 'exit_code': 124, 'timeout_seconds': 600, 'intended_python_files': 63, 'audited_actual_ufmt_files': 59, 'selection_audit': 'changed_static_selection_audit.json', 'completed_coverage_claim': False, 'superseded_by_completed_all_files_check': True}, 'invalid_baseline_attempts': [{'name': name, 'status': 'INVALID_NO_FILES_CHECKED', 'reported_exit_code': 0, 'excluded_from_pass_evidence': True} for name in ['format_baseline_ufmt_counterparts', 'format_baseline_mypy_counterparts']], 'invalid_selection_evidence': 'baseline_lintrunner_path_probe.log', 'invalid_report_archive': 'STATIC_CHECKS_INVALID_SELECTION_v1.md', 'baseline_verified': verified, 'diagnostics': 'static_diagnostics.json', 'resource_samples': 'static_resource_samples.jsonl', 'unresolved': ['19 UFMT formatting findings,5 Pylint errors,59 mypy errors remain unapplied;3 mypy advice messages are not counted as errors.']}
(root / 'static_checks_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
lines = ['# 정적 검사', '', '**전체 표준 검사 실제 완료: FAIL.** `./ccex format --no-apply-patches`를 원래 설정으로 재실행해835.68초 후 exit1로 완료했다. UFMT19개 formatting 경고, Pylint5개 오류, mypy59개 오류와3개 advisory note가 남았다. 노트3개는 오류 수에 포함하지 않았다. 제품 소스·원래 설정·formatter patch는 변경/적용하지 않았다.', '', '| 검사 | 최종 결과 | 수량 | 근거 |', '|---|---|---:|---|', '| 전체 표준 명령 |FAIL, 실제 완료|exit1 /835.68s|`format_all_retry.log`, `format_all_retry.command.json`|', '| UFMT |FAIL|19파일/19경고|`ufmt_findings.json`|', '| Pylint |FAIL|1파일/5오류|`static_diagnostics.json`|', '| mypy |FAIL|11파일/59오류 +3노트|`static_diagnostics.json`|', '| baseline UFMT 실제 경로 검증 |PASS|11개 원본 counterpart|`format_baseline_ufmt_counterparts_verified.log`|', '| baseline mypy 실제 경로 검증 |PASS|2개 원본 counterpart|`format_baseline_mypy_counterparts_verified.log`|', '', '## 최초 실행과 잘못된 baseline 근거의 철회', '', '- 최초 전체 명령은900초 timeout(exit124), 변경 파일 명령은600초 timeout(exit124)이었다. 원본 로그와 종료 코드를 보존했다.', '- 최초 baseline 명령은 exit0이었지만 verbose 감사에서 `Linting files: []`가 확인됐다. 이 결과는 **NO_FILES_CHECKED로 무효**이며 PASS 근거에서 제외했다. 당시 중간 보고서는 `STATIC_CHECKS_INVALID_SELECTION_v1.md`에 무효 표시로 보존했다.', '- 재실행은 `--paths-from`으로 경로를 명시했다. 두 verified 로그에서 baseline SHA `bb09ca8...`, 각각11개/2개 실제 baseline 절대 경로, 실제 UFMT/mypy adapter 호출, 종료 코드0을 모두 확인했다.', '- 변경 파일600초 명령도 positional 경로 선택을 감사했다. 의도한63개 대신 설정의 origin/main diff가 고른59개를 검사했다. `changed_static_selection_audit.json`에 누락4개를 남겼고 이 실행에 완료/전체 범위 주장을 하지 않는다. 이후 완료된 표준 `--all-files` 실행이 전체 범위를 검사했다.', '', '## Baseline과 PR 구분', '', '모든86개 diagnostic(오류/경고/노트 포함)은 PR diff의 파일에 있다. UFMT19개 중8개는 새 파일이고 기존11개는 verified baseline UFMT에서 통과했다. Pylint5개는 새 테스트 파일에 있다. mypy59개 오류 중57개는 새 파일이며, 기존 파일의2개 오류는 PR에서 추가한 표현식이고 verified baseline mypy counterpart2개가 실제 통과했다. 기존 baseline 실패로 분류한 static diagnostic은 없다.', '', '## 정확한 파일·행', '', '| UFMT 파일 | 첫 reformat 행 | baseline |', '|---|---:|---|']
for item in ufmt:
    lines.append(f"| `{item['path']}` |{item['first_reformat_line']}|" + ('새 파일' if not item['exists_in_baseline'] else '실제 counterpart UFMT PASS') + '|')
lines += ['', 'Pylint5개 오류는 모두 `test/quantization/wrapq/utils/test_static_export_inputs.py`의50,216,251,258,263행 `E1101`이다. `register_buffer()`로 동적으로 등록한 `inv_freq`의 타입/멤버 선언이 부족하다.', '', '기존 파일의 mypy 오류2개는 다음과 같다.', '', '- `test/quantization/recipes/test_export_llama_bundle.py:48`: `Tensor | Module`을 indexing하는 타입 문제.', '- `tico/quantization/recipes/export/llama.py:338`: `qlayer.wrapped`를 `prepare_llama_static_inputs()`에 넘길 때 구체적 decoder 타입으로 narrowing되지 않은 문제.', '', '나머지 mypy 오류·노트의 정확한 파일, 행, 코드, 메시지는 `static_diagnostics.json`과 ANSI를 제거한 `format_all_retry.plain.log`에 모두 남겼다.', '', '## 최소 수정 제안 (미적용)', '', '- 고정 UFMT 도구의19개 파일 patch를 검토·적용한다. 이 변경은 수치 tolerance/동작 수정과 분리한다.', '- 테스트 rotary fixture의 동적 tensor buffer를 명시적으로 선언/타이핑하고, 텐서를 사용하기 전에 구체적인 타입으로 좁힌다. 전체 `no-member` 검사나 타입 검사를 끄지 않는다.', '- exporter에서 이미 런타임으로 검증한 wrapper/tensor를 적절한 구체적 타입으로 좁히고, 새 debug script의 heterogeneous report dict/컨테이너·context manager·instrumentation callable 타입을 명시한다. 기존 exporter에서 사용하던 `cast(QuantLlamaDecoderLayer, qlayer.wrapped)`와 동등한 narrowing이 `:338`의 최소안이다.', '- 검증용 파일 선택은 positional 인자 대신 `--paths-from`을 사용하고, 실제 matched-file 수가0이면 성공으로 보고하지 않도록 검증한다. 원래 source/linter 설정은 이번 작업에서 수정하지 않았다.', '']
(root / 'STATIC_CHECKS.md').write_text('\n'.join(lines))
print(json.dumps({'status': 'FAIL', 'full_standard_completed': True, 'UFMT': 19, 'PYLINT_errors': 5, 'MYPY_errors': 59, 'MYPY_notes': 3, 'baseline_verified_counts': {key: value['matched_file_count'] for key, value in verified.items()}}))
