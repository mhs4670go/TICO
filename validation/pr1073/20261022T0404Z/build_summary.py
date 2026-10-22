"""Build an evidence index without promoting incomplete checks to PASS."""
import argparse, datetime, json, pathlib
p=argparse.ArgumentParser(); p.add_argument('--complete',action='store_true');a=p.parse_args()
r=pathlib.Path(__file__).resolve().parent

def read(path):
 try: return json.loads(path.read_text())
 except (FileNotFoundError,json.JSONDecodeError): return None

commands=[]
for path in sorted(r.glob('*.command.json')):
 d=read(path)
 if d is not None:
  commands.append({'name':d.get('name'),'command_record':path.name,'log':path.name.replace('.command.json','.log'),'exit_code':d.get('exit_code'),'elapsed_seconds':d.get('elapsed_seconds'),'timeout':d.get('timeout',False)})

runs=[]
for path in sorted(r.rglob('summary.json')):
 if path.parent==r or '.mypy_cache' in path.parts or 'mypy_cache' in path.parts: continue
 d=read(path)
 if not isinstance(d,dict): continue
 if not any(k in d for k in ['baseline_vs_candidate','candidate_before_vs_after_eager','comparisons_complete','baseline_comparison_performed']): continue
 item={'path':str(path.relative_to(r))}
 for key in ['status','mode','comparisons_complete','baseline_comparison_performed','atol','rtol','require_torch_parity','torch_parity_required','requested_scope']:
  if key in d:item[key]=d[key]
 for key,value in d.items():
  if isinstance(value,dict) and 'stages' in value:
   item[key]={'ok':value.get('ok'),'stage_count':len(value['stages']),'failures':value.get('failures',[])}
 runs.append(item)

indexes={}
for name in ['model_results.json','unit_results.json','static_checks_summary.json','pretrained_summary.json','pretrained_smoke_results.json','pretrained_protocol_comparison.json','pretrained_checkpoint_identity.json','pretrained_cached_checkpoint_identity.json','pretrained_observer_coverage.json','pretrained_cached_stage_differences.json','pretrained_selected_stage_differences.json','image_generation_analysis.json','llama_long_eager_analysis.json','blocked_resources.json','comparator_extra_negatives.json','unit_failure_classification.json','gemma_text_analysis.json','GEMMA_TEXT_PARITY_DIAGNOSIS.json','GEMMA_FULL_TEXT_DIAGNOSTIC.json','final_repository_environment.json']:
 d=read(r/name)
 if d is not None:indexes[name]=d

circle_index=read(r/'circle_validation_summary.json')
if circle_index is not None:indexes['circle_validation_summary.json']=circle_index

summary={'target':{'repository':'mhs4670go/TICO','branch':'ea','pr':'https://github.com/Samsung/TICO/pull/1073','head':'6b2f6a0511458d6a750e2a260d706080a78149dd','pre_pr_baseline':'bb09ca852b0ca1b1a6271cfe1e10c162104c997e'},'updated_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'validation_execution_complete':a.complete,'overall_status':'FAIL' if a.complete else 'IN_PROGRESS_WITH_FAILURES','pass_policy':'Only completed requested checks count as PASS. Unit/mock, tiny real-HF, tiny pretrained, Torch Export and actual Circle levels remain distinct; strict parity failures are retained.','product_code_modified':False,'committed_pushed_merged_or_pr_metadata_changed':False,'environment':'environment.json','commands_script':'commands.sh','report':'PR1073_VALIDATION.md','commands':commands,'circle_runs':runs,'evidence_indexes':indexes,'additional_evidence':{'tiny_rope':'llama-rope-checks.json','tiny_eager_frozen':'supplemental_eager_state/summary.json','generated_png_eager':'generated_image_eager/summary.json','llama_host_hf_and_same_external_inputs':'tiny-llama-old-new-v2/llama_rope_diagnosis.json','llama_bundle_evidence_inventory':'tiny-llama-old-new-v2/independent_evidence_inventory.json','llama_bundle_component_results':'tiny-llama-old-new-v2/component_summary.json','long_full_circle_inventory':'tiny-llama-native16384-long-eager-full/independent_full_evidence.json','baseline_parity_diagnosis':'PARITY_DIAGNOSIS.json'},'reports':sorted(x.name for x in r.glob('*.md'))}
summary['completion_note']='Task execution finished within resource/time budgets; this does not mark incomplete or blocked checks as completed or PASS.' if a.complete else 'Execution remains in progress.'
summary['artifact_manifest']='artifact_manifest.json'
summary['model_results']='model_results.json'
summary['unresolved_issues']=[
 {'status':'FAIL','scope':'unit_regression','evidence':'unit_test_analysis.md','details':'Owning quantization suite 2670 tests, 9 errors, 141 skips; 8 deterministic static-contract errors and one resource-sensitive 45-second child timeout.'},
 {'status':'FAIL','scope':'lint_type','evidence':'STATIC_CHECKS.md','details':'Completed standard command: 19 UFMT files, 5 Pylint errors, 59 mypy errors plus 3 advisory notes.'},
 {'status':'FAIL','scope':'normal_quant_generation','evidence':'PRETRAINED_SMOKE.md','details':'Llama and Qwen experts accessor API missing; actual baseline reproductions establish preexisting compatibility failure.'},
 {'status':'FAIL','scope':'strict_torch_circle_parity','evidence':'CIRCLE_VALIDATION.md','details':'Llama, Qwen per-layer and Gemma text strict numerical failures remain; old/new or before/after component equality does not override raw verdicts.'},
 {'status':'ERROR','scope':'gemma_vision_circle','evidence':'gemma-vision-failure-artifact/','details':'Actual rank-2 RMS_NORM rejected by reference runtime in both checkouts.'},
 {'status':'FAIL','scope':'comparator_evidence_negative_tests','evidence':'COMPARATOR_GAPS.md','details':'Both-side missing Torch/parity evidence or input evidence can incorrectly pass original comparator; completed actual artifacts were independently audited.'},
 {'status':'INCOMPLETE','scope':'full_standard_test','evidence':'full_unit_suite.command.json','details':'1800-second timeout, no final test-run summary.'},
 {'status':'INCOMPLETE','scope':'tiny_pretrained_full_bundle','evidence':'pretrained_cached_circle_isolation_16.command.json','details':'1200-second timeout, 9 before-eager stages and no after-eager stage; selected two-stage check reported separately.'},
 {'status':'BLOCKED','scope':'large_pretrained_2048_and_actual_VLM_video_evaluation','evidence':'blocked_resources.json','details':'No matching trusted frozen full-module checkpoints or benchmark/video assets; no multi-GB model download was performed.'}
]
(r/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
print(json.dumps({'overall_status':summary['overall_status'],'completed_command_records':len(commands),'export_runs':len(runs),'file':str(r/'summary.json')}))
