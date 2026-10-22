"""Read calibration/cache source paths and compare selected functions to baseline."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

REPO = Path('/workspace/TICO')
ROOT = Path(__file__).resolve().parent
BASELINE = 'bb09ca852b0ca1b1a6271cfe1e10c162104c997e'
FUNCTIONS = {
    'tico/quantization/recipes/adapters/qwen3_vl.py': ['Qwen3VLAdapter._disable_cache', 'Qwen3VLAdapter.forward_calibration', 'Qwen3VLAdapter.calibrate_prepared_model', 'Qwen3VLAdapter.build_calibration_inputs', 'Qwen3VLAdapter.build_ptq_config'],
    'tico/quantization/wrapq/wrappers/qwen_vl/quant_text_attention.py': ['QuantQwen3VLTextAttention._normalize_static_past_key_values', 'QuantQwen3VLTextAttention._build_present_key_values', 'QuantQwen3VLTextAttention._update_cache_like'],
    'tico/quantization/recipes/adapters/llama.py': ['LlamaAdapter.calibrate_prepared_model'],
    'tico/quantization/wrapq/wrappers/llama/quant_attention.py': ['QuantLlamaAttention._normalize_past_key_value', 'QuantLlamaAttention._get_layer_kv_from_cache'],
    'tico/quantization/recipes/adapters/gemma4.py': ['Gemma4Adapter._disable_cache', 'Gemma4Adapter.forward_calibration', 'Gemma4Adapter.calibrate_prepared_model'],
    'tico/quantization/wrapq/wrappers/gemma4/quant_text_attention.py': ['QuantGemma4TextAttention._build_present_key_value'],
    'tico/quantization/recipes/stages/ptq.py': ['PTQStage.run'],
}
EXCERPTS = {
    'tico/quantization/recipes/adapters/qwen3_vl.py': [(140,197),(212,235)],
    'tico/quantization/recipes/adapters/llama.py': [(152,164),(253,294)],
    'tico/quantization/recipes/adapters/gemma4.py': [(168,170),(201,217),(315,340)],
    'tico/quantization/recipes/stages/ptq.py': [(209,219)],
    'tico/quantization/recipes/data/vlm.py': [(81,114)],
    'tico/quantization/evaluation/vlm_eval_utils.py': [(1594,1609),(1621,1636),(1709,1715),(1850,1863),(1879,1891)],
    'tico/quantization/wrapq/wrappers/qwen_vl/quant_text_model.py': [(519,534),(618,629)],
    'tico/quantization/wrapq/wrappers/qwen_vl/quant_text_attention.py': [(179,185),(392,406),(518,548),(761,774)],
    'tico/quantization/wrapq/wrappers/llama/quant_attention.py': [(332,337),(480,514),(1034,1037)],
    'tico/quantization/wrapq/wrappers/gemma4/quant_text_attention.py': [(465,498),(788,812)],
    'tico/quantization/wrapq/wrappers/quant_module_base.py': [(104,131),(164,180),(189,194)],
    'tico/quantization/config/builders.py': [(45,52),(322,351),(510,534)],
    'tico/quantization/recipes/export/qwen3_vl.py': [(454,481)],
}

def functions(source):
    """Map qualified function names without importing any product modules."""
    found = {}
    def visit(node, prefix=''):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                key = prefix + child.name
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    found[key] = {'line': child.lineno, 'end_line': child.end_lineno, 'ast': ast.dump(child, include_attributes=False), 'source': ast.get_source_segment(source, child)}
                visit(child, key + '.')
            else:
                visit(child, prefix)
    visit(ast.parse(source))
    return found

comparisons = []
for filename, names in FUNCTIONS.items():
    current_source = (REPO / filename).read_text()
    old_source = subprocess.check_output(['git','show',BASELINE + ':' + filename], cwd=REPO, text=True)
    current = functions(current_source)
    old = functions(old_source)
    for name in names:
        a, b = old.get(name), current.get(name)
        comparisons.append({'file': filename, 'function': name, 'baseline_line': a['line'] if a else None, 'candidate_line': b['line'] if b else None, 'both_present': bool(a and b), 'ast_equal': bool(a and b and a['ast'] == b['ast']), 'source_equal': bool(a and b and a['source'] == b['source']), 'baseline_source_sha256': hashlib.sha256(a['source'].encode()).hexdigest() if a else None, 'candidate_source_sha256': hashlib.sha256(b['source'].encode()).hexdigest() if b else None})
excerpts = []
for filename, ranges in EXCERPTS.items():
    lines = (REPO / filename).read_text().splitlines()
    excerpts.append({'file': filename, 'windows': [{'start': start, 'end': end, 'text': '\n'.join(f'{n}: {lines[n-1]}' for n in range(start, min(end,len(lines))+1))} for start,end in ranges]})
result = {'scope':'Read-only source audit; no model/calibration/export run. Does not prove pretrained numerical quality.', 'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(), 'baseline':BASELINE, 'function_comparisons':comparisons,'source_excerpts':excerpts}
(ROOT/'CALIBRATION_PATH_AUDIT_EVIDENCE.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'functions_checked': len(comparisons),'function_comparisons':comparisons},indent=2))
