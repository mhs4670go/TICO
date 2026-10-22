"""Reproduce the actual generation API on unchanged baseline source."""
import hashlib
import json
from pathlib import Path
import sys
import traceback
import torch
import tico
from transformers import AutoModelForCausalLM, AutoTokenizer
from tico.quantization import prepare, convert
from tico.quantization.recipes.adapters.llama import LlamaAdapter
from tico.quantization.recipes.context import RecipeContext
ROOT=Path(__file__).resolve().parent
REPO=Path('/workspace/TICO-export-baseline')
if not Path(tico.__file__).resolve().is_relative_to(REPO):raise RuntimeError('Wrong source import')
torch.manual_seed(42);torch.set_num_threads(1)
model_path=ROOT/'pretrained_assets/Maykeye-TinyLLama-v0'
checkpoint=ROOT/'pretrained_tinyllama_quantized.pt'
report={'repo':str(REPO),'tico_file':tico.__file__,'python':sys.executable,'torch':torch.__version__,'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'scope':'Baseline generation API diagnosis only; fresh fallback PTQ is not old-new export equivalence','attempts':[]}
tokenizer=AutoTokenizer.from_pretrained(model_path,legacy=False,local_files_only=True)
tokenizer.pad_token=tokenizer.eos_token
inputs=tokenizer('The capital of France is',return_tensors='pt')
for mode in ['same_candidate_checkpoint','fresh_baseline_prepare']:
    record={'mode':mode,'status':'ERROR'}
    try:
        if mode=='same_candidate_checkpoint':model=torch.load(checkpoint,map_location='cpu',weights_only=False).eval()
        else:
            fp=AutoModelForCausalLM.from_pretrained(model_path,dtype=torch.float32,local_files_only=True).eval()
            adapter=LlamaAdapter();ctx=RecipeContext(cfg={'model_args':{'profile':'npu_export'},'pipeline':[]},adapter=adapter,model=fp)
            config=adapter.build_ptq_config(ctx,{'profile':'npu_export','activation':'int16','linear_weight':'uint8','embedding_weight':'uint8','lm_head_weight':'uint8','norm_weight':'int16','strict_wrap':True})
            model=prepare(fp,config).eval()
            calibration=tokenizer('The small blue boat crossed the quiet lake. A warm breeze moved through the trees while the passengers watched the birds above the water.',return_tensors='pt').input_ids[:,:32]
            with torch.no_grad():model(calibration,use_cache=False)
            model=convert(model).eval()
            record['calibration_input_ids']=calibration.tolist()
        record['model_class']=type(model).__name__
        with torch.no_grad():output=model.generate(**inputs,max_new_tokens=8,do_sample=False,use_cache=True,pad_token_id=tokenizer.pad_token_id)
        record.update(status='PASS',generated_ids=output.tolist())
    except Exception as exc:
        traceback.print_exc();record.update(status='FAIL',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
    report['attempts'].append(record)
    (ROOT/'baseline_pretrained_generation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
raise SystemExit(1 if any(r['status']!='PASS' for r in report['attempts']) else 0)
