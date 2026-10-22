"""Create a separately identified cache-calibrated tiny pretrained export fixture."""
import hashlib
import json
from pathlib import Path
import torch
from tico.quantization import prepare, convert
from tico.quantization.recipes.adapters.llama import LlamaAdapter
from tico.quantization.recipes.context import RecipeContext
from pretrained_llama_smoke import ROOT, REVISION, CALIBRATION_TEXTS, cfg, load_fp, state, write_json

torch.manual_seed(42);torch.set_num_threads(1)
checkpoint=ROOT/'pretrained_tinyllama_quantized_cached.pt'
if checkpoint.exists():raise FileExistsError(checkpoint)
model,tokenizer=load_fp()
adapter=LlamaAdapter();context=RecipeContext(cfg=cfg(),adapter=adapter,model=model,tokenizer=tokenizer)
stage={'name':'ptq','profile':'npu_export','activation':'int16','linear_weight':'uint8','embedding_weight':'uint8','lm_head_weight':'uint8','norm_weight':'int16','strict_wrap':True}
quantized=prepare(model,adapter.build_ptq_config(context,stage)).eval()
calibration_inputs=[]
with torch.no_grad():
    for text in CALIBRATION_TEXTS:
        ids=tokenizer(text,return_tensors='pt').input_ids[:,:32]
        if ids.shape[-1]!=32:raise ValueError('Each cache calibration passage must supply32tokens')
        calibration_inputs.append(ids.tolist())
        output=quantized(input_ids=ids[:,:28],use_cache=True,return_dict=True)
        cache=output.past_key_values
        for index in range(28,32):
            output=quantized(input_ids=ids[:,index:index+1],attention_mask=torch.ones((1,index+1),dtype=torch.long),past_key_values=cache,use_cache=True,return_dict=True)
            cache=output.past_key_values
quantized=convert(quantized).eval()
torch.save(quantized,checkpoint)
record={'checkpoint':str(checkpoint),'sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'model_id':'Maykeye/TinyLLama-v0','model_revision':REVISION,'tico_file':__import__('tico').__file__,'root_class':type(quantized).__name__,'wrapped_class':type(quantized.wrapped).__name__,'recorded_capacity':quantized.wrapped.config.max_position_embeddings,'parameter_dtypes':sorted({str(p.dtype) for p in quantized.parameters()}),'quantization_stage':stage,'calibration_prefill_tokens':28,'teacher_forced_decode_steps':4,'calibration_total_tokens':32,'calibration_inputs':calibration_inputs,'scope':'Separate new tiny pretrained cache-calibrated export fixture; not the original checkpoint and not an evaluation score replacement','original_checkpoint_unchanged':str(ROOT/'pretrained_tinyllama_quantized.pt')}
write_json(ROOT/'pretrained_cached_checkpoint_identity.json',record)
write_json(ROOT/'pretrained_cached_frozen_initial.json',state(quantized))
print(json.dumps(record,indent=2))
