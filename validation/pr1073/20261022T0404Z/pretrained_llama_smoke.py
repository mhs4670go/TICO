"""Run a tiny pretrained Llama quality smoke using normal public APIs."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / 'pretrained_assets' / 'Maykeye-TinyLLama-v0'
REVISION = '298338802ab94432b917bcce11382aa151aee50f'
CHECKPOINT = ROOT / 'pretrained_tinyllama_quantized.pt'
CALIBRATION_TEXTS = [
    'The small blue boat crossed the quiet lake. A warm breeze moved through the trees while the passengers watched the birds above the water. The captain smiled and continued toward the village before sunset.',
    'A student opened a book about science and read the first page carefully. She wrote several notes, checked the examples, and then explained the idea to a friend who was learning the same subject.',
    'In the kitchen, the cook mixed flour and water in a large bowl. After waiting for the dough to rise, he shaped a loaf and put it in the oven. The room soon smelled of fresh bread.',
    'The train arrived at the station early in the morning. People carried bags along the platform and looked for their seats. A conductor checked the tickets before the doors closed and the journey began.',
]
PROMPTS = [
    'The capital of France is',
    'A student opened a book about science and read the first page carefully. She wrote several notes, checked the examples, and then explained the idea to a friend who was learning the same subject. The friend asked a question about the experiment and they discussed the answer together. After a short break, they returned to the classroom and continued their work. Their teacher',
]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def download():
    MODEL.mkdir(parents=True, exist_ok=True)
    files = ['config.json', 'generation_config.json', 'model.safetensors', 'special_tokens_map.json', 'tokenizer.json', 'tokenizer.model', 'tokenizer_config.json']
    records = []
    for name in files:
        target = MODEL / name
        if target.exists():
            raise FileExistsError(target)
        url = f'https://huggingface.co/Maykeye/TinyLLama-v0/resolve/{REVISION}/{name}'
        with urllib.request.urlopen(url, timeout=90) as response:
            payload = response.read(32 * 1024 * 1024 + 1)
        if len(payload) > 32 * 1024 * 1024:
            raise RuntimeError('Refusing an unexpectedly large model asset')
        target.write_bytes(payload)
        records.append({'name': name, 'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest(), 'url': url})
    write_json(ROOT / 'pretrained_asset_manifest.json', {'model_id': 'Maykeye/TinyLLama-v0', 'revision': REVISION, 'scope': 'Tiny pretrained 4.62M parameter model; not Llama-3.2', 'files': records})
    print(json.dumps(records, indent=2))


def cfg(output='supplemental'):
    return {
        'model': {'family':'llama', 'name_or_path': str(MODEL), 'revision': REVISION, 'trust_remote_code':False, 'cache_dir':str(ROOT/'hf_cache')},
        'runtime': {'device':'cpu', 'dtype':'float32', 'seed':42, 'show_progress':False},
        'model_args': {'profile':'npu_export'},
        'calibration': {'seq_len':32, 'decode_steps':0},
        'pipeline': [],
        'evaluation': {'enabled':True, 'max_seq_len':128, 'use_cache':True, 'perplexity':{'dataset':'Salesforce/wikitext','dataset_config':'wikitext-2-raw-v1','split':'test[:12]','max_seq_len':64,'stride':32},'selected_tasks':['ppl'], 'output_dir':str(ROOT/output)},
        'export': {'enabled':False, 'max_seq_len':16},
    }


def load_fp():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL, legacy=False, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, local_files_only=True).eval()
    return model, tokenizer


def state(model):
    from tico.quantization.recipes.debug.export_regression import _state_contract
    contract = _state_contract(model)
    contract['profiles'] = {name:{'model_args':repr(module.qcfg.model_args),'attention_options':repr(getattr(module,'attn_options',None)),'return_type':getattr(module,'return_type',None)} for name,module in model.named_modules() if hasattr(module,'qcfg')}
    owner=getattr(model,'wrapped',model)
    contract['model_config']=owner.config.to_dict()
    return contract


def calibrate():
    import torch
    import yaml
    from tico.quantization import prepare, convert
    from tico.quantization.recipes.adapters.llama import LlamaAdapter
    from tico.quantization.recipes.context import RecipeContext
    torch.manual_seed(42); torch.set_num_threads(1)
    model, tokenizer = load_fp()
    adapter=LlamaAdapter(); ctx=RecipeContext(cfg=cfg(), adapter=adapter, model=model, tokenizer=tokenizer)
    stage={'name':'ptq','profile':'npu_export','activation':'int16','linear_weight':'uint8','embedding_weight':'uint8','lm_head_weight':'uint8','norm_weight':'int16','strict_wrap':True}
    config=adapter.build_ptq_config(ctx,stage)
    quantized=prepare(copy.deepcopy(model),config).eval()
    batches=[]
    with torch.no_grad():
        for text in CALIBRATION_TEXTS:
            ids=tokenizer(text,return_tensors='pt').input_ids[:,:32]
            batches.append(ids.tolist()); quantized(ids,use_cache=False)
    quantized=convert(quantized).eval()
    before=state(quantized)
    torch.save(quantized,CHECKPOINT)
    write_json(ROOT/'pretrained_calibration.json',{'model_id':'Maykeye/TinyLLama-v0','revision':REVISION,'checkpoint':str(CHECKPOINT),'checkpoint_sha256':hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),'parameter_dtypes':sorted({str(p.dtype) for p in quantized.parameters()}),'context_capacity':model.config.max_position_embeddings,'quantization_stage':stage,'calibration_inputs':batches,'calibration_texts':CALIBRATION_TEXTS,'note':'Four fixed short synthetic calibration passages, not a full calibration or quality benchmark'})
    write_json(ROOT/'pretrained_frozen_initial.json',before)
    (ROOT/'pretrained_eval.yaml').write_text(yaml.safe_dump(cfg(),sort_keys=False))
    print('Frozen tiny pretrained checkpoint prepared:',CHECKPOINT)


def supplemental():
    import torch
    import torch.nn.functional as F
    import traceback
    from tico.quantization.recipes.adapters.llama import LlamaAdapter
    from tico.quantization.recipes.context import RecipeContext
    from tico.quantization.recipes.evaluation.eager import prepare_eager_evaluation
    from tico.quantization.recipes.evaluation.llm import evaluate_perplexity
    torch.manual_seed(42); torch.set_num_threads(1)
    fp, tokenizer=load_fp()
    quant=torch.load(CHECKPOINT,map_location='cpu',weights_only=False).eval()
    report={'scope':'Tiny 4.62M pretrained Llama, actual FP/Quant eager likelihood and greedy generation; not Llama-3.2','revision':REVISION,'samples':[],'metrics':{},'protocol':{'seed':42,'dtype':'float32','generation':{'do_sample':False,'max_new_tokens':8,'use_cache':True},'ppl':cfg()['evaluation']['perplexity'],'context_cap':128,'fewshot':0,'prompt_mode':'raw','calibration_length':32,'export_length':16},'frozen_state_equal':False,'completed':False}
    initial=state(quant)
    from datasets import load_dataset
    dataset=load_dataset('Salesforce/wikitext','wikitext-2-raw-v1',split='test[:12]',cache_dir=str(ROOT/'hf_cache'))
    texts=dataset['text']
    corpus='\n\n'.join(texts)
    corpus_ids=tokenizer(corpus,return_tensors='pt').input_ids
    report['dataset']={'name':'Salesforce/wikitext','config':'wikitext-2-raw-v1','split':'test[:12]','sample_ids':list(range(len(dataset))),'fingerprint':dataset._fingerprint,'text_sha256':hashlib.sha256(corpus.encode()).hexdigest(),'token_ids_sha256':hashlib.sha256(corpus_ids.numpy().tobytes()).hexdigest(),'token_count':corpus_ids.numel(),'sample_text_sha256':[hashlib.sha256(text.encode()).hexdigest() for text in texts]}
    for label, model in [('fp',fp),('quant',quant)]:
        context=RecipeContext(cfg=cfg('pretrained_supplemental_'+label),adapter=LlamaAdapter(),model=model,tokenizer=tokenizer)
        context.artifacts['pretrained_context_length']=2048
        if label=='quant':context.artifacts['evaluation_checkpoint']=str(CHECKPOINT)
        resolved=prepare_eager_evaluation(context)
        ppl=evaluate_perplexity(model=model,tokenizer=tokenizer,device='cpu',cache_dir=str(ROOT/'hf_cache'),max_seq_len=64,stride=32,dataset_name='Salesforce/wikitext',dataset_config='wikitext-2-raw-v1',split='test[:12]')
        owner=getattr(model,'wrapped',model)
        report['metrics'][label]={'ppl_exact':ppl,'resolved_context':resolved,'recorded_capacity':owner.config.max_position_embeddings,'parameter_dtypes':sorted({str(parameter.dtype) for parameter in model.parameters()}),'seed':42,'source_revision':REVISION,'tokenizer_class':type(tokenizer).__name__,'tokenizer_path':str(MODEL),'dataset_fingerprint':dataset._fingerprint,'sample_ids':list(range(len(dataset))),'corpus_token_ids_sha256':report['dataset']['token_ids_sha256']}
        write_json(ROOT/'pretrained_smoke_results.json',report)
        for index,prompt in enumerate(PROMPTS):
            inputs=tokenizer(prompt,return_tensors='pt')
            ids=inputs.input_ids
            if ids.shape[-1]+8>128:raise RuntimeError('Prompt exceeds selected generation cap')
            with torch.no_grad():
                output=model(**inputs,use_cache=False)
                nll=F.cross_entropy(output.logits[:,:-1].float().reshape(-1,output.logits.shape[-1]),ids[:,1:].reshape(-1),reduction='sum')
            sample={'model':label,'sample_id':index,'prompt':prompt,'input_ids':ids.tolist(),'input_length':ids.shape[-1],'nll':float(nll),'target_tokens':ids.shape[-1]-1}
            try:
                with torch.no_grad():
                    generated=model.generate(**inputs,max_new_tokens=8,do_sample=False,use_cache=True,pad_token_id=tokenizer.pad_token_id)
                sample.update(generation_status='PASS',generated_ids=generated.tolist(),generated_text=tokenizer.batch_decode(generated,skip_special_tokens=True))
            except Exception as exc:
                traceback.print_exc()
                sample.update(generation_status='FAIL',generation_error_type=type(exc).__name__,generation_error=str(exc),generation_traceback=traceback.format_exc())
            report['samples'].append(sample)
            write_json(ROOT/'pretrained_smoke_results.json',report)
        report['metrics'][label]['generation_completed']=all(sample['generation_status']=='PASS' for sample in report['samples'] if sample['model']==label)
    report['completed']=True
    final=state(quant)
    report['frozen_state_equal']=initial==final
    report['status']='PASS' if initial==final and all(report['metrics'][label]['generation_completed'] for label in ['fp','quant']) else 'FAIL'
    report['metrics']['ppl_delta_quant_minus_fp']=report['metrics']['quant']['ppl_exact']-report['metrics']['fp']['ppl_exact']
    write_json(ROOT/'pretrained_frozen_after.json',final)
    write_json(ROOT/'pretrained_smoke_results.json',report)
    if initial!=final:raise AssertionError('Frozen Quant state changed during eager evaluation')
    print(json.dumps(report,indent=2))
    if report['status']!='PASS':raise SystemExit(1)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['download','calibrate','supplemental']);a=parser.parse_args()
    globals()[a.mode]()
