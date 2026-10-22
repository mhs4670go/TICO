"""Save disjoint FP and Quant metric and answer files from actual recorded calls."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
report=json.loads((ROOT/'pretrained_smoke_results.json').read_text())
for label in ['fp','quant']:
    output=ROOT/f'pretrained_supplemental_{label}'
    output.mkdir(exist_ok=True)
    value={'model':label,'metrics':report['metrics'][label],'protocol':report['protocol'],'dataset':report['dataset'],'samples':[sample for sample in report['samples'] if sample['model']==label]}
    (output/'actual_results.json').write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    print(output/'actual_results.json')
