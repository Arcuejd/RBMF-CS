"""Only recompute previously reported 2014 quantities from saved arrays."""
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from run_p0_diagnosis import dump,digest
out=ROOT/'outputs/p0_history_check_v1';out.mkdir(exist_ok=False)
rows=[]
for date in [201401,201404,201407,201410]:
 for budget in [4,12]:
  p=ROOT/f'outputs/comparison_formal_v1/cases/{date}_clean_s000/reconstruction/rbmf/n{budget:02d}.npz'
  with np.load(p) as z:
   e=z['evaluation'];w=z['weights'][e]
   # baseline_c is the physical seasonal baseline plus training centering.
   for name,key in [('original','prediction_c'),('climatology','baseline_c')]:
    err=z[key][e]-z['truth_c'][e]
    rows.append({'date':date,'budget':budget,'method':name,'rmse_c':float(np.sqrt(np.average(err**2,weights=w)))})
dump(out/'checks.json',{'rule':'saved clean minimum seed, all four historical months; no fitting/selection',
 'rows':rows,'means':{f'{name}_{budget}':float(np.mean([r['rmse_c'] for r in rows if r['method']==name and r['budget']==budget])) for name in ['original','climatology'] for budget in [4,12]},
 'original_model_sha256':digest(ROOT/'src/rbmf_cs/model.py'),
 'frozen_model_sha256':digest(ROOT/'outputs/comparison_formal_v1/runtime/src/rbmf_cs/model.py')})
print((out/'checks.json').read_text())
