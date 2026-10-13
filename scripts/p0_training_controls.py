"""Constructed train-only controls and saved historical checks; no new test fits."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import sys,json,gzip
from pathlib import Path
from dataclasses import replace
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from rbmf_cs.model import VBConfig
from rbmf_cs.p0_diagnostics import DiagnosticRBMF,precision_diagnostics
from rbmf_cs.comparison import fixed_domain
from run_p0_diagnosis import load_development,dump

out=ROOT/'outputs/p0_training_controls_v1';out.mkdir(exist_ok=False)
d=load_development(ROOT/'outputs/comparison_validation_v1/inputs');a=dict(np.load(ROOT/'outputs/comparison_validation_v1/inputs/training_artifacts.npz'))
psi=a['basis'][:,:32];scale=float(a['scale_c']);e,candidates=fixed_domain(d['original_grid_flat_ids']);w=np.cos(np.deg2rad(d['coordinates_latlon'][:,0]))
cfg=VBConfig(a_delta=2,b_delta=20,nu=3,max_iter=3000,tolerance=1e-6,convergence='factors')
design=json.loads((ROOT/'outputs/comparison_validation_v1/jobs/rbmf_000/validation_designs.json').read_text())[0]
ids=np.asarray(design['ids']);train=d['train_hf_model'][0];projected=psi@(psi.T@train)
# Both constructed low fields are training-derived diagnostic fixtures, NOT real LF.
fixtures={'representable_exact_LF':(projected,projected),
          'truncated_exact_LF':(train,train),
          'representable_incompatible_LF':(projected,np.roll(projected,173))}
records=[]
for fixture,(truth,low) in fixtures.items():
 for name,c in [('original_strict',cfg),('init_low_regularization',replace(cfg,initial_gamma=.001,initial_tau_delta=.001,initial_tau=10000.))]:
  previous=None
  for budget in [4,12]:
   model=DiagnosticRBMF(psi,c).fit(low,truth[ids[:budget]],indices=ids[:budget])
   record={'fixture':fixture,'configuration':name,'budget':budget,'factors':model.factor_summary(),
    'diagnostics':model.diagnostics_,'precision':precision_diagnostics(model,scale,e,w),
    'rmse_c':float(np.sqrt(np.average(((model.predict()-truth)*scale)[e]**2,weights=w[e]))),
    'climatology_rmse_c':float(np.sqrt(np.average((truth*scale)[e]**2,weights=w[e]))),
    'factor_trace':model.factor_trace}
   if previous is not None:record['additional_observation_rms_c']=float(np.sqrt(np.average(((model.predict()-previous)*scale)[e]**2,weights=w[e])))
   previous=model.predict();records.append(record)
with gzip.open(out/'constructed_diagnosis.json.gz','wt') as f:json.dump(records,f)
dump(out/'summary.json',[{k:v for k,v in r.items() if k not in ['factor_trace','diagnostics','precision']} for r in records])
print([(r['fixture'],r['configuration'],r['budget'],r['rmse_c']) for r in records])
