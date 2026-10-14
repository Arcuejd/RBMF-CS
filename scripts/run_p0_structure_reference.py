"""Predefined toy integration, exact no-data calculation, and train fixtures."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import sys,json,gzip,time
from pathlib import Path
from dataclasses import replace
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from rbmf_cs.model import VBConfig
from rbmf_cs.p0_diagnostics import DiagnosticRBMF
from rbmf_cs.structured_rbmf import StructuredRBMF,integrated_no_observation
from rbmf_cs.p0_isolation import install_development_guard
from rbmf_cs.comparison import fixed_domain
from p0_structure_reference import integrate,adaptive_reference
from run_p0_diagnosis import dump,digest,load_development

repaired='--repaired' in sys.argv
out=ROOT/'outputs'/('p0_structure_reference_v2' if repaired else 'p0_structure_reference_v1');install_development_guard(ROOT,out);out.mkdir(exist_ok=False)
start=time.perf_counter();cfg=VBConfig(a_delta=2.,b_delta=20.,nu=3.,convergence='factors',tolerance=1e-6,max_iter=3000)
dump(out/'freeze.json',{'protocol_sha256':digest(ROOT/'docs/p0_structure_protocol_v1.md'),'config':vars(cfg),
    'numerical_amendment_sha256':digest(ROOT/'docs/p0_structure_numerics_amendment_v1.md') if repaired else None})
b=np.ones(32)/np.sqrt(32);orth=np.r_[np.ones(16),-np.ones(16)]/np.sqrt(32)
rows=[];failures=[]
with gzip.open(out/'toy_traces.jsonl.gz','wt') as stream:
 for perpendicular in [0.,4.,8.]:
  low=4*b+perpendicular*orth
  for count in [0,4]:
   ids=np.arange(count);h=b[ids];y=h*4
   c=replace(cfg,robust_likelihood=False)
   integrations=[integrate(b,low,h,y,c,25,nodes=n,unbounded=repaired) for n in [512,1024]]
   err=max(abs(np.asarray(integrations[-1]['moments'])-integrations[-2]['moments']))
   if err>1e-4:
    integrations.append(integrate(b,low,h,y,c,25,nodes=2048,unbounded=repaired))
    err=max(abs(np.asarray(integrations[-1]['moments'])-integrations[-2]['moments']))
   ref=integrations[-1]
   if repaired:
    expanded=adaptive_reference(b,low,h,y,c,25) if perpendicular==0 and count==0 else ref
   else:
    expanded=integrate(b,low,h,y,c,25,nodes=2*ref['nodes'],alpha_limit=32,rho_limit=16)
   # Protocol allows expansion once; do not turn failed convergence into search.
   boundary=max(abs(np.asarray(expanded['moments'])-ref['moments']))
   passed=bool(err<=1e-4 and boundary<=1e-4)
   if not passed:failures.append({'perpendicular':perpendicular,'count':count,'node_change':float(err),'boundary_change':float(boundary)})
   entry={'perpendicular_norm':perpendicular,'observations':count,'quadrature':integrations,'expanded':expanded,
     'reference_passed':passed,'max_node_moment_change':float(err),'max_boundary_moment_change':float(boundary),'fits':[]}
   for family,cls in [('mf',DiagnosticRBMF),('joint',StructuredRBMF)]:
    for init,cc in [('default',c),('alternate',replace(c,initial_gamma=.001,initial_tau_delta=.001,initial_tau=10000.))]:
     m=cls(b[:,None],cc,fixed_tau=25.).fit(low,y,indices=ids)
     elbo=m.diagnostics_['history'][-1]['elbo']
     item={'family':family,'init':init,'mean_alpha':float(m.mean_alpha_[0]),'mean_rho':m.rho_mean_,
        'elbo':elbo,'converged':m.diagnostics_['converged'],'elbo_gap_to_reference':ref['log_evidence']-elbo,
        'coefficient_mean_error_to_reference':abs(float(m.mean_alpha_[0])-ref['mean_alpha']),
        'cross_covariance':float(m.cross_covariance_[0]) if family=='joint' else 0.}
     if elbo>ref['log_evidence']+1e-4:failures.append({'type':'elbo_above_evidence',**item})
     entry['fits'].append(item)
     stream.write(json.dumps({'perpendicular':perpendicular,'count':count,**item,'diagnostics':m.diagnostics_})+'\n')
   rows.append(entry);print('toy',perpendicular,count,'checked',passed,flush=True)
dump(out/'toy_reference.json',rows)

source=ROOT/'outputs/comparison_validation_v1/inputs';d=load_development(source);art=dict(np.load(source/'training_artifacts.npz'))
psi=art['basis'][:,:32];geometry=[]
for date,low in zip(d['validation_dates'],d['validation_lf_model']):
    exact=integrated_no_observation(psi,low,1.,1.)
    geometry.append({'date':int(date),'fixed_gamma':1.,'fixed_delta':1.,**exact})
dump(out/'exact_no_observation_2013_geometry.json',geometry)
train=d['train_hf_model'][0];projected=psi@(psi.T@train);scale=float(art['scale_c'])
e,_=fixed_domain(d['original_grid_flat_ids']);w=np.cos(np.deg2rad(d['coordinates_latlon'][e,0]))
ids=json.loads((ROOT/'outputs/comparison_validation_v1/jobs/rbmf_000/validation_designs.json').read_text())[0]['ids']
training=[]
with gzip.open(out/'training_traces.jsonl.gz','wt') as stream:
 for fixture,truth,low in [('representable',projected,projected),('truncated',train,train),('incompatible',projected,np.roll(projected,173))]:
  for family,cls in [('mf',DiagnosticRBMF),('joint',StructuredRBMF)]:
   for init,c in [('default',cfg),('alternate',replace(cfg,initial_gamma=.001,initial_tau_delta=.001,initial_tau=10000.))]:
    for budget in [4,12]:
     m=cls(psi,c).fit(low,truth[ids[:budget]],indices=ids[:budget]);pred=m.predict()
     r={'fixture':fixture,'family':family,'init':init,'budget':budget,'rmse_c':float(np.sqrt(np.average(((pred-truth)[e]*scale)**2,weights=w))),
        'elbo':m.diagnostics_['history'][-1]['elbo'],'converged':m.diagnostics_['converged'],'rho':m.rho_mean_}
     training.append(r);stream.write(json.dumps({**r,'diagnostics':m.diagnostics_,'factors':m.factor_summary()})+'\n')
dump(out/'training_summary.json',training)
dump(out/'verification.json',{'passed':not failures,'failures':failures,'seconds':time.perf_counter()-start,'test_fields_accessed':False})
print('reference verification',not failures,failures,flush=True)
