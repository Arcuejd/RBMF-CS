"""One pre-registered joint-factor candidate, finite two-start diagnostics."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,csv,ctypes,gzip,json,sys,time,zipfile
from pathlib import Path
from dataclasses import replace
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from rbmf_cs.model import VBConfig
from rbmf_cs.p0_diagnostics import DiagnosticRBMF
from rbmf_cs.structured_rbmf import StructuredRBMF
from rbmf_cs.p0_isolation import install_development_guard
from rbmf_cs.comparison import fixed_domain,reconstruct
from run_p0_diagnosis import dump,digest,load_development

def memory():
    class Memory(ctypes.Structure):
        _fields_=[('cb',ctypes.c_ulong),('PageFaultCount',ctypes.c_ulong)]+[(k,ctypes.c_size_t) for k in ['PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage']]
    value=Memory();value.cb=ctypes.sizeof(value)
    ctypes.windll.kernel32.GetCurrentProcess.restype=ctypes.c_void_p
    ctypes.windll.psapi.GetProcessMemoryInfo.argtypes=[ctypes.c_void_p,ctypes.POINTER(Memory),ctypes.c_ulong]
    ok=ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(),ctypes.byref(value),value.cb)
    return int(value.PeakWorkingSetSize) if ok else None

def choose(models):
    valid=[(i,m) for i,m in enumerate(models) if m is not None and m.diagnostics_['converged']]
    if not valid:return None,None
    i,m=max(valid,key=lambda pair:(pair[1].diagnostics_['history'][-1]['elbo'],-pair[0]))
    return i,m

def response_summary(m,scale,e,w):
    h=m.h_;obs=float(m.tau_.mean)*(h.T@(m.local_.mean[:,None]*h))
    if isinstance(m,StructuredRBMF):
        p,rhs=m.joint_precision_rhs();cov=np.linalg.inv(p);part=np.r_[float(m.tau_.mean)*(h.T@(m.local_.mean*m.y_)),0.]
        coefficient=(cov@part)[:-1];dof=float(np.trace(cov[:-1,:-1]@obs));cross=float(np.linalg.norm(m.cross_covariance_))
    else:
        p=np.diag(m.gamma_.mean)+float(m.delta_.mean)*m.gram+obs;cov=np.linalg.inv(p)
        coefficient=cov@(float(m.tau_.mean)*(h.T@(m.local_.mean*m.y_)));dof=float(np.trace(cov@obs));cross=0.
    projected=m.basis@coefficient*scale
    return {'effective_dof_frozen':dof,'cross_covariance_norm':cross,'rho':m.rho_mean_,
        'tau':float(m.tau_.mean),'tau_delta':float(m.delta_.mean),'gamma_mean':float(np.mean(m.gamma_.mean)),
        'observation_component_rms_c':float(np.sqrt(np.average(projected[e]**2,weights=w[e]))),
        'elbo':m.diagnostics_['history'][-1]['elbo'],'full_factor_change':m.diagnostics_['history'][-1]['relative_factor_change']}

def run(out,phase):
    # Read numerical gate only; no reference/test fields are opened here.
    ref=ROOT/'outputs/p0_structure_reference_v2/verification.json'
    if not json.loads(ref.read_text())['passed']:raise RuntimeError('numerical reference gate failed')
    gate_hash=digest(ref)
    install_development_guard(ROOT,out);out.mkdir(exist_ok=False);started=time.perf_counter()
    source=ROOT/'outputs/comparison_validation_v1/inputs';data=load_development(source);art=dict(np.load(source/'training_artifacts.npz'))
    selection=json.loads((ROOT/'outputs/comparison_validation_v1/selection.json').read_text());comp=selection['comparison']
    designs=json.loads((ROOT/'outputs/comparison_validation_v1/jobs/rbmf_000/validation_designs.json').read_text())
    psi=art['basis'][:,:32];scale=float(art['scale_c']);e,_=fixed_domain(data['original_grid_flat_ids']);w=np.cos(np.deg2rad(data['coordinates_latlon'][:,0]))
    c=VBConfig(max_iter=3000,tolerance=1e-6,convergence='factors',**selection['inference'])
    alternate=replace(c,initial_gamma=.001,initial_tau_delta=.001,initial_tau=10000.)
    scenarios=['gaussian'] if phase=='pilot' else ['clean','gaussian'];seeds=[1001] if phase=='pilot' else [1001,1002,1003]
    months=[1,7] if phase=='pilot' else list(range(1,13));budgets=[4,12] if phase=='pilot' else [4,6,8,10,12]
    dump(out/'freeze.json',{'phase':phase,'scenarios':scenarios,'months':months,'seeds':seeds,'budgets':budgets,
      'config':vars(c),'alternate_initialization':vars(alternate),'reference_gate_sha256':gate_hash,
      'protocol_sha256':digest(ROOT/'docs/p0_structure_protocol_v1.md'),
      'numerics_amendment_sha256':digest(ROOT/'docs/p0_structure_numerics_amendment_v1.md'),
      'input_sha256':digest(source/'aligned_fields.npz'),'training_sha256':digest(source/'training_artifacts.npz')})
    with zipfile.ZipFile(out/'source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for folder in ['src','scripts','docs','configs','tests']:
            for p in (ROOT/folder).rglob('*'):
                if p.suffix in ['.py','.md','.json'] and '__pycache__' not in p.parts:z.write(p,p.relative_to(ROOT))
    rows=[];failures=[];responses=[];fit_count=0
    def rms(v):return float(np.sqrt(np.average(v[e]**2,weights=w[e])))
    def fit(cls,cfg,low,y,ids,key):
        nonlocal fit_count
        if time.perf_counter()-started>290:raise TimeoutError('290s internal wall limit')
        fit_count+=1
        try:
            model=cls(psi,cfg).fit(low,y,indices=ids)
            if not model.diagnostics_['converged']:failures.append({'key':key,'status':'max_iter','iterations':model.diagnostics_['iterations']})
            return model
        except (ValueError,FloatingPointError,np.linalg.LinAlgError) as exc:
            failures.append({'key':key,'status':'exception','error':repr(exc)});return None
    trace=gzip.open(out/'traces.jsonl.gz','wt');arrays_dir=out/'arrays';arrays_dir.mkdir()
    for scenario in scenarios:
      for design in designs:
        date=design['date'];seed=design['seed']
        if seed not in seeds or date%100 not in months:continue
        truth=data['validation_hf_model'][design['i']];low=data['validation_lf_model'][design['i']]
        y=truth.copy() if scenario=='clean' else np.asarray(design['y']);previous={};arrays={}
        for budget in budgets:
            ids=np.asarray(design['ids'][:budget]);key=f'{scenario}_{seed}_{date}_{budget}'
            models={};chosen={}
            for family,cls in [('mf',DiagnosticRBMF),('joint',StructuredRBMF)]:
                pair=[]
                for init,cfg in [('default',c),('alternate',alternate)]:
                    m=fit(cls,cfg,low,y[ids],ids,key+'_'+family+'_'+init);models[family+'_'+init]=m;pair.append(m)
                    if m is not None:
                        trace.write(json.dumps({'key':key+'_'+family+'_'+init,'diagnostics':m.diagnostics_,
                            'factor_trace':getattr(m,'factor_trace',None),'factors':m.factor_summary()})+'\n')
                number,m=choose(pair);models[family+'_selected']=m;chosen[family]=number
            models['original']=fit(DiagnosticRBMF,replace(c,convergence='elbo'),low,y[ids],ids,key+'_original')
            methods=list(models)+['climatology','pod_ridge','pod_lasso','gaussian_sbl_vb']
            for method in methods:
                m=models.get(method);pred=None;elapsed=None
                if method in models:
                    if m is not None:pred=m.predict();diag=m.diagnostics_;elapsed=diag['runtime_seconds']
                    else:diag={'converged':False,'iterations':0}
                elif method=='climatology':pred=-art['mean_c']/scale;diag={'converged':True,'iterations':0}
                else:
                    pred,m,diag,elapsed=reconstruct(method,art['basis'][:,:comp['method_rank'][method]],low,ids,y[ids],
                        replace(c,convergence='elbo',**comp['method_inference'][method]),comp['regularization'][method])
                row={'scenario':scenario,'seed':seed,'date':date,'budget':budget,'method':method,
                     'converged':bool(diag['converged']),'iterations':diag['iterations'],'rmse_c':None,'mae_c':None,
                     'prediction_change_from_previous_budget_c':None,'vs_climatology_rms_c':None,
                     'selected_start':chosen.get(method.split('_')[0]) if method.endswith('_selected') else None,
                     'fit_seconds':elapsed}
                if pred is not None:
                    error=(pred-truth)*scale;row['rmse_c']=rms(error);row['mae_c']=float(np.average(abs(error[e]),weights=w[e]))
                    row['vs_climatology_rms_c']=rms(pred*scale+art['mean_c'])
                    if method in previous:row['prediction_change_from_previous_budget_c']=rms((pred-previous[method])*scale)
                    previous[method]=pred.copy();arrays[f'{budget}_{method}']=pred
                rows.append(row)
                if method in ['mf_selected','joint_selected','original'] and m is not None:
                    summary={'scenario':scenario,'seed':seed,'date':date,'budget':budget,'method':method,**response_summary(m,scale,e,w)}
                    if budget==12 and scenario=='gaussian' and seed==1001:
                        perturbed=[]
                        for sign in [-1,1]:
                            yp=y[ids].copy();yp[0]+=sign*.02/scale
                            if method=='original':pm=fit(DiagnosticRBMF,replace(c,convergence='elbo'),low,yp,ids,key+'_perturb')
                            else:
                                cls=StructuredRBMF if method=='joint_selected' else DiagnosticRBMF
                                pair=[fit(cls,cfg,low,yp,ids,key+'_perturb') for cfg in [c,alternate]];_,pm=choose(pair)
                            if pm is None or not pm.diagnostics_['converged']:
                                perturbed.append(None)
                            else:perturbed.append(pm.predict())
                        for index,label in [(0,'minus'),(1,'plus')]:
                            summary[label+'_0p02C_response_rms_c']=rms((perturbed[index]-pred)*scale) if perturbed[index] is not None else None
                        summary['central_response_gain']=rms((perturbed[1]-perturbed[0])*scale)/.04 if all(p is not None for p in perturbed) else None
                    responses.append(summary)
            arrays[f'{budget}_ids']=ids;arrays[f'{budget}_observations']=y[ids]
        np.savez_compressed(arrays_dir/f'{scenario}_{seed}_{date}.npz',**arrays)
        dump(out/'failures.json',failures)
        print(scenario,seed,date,'finished',flush=True)
    trace.close()
    with (out/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    dump(out/'responses.json',responses)
    dump(out/'resources.json',{'wall_seconds':time.perf_counter()-started,'fit_calls':fit_count,
        'metric_rows':len(rows),'peak_working_set_bytes':memory(),'output_bytes':sum(p.stat().st_size for p in out.rglob('*') if p.is_file()),
        'failure_count':len(failures),'single_thread':True})
    print('resources',(out/'resources.json').read_text(),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['pilot','full'],required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.output,a.phase)
