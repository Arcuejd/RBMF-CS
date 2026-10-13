"""Small, training/2013-only P0 experiment. Never reads sealed test data."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']: os.environ[key]='1'
import argparse, csv, gzip, hashlib, json, shutil, sys, time, zipfile
from pathlib import Path
from dataclasses import replace
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from rbmf_cs.model import VBConfig
from rbmf_cs.p0_diagnostics import DiagnosticRBMF, precision_diagnostics
from rbmf_cs.comparison import fixed_domain, reconstruct
from rbmf_cs.p0_isolation import install_development_guard

def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while b:=f.read(1024*1024): h.update(b)
    return h.hexdigest()

def dump(p,v):
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2,default=lambda x: x.tolist() if isinstance(x,np.ndarray) else x.item()),encoding='utf-8')

def load_development(directory):
    allowed=['coordinates_latlon','original_grid_flat_ids','train_dates','validation_dates',
             'train_hf_model','train_hf_insitu_c','train_hf_anomaly_c','validation_hf_model',
             'validation_hf_insitu_c','validation_lf_model','validation_ice_fraction']
    with np.load(directory/'aligned_fields.npz') as z:
        if any(k.startswith('test_') and z[k].size for k in z.files):
            raise ValueError('development input contains test arrays')
        d={k:z[k] for k in allowed}
    if not np.array_equal(d['train_dates'],[y*100+m for y in range(2000,2013) for m in range(1,13)]): raise ValueError('training dates')
    if not np.array_equal(d['validation_dates'],np.arange(201301,201313)): raise ValueError('validation dates')
    return d

def run(out,months,seed=1001,scenario='gaussian'):
    install_development_guard(ROOT,out)
    out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    source=ROOT/'outputs/comparison_validation_v1/inputs'
    (out/'inputs').mkdir()
    d=load_development(source)
    np.savez_compressed(out/'inputs/aligned_fields.npz',**d)
    shutil.copy2(source/'training_artifacts.npz',out/'inputs/training_artifacts.npz')
    art=dict(np.load(out/'inputs/training_artifacts.npz'))
    psi,scale=art['basis'],float(art['scale_c'])
    e,_=fixed_domain(d['original_grid_flat_ids']); w=np.cos(np.deg2rad(d['coordinates_latlon'][:,0]))
    def rms(x): return float(np.sqrt(np.average(x[e]**2,weights=w[e])))
    def mae(x): return float(np.average(abs(x[e]),weights=w[e]))
    selection=json.loads((ROOT/'outputs/comparison_validation_v1/selection.json').read_text())
    designs=json.loads((ROOT/'outputs/comparison_validation_v1/jobs/rbmf_000/validation_designs.json').read_text())
    cfg=VBConfig(max_iter=3000,tolerance=1e-6,**selection['inference'])
    strict=replace(cfg,convergence='factors')
    specs={
      'original':(cfg,None), 'factor_stop':(strict,None),
      'init_high_tau':(replace(strict,initial_tau=10000.),None),
      'ard_b1':(replace(strict,b_gamma=1.),None),
      'no_lf':(replace(strict,multifidelity=False),None),
      'known_noise_conditional':(strict,scale**2/(.02**2*(cfg.nu-2)/cfg.nu))}
    if scenario=='clean': specs['noise_floor_conditional']=specs.pop('known_noise_conditional')
    dump(out/'plan.json',{'months':months,'budgets':[4,6,8,10,12],'seed':seed,'scenario':scenario,
        'candidates':{k:{'config':vars(c),'fixed_tau':t} for k,(c,t) in specs.items()},
        'protocol_sha256':digest(ROOT/'docs/p0_research_protocol_v1.md'),
        'selection':'all months/budgets; failure ineligible; diagnostic controls not automatically deployable'})
    # Snapshot original and additions before any fit.
    with zipfile.ZipFile(out/'source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for folder in ['src','scripts','configs','docs']:
            for p in (ROOT/folder).rglob('*'):
                if p.suffix in ['.py','.json','.md'] and '__pycache__' not in p.parts: z.write(p,p.relative_to(ROOT))
    clim=art['monthly_climatology_c']
    train_clim=np.stack([d['train_hf_insitu_c'][d['train_dates']%100==m].mean(0) for m in range(1,13)])
    audit={'scale_c':scale,'basis_shape':list(psi.shape),'orthogonality_max':float(abs(psi.T@psi-np.eye(psi.shape[1])).max()),
      'climatology_max_error':float(abs(train_clim-clim).max()),
      'train_transform_max_error':float(abs((d['train_hf_insitu_c']-clim[d['train_dates']%100-1]-art['mean_c'])/scale-d['train_hf_model']).max()),
      'validation_transform_max_error':float(abs((d['validation_hf_insitu_c']-clim[d['validation_dates']%100-1]-art['mean_c'])/scale-d['validation_hf_model']).max()),
      'basis_column_norms':np.linalg.norm(psi,axis=0),'train_coeff_variance':np.var(d['train_hf_model']@psi,axis=0),
      'training_artifacts_hash_unchanged':digest(source/'training_artifacts.npz')==digest(out/'inputs/training_artifacts.npz'),
      'no_test_arrays':True,'source_input_sha256':digest(source/'aligned_fields.npz')}
    dump(out/'transform_audit.json',audit)
    rows=[]; arrays={}; timings=[]; projections=[]
    trace_stream=gzip.open(out/'diagnostics.jsonl.gz','wt',encoding='utf-8')
    for design in designs:
        date=design['date']
        if design['seed']!=seed or date%100 not in months: continue
        i=design['i']; low=d['validation_lf_model'][i]; truth=d['validation_hf_model'][i]; y=np.asarray(design['y'])
        if scenario=='clean': y=truth.copy()
        for rank in [4,16,32,70]:
            b=psi[:,:rank]; sw=np.sqrt(w[e]); a=np.linalg.lstsq(sw[:,None]*b[e],sw*truth[e],rcond=None)[0]
            projections.append({'date':date,'rank':rank,'offline_evaluation_projection_lower_bound_rmse_c':rms((b@a-truth)*scale)})
        empty=DiagnosticRBMF(psi[:,:32],cfg).fit(low,[],indices=[])
        previous={}
        for budget in [4,6,8,10,12]:
            ids=np.asarray(design['ids'][:budget]); assert len(ids)==budget
            methods=['climatology','no_observation',*specs,'pod_ridge','pod_lasso','gaussian_sbl_vb']
            for name in methods:
                start=time.perf_counter(); model=None
                if name=='climatology': pred=-art['mean_c']/scale; diag={'converged':True,'iterations':0}
                elif name=='no_observation': pred=empty.predict();model=empty;diag=empty.diagnostics_
                elif name in specs:
                    c,ft=specs[name]; model=DiagnosticRBMF(psi[:,:32],c,ft).fit(low,y[ids],indices=ids)
                    pred=model.predict();diag=model.diagnostics_
                else:
                    comp=selection['comparison']; c=replace(cfg,**comp['method_inference'][name])
                    pred,model,diag,_=reconstruct(name,psi[:,:comp['method_rank'][name]],low,ids,y[ids],c,comp['regularization'][name])
                elapsed=time.perf_counter()-start; err=(pred-truth)*scale
                row={'date':date,'budget':budget,'method':name,'rmse_c':rms(err),'mae_c':mae(err),
                     'converged':bool(diag['converged']),'iterations':diag['iterations'],'seconds':elapsed,
                     'vs_climatology_rms_c':rms(pred*scale+art['mean_c']),
                     'added_observation_response_rms_c':rms((pred-previous[name])*scale) if name in previous else None}
                previous[name]=pred.copy();rows.append(row);timings.append(elapsed)
                key=f'{date}_{budget}_{name}';arrays[key]=pred
                if name in specs or name=='no_observation':
                    info={'key':key,'diagnostics':diag,'precision':precision_diagnostics(model,scale,e,w),
                          'factors':model.factor_summary(),'factor_trace':model.factor_trace}
                    if budget in [4,12] and name in specs:
                        c,ft=specs[name]; yp=y[ids].copy();yp[0]+=.02/scale
                        pert=DiagnosticRBMF(psi[:,:32],c,ft).fit(low,yp,indices=ids)
                        info['full_refit_0p02C_first_observation_response_rms_c']=rms((pert.predict()-pred)*scale)
                        info['perturbed_converged']=pert.diagnostics_['converged']
                    trace_stream.write(json.dumps(info)+'\n')
            print(date,budget,'finished',flush=True)
    with (out/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    trace_stream.close()
    np.savez_compressed(out/'predictions_model.npz',**arrays)
    dump(out/'offline_projection.json',projections)
    summaries=[]
    for name in methods:
        rr=[r for r in rows if r['method']==name]
        summaries.append({'method':name,'rmse_c':float(np.mean([r['rmse_c'] for r in rr])),
            'mae_c':float(np.mean([r['mae_c'] for r in rr])),'convergence_rate':float(np.mean([r['converged'] for r in rr])),
            'seconds':sum(r['seconds'] for r in rr),'fits':len(rr)})
    dump(out/'summary.json',summaries)
    dump(out/'resources.json',{'wall_seconds':time.perf_counter()-started,'fit_seconds':sum(timings),
        'max_fit_seconds':max(timings),'output_bytes':sum(p.stat().st_size for p in out.rglob('*') if p.is_file()),
        'single_thread':True,'max_rss_available':False})
    print(json.dumps(summaries,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--months',type=int,nargs='+',default=[1,7]);p.add_argument('--seed',type=int,default=1001);p.add_argument('--scenario',choices=['clean','gaussian'],default='gaussian');a=p.parse_args();run(a.output,a.months,a.seed,a.scenario)
