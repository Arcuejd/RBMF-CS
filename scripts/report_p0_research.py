"""Read-only verification and figures from P0 saved development results."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import csv,gzip,json,sys,zipfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from run_p0_diagnosis import dump,digest,load_development
from rbmf_cs.comparison import fixed_domain

def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        ww=csv.DictWriter(f,fieldnames=list(rows[0]));ww.writeheader();ww.writerows(rows)

out=ROOT/'outputs/p0_report_v1';out.mkdir(exist_ok=False)
base=ROOT/'outputs/p0_validation_v1';rows=[];diagnostics=[];errors=[];projections=[];guards=[]
d=load_development(ROOT/'outputs/comparison_validation_v1/inputs');a=dict(np.load(ROOT/'outputs/comparison_validation_v1/inputs/training_artifacts.npz'))
e,_=fixed_domain(d['original_grid_flat_ids']);w=np.cos(np.deg2rad(d['coordinates_latlon'][e,0]));scale=float(a['scale_c'])
for folder in sorted(p for p in base.iterdir() if p.is_dir()):
    scenario,seed=folder.name.split('_');seed=int(seed)
    with zipfile.ZipFile(folder/'source.zip') as z:
        guards.append({'run':folder.name,'io_guard_in_source_snapshot':'install_development_guard' in z.read('scripts/run_p0_diagnosis.py').decode()})
    with np.load(folder/'predictions_model.npz') as pred:
      for row in csv.DictReader((folder/'metrics.csv').open()):
        for k in ['date','budget','iterations']:row[k]=int(row[k])
        for k in ['rmse_c','mae_c','seconds','vs_climatology_rms_c']:row[k]=float(row[k])
        row['converged']=row['converged']=='True';row['scenario']=scenario;row['seed']=seed
        key=f"{row['date']}_{row['budget']}_{row['method']}";truth=d['validation_hf_model'][row['date']%100-1]
        err=(pred[key]-truth)[e]*scale
        for k,v in [('rmse_c',np.sqrt(np.average(err**2,weights=w))),('mae_c',np.average(abs(err),weights=w))]:
            if abs(v-row[k])>1e-12:errors.append([folder.name,key,k,float(v-row[k])])
        rows.append(row)
    with gzip.open(folder/'diagnostics.jsonl.gz','rt') as f:
      for line in f:
        r=json.loads(line);date,budget,name=r['key'].split('_',2);p=r['precision'];fac=r['factors'];hist=r['diagnostics']['history']
        tau=fac['tau'].get('mean',fac['tau'].get('constant_precision'))
        var=3/tau if 'constant_precision' in fac['tau'] else (3*fac['tau']['rate']/(fac['tau']['shape']-1) if fac['tau']['shape']>1 else None)
        diagnostics.append({'scenario':scenario,'seed':seed,'date':int(date),'budget':int(budget),'method':name,
          'rho':fac['rho']['mean'],'tau':tau,'tau_delta':fac['tau_delta']['mean'],
          'physical_noise_marginal_sd_c':float(np.sqrt(var)*scale) if var is not None else None,
          'gamma_mean':float(np.mean(fac['gamma']['mean'])),
          'lambda_min':min(fac['lambda']['mean'],default=None),'lambda_max':max(fac['lambda']['mean'],default=None),
          'dof':p['effective_dof_frozen'],'obs_contribution_rms_c':p['observation_contribution_rms_c'],
          'lf_contribution_rms_c':p['lf_contribution_rms_c'],'sweep_residual':p['sweep_mean_residual'],
          'perturb_response_rms_c':r.get('full_refit_0p02C_first_observation_response_rms_c'),
          'final_relative_factor_change':hist[-1]['relative_factor_change'],
          'elbo_decreases':r['diagnostics']['elbo_decreases'],
          'obs_rayleigh_mean':float(np.mean([v['observation'] for v in p['observed_direction_rayleigh']])) if p['observed_direction_rayleigh'] else None,
          'ard_rayleigh_mean':float(np.mean([v['ard'] for v in p['observed_direction_rayleigh']])) if p['observed_direction_rayleigh'] else None,
          'lf_rayleigh_mean':float(np.mean([v['lf'] for v in p['observed_direction_rayleigh']])) if p['observed_direction_rayleigh'] else None})
    if folder.name=='clean_1001':projections=json.loads((folder/'offline_projection.json').read_text())
write_csv(out/'all_metrics.csv',rows);write_csv(out/'response_diagnostics.csv',diagnostics)
groups=[];budget_rows=[];month_rows=[]
for scenario in ['clean','gaussian']:
    sr=[r for r in rows if r['scenario']==scenario]
    climate=np.mean([r['rmse_c'] for r in sr if r['method']=='climatology'])
    for method in sorted(set(r['method'] for r in sr)):
        rr=[r for r in sr if r['method']==method]
        means={b:float(np.mean([r['rmse_c'] for r in rr if r['budget']==b])) for b in [4,6,8,10,12]}
        improvement=100*(climate-np.mean([r['rmse_c'] for r in rr]))/climate
        gain=100*(means[4]-means[12])/means[4]
        wins=0
        for date in range(201301,201313):
            mr=[r for r in rr if r['date']==date];cl=np.mean([r['rmse_c'] for r in sr if r['method']=='climatology' and r['date']==date])
            rmse=float(np.mean([r['rmse_c'] for r in mr]));wins+=rmse<cl
            month_rows.append({'scenario':scenario,'method':method,'date':date,'rmse_c':rmse,
                'mae_c':float(np.mean([r['mae_c'] for r in mr])),'convergence_rate':float(np.mean([r['converged'] for r in mr]))})
        for b in means:
            br=[r for r in rr if r['budget']==b]
            budget_rows.append({'scenario':scenario,'method':method,'budget':b,'rmse_c':means[b],
                'mae_c':float(np.mean([r['mae_c'] for r in br])),'convergence_rate':float(np.mean([r['converged'] for r in br]))})
        converged=all(r['converged'] for r in rr)
        groups.append({'scenario':scenario,'method':method,'rmse_c':float(np.mean([r['rmse_c'] for r in rr])),
            'mae_c':float(np.mean([r['mae_c'] for r in rr])),'convergence_rate':float(np.mean([r['converged'] for r in rr])),
            'improvement_vs_climatology_pct':float(improvement),'months_better':int(wins),'gain_4_to_12_pct':gain,
            'three_thresholds_met':bool(improvement>=5 and wins>=8 and gain>=2 and converged),
            'selection_eligible':converged and method not in ['known_noise_conditional','noise_floor_conditional','no_lf'],
            'n_rows':len(rr)})
write_csv(out/'overall.csv',groups);write_csv(out/'by_budget.csv',budget_rows);write_csv(out/'by_month.csv',month_rows)
dump(out/'verification.json',{'passed':not errors and len(rows)==3960,'recomputed_predictions':len(rows),'errors':errors,
 'nonconverged_rows':[r for r in rows if not r['converged']], 'guard_rollout':guards,
 'training_hash_preserved':all(digest(p/'inputs/training_artifacts.npz')==digest(ROOT/'outputs/comparison_validation_v1/inputs/training_artifacts.npz') for p in base.iterdir() if p.is_dir())})
show=['climatology','original','factor_stop','ard_b1','pod_ridge','pod_lasso','gaussian_sbl_vb']
fig,ax=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
for axis,scenario in zip(ax,['clean','gaussian']):
 for method in show:
  rr=[r for r in budget_rows if r['scenario']==scenario and r['method']==method]
  axis.plot([r['budget'] for r in rr],[r['rmse_c'] for r in rr],marker='o',label=method)
 axis.set(title=scenario,xlabel='Node budget',ylabel='Area-weighted RMSE (C)',xticks=[4,6,8,10,12]);axis.grid(alpha=.2)
ax[1].legend(fontsize=8);fig.suptitle('2013 development validation; 3 paired seeds; no independent test')
fig.savefig(out/'budget_curves.png',dpi=200);fig.savefig(out/'budget_curves.svg');plt.close(fig)
fig,ax=plt.subplots(figsize=(10,4.8),layout='constrained')
for method in show:
 rr=[r for r in month_rows if r['scenario']=='gaussian' and r['method']==method]
 ax.plot([r['date']%100 for r in rr],[r['rmse_c'] for r in rr],marker='.',label=method)
ax.set(xlabel='2013 month',ylabel='Area-weighted RMSE (C)',xticks=list(range(1,13)),title='Gaussian 0.02 C; mean over budgets and seeds')
ax.legend(fontsize=8,ncol=3);ax.grid(alpha=.2);fig.savefig(out/'monthly_curves.png',dpi=200);plt.close(fig)
dump(out/'summary.json',groups)
dump(out/'resources.json',{'batch_wall_seconds':json.loads((base/'wall.json').read_text())['wall_seconds'],
    'batch_output_bytes':sum(p.stat().st_size for p in base.rglob('*') if p.is_file()),
    'offline_rank32_lower_bound_mean_c':float(np.mean([r['offline_evaluation_projection_lower_bound_rmse_c'] for r in projections if r['rank']==32])),
    'failure_count':sum(not r['converged'] for r in rows)})
print(json.dumps(groups,indent=2))
