"""Verify saved development results and enforce the registered stop decision."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import csv,gzip,json,sys
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
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

base=ROOT/'outputs/p0_structure_validation_v1';out=ROOT/'outputs/p0_structure_report_v1';out.mkdir(exist_ok=False)
data=load_development(ROOT/'outputs/comparison_validation_v1/inputs');art=dict(np.load(ROOT/'outputs/comparison_validation_v1/inputs/training_artifacts.npz'))
e,_=fixed_domain(data['original_grid_flat_ids']);w=np.cos(np.deg2rad(data['coordinates_latlon'][e,0]));scale=float(art['scale_c'])
designs=json.loads((ROOT/'outputs/comparison_validation_v1/jobs/rbmf_000/validation_designs.json').read_text())
lookup={(d['date'],d['seed']):d for d in designs}
rows=[];errors=[];trace={};decreases=[]
with gzip.open(base/'traces.jsonl.gz','rt') as f:
 for line in f:
    r=json.loads(line);diag=r['diagnostics'];trace[r['key']]={'converged':diag['converged'],'elbo':diag['history'][-1]['elbo']}
    if diag['elbo_decreases']:decreases.append({'key':r['key'],'count':diag['elbo_decreases']})
for r in csv.DictReader((base/'metrics.csv').open()):
    for k in ['seed','date','budget','iterations']:r[k]=int(r[k])
    for k in ['rmse_c','mae_c','prediction_change_from_previous_budget_c','vs_climatology_rms_c','fit_seconds']:r[k]=float(r[k]) if r[k] else None
    r['converged']=r['converged']=='True';r['selected_start']=int(r['selected_start']) if r['selected_start'] else None
    rows.append(r)
for scenario in ['clean','gaussian']:
 for date in range(201301,201313):
  for seed in [1001,1002,1003]:
   rr=[r for r in rows if (r['scenario'],r['date'],r['seed'])==(scenario,date,seed)]
   with np.load(base/'arrays'/f'{scenario}_{seed}_{date}.npz') as z:
    design=lookup[(date,seed)];truth=data['validation_hf_model'][date%100-1]
    y=truth if scenario=='clean' else np.asarray(design['y'])
    for budget in [4,6,8,10,12]:
        ids=z[f'{budget}_ids']
        if not np.array_equal(ids,design['ids'][:budget]) or np.intersect1d(e,ids).size:errors.append('node pairing')
        if not np.array_equal(z[f'{budget}_observations'],y[ids]):errors.append('observation pairing')
    for row in rr:
        key=f"{row['budget']}_{row['method']}"
        if key not in z:
            errors.append('missing prediction '+key);continue
        pred=z[key];err=(pred-truth)[e]*scale
        for name,value in [('rmse_c',np.sqrt(np.average(err**2,weights=w))),('mae_c',np.average(abs(err),weights=w))]:
            if abs(value-row[name])>1e-12:errors.append('metric '+name)
        if row['method'].endswith('_selected'):
            family=row['method'].split('_')[0];prefix=f"{scenario}_{seed}_{date}_{row['budget']}_{family}_"
            available=[(j,trace[prefix+name]) for j,name in enumerate(['default','alternate']) if trace[prefix+name]['converged']]
            expected=max(available,key=lambda v:(v[1]['elbo'],-v[0]))[0]
            if row['selected_start']!=expected:errors.append('ELBO selection')
            raw=f"{row['budget']}_{family}_{['default','alternate'][expected]}"
            if not np.array_equal(pred,z[raw]):errors.append('selected prediction')
summary=[];monthly=[];budgets=[]
for scenario in ['clean','gaussian']:
 sr=[r for r in rows if r['scenario']==scenario];climate=np.mean([r['rmse_c'] for r in sr if r['method']=='climatology'])
 for method in sorted(set(r['method'] for r in sr)):
    rr=[r for r in sr if r['method']==method];success=all(r['converged'] and r['rmse_c'] is not None for r in rr)
    rmse=float(np.mean([r['rmse_c'] for r in rr if r['rmse_c'] is not None]));mae=float(np.mean([r['mae_c'] for r in rr if r['mae_c'] is not None]))
    wins=0;bb={}
    for month in range(1,13):
        mm=[r for r in rr if r['date']%100==month];cl=np.mean([r['rmse_c'] for r in sr if r['date']%100==month and r['method']=='climatology'])
        value=float(np.mean([r['rmse_c'] for r in mm]));wins+=value<cl
        monthly.append({'scenario':scenario,'month':month,'method':method,'rmse_c':value,'mae_c':float(np.mean([r['mae_c'] for r in mm])),
            'convergence_rate':float(np.mean([r['converged'] for r in mm]))})
    for budget in [4,6,8,10,12]:
        mm=[r for r in rr if r['budget']==budget];bb[budget]=float(np.mean([r['rmse_c'] for r in mm]))
        budgets.append({'scenario':scenario,'budget':budget,'method':method,'rmse_c':bb[budget],'mae_c':float(np.mean([r['mae_c'] for r in mm])),
            'convergence_rate':float(np.mean([r['converged'] for r in mm]))})
    improvement=100*(climate-rmse)/climate;gain=100*(bb[4]-bb[12])/bb[4]
    summary.append({'scenario':scenario,'method':method,'rmse_c':rmse,'mae_c':mae,'improvement_pct':improvement,'months_better':int(wins),
       'gain_4_to_12_pct':gain,'convergence_rate':float(np.mean([r['converged'] for r in rr])),
       'accepted_thresholds':bool(success and improvement>=5 and wins>=8 and gain>=2),
       'alternate_start_selected_count':sum(r['selected_start']==1 for r in rr),'case_count':len(rr)})
write_csv(out/'overall.csv',summary);write_csv(out/'by_month.csv',monthly);write_csv(out/'by_budget.csv',budgets)
dump(out/'summary.json',summary)
joint_passed=all(r['accepted_thresholds'] for r in summary if r['method']=='joint_selected')
responses=json.loads((base/'responses.json').read_text());rs=[]
for name in ['original','mf_selected','joint_selected']:
 rr=[r for r in responses if r['method']==name and r['scenario']=='gaussian' and r['budget']==12]
 row={'method':name,'dof_mean':float(np.mean([r['effective_dof_frozen'] for r in rr])),
     'rho_mean':float(np.mean([r['rho'] for r in rr])),'cross_covariance_norm_mean':float(np.mean([r['cross_covariance_norm'] for r in rr])),
     'observation_component_rms_c':float(np.mean([r['observation_component_rms_c'] for r in rr]))}
 for key in ['minus_0p02C_response_rms_c','plus_0p02C_response_rms_c','central_response_gain']:
    values=[r[key] for r in rr if r.get(key) is not None]
    row[key+'_median']=float(np.median(values));row[key+'_max']=max(values)
 rs.append(row)
write_csv(out/'response_summary.csv',rs)
seal_path=ROOT/'outputs/p0_holdout_v1/seal.json';seal=json.loads(seal_path.read_text())
holdout_manifest_unchanged=digest(ROOT/'outputs/p0_holdout_v1/manifest.json')==seal['manifest_sha256']
training_unchanged=digest(ROOT/'outputs/comparison_validation_v1/inputs/training_artifacts.npz')==json.loads((base/'freeze.json').read_text())['training_sha256']
dump(out/'verification.json',{'passed':not errors and len(rows)==3960 and training_unchanged and holdout_manifest_unchanged,
 'recomputed_predictions':len(rows),'metric_tolerance':1e-12,'paired_designs_checked':360,'selected_predictions_checked':720,
 'errors':errors,'elbo_decreases':decreases,'training_unchanged':training_unchanged,'holdout_manifest_unchanged':holdout_manifest_unchanged,
 'holdout_release_authorized':seal['release_authorized']})
decision={'joint_candidate_meets_three_thresholds_both_scenarios':joint_passed,'further_rbmf_search_authorized_by_this_round':False,
    'action':'retain_development_candidate_pending_response_review' if joint_passed else 'stop_rbmf_search_and_freeze_strong_baselines',
    'independent_test_released':False,'primary_baseline':'pod_ridge','source':'2013 development only'}
dump(out/'decision.json',decision)
if not joint_passed:
    baseline={'status':'development_frozen_not_independent_test_authorized','primary':'pod_ridge',
      'methods':{m:{'rank':r,'regularization':reg,'inference':inf} for m,r,reg,inf in [
        ('pod_ridge',16,.01,{}),('pod_lasso',16,.1,{}),('gaussian_sbl_vb',4,.1,{'a_gamma':.001,'b_gamma':.001})]},
      'training_artifacts_sha256':digest(ROOT/'outputs/comparison_validation_v1/inputs/training_artifacts.npz'),
      'historical_selection_sha256':digest(ROOT/'outputs/comparison_validation_v1/selection.json'),
      'test_gate':'rbmf_cs.p0_holdout_gate.require_qualified_release',
      'next_work':'Complete paired independent data and version/QC audit; do not tune against sealed fields',
      'rbmf_status':'archived research comparator; no further search in this round'}
    dump(out/'strong_baselines_frozen.json',baseline)
fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
show=['climatology','original','mf_selected','joint_selected','pod_ridge','pod_lasso','gaussian_sbl_vb']
for ax,scenario in zip(axes,['clean','gaussian']):
 for name in show:
    rr=[r for r in budgets if r['scenario']==scenario and r['method']==name]
    ax.plot([r['budget'] for r in rr],[r['rmse_c'] for r in rr],marker='o',label=name)
 ax.set(title=scenario,xlabel='Node budget',ylabel='Area RMSE (C)',xticks=[4,6,8,10,12]);ax.grid(alpha=.2)
axes[1].legend(fontsize=8);fig.suptitle('2013 development: joint alpha-rho does not change the probability model')
fig.savefig(out/'budget_comparison.png',dpi=180);fig.savefig(out/'budget_comparison.svg');plt.close(fig)
fig,ax=plt.subplots(figsize=(10,4.5),layout='constrained')
for name in show:
 rr=[r for r in monthly if r['scenario']=='gaussian' and r['method']==name]
 ax.plot([r['month'] for r in rr],[r['rmse_c'] for r in rr],marker='.',label=name)
ax.set(xlabel='2013 month',ylabel='Area RMSE (C)',xticks=range(1,13),title='Gaussian 0.02 C: paired monthly comparison')
ax.legend(fontsize=8,ncol=3);ax.grid(alpha=.2);fig.savefig(out/'monthly_comparison.png',dpi=180);plt.close(fig)
print(json.dumps(summary,indent=2));print('decision',decision,flush=True)
