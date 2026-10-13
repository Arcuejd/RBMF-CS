"""Run deterministic synthetic reconstruction and sequential assimilation smoke demo."""
import argparse, hashlib, json, os, platform, sys
from pathlib import Path
from time import perf_counter, time
from datetime import datetime, timezone
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import scipy
from rbmf_cs import RBMFCS, VBConfig, TrainingTransform, PODBasis, TrainingCovariance, SyntheticConfig, make_synthetic


def save_json(path, value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')


def run(config_path, output):
    start=perf_counter()
    time_start=time()
    config=json.loads(Path(config_path).read_text(encoding='utf-8-sig'))
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    save_json(output/'config.json',config)
    data=make_synthetic(SyntheticConfig(**config['synthetic']))
    transform=TrainingTransform.fit(data['train_hf'],config['preprocessing']['scale_floor'])
    training=transform.transform(data['train_hf'])
    pod=PODBasis.fit(training,config['preprocessing']['n_modes'])
    structural=TrainingCovariance.fit(training)
    obs=config['observation']
    truth=data['test_hf'][obs['test_snapshot']]
    low=transform.transform(data['test_lf'][obs['test_snapshot']])
    rng=np.random.default_rng(obs['seed'])
    total=obs['initial_count']+obs['additional_count']
    indices=rng.choice(len(truth),total,replace=False)
    if obs['noise_squared_scale']<=0 or obs['student_t_df']<=0: raise ValueError('invalid observation noise')
    def acquire(position):
        # This is the only acquisition function accessing held-out HF values.
        index=indices[position]
        y=(truth[index]-transform.mean[index])/transform.scale
        y+=np.sqrt(obs['noise_squared_scale'])*rng.standard_t(obs['student_t_df'])
        if position in obs['outlier_positions']: y+=obs['outlier_offset']
        if position in obs['missing_positions']: y=float('nan')
        return y
    initial_count=obs['initial_count']
    observations=[acquire(i) for i in range(initial_count)]
    model=RBMFCS(pod.matrix,VBConfig(**config['inference']))
    model.fit(low,observations,indices=indices[:initial_count])
    records=[]
    def record(step, acquired):
        predicted=transform.inverse(model.predict())
        variance=transform.inverse_variance(model.posterior_variance())
        entry={'step':step,'acquired_node':acquired,'observed_count':len(model.y_),
               'rmse_physical':float(np.sqrt(np.mean((predicted-truth)**2))),
               'total_latent_variance_physical':float(variance.sum()),
               'converged':model.diagnostics_['converged'],'iterations':model.diagnostics_['iterations'],
               'fit_count':model.fit_count_,'runtime_seconds':model.diagnostics_['runtime_seconds']}
        records.append(entry)
        save_json(output/f'fit_{step:02d}.json',{'factors':model.factor_summary(),'diagnostics':model.diagnostics_})
        np.savez_compressed(output/f'posterior_{step:02d}.npz',mean_physical=predicted,
                            variance_physical=variance,mean_alpha=model.mean_alpha_,cov_alpha=model.cov_alpha_)
    record(0,None)
    for step,position in enumerate(range(initial_count,total),start=1):
        y=acquire(position); observations.append(y)
        model.update([y],indices=[int(indices[position])])
        record(step,int(indices[position]))
    np.savez_compressed(output/'synthetic_data.npz',**data)
    np.savez_compressed(output/'training_artifacts.npz',basis=pod.matrix,mean=transform.mean,
                        scale=transform.scale,singular_values=pod.singular_values,
                        covariance_factor=structural.factor)
    np.savez_compressed(output/'observations.npz',indices=indices,values=np.array(observations))
    files=sorted((ROOT/'src').rglob('*.py'))+sorted((ROOT/'scripts').glob('*.py'))+sorted((ROOT/'tests').glob('*.py'))
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    source=ROOT/'docs/source/source_manifest.json'
    manifest={'label':config['label'],'status':'completed' if all(r['converged'] for r in records) else 'completed_with_nonconvergence',
              'utc_started':datetime.fromtimestamp(time_start,timezone.utc).isoformat(),
              'python':sys.version,'python_executable':sys.executable,'numpy':np.__version__,'scipy':scipy.__version__,
              'platform':platform.platform(),'threads':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
              'configuration_sha256':hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
              'source_hashes':hashes,'paper':json.loads(source.read_text()) if source.exists() else None,
              'runtime_seconds':perf_counter()-start,'pod_training_energy_fraction':pod.energy_fraction,
              'trajectory':records,'limitations':['synthetic only','predeclared random acquisition order, no MC-EIR or engineering feasibility',
              'latent mean-field uncertainty excludes basis/truncation/measurement uncertainty']}
    save_json(output/'run.json',manifest)
    print(json.dumps({'status':manifest['status'],'output':str(output.resolve()),'trajectory':records},indent=2))
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/synthetic_demo.json')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/synthetic_demo')
    args=parser.parse_args()
    result=run(args.config,args.output)
    sys.exit(0 if result['status']=='completed' else 2)
