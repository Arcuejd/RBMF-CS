"""Read historical results and independently refit; never overwrite old runs."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse
from collections import Counter
from pathlib import Path
import sys
import json
import hashlib
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from rbmf_cs import RBMFCS, VBConfig


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads((ROOT/'configs/polar_deployment.json').read_text())
    fields = np.load(Path(config['data_directory'])/'aligned_fields.npz')
    art = np.load(Path(config['data_directory'])/'training_artifacts.npz')
    basis, scale = art['basis'], float(art['scale_c'])
    singular = np.linalg.svd(fields['train_hf_model'], compute_uv=False)
    eig = singular**2
    report = {'status': 'exploratory_diagnostic_not_parameter_selection',
              'development_years': {'model_training': '2000-2012', 'validation_reserved': '2013',
                'exploratory_evaluation': '2014 entire year conservatively reserved as exposed',
                'earlier_availability_audit': '2000-2026; not a performance test'},
              'missing': [k for k in ['train_lf_model', 'validation_lf_model', 'validation_ice_fraction'] if k not in fields],
              'pod': {'rank': basis.shape[1], 'participation_rank': float(eig.sum()**2/(eig@eig)),
                'modes_90_percent': int(np.searchsorted(np.cumsum(eig)/eig.sum(), .9)+1),
                'energy_retained': float(art['pod_energy_fraction'])}, 'months': {}}
    for t, date in enumerate(fields['test_dates']):
        date = int(date)
        low = fields['test_lf_model'][t]
        prior = RBMFCS(basis, VBConfig(**config['inference'])).fit(low, [], indices=[])
        rows = {}
        for method in config['methods']:
            folder = ROOT/'outputs/polar_deployment'/str(date)/method
            deploy = json.loads((folder/'deployment.json').read_text())
            obs = json.loads((folder/'acquisitions.json').read_text())
            model = RBMFCS(basis, VBConfig(**config['inference']))
            model.fit(low, [o['value_model'] for o in obs[:4]], indices=[o['node'] for o in obs[:4]])
            steps = []
            for i in range(4, len(obs)+1):
                if i > 4:
                    model.update([obs[i-1]['value_model']], indices=[obs[i-1]['node']])
                obs_precision = float(model.tau_.mean)*(model.h_.T@(model.local_.mean[:, None]*model.h_))
                lf_precision = float(model.delta_.mean)*model.gram
                rhs_lf = float(model.delta_.mean)*model.rho_mean_*model.low_projection_
                rhs_obs = float(model.tau_.mean)*(model.h_.T@(model.local_.mean*model.y_))
                leverage = np.trace(model.cov_alpha_@obs_precision)
                steps.append({'nodes': i, 'observation_precision_trace': float(np.trace(obs_precision)),
                    'lf_precision_trace': float(np.trace(lf_precision)),
                    'observation_to_lf_precision_trace': float(np.trace(obs_precision)/np.trace(lf_precision)),
                    'observation_rhs_norm': float(np.linalg.norm(rhs_obs)), 'lf_rhs_norm': float(np.linalg.norm(rhs_lf)),
                    'observation_effective_dof': float(leverage),
                    'lambda_min': float(model.local_.mean.min()), 'lambda_max': float(model.local_.mean.max()),
                    'lambda_mean': float(model.local_.mean.mean()), 'tau': float(model.tau_.mean),
                    'tau_delta': float(model.delta_.mean), 'rho': float(model.rho_mean_),
                    'posterior_participation_rank': float(np.trace(model.cov_alpha_)**2/np.sum(model.cov_alpha_**2)),
                    'change_from_no_observations_rms_c': float(scale*np.sqrt(np.mean((model.predict()-prior.predict())**2))),
                    'converged': bool(model.diagnostics_['converged'])})
            with np.load(folder/'posterior_000.npz') as seed, np.load(folder/'final.npz') as final:
                delta = final['prediction_insitu_c']-seed['prediction_insitu_c']
                rows[method] = {'selected': deploy['selected'], 'steps': steps,
                    'seed_to_final_prediction_rms_c': float(np.sqrt(np.mean(delta**2))),
                    'seed_to_final_prediction_max_c': float(abs(delta).max()),
                    'constraint_reasons': dict(Counter(reason for d in deploy['decisions']
                        for r in d['prospective_feasibility'].values() for reason in r['reasons'])),
                    'source_sha256': hashlib.sha256((folder/'deployment.json').read_bytes()).hexdigest()}
        pairs = []
        for i, a in enumerate(rows):
            for b in list(rows)[i+1:]:
                pa = np.load(ROOT/'outputs/polar_deployment'/str(date)/a/'final.npz')['prediction_insitu_c']
                pb = np.load(ROOT/'outputs/polar_deployment'/str(date)/b/'final.npz')['prediction_insitu_c']
                aa, bb = set(rows[a]['selected']), set(rows[b]['selected'])
                pairs.append({'a': a, 'b': b, 'jaccard': len(aa&bb)/len(aa|bb),
                              'prediction_rms_difference_c': float(np.sqrt(np.mean((pa-pb)**2)))})
        report['months'][str(date)] = {'methods': rows, 'pairs': pairs}
    (output/'diagnosis.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(output/'diagnosis.json')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
