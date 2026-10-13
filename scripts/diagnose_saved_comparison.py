"""Diagnose archived clean, minimum-seed deployments without extra fitting."""
import argparse
import gzip
import json
from itertools import combinations
from pathlib import Path
import numpy as np
from run_comparison import write_json


def run(root, output):
    config = json.loads((root/'config.json').read_text(encoding='utf-8'))
    seed = min(config['comparison']['seeds'])
    report = {'sample_rule': 'clean, minimum planned seed, every planned month', 'months': {}}
    for date in config['comparison']['dates']:
        folder = root/'cases'/f'{date}_clean_s{seed:03d}'
        if not (folder/'realization.npz').exists():
            report['months'][date] = {'status': 'realization unavailable'}
            continue
        realization = np.load(folder/'realization.npz')
        basis = realization['basis']
        result = {'methods': {}, 'pairs': []}
        final = {}
        for method in ['random', 'maximin', 'mc_eir']:
            loc = folder/'deployment'/method
            paths = sorted(loc.glob('n*.npz'))
            if not paths:
                result['methods'][method] = {'status': 'prediction unavailable'}
                continue
            steps = []
            for path in paths:
                data = np.load(path)
                meta = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
                factors = meta['factors']
                ids = data['selected'][np.isfinite(data['observations_model'])]
                h = basis[ids]
                local = np.asarray(factors['lambda']['mean'])
                obs_precision = factors['tau']['mean']*(h.T@(local[:, None]*h))
                lf_precision = factors['tau_delta']['mean']*(basis.T@basis)
                y = data['observations_model'][np.isfinite(data['observations_model'])]
                rhs_obs = factors['tau']['mean']*(h.T@(local*y))
                rhs_lf = factors['tau_delta']['mean']*factors['rho']['mean']*(basis.T@realization['low_model'])
                cov = data['cov_alpha']
                steps.append({'nodes': len(data['selected']), 'converged': meta['diagnostics']['converged'],
                    'observation_to_lf_precision_trace': float(np.trace(obs_precision)/np.trace(lf_precision)),
                    'observation_precision_trace': float(np.trace(obs_precision)),
                    'lf_precision_trace': float(np.trace(lf_precision)),
                    'ard_precision_trace': float(np.sum(factors['gamma']['mean'])),
                    'lf_scale_rho': factors['rho']['mean'],
                    'observation_rhs_norm': float(np.linalg.norm(rhs_obs)),
                    'lf_rhs_norm': float(np.linalg.norm(rhs_lf)),
                    'observation_effective_dof': float(np.trace(cov@obs_precision)),
                    'lambda_min': float(local.min()) if len(local) else None,
                    'lambda_max': float(local.max()) if len(local) else None,
                    'posterior_participation_rank': float(np.trace(cov)**2/np.sum(cov**2))})
            first, last = np.load(paths[0]), np.load(paths[-1])
            final[method] = set(last['selected'].tolist())
            info = {'steps': steps, 'seed_to_final_prediction_rms_c': float(np.sqrt(np.mean((last['prediction_c']-first['prediction_c'])**2)))}
            trace = loc/'deployment.json.gz'
            if trace.exists():
                saved = json.loads(gzip.decompress(trace.read_bytes()))
                info['stop_reason'] = saved['stop_reason']
                # Preserve full audits; this summary is a convenience, not their replacement.
                info['constraint_audit_artifact'] = str(trace.relative_to(root))
            result['methods'][method] = info
        for a, b in combinations(final, 2):
            result['pairs'].append({'a': a, 'b': b, 'jaccard': len(final[a]&final[b])/len(final[a]|final[b])})
        report['months'][date] = result
    write_json(output, report)
    print(output)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    run(a.root, a.output)
