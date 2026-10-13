"""Validation-only grid search. Fail closed if 2013 LF/ice is absent.

Never falls back to 2014. A complete search is expensive; all candidates and
failures are retained, and a selection is written only after every candidate.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import argparse
from dataclasses import replace
from itertools import product
from pathlib import Path
from time import perf_counter
import json
import gzip
import sys
import numpy as np
from run_comparison import (ROOT, write_json, digest, preflight, seed_nodes,
    RBMFCS, VBConfig, TrainingCovariance, ConstraintConfig, DeploymentConstraints,
    PlanningConfig, run_closed_loop, ComparisonPlanner)
from rbmf_cs.comparison import fixed_domain, perturbation, entropy_weights, reconstruct, metrics, SelectedObserver
from rbmf_cs.constraints import distances_km

_plain_json = write_json
def write_json(path, value):
    # Lossless storage of large per-candidate audit trails.
    if path.stem.rsplit('_', 1)[-1].isdigit():
        from run_comparison import clean_json
        with gzip.GzipFile(filename=str(path)+'.gz', mode='wb', mtime=0) as stream:
            stream.write(json.dumps(clean_json(value), ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8'))
    else:
        _plain_json(path, value)

def search_grids(ranks):
    grids = {}
    for method in ['pod_ridge', 'pod_lasso', 'deterministic_mf']:
        grids[method] = [{'rank': r, 'regularization': reg, 'inference': {}}
                         for r, reg in product(ranks, [1e-4, 1e-3, 1e-2, .1, 1., 10.])]
    grids['gaussian_sbl_vb'] = [{'rank': r, 'regularization': .1, 'inference': {'a_gamma': a, 'b_gamma': a}}
                              for r, a in product(ranks, [.001, .01, .1])]
    grids['rbmf'] = [{'rank': r, 'regularization': .1, 'inference': {'a_delta': 2., 'b_delta': d, 'nu': nu}}
                    for r, d, nu in product(ranks, [.2, 2., 20.], [3., 4., 8.])]
    return grids


def run(config_path, output, candidate=None, reconstruction_selection=None, planning_index=None):
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads(config_path.read_text(encoding='utf-8'))
    check = preflight(config)
    write_json(output/'preflight.json', check)
    if not check['formal_ready']:
        write_json(output/'status.json', {'status': 'blocked', 'reasons': check['blocking_issues']})
        print('Validation search blocked:', '; '.join(check['blocking_issues']))
        return 2
    directory = Path(config['data_directory'])
    with np.load(directory/'aligned_fields.npz') as source:
        # Explicit allowlist: target test arrays never materialized in this search.
        data = {key: source[key] for key in ('train_hf_model', 'train_hf_anomaly_c',
            'validation_dates', 'validation_hf_model', 'validation_lf_model',
            'validation_ice_fraction', 'coordinates_latlon', 'original_grid_flat_ids')}
    art = np.load(directory/'training_artifacts.npz')
    basis, scale = art['basis'], float(art['scale_c'])
    eval_ids, candidates = fixed_domain(data['original_grid_flat_ids'])
    weights = np.cos(np.deg2rad(data['coordinates_latlon'][:, 0]))
    denominator = float(np.sqrt(np.average(np.mean(data['train_hf_anomaly_c'][:, eval_ids]**2, axis=0), weights=weights[eval_ids])))
    shape = data['validation_hf_model'].shape
    if shape != data['validation_lf_model'].shape or shape != data['validation_ice_fraction'].shape:
        raise ValueError('aligned validation shapes differ')
    if not np.isfinite(data['validation_lf_model']).all():
        raise ValueError('nonfinite validation LF')
    designs = []
    distances = distances_km(data['coordinates_latlon'], 'latlon_degrees')
    for i, date in enumerate(data['validation_dates']):
        ice = data['validation_ice_fraction'][i]
        initial = seed_nodes(data['coordinates_latlon'], ice, candidates, config)
        constraints = DeploymentConstraints(data['coordinates_latlon'], ice,
            replace(ConstraintConfig(**config['constraints']), max_sensors=12), gateways=np.array([config['gateway_latlon']]))
        selected = initial.copy()
        while len(selected) < 12:
            feasible, _ = constraints.filter_candidates(selected, candidates)
            if not feasible:
                break
            selected.append(min(feasible, key=lambda j: (-distances[j, selected].min(), j)))
        for seed in [1001, 1002, 1003]:
            noise, _ = perturbation(shape[1], seed, int(date), 'gaussian', stream=20)
            designs.append({'date': int(date), 'i': i, 'seed': seed, 'initial': initial,
                            'ids': selected, 'y': data['validation_hf_model'][i]+noise/scale})
    write_json(output/'validation_designs.json', designs)
    ranks = [r for r in [4, 8, 16, 32, 70] if r <= basis.shape[1]]
    grids = {}
    for method in ['pod_ridge', 'pod_lasso', 'deterministic_mf']:
        grids[method] = [{'rank': r, 'regularization': reg, 'inference': {}}
                         for r, reg in product(ranks, [1e-4, 1e-3, 1e-2, .1, 1., 10.])]
    grids['gaussian_sbl_vb'] = [{'rank': r, 'regularization': .1, 'inference': {'a_gamma': a, 'b_gamma': a}}
                              for r, a in product(ranks, [.001, .01, .1])]
    grids['rbmf'] = [{'rank': r, 'regularization': .1, 'inference': {'a_delta': 2., 'b_delta': d, 'nu': nu}}
                    for r, d, nu in product(ranks, [.2, 2., 20.], [3., 4., 8.])]
    write_json(output/'search_space.json', grids)
    chosen, all_scores = {}, []
    for method, grid in grids.items():
        if planning_index is not None:
            break
        if candidate is not None and method != candidate.split(':')[0]:
            continue
        scores = []
        for number, params in enumerate(grid):
            if candidate is not None and number != int(candidate.split(':')[1]):
                continue
            begun = perf_counter()
            errors, fits, predictions = [], [], []
            valid = True
            cfg = replace(VBConfig(**config['inference']), **params['inference'])
            for design in designs:
                for budget in [4, 8, 12]:
                    ids = design['ids'][:budget]
                    if len(ids) < budget:
                        valid = False
                        fits.append({'status': 'infeasible_budget', 'budget': budget, 'date': design['date']})
                        continue
                    try:
                        i = design['i']
                        pred, model, diag, elapsed = reconstruct(method, basis[:, :params['rank']], data['validation_lf_model'][i],
                            ids, design['y'][ids], cfg, params['regularization'])
                        value = metrics(pred*scale, data['validation_hf_model'][i]*scale, eval_ids, weights, denominator)['rmse_c']
                        errors.append(value)
                        predictions.append(pred)
                        valid &= diag['converged']
                        fits.append({'date': design['date'], 'seed': design['seed'], 'budget': budget,
                            'ids': ids, 'y': design['y'][ids], 'rmse_c': value, 'diagnostics': diag,
                            'factors': model.factor_summary() if model else None, 'seconds': elapsed})
                        if model:
                            fits[-1]['posterior_summary'] = {'mean_alpha': model.mean_alpha_, 'variance_alpha': np.diag(model.cov_alpha_)}
                    except Exception as exc:
                        valid = False
                        fits.append({'date': design['date'], 'seed': design['seed'], 'budget': budget, 'error': repr(exc)})
            score = float(np.mean(errors)) if valid and errors else None
            record = {'method': method, 'index': number, 'parameters': params, 'score': score,
                      'valid': valid, 'seconds': perf_counter()-begun}
            scores.append(record)
            all_scores.append(record)
            np.savez_compressed(output/f'{method}_{number:03d}.npz', predictions=np.asarray(predictions))
            write_json(output/f'{method}_{number:03d}.json', {'candidate': record, 'fits': fits})
            write_json(output/'scores.json', all_scores)
        if candidate is not None:
            return 0
        admissible = [s for s in scores if s['score'] is not None]
        if not admissible:
            write_json(output/'status.json', {'status': 'blocked', 'reasons': ['all validation candidates failed: '+method]})
            return 2
        chosen[method] = min(admissible, key=lambda s: (s['score'], s['index']))['parameters']
    # Tune MC-EIR only against validation fields, with RBMF held fixed.
    if reconstruction_selection is not None:
        chosen = json.loads(reconstruction_selection.read_text(encoding='utf-8'))
    rbmf = chosen['rbmf']
    model_basis = basis[:, :rbmf['rank']]
    vb = replace(VBConfig(**config['inference']), **rbmf['inference'])
    covariance = TrainingCovariance.fit(data['train_hf_model'])
    ew = entropy_weights(data['train_hf_model'])
    planning_scores = []
    for number, (beta, eta, reference) in enumerate(product([.25, 1., 4.], [0., .2, .5], [.001, .01, .1])):
        if planning_index is not None and number != planning_index:
            continue
        params = {'beta': beta, 'eta': eta, 'entropy_reference_variance': reference}
        pcfg = replace(PlanningConfig(**config['planning']), **params)
        errors, details = [], []
        valid = True
        for design in designs:
            i = design['i']
            ice = data['validation_ice_fraction'][i]
            constraints = DeploymentConstraints(data['coordinates_latlon'], ice,
                replace(ConstraintConfig(**config['constraints']), max_sensors=12), gateways=np.array([config['gateway_latlon']]))
            observer = SelectedObserver(design['y'], candidates)
            safe_ice = np.nan_to_num(ice)
            planner = ComparisonPlanner(covariance, np.full(shape[1], config['costs']['base']),
                safe_ice*config['costs']['maintenance_per_ice'], safe_ice*config['costs']['failure_per_ice'], pcfg,
                method='mc_eir', coordinates=data['coordinates_latlon'], priorities=np.zeros(shape[1]), entropy_weight=ew)
            try:
                result = run_closed_loop(RBMFCS(model_basis, vb), data['validation_lf_model'][i], design['initial'],
                    [observer.acquire(j) for j in design['initial']], constraints, planner, observer.acquire, candidates=candidates)
                snapshots = result.pop('snapshots')
                for budget in [4, 8, 12]:
                    if len(snapshots) <= budget-4 or not snapshots[budget-4]['diagnostics']['converged']:
                        valid = False
                    else:
                        errors.append(metrics(snapshots[budget-4]['mean']*scale, data['validation_hf_model'][i]*scale,
                            eval_ids, weights, denominator)['rmse_c'])
                details.append({'date': design['date'], 'seed': design['seed'], 'deployment': result, 'snapshots': snapshots})
            except Exception as exc:
                valid = False
                details.append({'date': design['date'], 'seed': design['seed'], 'error': repr(exc)})
        entry = {'index': number, 'parameters': params, 'score': float(np.mean(errors)) if valid and errors else None}
        planning_scores.append(entry)
        write_json(output/f'planning_{number:03d}.json', {'candidate': entry, 'fits': details})
        write_json(output/'planning_scores.json', planning_scores)
    if planning_index is not None:
        return 0
    admissible = [s for s in planning_scores if s['score'] is not None]
    if not admissible:
        write_json(output/'status.json', {'status': 'blocked', 'reasons': ['all deployment validation candidates failed']})
        return 2
    planning = min(admissible, key=lambda s: (s['score'], s['index']))['parameters']
    selection = {'status': 'validation_selected', 'data_sha256': digest(directory/'aligned_fields.npz'),
        'validation_dates': data['validation_dates'].tolist(), 'validation_seeds': [1001, 1002, 1003],
        'selection_rule': 'mean RMSE over months, seeds, budgets; failures infinite; first tie',
        'inference': rbmf['inference'], 'planning': planning,
        'comparison': {'rank': rbmf['rank'], 'method_rank': {m: p['rank'] for m, p in chosen.items()},
            'regularization': {m: p['regularization'] for m, p in chosen.items()},
            'method_inference': {m: p['inference'] for m, p in chosen.items()}}}
    write_json(output/'selection.json', selection)
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/polar_deployment.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--candidate')
    parser.add_argument('--reconstruction-selection', type=Path)
    parser.add_argument('--planning-index', type=int)
    args = parser.parse_args()
    sys.exit(run(args.config, args.output, args.candidate, args.reconstruction_selection, args.planning_index))
