"""Audit-first comparison runner. Existing outputs are never overwritten.

Examples: --mode preflight|pilot|formal --output outputs/unique_run
Formal runs require real validation LF/ice and a frozen validated selection.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import argparse
from dataclasses import replace
from pathlib import Path
from time import perf_counter
import datetime
import hashlib
import importlib.metadata
import json
import gzip
from concurrent.futures import ThreadPoolExecutor, as_completed
import platform
import shutil
import subprocess
import sys
import traceback
import zipfile
import numpy as np
import scipy
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from rbmf_cs import RBMFCS, VBConfig, TrainingCovariance, ConstraintConfig, DeploymentConstraints, PlanningConfig, run_closed_loop
from rbmf_cs.model import GammaFactor
from rbmf_cs.basis import PODBasis
from rbmf_cs.constraints import distances_km
from rbmf_cs.comparison import (RECONSTRUCTION, DEPLOYMENT, ComparisonPlanner, entropy_weights,
    reconstruct, perturbation, SelectedObserver, fixed_domain, metrics, future_interval)

SCENARIOS = ['clean', 'gaussian', 'student', 'outlier', 'missing', 'lf_bias',
             'train_gaussian', 'train_student', 'train_outlier', 'train_missing']


def clean_json(x):
    if isinstance(x, dict):
        return {str(k): clean_json(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return [clean_json(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean_json(x.tolist())
    if isinstance(x, (np.integer, np.bool_)):
        return x.item()
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    return x


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.name == 'deployment.json':
        with gzip.GzipFile(filename=str(path)+'.gz', mode='wb', mtime=0) as stream:
            stream.write(json.dumps(clean_json(value), ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8'))
        return
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(clean_json(value), indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def preflight(config):
    directory = Path(config['data_directory'])
    issues = []
    with np.load(directory/'aligned_fields.npz') as data, np.load(directory/'training_artifacts.npz') as art:
        required = ['validation_lf_model', 'validation_ice_fraction']
        issues.extend('missing '+k for k in required if k not in data)
        train, val, test = [set(data[k].tolist()) for k in ('train_dates', 'validation_dates', 'test_dates')]
        if train & val or train & test or val & test:
            issues.append('overlapping temporal splits')
        if train != set(art['train_dates'].tolist()) or any(d//100 > 2012 for d in train):
            issues.append('training artifact provenance mismatch')
        if val != set(range(201301, 201313)):
            issues.append('validation split must remain all of 2013')
        schema = {k: list(data[k].shape) for k in data.files}
    return {'formal_ready': not issues, 'blocking_issues': issues, 'schema': schema,
            'evaluation_label': 'exploratory_2014_no_unseen_paired_test_data',
            'unchanged_split': {'train': '2000-2012', 'validation': '2013', 'exploratory': '2014'}}


def seed_nodes(coordinates, ice, candidates, config):
    gateway = np.array([config['gateway_latlon']])
    distance = distances_km(np.vstack((coordinates, gateway)), 'latlon_degrees')[:-1, -1]
    cfg = ConstraintConfig(**config['constraints'])
    eligible = candidates[np.isfinite(ice[candidates]) & (ice[candidates] <= cfg.ice_threshold)
                          & (distance[candidates] <= cfg.communication_radius_km)]
    seed = eligible[np.argsort(distance[eligible], kind='stable')[:4]].tolist()
    if len(seed) != 4:
        raise ValueError('no four-node feasible seed')
    return seed


def save_prediction(folder, count, prediction, ids, y, baseline, scale, truth, future_truth,
                    evaluation, weights, denominator, constraints, costs, timings,
                    diagnostics, model=None, snapshot=None, interval_seed=0):
    folder.mkdir(parents=True, exist_ok=True)
    prediction_c = baseline+scale*prediction
    arrays = {'prediction_c': prediction_c, 'selected': np.asarray(ids, int),
              'observations_model': np.asarray(y), 'truth_c': truth, 'future_truth_c': future_truth,
              'evaluation': evaluation, 'weights': weights, 'baseline_c': baseline, 'scale_c': scale}
    row = dict(actual_nodes=len(ids), requested_nodes=count, effective_observations=int(np.isfinite(y).sum()),
               **metrics(prediction_c, truth, evaluation, weights, denominator), **timings,
               converged=bool(diagnostics['converged']),
               status='complete' if diagnostics['converged'] else 'nonconverged',
               cost=float(np.sum(costs[ids])))
    check = constraints.evaluate(ids)
    row.update(feasible=check['feasible'], mission_energy_j=check['energy']['total_j'] if check['energy'] else None)
    factors = None
    if snapshot is not None:
        # Restore a detached snapshot only for saved interval calculation; no refit.
        model.mean_alpha_, model.cov_alpha_ = snapshot['mean_alpha'], snapshot['cov_alpha']
        factors = snapshot['factors']
        model.tau_ = GammaFactor(np.asarray(factors['tau']['shape']), np.asarray(factors['tau']['rate']))
    if model is not None:
        variance = model.posterior_variance()*scale**2
        latent = np.stack((prediction_c-1.959963984540054*np.sqrt(variance),
                           prediction_c+1.959963984540054*np.sqrt(variance)))
        future = baseline+scale*future_interval(model, interval_seed)
        arrays.update(variance_c2=variance, latent_interval_c=latent, future_interval_c=future,
                      mean_alpha=model.mean_alpha_, cov_alpha=model.cov_alpha_)
        for name, target, interval in [('latent', truth, latent), ('future', future_truth, future)]:
            ev = evaluation[np.isfinite(target[evaluation])]
            row[name+'_coverage95'] = float(np.average((target[ev] >= interval[0, ev]) &
                (target[ev] <= interval[1, ev]), weights=weights[ev])) if len(ev) else None
            row[name+'_coverage_nodes'] = len(ev)
            row[name+'_width95_c'] = float(np.average(interval[1, evaluation]-interval[0, evaluation], weights=weights[evaluation]))
        factors = factors or model.factor_summary()
    np.savez_compressed(folder/f'n{count:02d}.npz', **arrays)
    write_json(folder/f'n{count:02d}.json', {'metrics': row, 'diagnostics': diagnostics,
               'model_config': model.configuration() if model is not None else None,
               'factors': factors, 'constraints': check, 'interval_seed': interval_seed,
               'interval_samples': 4096, 'uncertainty_omissions': ['basis learning', 'truncation', 'model discrepancy', 'mean-field cross covariance']})
    return row


def run_case(root, case):
    cfg = json.loads((root/'config.json').read_text(encoding='utf-8'))
    settings = cfg['comparison']
    output = root/'cases'/case['id']
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'case.json', case)
    start = perf_counter()
    fields = np.load(root/'inputs/aligned_fields.npz')
    art = np.load(root/'inputs/training_artifacts.npz')
    coords, scale = fields['coordinates_latlon'], float(art['scale_c'])
    date, seed, scenario = case['date'], case['seed'], case['scenario']
    t = int(np.flatnonzero(fields['test_dates'] == date)[0])
    evaluation, candidates = fixed_domain(fields['original_grid_flat_ids'])
    weights = np.cos(np.deg2rad(coords[:, 0]))
    denominator = float(np.sqrt(np.average(np.mean(fields['train_hf_anomaly_c'][:, evaluation]**2, axis=0), weights=weights[evaluation])))
    baseline = art['monthly_climatology_c'][date % 100-1]+art['mean_c']
    training = fields['train_hf_model'].copy()
    if scenario.startswith('train_'):
        noise, missing = perturbation(training.shape, seed, 0, scenario[6:], stream=10)
        training += noise/scale
        training[missing] = np.nan
        means = np.nanmean(training, axis=0)
        if not np.isfinite(means).all():
            raise ValueError('all-missing training column')
        training = np.where(np.isnan(training), means, training)
        online = 'clean'
    else:
        noise, missing = np.zeros_like(training), np.zeros_like(training, bool)
        online = scenario
    rank = settings.get('rank', art['basis'].shape[1])
    full_basis = PODBasis.fit(training, int(art['basis'].shape[1])).matrix if scenario.startswith('train_') else art['basis']
    basis = full_basis[:, :rank]
    covariance = TrainingCovariance.fit(training)
    ew = entropy_weights(training)
    perturb, mask = perturbation(len(coords), seed, date, online, stream=20)
    future_noise, future_missing = perturbation(len(coords), seed, date, online, stream=21)
    truth = fields['test_hf_insitu_c'][t]
    future_truth = truth+future_noise
    future_truth[future_missing] = np.nan
    obs = (truth+perturb-baseline)/scale
    obs[mask] = np.nan
    low = fields['test_lf_model'][t].copy() + (.2/scale if online == 'lf_bias' else 0.)
    ice = fields['ice_fraction'][t]
    initial = seed_nodes(coords, ice, candidates, cfg)
    priorities = np.random.default_rng(np.random.SeedSequence([seed, date, 30])).random(len(coords))
    main_cfg = ConstraintConfig(**cfg['constraints'])
    main_cfg = replace(main_cfg, max_sensors=max(settings['budgets']))
    original = DeploymentConstraints(coords, ice, main_cfg, gateways=np.array([cfg['gateway_latlon']]))
    if not original.evaluate(initial)['feasible']:
        raise ValueError('initial seed infeasible')
    base = np.full(len(coords), cfg['costs']['base'])
    maintenance = cfg['costs']['maintenance_per_ice']*np.nan_to_num(ice)
    failure = cfg['costs']['failure_per_ice']*np.nan_to_num(ice)
    planning_cfg = PlanningConfig(**cfg['planning'])
    costs = base+planning_cfg.maintenance_weight*maintenance+planning_cfg.failure_weight*failure
    inference_cfg = VBConfig(**cfg['inference'])
    preprocessing = perf_counter()-start
    np.savez_compressed(output/'realization.npz', noise_c=perturb, missing=mask,
        future_noise_c=future_noise, future_missing=future_missing, training_noise_c=noise,
        training_missing=missing, training_model=training, basis=basis, full_basis=full_basis, covariance_factor=covariance.factor,
        entropy_weights=ew, evaluation=evaluation, candidates=candidates, initial=initial,
        coordinates_latlon=coords, low_model=low, random_priorities=priorities)
    result_summary = {'preprocessing_seconds': preprocessing, 'methods': {}, 'failures': []}
    fixed_data = None
    # Maximin is always first to produce the shared reconstruction observations.
    methods = ['maximin']+[m for m in settings['deployment'] if m != 'maximin']
    if settings.get('constraint_cost', False):
        methods += ['unconstrained_pod_qr', 'unconstrained_mc_eir']
    for method in methods:
        folder = output/'deployment'/method
        folder.mkdir(parents=True, exist_ok=True)
        try:
            unconstrained = method.startswith('unconstrained_')
            constraints = original
            if unconstrained:
                constraints = DeploymentConstraints(coords, np.zeros(len(coords)), replace(main_cfg,
                    ice_threshold=1., communication_radius_km=21000., energy_budget_j=1e30),
                    gateways=np.array([cfg['gateway_latlon']]))
            observer = SelectedObserver(obs, candidates)
            planner_start = perf_counter()
            planner = ComparisonPlanner(covariance, base, maintenance, failure, planning_cfg,
                method=method.removeprefix('unconstrained_'), coordinates=coords, priorities=priorities, entropy_weight=ew)
            planner_setup = perf_counter()-planner_start
            model = RBMFCS(basis, inference_cfg)
            result = run_closed_loop(model, low, initial, [observer.acquire(j) for j in initial],
                constraints, planner, observer.acquire, candidates=candidates)
            snapshots = result.pop('snapshots')
            write_json(folder/'deployment.json', result)
            write_json(folder/'observations.json', observer.log)
            selection_time = planner_setup
            reconstruction_time = 0.
            for step, snapshot in enumerate(snapshots):
                ids = initial if step == 0 else result['trajectory'][step-1]['selected']
                count = len(ids)
                if step:
                    selection_time += result['decisions'][step-1]['scoring_runtime_seconds']
                reconstruction_time += snapshot['diagnostics']['runtime_seconds']
                values = np.array([np.nan if r['value'] is None else r['value'] for r in observer.log[:count]])
                save_prediction(folder, count, snapshot['mean'], ids, values, baseline, scale, truth,
                    future_truth, evaluation, weights, denominator, original, costs,
                    dict(preprocessing_seconds=preprocessing, selection_seconds=selection_time,
                         reconstruction_seconds=reconstruction_time), snapshot['diagnostics'],
                    model=model, snapshot=snapshot, interval_seed=seed+date)
            if method == 'maximin':
                fixed_data = (result['selected'], observer.log)
            result_summary['methods'][method] = {'actual_nodes': len(result['selected']),
                'stop_reason': result['stop_reason'], 'all_converged': all(s['diagnostics']['converged'] for s in snapshots)}
            for budget in settings['budgets']:
                if len(result['selected']) < budget:
                    write_json(folder/f'n{budget:02d}.json', {'metrics': {'requested_nodes': budget,
                        'actual_nodes': len(result['selected']), 'status': result['stop_reason'], 'converged': False}})
        except Exception as exc:
            failure_record = {'method': method, 'error': repr(exc), 'traceback': traceback.format_exc()}
            result_summary['failures'].append(failure_record)
            write_json(folder/'failure.json', failure_record)
        write_json(output/'summary.json', result_summary)
    if fixed_data is not None:
        ids_all, observations = fixed_data
        for method in settings['reconstruction']:
            for budget in settings['budgets']:
                folder = output/'reconstruction'/method
                if len(ids_all) < budget:
                    write_json(folder/f'n{budget:02d}.json', {'metrics': {'requested_nodes': budget,
                        'actual_nodes': len(ids_all), 'status': 'shared_design_incomplete', 'converged': False}})
                    continue
                ids = ids_all[:budget]
                y = np.array([np.nan if r['value'] is None else r['value'] for r in observations[:budget]])
                try:
                    penalty = settings.get('regularization', {}).get(method, .1)
                    method_vb = replace(inference_cfg, **settings.get('method_inference', {}).get(method, {}))
                    method_basis = full_basis[:, :settings.get('method_rank', {}).get(method, rank)]
                    mean, model, diagnostic, elapsed = reconstruct(method, method_basis, low, ids, y, method_vb, penalty)
                    save_prediction(folder, budget, mean, ids, y, baseline, scale, truth, future_truth,
                        evaluation, weights, denominator, original, costs,
                        dict(preprocessing_seconds=preprocessing, selection_seconds=0., reconstruction_seconds=elapsed),
                        diagnostic, model=model, interval_seed=seed+date)
                except Exception as exc:
                    record = {'method': method, 'budget': budget, 'error': repr(exc), 'traceback': traceback.format_exc()}
                    result_summary['failures'].append(record)
                    write_json(folder/f'failure_n{budget:02d}.json', record)
    else:
        result_summary['failures'].append({'error': 'shared Maximin design unavailable; reconstruction not run'})
    result_summary['wall_seconds_including_audit_io'] = perf_counter()-start
    write_json(output/'summary.json', result_summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['preflight', 'pilot', 'formal', 'worker'], default='preflight')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/polar_deployment.json')
    parser.add_argument('--case', type=str)
    parser.add_argument('--seeds', type=int, default=2)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--full-pilot-budgets', action='store_true')
    parser.add_argument('--scenarios', nargs='+', choices=SCENARIOS, default=SCENARIOS)
    parser.add_argument('--selection', type=Path)
    parser.add_argument('--pilot-verification', type=Path)
    args = parser.parse_args()
    root = args.output.resolve()
    if args.mode == 'worker':
        run_case(root, json.loads(args.case))
        return 0
    root.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_text(encoding='utf-8'))
    check = preflight(config)
    write_json(root/'preflight.json', check)
    if args.mode == 'preflight':
        print(json.dumps(check, indent=2))
        return 0 if check['formal_ready'] else 2
    selection = json.loads(args.selection.read_text(encoding='utf-8')) if args.selection else {}
    if selection and (selection.get('status') != 'validation_selected' or selection.get('data_sha256') != digest(Path(config['data_directory'])/'aligned_fields.npz')):
        raise ValueError('invalid validation selection or input hash')
    if args.mode == 'formal':
        if args.selection is None or args.pilot_verification is None:
            check['blocking_issues'].append('frozen validation selection and passed pilot verification required')
        else:
            verification = json.loads(args.pilot_verification.read_text(encoding='utf-8'))
            selection = json.loads(args.selection.read_text(encoding='utf-8'))
            if not verification.get('passed') or selection.get('status') != 'validation_selected':
                check['blocking_issues'].append('invalid pilot or selection status')
            if verification.get('selection_sha256') != digest(args.selection):
                check['blocking_issues'].append('pilot did not verify this frozen validation selection')
            audit_path = args.selection.parent/'verification.json'
            audit = json.loads(audit_path.read_text(encoding='utf-8')) if audit_path.exists() else {}
            if not audit.get('passed') or audit.get('selection_sha256') != digest(args.selection):
                check['blocking_issues'].append('independent validation-search audit required')
            if selection.get('data_sha256') != digest(Path(config['data_directory'])/'aligned_fields.npz'):
                check['blocking_issues'].append('selection input hash mismatch')
        if check['blocking_issues']:
            write_json(root/'status.json', {'status': 'blocked', 'reasons': check['blocking_issues']})
            print('Formal experiment blocked:', '; '.join(check['blocking_issues']))
            return 2
    settings = {'budgets': [4, 6] if args.mode == 'pilot' and not args.full_pilot_budgets else [4, 6, 8, 10, 12],
        'seeds': list(range(args.seeds if args.mode == 'pilot' else 30)),
        'dates': config['months'][:1] if args.mode == 'pilot' else config['months'],
        'scenarios': args.scenarios, 'deployment': list(DEPLOYMENT), 'reconstruction': list(RECONSTRUCTION),
        'constraint_cost': True, 'task_timeout_seconds': 600,
        'parameter_status': 'validation_selected' if selection else 'inherited_untuned_pilot'}
    settings.update(selection.get('comparison', {}))
    if 'inference' in selection:
        config['inference'].update(selection['inference'])
    if 'planning' in selection:
        config['planning'].update(selection['planning'])
    config['comparison'] = settings
    write_json(root/'config.json', config)
    if selection:
        shutil.copyfile(args.selection, root/'selection.json')
    if args.pilot_verification:
        shutil.copyfile(args.pilot_verification, root/'pilot_verification.json')
    (root/'inputs').mkdir()
    hashes = {}
    for name in ('aligned_fields.npz', 'training_artifacts.npz'):
        path = Path(config['data_directory'])/name
        hashes[str(path)] = digest(path)
        shutil.copyfile(path, root/'inputs'/name)
    sources = sorted([*ROOT.glob('src/**/*.py'), *ROOT.glob('scripts/*.py'), *ROOT.glob('tests/*.py'),
        *ROOT.glob('docs/source/comparison_reference_v1/*'),
        *ROOT.glob('docs/comparison_protocol*.md'), ROOT/'pyproject.toml'])
    with zipfile.ZipFile(root/'source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for source in sources:
            archive.write(source, source.relative_to(ROOT))
    # Execute workers from the archived source, immune to live-workspace edits.
    with zipfile.ZipFile(root/'source.zip') as archive:
        archive.extractall(root/'runtime')
    manifest = {'mode': args.mode, 'label': check['evaluation_label'],
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'python': sys.version, 'numpy': np.__version__, 'scipy': scipy.__version__,
        'installed_packages': {d.metadata['Name']: d.version for d in importlib.metadata.distributions() if d.metadata['Name']},
        'platform': platform.platform(), 'processor': platform.processor(), 'cpu_count': os.cpu_count(),
        'threads': {'OPENBLAS_NUM_THREADS': os.environ['OPENBLAS_NUM_THREADS'], 'OMP_NUM_THREADS': os.environ['OMP_NUM_THREADS']},
        'input_sha256': hashes, 'source_sha256': {str(p.relative_to(ROOT)): digest(p) for p in sources},
        'config_sha256': digest(root/'config.json'), 'source_archive_sha256': digest(root/'source.zip'),
        'protocol_sha256': digest(ROOT/'docs/comparison_protocol.md'), 'workers': args.workers,
        'parameter_status': settings['parameter_status'],
        'selection_sha256': digest(root/'selection.json') if selection else None, 'status': 'running'}
    write_json(root/'manifest.json', manifest)
    cases = [{'id': f'{date}_{scenario}_s{seed:03d}', 'date': date, 'scenario': scenario, 'seed': seed}
             for date in settings['dates'] for scenario in settings['scenarios'] for seed in settings['seeds']]
    write_json(root/'planned_cases.json', cases)
    ledger = []
    batch_start = perf_counter()
    def execute(case):
        print('Running', case['id'], flush=True)
        begun = perf_counter()
        try:
            completed = subprocess.run([sys.executable, str(root/'runtime/scripts/run_comparison.py'), '--mode', 'worker',
                '--output', str(root), '--case', json.dumps(case)], capture_output=True, text=True,
                timeout=settings['task_timeout_seconds'])
            entry = dict(case, status='completed' if completed.returncode == 0 else 'failed',
                         returncode=completed.returncode, stdout=completed.stdout, stderr=completed.stderr)
            summary_path = root/'cases'/case['id']/'summary.json'
            if summary_path.exists():
                summary = json.loads(summary_path.read_text(encoding='utf-8'))
                entry['method_failures'] = len(summary['failures'])
        except subprocess.TimeoutExpired as exc:
            entry = dict(case, status='timeout', reason=str(exc))
        entry['wall_seconds'] = perf_counter()-begun
        return entry
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        pending = [executor.submit(execute, case) for case in cases]
        for future in as_completed(pending):
            ledger.append(future.result())
            write_json(root/'ledger.json', sorted(ledger, key=lambda item: item['id']))
    manifest['status'] = 'complete' if all(e['status'] == 'completed' and not e.get('method_failures', 0) for e in ledger) else 'complete_with_failures'
    manifest['wall_seconds'] = perf_counter()-batch_start
    manifest['sum_case_seconds'] = sum(e['wall_seconds'] for e in ledger)
    write_json(root/'manifest.json', manifest)
    if args.mode == 'pilot':
        total = manifest['wall_seconds']
        # Two vs eight added nodes; reconstruction checkpoints 2 vs 5. Upper heuristic, not a promise.
        ratio = 30/len(settings['seeds'])*4/max(1, len(settings['dates']))
        write_json(root/'scale_estimate.json', {'pilot_wall_seconds': total,
            'proposed_formal_seeds': 30, 'proposed_formal_dates': 4, 'proposed_budgets': [4, 6, 8, 10, 12],
            'rough_formal_seconds_lower': total*ratio, 'rough_formal_seconds_upper': total*ratio*4,
            'status': 'measured_selected_pilot' if selection else 'untuned_pilot', 'assumption': 'same scenarios and concurrency; 1x to 4x pilot per-case work; no precision-based scale selection'})
    print(root)
    return 0 if manifest['status'] == 'complete' else 2


if __name__ == '__main__':
    sys.exit(main())
