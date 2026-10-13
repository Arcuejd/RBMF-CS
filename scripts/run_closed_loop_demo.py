"""Synthetic Algorithm 2 demonstration, with persisted decisions and posteriors."""
import argparse
import csv
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
import scipy
from rbmf_cs import (RBMFCS, VBConfig, TrainingTransform, PODBasis, TrainingCovariance,
                     SyntheticConfig, make_synthetic, ConstraintConfig, DeploymentConstraints,
                     PlanningConfig, MCEIR, run_closed_loop)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def run(config_path, output):
    start = perf_counter()
    started = datetime.now(timezone.utc).isoformat()
    config = json.loads(Path(config_path).read_text(encoding='utf-8-sig'))
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    write_json(output/'config.json', config)
    data = make_synthetic(SyntheticConfig(**config['synthetic']))
    transform = TrainingTransform.fit(data['train_hf'], config['preprocessing']['scale_floor'])
    training = transform.transform(data['train_hf'])
    pod = PODBasis.fit(training, config['preprocessing']['n_modes'])
    covariance = TrainingCovariance.fit(training)
    deployment = config['deployment']
    constraints = DeploymentConstraints(data['coordinates_km'], data['ice_fraction'],
        ConstraintConfig(**config['constraints']), gateways=deployment['gateways_km'],
        anchors=deployment['sensing_anchors'])
    costs = config['cost_model']
    # Declared deterministic, exogenous synthetic costs, not random scores.
    distance_from_origin = np.linalg.norm(data['coordinates_km'], axis=1)
    base = costs['base_constant'] + costs['base_per_km']*distance_from_origin
    maintenance = costs['maintenance_constant'] + costs['maintenance_per_ice_fraction']*data['ice_fraction']
    risk = costs['failure_risk_constant'] + costs['failure_risk_per_ice_fraction']*data['ice_fraction']
    planner = MCEIR(covariance, base, maintenance, risk, PlanningConfig(**config['planning']),
                   historical_reliability=np.full(len(base), costs['historical_reliability']))
    model = RBMFCS(pod.matrix, VBConfig(**config['inference']))
    observation = config['observation']
    if observation['student_t_df'] <= 0 or observation['noise_squared_scale'] <= 0:
        raise ValueError('positive observation df and squared scale required')
    low = transform.transform(data['test_lf'][observation['test_snapshot']])
    rng = np.random.default_rng(observation['seed'])
    acquisition_log = []
    def acquire(j):
        # The planner never receives truth, future noise, or this closure.
        position = len(acquisition_log)
        value = (data['test_hf'][observation['test_snapshot'], j]-transform.mean[j])/transform.scale
        value += np.sqrt(observation['noise_squared_scale'])*rng.standard_t(observation['student_t_df'])
        if position in observation['outlier_acquisition_positions']:
            value += observation['outlier_offset']
        if position in observation['missing_acquisition_positions']:
            value = float('nan')
        acquisition_log.append({'position': position, 'node': j, 'value': float(value) if np.isfinite(value) else None})
        return value
    seed = deployment['seed']
    # Verify the supplied seed before calling the observation source.
    if not seed or not constraints.evaluate(seed)['feasible']:
        raise ValueError('demo requires a nonempty feasible connected seed')
    seed_y = [acquire(j) for j in seed]
    result = run_closed_loop(model, low, seed, seed_y, constraints, planner, acquire)
    snapshots = result.pop('snapshots')
    snapshot_files = []
    for step, snapshot in enumerate(snapshots):
        posterior_file = f'posterior_{step:03d}.npz'
        fit_file = f'fit_{step:03d}.json'
        np.savez_compressed(output/posterior_file, mean_model=snapshot['mean'],
            variance_model=snapshot['variance'], mean_physical=transform.inverse(snapshot['mean']),
            variance_physical=transform.inverse_variance(snapshot['variance']),
            mean_alpha=snapshot['mean_alpha'], cov_alpha=snapshot['cov_alpha'])
        write_json(output/fit_file, {'factors': snapshot['factors'], 'diagnostics': snapshot['diagnostics']})
        snapshot_files.append({'posterior': posterior_file, 'inference': fit_file})
    write_json(output/'deployment.json', result)
    write_json(output/'acquisitions.json', acquisition_log)
    np.savez_compressed(output/'synthetic_data.npz', **data)
    np.savez_compressed(output/'training_artifacts.npz', basis=pod.matrix, mean=transform.mean,
        scale=transform.scale, covariance_factor=covariance.factor, base_cost=base,
        maintenance_cost=maintenance, failure_risk=risk, reliability=planner.reliability)
    fields = ['round', 'selected_node', 'observation_missing', 'n_observations',
              'delta_h_raw', 'delta_u_raw', 'redundancy_raw', 'delta_h_normalized',
              'delta_u_normalized', 'redundancy_normalized', 'eir', 'cumulative_mission_energy_j',
              'energy_change_j', 'connected', 'selected_ice', 'total_posterior_uncertainty',
              'posterior_fit_count', 'inference_converged', 'inference_iterations', 'runtime_seconds']
    with (output/'trajectory.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for row in result['trajectory']:
            record = {key: row[key] for key in fields if key in row}
            record.update({key: row['score'][key] for key in fields if key in row['score']})
            record['connected'] = row['feasibility']['connected']
            record['selected_ice'] = row['feasibility']['ice'][str(row['selected_node'])]
            writer.writerow(record)
    files = sorted((ROOT/'src').rglob('*.py'))+sorted((ROOT/'scripts').glob('*.py'))+sorted((ROOT/'tests').glob('*.py'))
    status = 'completed' if result['stop_reason'] in ('sensor_budget_reached','no_feasible_candidate','insufficient_marginal_value') else 'stopped_for_attention'
    manifest = {'label': config['label'], 'status': status, 'stop_reason': result['stop_reason'],
                'utc_started': started, 'selected': result['selected'], 'acquisitions': len(acquisition_log),
                'posterior_fits': len(snapshots), 'all_fits_converged': all(s['diagnostics']['converged'] for s in snapshots),
                'final_feasibility': result['terminal_feasibility'], 'snapshots': snapshot_files,
                'runtime_seconds': perf_counter()-start, 'python': sys.version, 'numpy': np.__version__,
                'scipy': scipy.__version__, 'platform': platform.platform(),
                'config_sha256': hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
                'source_hashes': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                'units': {'distance':'km','mission_energy':'J','ice':'fraction','field':'synthetic arbitrary units',
                          'cost':'synthetic cost units','planning_variance':'model anomaly units squared'},
                'limitations': ['synthetic software demonstration only', 'fixed-horizon forecast energy, no real mission telemetry',
                                'no receive/idle radio energy model', 'no baseline comparison or real-data conclusions']}
    write_json(output/'run.json', manifest)
    print(json.dumps({key: manifest[key] for key in ('status','stop_reason','selected','posterior_fits','all_fits_converged','runtime_seconds')}, indent=2))
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/closed_loop_demo.json')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/closed_loop_demo')
    args = parser.parse_args()
    manifest = run(args.config, args.output)
    sys.exit(0 if manifest['status']=='completed' else 2)
