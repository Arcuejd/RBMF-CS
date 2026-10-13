"""Saved-results-only reporting. Never imports a fitter or runs an experiment."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from rbmf_cs.comparison_statistics import paired_intervals


def json_write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def csv_write(path, rows):
    columns = sorted({k for row in rows for k in row})
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def save(fig, output, name):
    fig.suptitle(RUN_LABEL, fontsize=10, color='#844000')
    for ext in ('svg', 'pdf', 'png'):
        fig.savefig(output/f'{name}.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)


def run(root, output):
    global RUN_LABEL
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads((root/'config.json').read_text(encoding='utf-8'))
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    RUN_LABEL = ('Exploratory pilot - inherited parameters' if config['comparison'].get('parameter_status') != 'validation_selected'
                 else 'Exploratory 2014 evaluation - validation-selected parameters')
    RUN_LABEL += f" / {len(config['comparison']['seeds'])} seeds, {len(config['comparison']['dates'])} months"
    budgets = config['comparison']['budgets']
    maximum = max(budgets)
    rows, lookup, failures = [], {}, []
    for case_folder in sorted((root/'cases').iterdir()):
        if not (case_folder/'case.json').exists():
            continue
        case = json.loads((case_folder/'case.json').read_text(encoding='utf-8'))
        for record in sorted(case_folder.glob('*/*/n*.json')):
            content = json.loads(record.read_text(encoding='utf-8'))
            row = dict(case=case['id'], date=case['date'], seed=case['seed'], scenario=case['scenario'],
                       branch=record.parent.parent.name, method=record.parent.name, **content['metrics'])
            if row['requested_nodes'] not in budgets:
                continue
            rows.append(row)
            if record.with_suffix('.npz').exists():
                lookup[(row['date'], row['seed'], row['scenario'], row['branch'], row['method'], row['requested_nodes'])] = record.with_suffix('.npz')
        for path in case_folder.glob('*/*/failure*.json'):
            failures.append(dict(case=case['id'], artifact=str(path.relative_to(root)), **json.loads(path.read_text(encoding='utf-8'))))
    planned = json.loads((root/'planned_cases.json').read_text(encoding='utf-8'))
    present = {(r['case'], r['branch'], r['method'], r['requested_nodes']) for r in rows}
    ledger = {r['id']: r for r in json.loads((root/'ledger.json').read_text(encoding='utf-8'))}
    for case in planned:
        for branch in ('deployment', 'reconstruction'):
            methods = list(config['comparison'][branch])
            if branch == 'deployment' and config['comparison'].get('constraint_cost'):
                methods += ['unconstrained_pod_qr', 'unconstrained_mc_eir']
            for method in methods:
                for budget in budgets:
                    if (case['id'], branch, method, budget) not in present:
                        rows.append(dict(case=case['id'], date=case['date'], seed=case['seed'], scenario=case['scenario'],
                            branch=branch, method=method, requested_nodes=budget, actual_nodes=None,
                            converged=False, status='unavailable_'+ledger.get(case['id'], {}).get('status', 'not_run')))
    csv_write(output/'metrics.csv', rows)
    json_write(output/'failures.json', failures)
    groups = defaultdict(list)
    for row in rows:
        groups[(row['branch'], row['method'], row['scenario'], row['requested_nodes'])].append(row)
    aggregated = []
    for (branch, method, scenario, count), group in groups.items():
        expected = sum(c['scenario'] == scenario for c in planned)
        valid = [r for r in group if r.get('converged') and r.get('actual_nodes') == count and 'rmse_c' in r]
        row = {'branch': branch, 'method': method, 'scenario': scenario, 'requested_nodes': count,
            'planned_cases': expected, 'recorded_cases': len(group), 'converged_same_count_cases': len(valid),
            'missing_cases': expected-len(group), 'feasible_cases': sum(bool(r.get('feasible')) for r in group),
            'feasible_rate_of_planned': sum(bool(r.get('feasible')) for r in group)/expected}
        for key in ('rmse_c', 'mae_c', 'nrmse', 'cost', 'mission_energy_j', 'selection_seconds', 'reconstruction_seconds',
                    'preprocessing_seconds', 'latent_coverage95', 'future_coverage95', 'latent_width95_c', 'future_width95_c'):
            values = [r[key] for r in valid if r.get(key) is not None]
            row[key] = float(np.mean(values)) if values else None
        aggregated.append(row)
    csv_write(output/'summary.csv', aggregated)
    pairs = []
    for branch in ('reconstruction', 'deployment'):
        reference = 'rbmf' if branch == 'reconstruction' else 'mc_eir'
        methods = sorted({r['method'] for r in rows if r['branch'] == branch and not r['method'].startswith('unconstrained_')})
        for scenario in config['comparison']['scenarios']:
            for budget in budgets:
                for method in methods:
                    if method == reference:
                        continue
                    ea, eb = [], []
                    paired_seeds = []
                    dates = config['comparison']['dates']
                    for seed in config['comparison']['seeds']:
                        paths = [(lookup.get((date, seed, scenario, branch, method, budget)),
                                  lookup.get((date, seed, scenario, branch, reference, budget))) for date in dates]
                        if not all(a and b for a, b in paths):
                            continue
                        good = True
                        aa, bb = [], []
                        for pa, pb in paths:
                            ma = json.loads(pa.with_suffix('.json').read_text())['metrics']
                            mb = json.loads(pb.with_suffix('.json').read_text())['metrics']
                            if not (ma.get('converged') and mb.get('converged') and ma.get('feasible') and mb.get('feasible')):
                                good = False
                                break
                            with np.load(pa) as a, np.load(pb) as b:
                                ev = a['evaluation']
                                aa.append(a['prediction_c'][ev]-a['truth_c'][ev])
                                bb.append(b['prediction_c'][ev]-b['truth_c'][ev])
                                w = a['weights'][ev]
                            realization = np.load(pa.parents[2]/'realization.npz')
                            coords = realization['coordinates_latlon'][ev]
                        if good:
                            ea.append(aa)
                            eb.append(bb)
                            paired_seeds.append(seed)
                    if ea:
                        result = paired_intervals(ea, eb, w, coords)
                    else:
                        result = {'paired_mean_difference_c': None, 'reason': 'no complete successful paired seed',
                                  'environment_p_value': None, 'holm_adjusted_p_value': None}
                    pairs.append(dict(branch=branch, method=method, reference=reference, scenario=scenario,
                        budget=budget, paired_seeds=paired_seeds, planned_seeds=len(config['comparison']['seeds']), **result))
    json_write(output/'paired_statistics.json', pairs)
    csv_write(output/'paired_statistics.csv', [{k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in p.items()} for p in pairs])

    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    def subset(branch, scenario='clean', count=None, methods=None):
        return [r for r in aggregated if r['branch'] == branch and r['scenario'] == scenario
                and (count is None or r['requested_nodes'] == count) and (methods is None or r['method'] in methods)]
    for branch in ['reconstruction', 'deployment']:
        methods = config['comparison'][branch]
        core = [m for m in methods if not m.startswith('no_')]
        fig, ax = plt.subplots(figsize=(8, 4.6), layout='constrained')
        for method in core:
            points = sorted([r for r in subset(branch) if r['method'] == method and r['rmse_c'] is not None], key=lambda r: r['requested_nodes'])
            ax.plot([r['requested_nodes'] for r in points], [r['rmse_c'] for r in points], 'o-', label=method, markersize=4)
        ax.set(xlabel='Actual sensors (completed budgets only)', ylabel='Area-weighted RMSE (C)', title=branch+' / clean')
        ax.set_xticks(budgets)
        ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc='upper left')
        save(fig, output, 'error_nodes_'+branch)

        scenarios = config['comparison']['scenarios']
        matrix = np.full((len(methods), len(scenarios)), np.nan)
        for i, method in enumerate(methods):
            for j, scenario in enumerate(scenarios):
                values = [r['rmse_c'] for r in subset(branch, scenario, maximum, [method]) if r['rmse_c'] is not None]
                if values:
                    matrix[i, j] = values[0]
        fig, ax = plt.subplots(figsize=(11, max(4, len(methods)*.35)), layout='constrained')
        im = ax.pcolormesh(np.ma.masked_invalid(matrix), cmap='viridis', shading='flat')
        ax.set_ylim(len(methods), 0)
        ax.set_yticks(np.arange(len(methods))+.5, methods)
        ax.set_xticks(np.arange(len(scenarios))+.5, scenarios, rotation=45, ha='right')
        ax.set_title(f'{branch} robustness / {maximum} nodes (blank = unavailable)')
        fig.colorbar(im, ax=ax, label='Area-weighted RMSE (C)')
        save(fig, output, 'robustness_'+branch)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout='constrained')
    cost_rows = [r for r in subset('deployment', count=maximum) if not r['method'].startswith('no_')]
    colors = {r['method']: plt.get_cmap('tab20')(i) for i, r in enumerate(cost_rows)}
    for ax, key in zip(axes, ['cost', 'mission_energy_j']):
        for row in cost_rows:
            if row[key] is None or row['rmse_c'] is None:
                continue
            ax.scatter(row[key], row['rmse_c'], marker='x' if row['feasible_rate_of_planned'] < 1 else 'o',
                       color=colors[row['method']], label=row['method'])
        ax.set(xlabel=key, ylabel='RMSE (C)', title='Accuracy / cost; x = engineering violations')
        ax.set_ylim(0, max(r['rmse_c'] for r in cost_rows if r['rmse_c'] is not None)*1.08)
    handles, labels = axes[0].get_legend_handles_labels()
    axes[1].legend(handles, labels, fontsize=7, bbox_to_anchor=(1.02, 1), loc='upper left')
    save(fig, output, 'accuracy_cost')

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    for ax, branch in zip(axes, ['reconstruction', 'deployment']):
        points = [r for r in pairs if r['branch'] == branch and r['scenario'] == 'clean'
                  and r['budget'] == maximum and r['method'].startswith('no_') and r['paired_mean_difference_c'] is not None]
        ax.barh([r['method'] for r in points], [r['paired_mean_difference_c'] for r in points])
        ax.axvline(0, color='black', linewidth=.6)
        ax.set(xlabel='RMSE difference from full method (C)', title=branch+' ablations')
    save(fig, output, 'ablations')

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    for ax, kind in zip(axes, ['latent', 'future']):
        points = [r for r in subset('reconstruction', 'gaussian', maximum) if r.get(kind+'_coverage95') is not None]
        ax.barh([r['method'] for r in points], [r[kind+'_coverage95'] for r in points])
        ax.axvline(.95, color='red', linestyle='--', label='nominal 95%')
        ax.set(xlim=(0, 1), xlabel='Area-weighted coverage', title=kind+' interval / Gaussian scenario')
        ax.legend()
    save(fig, output, 'calibration')

    # Prespecified case, not the case with best performance.
    date, seed = min(config['comparison']['dates']), min(config['comparison']['seeds'])
    selected = [('reconstruction', 'lf_only'), ('reconstruction', 'pod_ridge'), ('reconstruction', 'rbmf'),
                ('deployment', 'mc_eir'), ('deployment', 'random')]
    samples = []
    for branch, method in selected:
        path = lookup.get((date, seed, 'clean', branch, method, maximum))
        if path:
            with np.load(path) as z:
                samples.append((method, {k: z[k].copy() for k in z.files}))
            coords = np.load(path.parents[2]/'realization.npz')['coordinates_latlon']
    if samples:
        truth = samples[0][1]['truth_c']
        vmin = min(float(truth.min()), min(float(z['prediction_c'].min()) for _, z in samples))
        vmax = max(float(truth.max()), max(float(z['prediction_c'].max()) for _, z in samples))
        fig, axes = plt.subplots(2, 3, figsize=(10, 6), layout='constrained')
        panels = [('IAP reference', truth, [])]+[(name, z['prediction_c'], z['selected']) for name, z in samples]
        for ax, (label, field, ids) in zip(axes.flat, panels):
            im = ax.scatter(coords[:, 1], coords[:, 0], c=field, s=13, cmap='viridis', vmin=vmin, vmax=vmax)
            if len(ids):
                ax.scatter(coords[ids, 1], coords[ids, 0], facecolors='none', edgecolors='black', s=50)
            ax.set(title=label, xlabel='Longitude', ylabel='Latitude')
        fig.colorbar(im, ax=axes.ravel().tolist(), label='Temperature (C)', shrink=.8)
        save(fig, output, 'spatial_common_scale')
        extent = max(float(np.max(np.abs(z['prediction_c']-truth))) for _, z in samples)
        extent = max(extent, np.finfo(float).eps)
        fig, axes = plt.subplots(2, 3, figsize=(10, 6), layout='constrained')
        for ax, (name, z) in zip(axes.flat, samples):
            im = ax.scatter(coords[:, 1], coords[:, 0], c=z['prediction_c']-truth,
                s=13, cmap='RdBu_r', vmin=-extent, vmax=extent)
            ids = z['selected']
            ax.scatter(coords[ids, 1], coords[ids, 0], facecolors='none', edgecolors='black', s=50)
            ax.set(title=name, xlabel='Longitude', ylabel='Latitude')
        for ax in list(axes.flat)[len(samples):]:
            ax.set_visible(False)
        fig.colorbar(im, ax=axes.ravel().tolist(), label='Prediction minus IAP reference (C)', shrink=.8)
        save(fig, output, 'spatial_error_common_scale')
        json_write(output/'spatial_selection.json', {'date': date, 'seed': seed, 'scenario': 'clean',
            'budget': maximum, 'rule': 'earliest month; minimum seed; largest planned budget',
            'color_limits_c': [vmin, vmax], 'error_color_limits_c': [-extent, extent],
            'missing_panels': [m for _, m in selected if m not in [s[0] for s in samples]]})
    json_write(output/'plot_manifest.json', {'source_run': str(root.resolve()),
        'plot_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'statistics_source_sha256': hashlib.sha256((ROOT/'src/rbmf_cs/comparison_statistics.py').read_bytes()).hexdigest(),
        'source_manifest_sha256': hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),
        'source_config_sha256': hashlib.sha256((root/'config.json').read_bytes()).hexdigest(),
        'figures': sorted(p.name for p in output.glob('*.svg')),
        'smoothing': False, 'monotonicity_enforcement': False,
        'statistics_scope': 'conditional seed CI; environment inference unavailable',
        'observed_records': len(rows), 'failures': len(failures)})
    print(output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    run(args.root.resolve(), args.output.resolve())
