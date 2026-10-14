"""User-run independent saved-array audit; does not write a conclusion report."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.special import ndtri
from clbr_common import (ROOT, preservation, source_hash, digest, write,
                         load_inputs, install_development_guard)


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def area_average(values, weights):
    """Independent float64 reduction, including boolean coverage and denominator."""
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    return np.sum(weights*values, dtype=np.float64)/np.sum(weights, dtype=np.float64)


def verify(directory):
    out = Path(directory).resolve()
    if out.parent != ROOT/'outputs' or not out.name.startswith('clbr_p1_'):
        raise ValueError('only an explicit CLBR run directory is permitted')
    target = out/'verification.json'
    if target.exists():
        raise FileExistsError('verification is immutable; use a new run directory after repairs')
    preserved = preservation()
    current_source = source_hash()
    install_development_guard(ROOT, out)
    freeze = json.loads((out/'freeze.json').read_text())
    errors = []

    def require(ok, description):
        if not bool(ok):
            errors.append(description)

    def close(actual, expected, label, tol=1e-10):
        require(np.allclose(actual, expected, atol=tol, rtol=0, equal_nan=False), label)

    require(freeze['source_sha256'] == current_source, 'source changed since run')
    require(freeze['preservation'] == preserved, 'protected artifacts changed')
    require(json.loads((out/'status.json').read_text())['completed'], 'batch incomplete')
    data, art, e, w, designs, selected = load_inputs()
    require(freeze['input_sha256'] == digest(ROOT/'outputs/comparison_validation_v1/inputs/aligned_fields.npz'), 'input changed')
    scale = float(art['scale_c'])
    plan = freeze['plan']
    expected_cases = len(plan['outer_folds'])*3*len(plan['scenarios'])*len(plan['seeds'])*len(plan['budgets'])
    points = read_csv(out/'point_metrics.csv')
    intervals = read_csv(out/'interval_metrics.csv')
    inner_rows = read_csv(out/'inner_metrics.csv')
    manifests = json.loads((out/'fold_manifest.json').read_text())
    require(len(manifests) == len(plan['outer_folds']), 'missing outer folds')
    require(len(points) == expected_cases*10, 'missing point methods/cases')
    require(len(intervals) == expected_cases*3*2*4, 'missing interval methods/cases')
    require(len(inner_rows) == expected_cases*3*9, 'missing inner candidates/cases')
    require(len(list((out/'outer').glob('*.npz'))) == expected_cases, 'outer file count')
    require(len(list((out/'inner').glob('*.npz'))) == expected_cases*3, 'inner file count')
    lookup = {f['fold']: f for f in manifests}
    for f in manifests:
        train, evaluation = set(f['outer_training']), set(f['outer_evaluation'])
        require(not train & evaluation and train | evaluation == set(range(201301, 201313)), 'outer partition')
        require(f['outer_prior']['training_dates'] == f['outer_training'], 'outer learner provenance')
        seen = []
        for inner in f['inner']:
            it, ie = set(inner['training']), set(inner['evaluation'])
            require(not it & ie and it | ie == train and not (it | ie) & evaluation, 'inner partition')
            require(inner['prior']['training_dates'] == inner['training'], 'inner learner provenance')
            seen += inner['evaluation']
        require(sorted(seen) == sorted(train), 'inner coverage')
        for family in ('clbr', 'ridge'):
            scores = []
            count = 3 if family == 'clbr' else 6
            for j in range(count):
                group = [r for r in inner_rows if int(r['fold']) == f['fold'] and r['family'] == family and int(r['candidate']) == j]
                require(len(group) == 9*len(plan['scenarios'])*len(plan['seeds'])*len(plan['budgets']), 'inner candidate denominator')
                scores.append(np.mean([float(r['rmse_c']) for r in group]) if all(r['success']=='True' for r in group) else np.inf)
            require(int(np.argmin(scores)) == f['choices'][family], 'selection not inner-only minimum')
            for a, b in zip(scores, f[family+'_candidate_scores']):
                if b is None:
                    require(not np.isfinite(a), 'failed candidate not excluded')
                else:
                    close(a, b, 'candidate score')
        for quantile in f['quantiles'].values():
            require(set(quantile['calibration_dates']) == train, 'uncertainty calibration contaminated')

    point_count = interval_count = inner_count = 0
    # Check the original paired designs and independent future RNG, separately from the runner.
    def design_check(path, z):
        fold, date, scenario, seed, budget = path.stem.split('_')
        fold, date, seed, budget = map(int, (fold, date, seed, budget))
        row = designs[(date, seed)]
        ids = np.asarray(row['ids'][:budget], int)
        i = row['i']
        close(z['ids'], ids, 'paired node ids')
        require(not set(ids) & set(e), 'evaluation node observed')
        close(z['observations'], data['validation_hf_model'][i, ids] if scenario == 'clean' else np.asarray(row['y'])[ids], 'paired observations')
        close(z['truth'], data['validation_hf_model'][i], 'saved target')
        noise = np.zeros(len(w)) if scenario == 'clean' else np.random.default_rng(np.random.SeedSequence([seed, date, 1])).normal(0, .02, len(w))
        close(z['future'], z['truth']+noise/scale, 'independent future realization')
        return fold, date, scenario, seed, budget

    for path in (out/'inner').glob('*.npz'):
        with np.load(path, allow_pickle=False) as z:
            fold, date, scenario, seed, budget = design_check(path, z)
            f = lookup[fold]
            inner = next(item for item in f['inner'] if date in item['evaluation'])
            require(z['training_dates'].tolist() == inner['training'], 'saved inner learner months')
            group = [r for r in inner_rows if int(r['fold']) == fold and int(r['date']) == date and r['scenario'] == scenario and int(r['seed']) == seed and int(r['budget']) == budget]
            for r in group:
                pred = z['clbr_mean' if r['family']=='clbr' else 'ridge_mean'][int(r['candidate'])]
                if r['success'] == 'True':
                    error = (pred[e]-z['truth'][e])*scale
                    close(float(r['rmse_c']), np.sqrt(area_average(error**2, w[e])), 'inner saved RMSE')
                else:
                    require(not r['rmse_c'], 'failed inner candidate finite score')
                inner_count += 1
    # Independently regenerate weighted empirical quantiles from held-out inner arrays.
    for f in manifests:
        for key, q in f['quantiles'].items():
            scenario, budget, target_name = key.split('_'); budget = int(budget)
            scores = {'clbr': [], 'ridge': []}
            for path in (out/'inner').glob(f"{f['fold']}_*_{scenario}_*_{budget}.npz"):
                with np.load(path, allow_pickle=False) as z:
                    truth = z['truth'] if target_name == 'field' else z['future']
                    cj, rj = f['choices']['clbr'], f['choices']['ridge']
                    sd = np.sqrt(z['clbr_var'][cj] + (z['noise_var'][cj] if target_name == 'future' else 0))
                    scores['clbr'].append(abs(truth[e]-z['clbr_mean'][cj, e])/np.maximum(sd[e], 1e-12))
                    scores['ridge'].append(abs(truth[e]-z['ridge_mean'][rj, e]))
            for method, values in scores.items():
                array = np.concatenate(values)
                order = np.argsort(array, kind='stable')
                # Same normalized sum order as the registered weighting convention.
                weights = np.asarray(np.tile(w[e]/w[e].sum()/len(values), len(values)), dtype=np.float64)
                cumulative = np.cumsum(weights[order])/weights.sum()
                expected = [array[order[min(np.searchsorted(cumulative, level), len(array)-1)]] for level in (.5, .8, .9, .95)]
                close(q[method], expected, 'empirical quantile learned only inside outer training')
    for path in (out/'outer').glob('*.npz'):
        with np.load(path, allow_pickle=False) as z:
            fold, date, scenario, seed, budget = design_check(path, z)
            require(z['training_dates'].tolist() == lookup[fold]['outer_training'], 'outer fit months')
            close(z['low'], data['validation_lf_model'][date%100-1], 'LF paired field')
            pgroup = [r for r in points if int(r['fold']) == fold and int(r['date']) == date and r['scenario'] == scenario and int(r['seed']) == seed and int(r['budget']) == budget]
            for r in pgroup:
                pred = z['mean_'+r['method']]
                if np.isfinite(pred).all():
                    error = (pred[e]-z['truth'][e])*scale
                    close(float(r['rmse_c']), np.sqrt(area_average(error**2, w[e])), 'outer saved RMSE', 1e-12)
                    close(float(r['mae_c']), area_average(abs(error), w[e]), 'outer saved MAE', 1e-12)
                else:
                    require(r['success']=='False', 'nonfinite method marked success')
                point_count += 1
            if 'clbr_var' in z:
                close(z['mean_residual_ridge'][e], z['mean_clbr'][e], 'deterministic/Bayes equality', 1e-10)
            igroup = [r for r in intervals if int(r['fold']) == fold and int(r['date']) == date and r['scenario'] == scenario and int(r['seed']) == seed and int(r['budget']) == budget]
            for r in igroup:
                target_name, method = r['target'], r['method']
                level = float(r['level']); k = [.5, .8, .9, .95].index(level)
                lo = z[f'{target_name}_{method}_lower'][e]*scale
                hi = z[f'{target_name}_{method}_upper'][e]*scale
                center = z['mean_ridge_nested' if method=='ridge_constant' else 'mean_clbr'][e]*scale
                truth = z['truth' if target_name=='field' else 'future'][e]*scale
                q = lookup[fold]['quantiles'][f'{scenario}_{budget}_{target_name}']
                if method == 'ridge_constant':
                    expected_half = np.broadcast_to(np.asarray(q['ridge'])*scale, lo.shape)
                else:
                    factor = ndtri((1+np.array([.5, .8, .9, .95]))/2) if method=='clbr_conditional' else np.asarray(q['clbr'])
                    sd = np.sqrt(z['clbr_var'][e]+(z['noise_var'] if target_name=='future' else 0))*scale
                    expected_half = sd[:, None]*factor
                close(lo, center[:, None]-expected_half, 'interval construction lower')
                close(hi, center[:, None]+expected_half, 'interval construction upper')
                covered = (truth>=lo[:, k]) & (truth<=hi[:, k])
                width = hi[:, k]-lo[:, k]
                score = width+2/(1-level)*(np.maximum(lo[:, k]-truth, 0)+np.maximum(truth-hi[:, k], 0))
                close(float(r['coverage']), area_average(covered, w[e]), 'coverage')
                close(float(r['coverage_unweighted']), covered.mean(), 'unweighted coverage')
                close(float(r['width']), area_average(width, w[e]), 'width')
                close(float(r['interval_score']), area_average(score, w[e]), 'interval score')
                wis = .5*abs(truth-center)
                for j, nominal in enumerate([.5, .8, .9, .95]):
                    a = 1-nominal
                    wis += a/2*(hi[:, j]-lo[:, j]) + np.maximum(lo[:, j]-truth, 0)+np.maximum(truth-hi[:, j], 0)
                close(float(r['wis_c']), area_average(wis, w[e])/4.5, 'WIS')
                interval_count += 1
    failures = json.loads((out/'failures.json').read_text())
    # Counts fail closed; no successful-subset promotion after dropping failures.
    require(not failures, 'experiment failures retained; not eligible for promotion')
    report = {'passed': not errors, 'phase': freeze['phase'], 'source_sha256': current_source,
              'point_rows_checked': point_count, 'inner_rows_checked': inner_count,
              'interval_rows_checked': interval_count, 'paired_cases_checked': expected_cases*4,
              'preservation': preserved, 'errors': errors, 'failure_count': len(failures)}
    write(target, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return not errors


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--run', required=True)
    raise SystemExit(0 if verify(p.parse_args().run) else 1)
