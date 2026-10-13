"""Independent audit of frozen validation search, including recomputed objectives."""
import argparse
import gzip
import json
from pathlib import Path
import sys
import numpy as np
from run_comparison import digest, write_json
from rbmf_cs.comparison import fixed_domain


def read(path):
    return json.loads(gzip.decompress(path.read_bytes()) if path.suffix == '.gz' else path.read_text(encoding='utf-8'))


def verify(root):
    errors, checks, predictions = [], 0, 0
    def check(value, label):
        nonlocal checks
        checks += 1
        if not value:
            errors.append(label)
    manifest, selection = read(root/'manifest.json'), read(root/'selection.json')
    data = np.load(root/'inputs/aligned_fields.npz')
    art = np.load(root/'inputs/training_artifacts.npz')
    check(not any(k.startswith('test_') and k != 'test_dates' for k in data.files), 'no test truth or LF')
    check(data['test_dates'].size == 0, 'no test rows')
    check(digest(root/'inputs/aligned_fields.npz') == manifest['validation_only_archive_sha256'], 'validation archive hash')
    for path, sha in manifest['input_sha256'].items():
        check(digest(Path(path)) == sha, 'original hash '+path)
    check(set(data['train_dates']).isdisjoint(data['validation_dates']), 'disjoint years')
    check(np.array_equal(data['train_dates'], art['train_dates']), 'training provenance')
    ev, pool = fixed_domain(data['original_grid_flat_ids'])
    w = np.cos(np.deg2rad(data['coordinates_latlon'][:, 0]))[ev].astype(np.float64)
    scale = float(art['scale_c'])
    dates = data['validation_dates'].tolist()
    def rmse(pred, date):
        error = (np.asarray(pred)[ev]-data['validation_hf_model'][dates.index(date), ev])*scale
        return float(np.sqrt(np.sum(w*error**2)/np.sum(w)))
    scores, grids = read(root/'scores.json'), read(root/'search_space.json')
    check(len(scores) == sum(map(len, grids.values())), 'every reconstruction candidate recorded')
    for method, grid in grids.items():
        group = [s for s in scores if s['method'] == method]
        check({s['index'] for s in group} == set(range(len(grid))), 'complete grid '+method)
        valid = [s for s in group if s['score'] is not None]
        winner = min(valid, key=lambda s: (s['score'], s['index']))['parameters']
        check(winner['rank'] == selection['comparison']['method_rank'][method], 'selected rank '+method)
        check(winner['regularization'] == selection['comparison']['regularization'][method], 'selected penalty '+method)
        check(winner['inference'] == selection['comparison']['method_inference'][method], 'selected priors '+method)
        for item in group:
            name = f"{method}_{item['index']:03d}"
            check(item['parameters'] == grid[item['index']], 'grid parameters '+name)
            path = root/'jobs'/name/(name+'.json.gz')
            if not path.exists():
                check(item['score'] is None, 'missing candidate must fail '+name)
                continue
            fits = read(path)['fits']
            pred = np.load(path.with_name(name+'.npz'))['predictions']
            successful = [f for f in fits if 'rmse_c' in f]
            check(len(pred) == len(successful), 'prediction count '+name)
            values = []
            for field, fit in zip(pred, successful):
                value = rmse(field, fit['date'])
                check(abs(value-fit['rmse_c']) < 1e-12, 'validation RMSE '+name)
                check(set(fit['ids']).issubset(pool), 'validation evaluation exclusion '+name)
                values.append(value)
                predictions += 1
            if item['score'] is not None:
                check(len(fits) == 108 and len(successful) == 108 and all(f['diagnostics']['converged'] for f in successful), 'all fits converged '+name)
                check(abs(np.mean(values)-item['score']) < 1e-12, 'candidate objective '+name)
    planning = read(root/'planning_scores.json')
    check({p['index'] for p in planning} == set(range(27)), 'complete planning grid')
    chosen = min((p for p in planning if p['score'] is not None), key=lambda p: (p['score'], p['index']))
    check(chosen['parameters'] == selection['planning'], 'planning argmin')
    for item in planning:
        if item['score'] is None:
            continue
        name = f"planning_{item['index']:03d}"
        fits = read(root/'jobs'/name/(name+'.json.gz'))['fits']
        values = []
        for fit in fits:
            for budget in [4, 8, 12]:
                snapshot = fit['snapshots'][budget-4]
                check(snapshot['diagnostics']['converged'], 'planning convergence '+name)
                values.append(rmse(snapshot['mean'], fit['date']))
                predictions += 1
        check(len(values) == 108 and abs(np.mean(values)-item['score']) < 1e-12, 'planning objective '+name)
    result = {'passed': not errors, 'checks': checks, 'recomputed_predictions': predictions, 'errors': errors,
              'selection_sha256': digest(root/'selection.json'), 'test_truth_available': False}
    write_json(root/'verification.json', result)
    print(json.dumps(result, indent=2))
    return not errors


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    a = p.parse_args()
    sys.exit(0 if verify(a.root) else 2)
