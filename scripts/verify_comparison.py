"""Verify saved predictions, paired observations, provenance and repeatability."""
import argparse
import json
import gzip
from pathlib import Path
import sys
import numpy as np
from run_comparison import ROOT, digest, write_json
from rbmf_cs.comparison import fixed_domain
from rbmf_cs import ConstraintConfig, DeploymentConstraints


def without_timing(value):
    if isinstance(value, dict):
        return {k: without_timing(v) for k, v in value.items() if 'seconds' not in k and k not in ('created_utc',)}
    if isinstance(value, list):
        return [without_timing(v) for v in value]
    return value


def verify(root, repeat=None):
    errors, warnings = [], []
    checks = 0
    def check(ok, label):
        nonlocal checks
        checks += 1
        if not ok:
            errors.append(label)
    config = json.loads((root/'config.json').read_text(encoding='utf-8'))
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    check(digest(root/'config.json') == manifest['config_sha256'], 'config hash')
    check(digest(root/'source.zip') == manifest['source_archive_sha256'], 'source snapshot hash')
    if manifest.get('selection_sha256'):
        check(digest(root/'selection.json') == manifest['selection_sha256'], 'frozen selection hash')
    for path, sha in manifest['input_sha256'].items():
        check(digest(root/'inputs'/Path(path).name) == sha, 'input copy '+path)
        check(digest(Path(path)) == sha, 'original remains unchanged '+path)
    fields = np.load(root/'inputs/aligned_fields.npz')
    art = np.load(root/'inputs/training_artifacts.npz')
    check(not set(fields['train_dates']) & set(fields['validation_dates']), 'train-validation overlap')
    check(not set(fields['train_dates']) & set(fields['test_dates']), 'train-test overlap')
    check(np.array_equal(fields['train_dates'], art['train_dates']), 'train artifact dates')
    eval_ids, candidates = fixed_domain(fields['original_grid_flat_ids'])
    requested = set(config['comparison']['budgets'])
    n_arrays, n_metrics = 0, 0
    ledger = json.loads((root/'ledger.json').read_text(encoding='utf-8'))
    planned = json.loads((root/'planned_cases.json').read_text(encoding='utf-8'))
    check({c['id'] for c in planned} == {c['id'] for c in ledger}, 'every planned case recorded')
    for entry in ledger:
        if entry['status'] != 'completed':
            warnings.append(entry['id']+': '+entry['status'])
        folder = root/'cases'/entry['id']
        if not (folder/'realization.npz').exists():
            check(entry['status'] != 'completed', 'missing realization not recorded as failure')
            continue
        data = np.load(folder/'realization.npz')
        check(np.array_equal(data['evaluation'], eval_ids), 'fixed evaluation '+entry['id'])
        check(not set(data['initial']) & set(eval_ids), 'seed eval exclusion')
        month = int(np.flatnonzero(fields['test_dates'] == entry['date'])[0])
        constraints = DeploymentConstraints(fields['coordinates_latlon'], fields['ice_fraction'][month],
            ConstraintConfig(**config['constraints']), gateways=np.array([config['gateway_latlon']]))
        shared = {}
        for record in folder.glob('*/*/n*.npz'):
            saved = np.load(record)
            meta = json.loads(record.with_suffix('.json').read_text(encoding='utf-8'))
            metric = meta['metrics']
            ids = saved['selected']
            check(not set(ids) & set(eval_ids), 'evaluation sampled '+str(record))
            check(np.array_equal(saved['evaluation'], eval_ids), 'metric domain '+str(record))
            expected = (saved['truth_c'][ids]+data['noise_c'][ids]-saved['baseline_c'][ids])/float(saved['scale_c'])
            expected[data['missing'][ids]] = np.nan
            check(np.allclose(saved['observations_model'], expected, atol=1e-12, rtol=0, equal_nan=True), 'paired noise '+str(record))
            e = saved['prediction_c'][eval_ids]-saved['truth_c'][eval_ids]
            w = saved['weights'][eval_ids].astype(np.float64)
            rmse = np.sqrt(np.sum(w*e*e)/np.sum(w))
            check(abs(rmse-metric['rmse_c']) < 1e-12, 'RMSE recompute '+str(record))
            check(abs(np.sum(w*abs(e))/np.sum(w)-metric['mae_c']) < 1e-12, 'MAE recompute')
            check(abs(rmse/metric['nrmse_denominator_c']-metric['nrmse']) < 1e-12, 'NRMSE recompute')
            check(constraints.evaluate(ids.tolist())['feasible'] == metric['feasible'], 'original constraints '+str(record))
            if record.parent.parent.name == 'reconstruction':
                count = len(ids)
                if count in shared:
                    check(np.array_equal(ids, shared[count]), 'shared reconstruction design')
                shared[count] = ids
            if not metric['converged']:
                warnings.append(str(record.relative_to(root))+': nonconverged retained')
            if repeat is not None:
                other = repeat/record.relative_to(root)
                check(other.exists(), 'repeat missing '+str(other))
                if other.exists():
                    with np.load(other) as reference:
                        for key in saved.files:
                            check(np.allclose(saved[key], reference[key], atol=1e-12, rtol=0, equal_nan=True), 'repeat '+str(record)+' '+key)
                            n_arrays += 1
                    other_meta = json.loads(other.with_suffix('.json').read_text(encoding='utf-8'))
                    check(without_timing(meta) == without_timing(other_meta), 'repeat factors/diagnostics '+str(record))
            n_metrics += 1
        for branch in ['deployment', 'reconstruction']:
            for method in config['comparison'][branch]:
                location = folder/branch/method
                for count in requested:
                    exists = (location/f'n{count:02d}.json').exists() or (location/'failure.json').exists() or (location/f'failure_n{count:02d}.json').exists()
                    check(exists or entry['status'] != 'completed', 'missing result not explained '+str(location)+str(count))
        if repeat is not None:
            for path in folder.glob('deployment/*/deployment.json*'):
                other = repeat/path.relative_to(root)
                check(other.exists(), 'repeat missing deployment')
                if other.exists():
                    def load(p):
                        return json.loads(gzip.decompress(p.read_bytes()).decode('utf-8')) if p.suffix == '.gz' else json.loads(p.read_text())
                    check(without_timing(load(path)) == without_timing(load(other)), 'repeat decisions/constraints '+str(path))
    result = {'passed': not errors, 'checks': checks, 'recomputed_predictions': n_metrics,
              'repeated_arrays': n_arrays, 'errors': errors, 'warnings': warnings,
              'root_config_sha256': digest(root/'config.json'), 'source_sha256': manifest['source_archive_sha256'],
              'selection_sha256': manifest.get('selection_sha256'),
              'scope': 'artifact integrity; algorithm unit tests are separate',
              'repeat': str(repeat) if repeat else None}
    write_json(root/('repeat_verification.json' if repeat else 'verification.json'), result)
    print(json.dumps({k: v for k, v in result.items() if k != 'warnings'}, indent=2))
    return result['passed']

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--repeat', type=Path)
    args = parser.parse_args()
    sys.exit(0 if verify(args.root.resolve(), args.repeat.resolve() if args.repeat else None) else 2)
