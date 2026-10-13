"""Execute the preregistered validation grid in independent archived workers."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
import argparse
import json
import subprocess
import sys
import zipfile
import shutil
import importlib.metadata
import platform
import numpy as np
from run_comparison import ROOT, write_json, digest, preflight
from tune_comparison import search_grids


def run(config_path, output, workers):
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads(config_path.read_text(encoding='utf-8'))
    check = preflight(config)
    write_json(output/'preflight.json', check)
    if not check['formal_ready']:
        raise ValueError(check['blocking_issues'])
    source_data = Path(config['data_directory'])
    (output/'inputs').mkdir()
    with np.load(source_data/'aligned_fields.npz') as z:
        # Physical barrier: workers receive no test truth or test LF arrays.
        validation = {k: z[k] for k in z.files if not k.startswith('test_') and k != 'ice_fraction'}
    # preflight needs disjoint test date metadata, but no test fields.
    validation['test_dates'] = np.array([], dtype=int)
    np.savez_compressed(output/'inputs/aligned_fields.npz', **validation)
    shutil.copyfile(source_data/'training_artifacts.npz', output/'inputs/training_artifacts.npz')
    config['data_directory'] = str((output/'inputs').resolve())
    write_json(output/'config.json', config)
    sources = sorted([*ROOT.glob('src/**/*.py'), *ROOT.glob('scripts/*.py'), *ROOT.glob('docs/comparison_protocol*.md')])
    with zipfile.ZipFile(output/'source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for p in sources:
            archive.write(p, p.relative_to(ROOT))
    with zipfile.ZipFile(output/'source.zip') as archive:
        archive.extractall(output/'runtime')
    manifest = {'status': 'running', 'workers': workers, 'timeout_per_candidate_seconds': 600,
        'source_sha256': {str(p.relative_to(ROOT)): digest(p) for p in sources},
        'input_sha256': {str(source_data/n): digest(source_data/n) for n in ['aligned_fields.npz', 'training_artifacts.npz']},
        'validation_only_archive_sha256': digest(output/'inputs/aligned_fields.npz'),
        'test_fields_available_to_workers': False, 'python': sys.version, 'numpy': np.__version__,
        'platform': platform.platform(),
        'installed_packages': {d.metadata['Name']: d.version for d in importlib.metadata.distributions() if d.metadata['Name']}}
    write_json(output/'manifest.json', manifest)
    grids = search_grids([4, 8, 16, 32, 70])
    write_json(output/'search_space.json', grids)
    def execute(task):
        name, flags = task
        begun = perf_counter()
        args = [sys.executable, str(output/'runtime/scripts/tune_comparison.py'), '--config', str(output/'config.json'),
                '--output', str(output/'jobs'/name), *flags]
        try:
            result = subprocess.run(args, capture_output=True, text=True, timeout=600)
            entry = {'task': name, 'returncode': result.returncode, 'stderr': result.stderr, 'stdout': result.stdout,
                     'status': 'completed' if result.returncode == 0 else 'failed'}
        except subprocess.TimeoutExpired:
            entry = {'task': name, 'status': 'timeout', 'reason': '600 second candidate timeout'}
        entry['wall_seconds'] = perf_counter()-begun
        return entry
    ledger = []
    def phase(tasks):
        with ThreadPoolExecutor(max_workers=workers) as executor:
            pending = [executor.submit(execute, t) for t in tasks]
            for future in as_completed(pending):
                item = future.result()
                ledger.append(item)
                write_json(output/'ledger.json', ledger)
                print(f"{len(ledger)} candidates finished: {item['task']} {item['status']}", flush=True)
    begun = perf_counter()
    tasks = [(f'{m}_{i:03d}', ['--candidate', f'{m}:{i}']) for m, g in grids.items() for i in range(len(g))]
    phase(tasks)
    chosen, scores = {}, []
    for method, grid in grids.items():
        candidates = []
        for i, params in enumerate(grid):
            score_file = output/'jobs'/f'{method}_{i:03d}'/'scores.json'
            entry = json.loads(score_file.read_text(encoding='utf-8'))[0] if score_file.exists() else {
                'method': method, 'index': i, 'parameters': params, 'score': None, 'valid': False, 'reason': 'worker failure or timeout'}
            scores.append(entry)
            if entry['score'] is not None:
                candidates.append(entry)
        if not candidates:
            manifest.update(status='blocked', reason='all validation candidates failed: '+method)
            write_json(output/'manifest.json', manifest)
            write_json(output/'scores.json', scores)
            return 2
        chosen[method] = min(candidates, key=lambda x: (x['score'], x['index']))['parameters']
    write_json(output/'scores.json', scores)
    write_json(output/'reconstruction_selection.json', chosen)
    phase([(f'planning_{i:03d}', ['--planning-index', str(i), '--reconstruction-selection', str(output/'reconstruction_selection.json')]) for i in range(27)])
    planning_scores = []
    for i in range(27):
        path = output/'jobs'/f'planning_{i:03d}'/'planning_scores.json'
        planning_scores.append(json.loads(path.read_text())[0] if path.exists() else {'index': i, 'score': None, 'reason': 'worker failure or timeout'})
    write_json(output/'planning_scores.json', planning_scores)
    valid = [s for s in planning_scores if s['score'] is not None]
    if not valid:
        manifest.update(status='blocked', reason='all deployment validation candidates failed')
        write_json(output/'manifest.json', manifest)
        return 2
    rbmf = chosen['rbmf']
    selection = {'status': 'validation_selected', 'data_sha256': digest(source_data/'aligned_fields.npz'),
        'validation_dates': list(range(201301, 201313)), 'validation_seeds': [1001, 1002, 1003],
        'inference': rbmf['inference'], 'planning': min(valid, key=lambda s: (s['score'], s['index']))['parameters'],
        'comparison': {'rank': rbmf['rank'], 'method_rank': {m: p['rank'] for m, p in chosen.items()},
            'regularization': {m: p['regularization'] for m, p in chosen.items()},
            'method_inference': {m: p['inference'] for m, p in chosen.items()}},
        'search_space_sha256': digest(output/'search_space.json'), 'selection_rule': 'all 12 months x 3 seeds x 3 budgets; failed candidate infinite; first index tie'}
    write_json(output/'selection.json', selection)
    manifest.update(status='complete', wall_seconds=perf_counter()-begun)
    write_json(output/'manifest.json', manifest)
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    sys.exit(run(args.config.resolve(), args.output.resolve(), args.workers))
