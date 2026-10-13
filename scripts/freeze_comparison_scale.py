"""Freeze batch size from pilot resource use, never from reconstruction errors."""
import argparse
import json
import shutil
from pathlib import Path
from run_comparison import digest, write_json


def run(pilot, output):
    config = json.loads((pilot/'config.json').read_text(encoding='utf-8'))
    manifest = json.loads((pilot/'manifest.json').read_text(encoding='utf-8'))
    verification = json.loads((pilot/'repeat_verification.json').read_text(encoding='utf-8'))
    planned = json.loads((pilot/'planned_cases.json').read_text(encoding='utf-8'))
    if not verification['passed'] or not verification['repeat'] or manifest['parameter_status'] != 'validation_selected':
        raise ValueError('validation-selected, repeat-verified pilot required')
    if config['comparison']['budgets'] != [4, 6, 8, 10, 12]:
        raise ValueError('full-budget pilot required for this scale estimate')
    case_bytes = sum(p.stat().st_size for p in (pilot/'cases').rglob('*') if p.is_file())
    ratio = (30*4*10)/len(planned)
    free = shutil.disk_usage(pilot).free
    expected = int(case_bytes*ratio)
    result = {'status': 'frozen', 'seeds': list(range(30)), 'dates': [201401, 201404, 201407, 201410],
        'budgets': [4, 6, 8, 10, 12], 'scenarios': config['comparison']['scenarios'], 'planned_cases': 1200,
        'workers': manifest['workers'], 'pilot_cases': len(planned), 'pilot_wall_seconds': manifest['wall_seconds'],
        'estimated_wall_seconds': manifest['wall_seconds']*ratio, 'pilot_case_bytes': case_bytes,
        'estimated_case_bytes': expected, 'disk_free_bytes': free,
        'pilot_config_sha256': digest(pilot/'config.json'), 'pilot_repeat_verification_sha256': digest(pilot/'repeat_verification.json'),
        'selection_sha256': manifest['selection_sha256'],
        'selection_basis': 'runtime and storage only; no error values read',
        'uncertainty': 'other months may differ in convergence and feasible candidates; estimate is not a guarantee'}
    if expected*1.3 + 2*1024**3 > free:
        result['status'] = 'storage_review_required'
    write_json(output, result)
    print(json.dumps(result, indent=2))
    return result['status'] == 'frozen'


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('pilot', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    raise SystemExit(0 if run(a.pilot, a.output) else 2)
