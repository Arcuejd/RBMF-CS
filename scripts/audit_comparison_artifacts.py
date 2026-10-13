"""Read-only numerical/export checks of already completed comparison results.

No fitter, experiment runner, or plotting call is used. Visual review is a
separate human inspection of the rendered PDFs, recorded at final acceptance.
"""
import argparse
from collections import defaultdict
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean
import xml.etree.ElementTree as ET
from PIL import Image
from pypdf import PdfReader


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root, report, output):
    errors, checks = [], 0
    def check(ok, label):
        nonlocal checks
        checks += 1
        if not ok:
            errors.append(label)
    cfg, manifest = read(root/'config.json'), read(root/'manifest.json')
    verify = read(root/'verification.json')
    check(verify['passed'] and not verify['errors'], 'saved-prediction verification passed')
    check(verify['root_config_sha256'] == sha(root/'config.json'), 'verified config unchanged')
    check(verify['source_sha256'] == sha(root/'source.zip'), 'verified source unchanged')
    planned, ledger = read(root/'planned_cases.json'), read(root/'ledger.json')
    check({c['id'] for c in planned} == {c['id'] for c in ledger}, 'all planned cases recorded')
    figs = root/'figures'
    pm = read(figs/'plot_manifest.json')
    check(pm['source_manifest_sha256'] == sha(root/'manifest.json'), 'plot input manifest')
    check(pm['source_config_sha256'] == sha(root/'config.json'), 'plot input config')
    groups = defaultdict(list)
    with (figs/'metrics.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        groups[(row['branch'], row['method'], row['scenario'], row['requested_nodes'])].append(row)
    with (figs/'summary.csv').open(encoding='utf-8-sig', newline='') as stream:
        summaries = list(csv.DictReader(stream))
    summary_lookup = {}
    for summary in summaries:
        key = tuple(summary[k] for k in ['branch', 'method', 'scenario', 'requested_nodes'])
        group = groups[key]
        good = [r for r in group if r['converged'] == 'True' and r['actual_nodes'] == r['requested_nodes'] and r['rmse_c'] != '']
        check(len(good) == int(summary['converged_same_count_cases']), 'summary success count '+str(key))
        check(len(group) == int(summary['recorded_cases']), 'summary record count '+str(key))
        check(sum(r.get('feasible') == 'True' for r in group) == int(summary['feasible_cases']), 'summary feasible count '+str(key))
        for field in ['rmse_c', 'mae_c', 'nrmse', 'cost', 'mission_energy_j', 'preprocessing_seconds',
                      'selection_seconds', 'reconstruction_seconds', 'latent_coverage95', 'future_coverage95',
                      'latent_width95_c', 'future_width95_c']:
            values = [float(r[field]) for r in good if r.get(field) not in (None, '')]
            expected = fmean(values) if values else None
            actual = float(summary[field]) if summary[field] != '' else None
            check(actual is None if expected is None else actual is not None and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), 'summary mean '+str(key)+' '+field)
        summary_lookup[key] = summary
    pairs = read(figs/'paired_statistics.json')
    for pair in pairs:
        check(pair.get('environment_p_value') is None and pair.get('holm_adjusted_p_value') is None, 'no unsupported environment test')
        if len(pair['paired_seeds']) == len(cfg['comparison']['seeds']) and pair['paired_mean_difference_c'] is not None:
            key = (pair['branch'], pair['method'], pair['scenario'], str(pair['budget']))
            refkey = (pair['branch'], pair['reference'], pair['scenario'], str(pair['budget']))
            expected = float(summary_lookup[key]['rmse_c'])-float(summary_lookup[refkey]['rmse_c'])
            check(abs(expected-pair['paired_mean_difference_c']) < 1e-12, 'paired sign and estimate '+str(key))
    exports = []
    for name in [Path(p).stem for p in pm['figures']]:
        paths = [figs/(name+'.'+ext) for ext in ['pdf', 'svg', 'png']]
        for path in paths:
            check(path.is_file() and path.stat().st_size > 0, 'export exists '+path.name)
        pdf = PdfReader(paths[0])
        check(len(pdf.pages) == 1 and bool(pdf.pages[0].extract_text()), 'PDF page/text '+name)
        check(ET.parse(paths[1]).getroot().tag.endswith('svg'), 'SVG parses '+name)
        with Image.open(paths[2]) as im:
            check(min(im.size) >= 1000, 'PNG resolution '+name)
            dpi = im.info.get('dpi', (0, 0))
            check(all(abs(d-300) < .1 for d in dpi), 'PNG 300dpi '+name)
        exports += [{'path': str(p.relative_to(root)), 'sha256': sha(p), 'bytes': p.stat().st_size} for p in paths]
    check(report.is_file(), 'final report exists')
    result = {'passed': not errors, 'checks': checks, 'errors': errors,
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'audit_source_sha256': sha(Path(__file__)), 'run_status': manifest['status'],
        'planned_cases': len(planned), 'recorded_cases': len(ledger), 'requested_budget_metric_rows': len(rows),
        'summary_rows': len(summaries), 'paired_comparisons': len(pairs), 'figures': len(pm['figures']),
        'report_sha256': sha(report), 'verification_sha256': sha(root/'verification.json'),
        'exports': exports, 'scope': 'saved artifact aggregation, pairing and export integrity; visual review separate'}
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'exports'}, indent=2))
    return not errors


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    raise SystemExit(0 if audit(a.root, a.report, a.output) else 2)
