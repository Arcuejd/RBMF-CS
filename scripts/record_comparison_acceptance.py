"""Record completed numerical and visual acceptance; never run experiments."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p, value):
    p.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')


def run(root, report, completion, tuning, pilot, confirmed, visual_only=False, revision=False):
    if not confirmed:
        raise ValueError('actual rendered-PDF visual inspection must precede this record')
    existing = (root/'acceptance.json').exists()
    if existing and not revision:
        raise FileExistsError('final acceptance already exists; preserve it')
    state = read(completion/'status.json')
    allowed = ['complete_pending_visual_review']+(['complete'] if revision else [])
    if state['status'] not in allowed or state['active_stage'] is not None:
        raise ValueError('pipeline has not finished')
    artifacts = read(root/'artifact_verification.json')
    numerical = read(root/'verification.json')
    validation = read(tuning/'verification.json')
    repeated = read(pilot/'repeat_verification.json')
    if not all(v['passed'] and not v['errors'] for v in [artifacts, numerical, validation, repeated]):
        raise ValueError('verification failed')
    if existing:
        previous = read(root/'acceptance.json')
        if previous['source_archive_sha256'] != sha(root/'source.zip'):
            raise ValueError('numerical source archive changed')
        history = root/'acceptance_history'/datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S')
        history.mkdir(parents=True, exist_ok=False)
        for name in ['acceptance.json', 'final_report.md', 'figures_bundle.zip', 'visual_review.json']:
            shutil.copyfile(root/name, history/name)
        shutil.copytree(root/'analysis_source', history/'analysis_source')
    names = [Path(p).stem for p in read(root/'figures/plot_manifest.json')['figures']]
    rendered = []
    for name in names:
        image = root/'figure_review'/(name+'.png')
        if not image.exists():
            raise FileNotFoundError(image)
        rendered.append({'figure': name, 'pdf_sha256': sha(root/'figures'/(name+'.pdf')),
                         'rendered_png_sha256': sha(image)})
    write(root/'visual_review.json', {'passed': True, 'review_method': 'each actual PDF rendered and visually inspected',
        'checks': ['titles/labels readable', 'no clipped legends', 'shared spatial field color limits',
                   'shared symmetric spatial error limits', 'prespecified spatial sample', 'node markers readable'],
        'interpretation': 'zoomed deployment axes show tiny differences; not a practical superiority claim',
        'figures': rendered})
    if visual_only:
        print(root/'visual_review.json')
        return
    if sha(report) != artifacts['report_sha256']:
        raise ValueError('final report differs from independently audited report')
    # Capture final reporting sources separately from the immutable numerical worker snapshot.
    analysis = root/'analysis_source'
    analysis.mkdir(exist_ok=revision)
    workspace = Path(__file__).resolve().parents[1]
    sources = ['scripts/report_comparison_formal.py', 'scripts/plot_comparison.py',
        'scripts/diagnose_saved_comparison.py', 'scripts/audit_comparison_artifacts.py',
        'scripts/record_comparison_acceptance.py', 'src/rbmf_cs/comparison_statistics.py']
    hashes = {}
    for source in sources:
        target = analysis/source
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(workspace/source, target)
        hashes[source] = sha(target)
    shutil.copyfile(report, root/'final_report.md')
    with zipfile.ZipFile(root/'figures_bundle.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for p in sorted((root/'figures').iterdir()):
            if p.is_file():
                archive.write(p, 'figures/'+p.name)
        archive.write(report, 'comparison_formal_report.md')
    acceptance = {'status': 'complete', 'accepted_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'planned_cases': artifacts['planned_cases'], 'recorded_cases': artifacts['recorded_cases'],
        'numerical_verification_checks': numerical['checks'], 'recomputed_predictions': numerical['recomputed_predictions'],
        'artifact_verification_checks': artifacts['checks'], 'visual_review_passed': True, 'figures': len(names),
        'evaluation_label': 'exploratory_2014_no_unseen_paired_test_data',
        'reference_field': 'IAP gridded temperature; simulated sensor acquisitions',
        'validation_verification_sha256': sha(tuning/'verification.json'),
        'pilot_repeat_verification_sha256': sha(pilot/'repeat_verification.json'),
        'source_archive_sha256': sha(root/'source.zip'), 'final_analysis_sources': hashes,
        'report_sha256': sha(report), 'visual_review_sha256': sha(root/'visual_review.json'),
        'artifact_verification_sha256': sha(root/'artifact_verification.json'),
        'figures_bundle_sha256': sha(root/'figures_bundle.zip'),
        'note': 'all completed cases reused; no experiments rerun during final review'}
    write(root/'acceptance.json', acceptance)
    if not (completion/'status_pre_visual_review.json').exists():
        shutil.copyfile(completion/'status.json', completion/'status_pre_visual_review.json')
    state.update(status='complete', active_stage=None, acceptance_record=str((root/'acceptance.json').resolve()),
                 final_report=str(report.resolve()), visual_review='passed')
    write(completion/'status.json', state)
    print(json.dumps(acceptance, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('--report', required=True, type=Path)
    p.add_argument('--completion', required=True, type=Path)
    p.add_argument('--tuning', required=True, type=Path)
    p.add_argument('--pilot', required=True, type=Path)
    p.add_argument('--confirm-visual-review', action='store_true')
    p.add_argument('--visual-only', action='store_true')
    p.add_argument('--revision', action='store_true')
    a = p.parse_args()
    run(a.root, a.report, a.completion, a.tuning, a.pilot, a.confirm_visual_review, a.visual_only, a.revision)
