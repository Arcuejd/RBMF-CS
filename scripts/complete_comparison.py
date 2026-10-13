"""Sequential completion of the authorized experiment, with durable stage logs.

Every stage uses its own directory. Stops on failed verification or storage gate;
scientific failures inside a completed batch remain reportable results.
"""
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/comparison_completion_v1'


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    state = {'status': 'running', 'stages': []}
    def save():
        target = OUT/'status.json'
        temp = OUT/'status.tmp'
        temp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
        temp.replace(target)
    def stage(name, args, allowed=(0,)):
        state['active_stage'] = name
        save()
        start = perf_counter()
        with (OUT/(name+'.log')).open('w', encoding='utf-8') as log:
            done = subprocess.run([sys.executable, *args], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        state['stages'].append({'name': name, 'command': [sys.executable, *args],
                                'returncode': done.returncode, 'wall_seconds': perf_counter()-start})
        if done.returncode not in allowed:
            state['status'] = 'blocked'
            save()
            raise RuntimeError(f'{name} failed; see durable stage log')
        save()
    pilot = 'outputs/comparison_selected_pilot_v1'
    repeat = 'outputs/comparison_selected_repeat_v1'
    tuning = 'outputs/comparison_validation_v1'
    formal = 'outputs/comparison_formal_v1'
    figs = formal+'/figures'
    selection = tuning+'/selection.json'
    stage('verify_pilot', ['scripts/verify_comparison.py', pilot])
    stage('repeat_pilot', ['scripts/run_comparison.py', '--mode', 'pilot', '--config', 'configs/comparison_2013_v1.json',
        '--selection', selection, '--full-pilot-budgets', '--workers', '6', '--output', repeat], allowed=(0, 2))
    stage('verify_repeat', ['scripts/verify_comparison.py', pilot, '--repeat', repeat])
    stage('freeze_scale', ['scripts/freeze_comparison_scale.py', pilot, '--output', 'outputs/comparison_scale_2013_v1.json'])
    stage('formal_batch', ['scripts/run_comparison.py', '--mode', 'formal', '--config', 'configs/comparison_2013_v1.json',
        '--selection', selection, '--pilot-verification', pilot+'/repeat_verification.json',
        '--workers', '6', '--output', formal], allowed=(0, 2))
    # A preflight-blocked directory has no ledger and fails verification; never skip gates.
    stage('verify_formal', ['scripts/verify_comparison.py', formal])
    stage('saved_diagnosis', ['scripts/diagnose_saved_comparison.py', formal, '--output', formal+'/saved_diagnosis.json'])
    stage('plots', ['scripts/plot_comparison.py', formal, '--output', figs])
    stage('report', ['scripts/report_comparison_formal.py', formal, '--figures', figs, '--tuning', tuning,
        '--output', 'docs/comparison_formal_report.md'])
    state['status'] = 'complete_pending_visual_review'
    state['active_stage'] = None
    save()
    print(OUT/'status.json')


if __name__ == '__main__':
    main()
