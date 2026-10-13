"""Repeat the closed-loop demo in fresh processes; compare complete decisions."""
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]


def without_runtime(value):
    if isinstance(value,dict):
        return {k:without_runtime(v) for k,v in value.items() if not k.endswith('runtime_seconds')}
    if isinstance(value,list):
        return [without_runtime(v) for v in value]
    return value


def main():
    directories=[ROOT/'outputs/closed_loop_reproducibility/run_a',ROOT/'outputs/closed_loop_reproducibility/run_b']
    for directory in directories:
        process=subprocess.run([sys.executable,str(ROOT/'scripts/run_closed_loop_demo.py'),
            '--output',str(directory)],capture_output=True,text=True)
        directory.mkdir(parents=True,exist_ok=True)
        (directory/'process.log').write_text(process.stdout+process.stderr,encoding='utf-8')
        if process.returncode:
            raise RuntimeError(f'demo failed: {directory}/process.log')
    manifest=json.loads((directories[0]/'run.json').read_text())
    arrays=['synthetic_data.npz','training_artifacts.npz']+[s['posterior'] for s in manifest['snapshots']]
    checked=0
    for filename in arrays:
        with np.load(directories[0]/filename) as a,np.load(directories[1]/filename) as b:
            assert set(a.files)==set(b.files)
            for key in a.files:
                np.testing.assert_allclose(a[key],b[key],rtol=1e-12,atol=1e-12,equal_nan=True)
                checked+=1
    json_files=['deployment.json','acquisitions.json']+[s['inference'] for s in manifest['snapshots']]
    for filename in json_files:
        a=without_runtime(json.loads((directories[0]/filename).read_text()))
        b=without_runtime(json.loads((directories[1]/filename).read_text()))
        assert a==b, f'mismatch: {filename}'
    report={'success':True,'fresh_processes':2,'array_comparisons':checked,
            'json_comparisons':len(json_files),'selected':manifest['selected'],
            'ignored':'runtime_seconds fields; timestamp-bearing run manifest not compared'}
    (ROOT/'outputs/closed_loop_reproducibility/report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
