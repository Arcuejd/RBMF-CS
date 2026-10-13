"""Execute two fresh demo processes and compare numerical outputs, not timing."""
import json, subprocess, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
paths=[ROOT/'outputs/reproducibility/run_a',ROOT/'outputs/reproducibility/run_b']
for path in paths:
    result=subprocess.run([sys.executable,str(ROOT/'scripts/run_synthetic_demo.py'),
                           '--config',str(ROOT/'configs/synthetic_demo.json'),'--output',str(path)],
                          capture_output=True,text=True)
    path.mkdir(parents=True,exist_ok=True)
    (path/'process.log').write_text(result.stdout+result.stderr,encoding='utf-8')
    if result.returncode:
        raise RuntimeError(f'demo failed: {path}/process.log')
checks=[]
for left in sorted(paths[0].glob('*.npz')):
    right=paths[1]/left.name
    with np.load(left) as a,np.load(right) as b:
        if set(a.files)!=set(b.files): raise AssertionError(left.name)
        for key in a.files:
            np.testing.assert_allclose(a[key],b[key],rtol=1e-12,atol=1e-12,equal_nan=True)
            checks.append(f'{left.name}:{key}')
for left in sorted(paths[0].glob('fit_*.json')):
    a=json.loads(left.read_text()); b=json.loads((paths[1]/left.name).read_text())
    if a['factors']!=b['factors']: raise AssertionError(f'factor mismatch: {left.name}')
    if a['diagnostics']['history']!=b['diagnostics']['history']: raise AssertionError(f'history mismatch: {left.name}')
    checks.append(f'{left.name}:factors_and_history')
report={'success':True,'fresh_processes':2,'numerical_rtol':1e-12,'numerical_atol':1e-12,
        'ignored_fields':['wallclock timestamps','runtime_seconds'], 'checks':checks}
(ROOT/'outputs/reproducibility/report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(f'Reproducibility passed: {len(checks)} comparisons across two fresh processes.')
