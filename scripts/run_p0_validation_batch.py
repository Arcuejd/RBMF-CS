"""Authorized finite P0 validation, sequential bounded child runs."""
import json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'outputs/p0_validation_v1'
out.mkdir(exist_ok=False)
start=time.perf_counter();ledger=[]
for scenario in ['clean','gaussian']:
    for seed in [1001,1002,1003]:
        name=f'{scenario}_{seed}';begun=time.perf_counter()
        cmd=[sys.executable,str(ROOT/'scripts/run_p0_diagnosis.py'),'--output',str(out/name),
            '--months',*[str(m) for m in range(1,13)],'--scenario',scenario,'--seed',str(seed)]
        with (out/f'{name}.log').open('w',encoding='utf-8') as log:
            try:
                result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=300)
                row={'name':name,'exit_code':result.returncode}
            except subprocess.TimeoutExpired: row={'name':name,'exit_code':None,'failure':'timeout_300s'}
        row['seconds']=time.perf_counter()-begun;ledger.append(row)
        (out/'ledger.json').write_text(json.dumps(ledger,indent=2))
        print(row,flush=True)
(out/'wall.json').write_text(json.dumps({'wall_seconds':time.perf_counter()-start}))
