"""Bounded launcher: use measured pilot, preserve partial output on timeout."""
import json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'outputs/p0_structure_launch_v1';out.mkdir(exist_ok=False)
pilot=json.loads((ROOT/'outputs/p0_structure_pilot_v1/resources.json').read_text())
# Conservative total-runs extrapolation; one-time source snapshot overestimates.
estimate=pilot['wall_seconds']*60
if estimate>300 or pilot['peak_working_set_bytes']>512*1024**2:
    raise RuntimeError('pilot resource gate exceeds bounded scope')
(out/'resource_gate.json').write_text(json.dumps({'pilot':pilot,'estimated_wall_seconds':estimate,
    'timeout_seconds':300,'estimated_output_bytes':pilot['output_bytes']*60,'approved_scope':'one finite structural round within previous batch size'},indent=2))
started=time.perf_counter()
with (out/'full.log').open('w',encoding='utf-8') as f:
    try:
        p=subprocess.run([sys.executable,str(ROOT/'scripts/run_p0_structure.py'),'--phase','full','--output',str(ROOT/'outputs/p0_structure_validation_v1')],
                         stdout=f,stderr=subprocess.STDOUT,timeout=300)
        result={'exit_code':p.returncode,'timeout':False}
    except subprocess.TimeoutExpired:result={'exit_code':None,'timeout':True}
result['wall_seconds']=time.perf_counter()-started
(out/'status.json').write_text(json.dumps(result,indent=2));print(result,flush=True)
