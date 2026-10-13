"""Final replay and seal verification. Never evaluates holdout performance."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import ctypes,json,sys,time,zipfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from run_p0_diagnosis import digest,dump,load_development
from rbmf_cs.model import VBConfig
from rbmf_cs.p0_diagnostics import DiagnosticRBMF
out=ROOT/'outputs/p0_release_v1';out.mkdir(exist_ok=False)
start=time.perf_counter()
d=load_development(ROOT/'outputs/comparison_validation_v1/inputs');art=dict(np.load(ROOT/'outputs/comparison_validation_v1/inputs/training_artifacts.npz'))
design=json.loads((ROOT/'outputs/comparison_validation_v1/jobs/rbmf_000/validation_designs.json').read_text())[0]
ids=design['ids'];y=np.asarray(design['y']);scale=float(art['scale_c']);fixed=scale**2/(.02**2/3)
c=VBConfig(max_iter=3000,tolerance=1e-6,convergence='factors',a_delta=2,b_delta=20,nu=3)
m=DiagnosticRBMF(art['basis'][:,:32],c,fixed).fit(d['validation_lf_model'][0],y[ids],indices=ids)
with np.load(ROOT/'outputs/p0_validation_v1/gaussian_1001/predictions_model.npz') as z:
    difference=float(abs(z['201301_12_known_noise_conditional']-m.predict()).max())
assert difference<1e-12
assert m.diagnostics_['initial_factors']['tau']=={'constant_precision':fixed}
seal=json.loads((ROOT/'outputs/p0_holdout_v1/seal.json').read_text());manifest=ROOT/'outputs/p0_holdout_v1/manifest.json'
assert digest(manifest)==seal['manifest_sha256']
sealed=ROOT/'outputs/p0_holdout_v1/sealed/temperature_anomaly_50m_candidates.npz'
assert digest(sealed)==seal['sealed_sha256'] and seal['release_authorized'] is False
class Memory(ctypes.Structure):
    _fields_=[('cb',ctypes.c_ulong),('PageFaultCount',ctypes.c_ulong)]+[(n,ctypes.c_size_t) for n in ['PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage']]
mem=Memory();mem.cb=ctypes.sizeof(mem)
ctypes.windll.kernel32.GetCurrentProcess.restype=ctypes.c_void_p
ctypes.windll.psapi.GetProcessMemoryInfo.argtypes=[ctypes.c_void_p,ctypes.POINTER(Memory),ctypes.c_ulong]
ok=ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(),ctypes.byref(mem),mem.cb)
dump(out/'verification.json',{'passed':True,'fixed_noise_initial_metadata_corrected':True,
 'numeric_replay_max_difference':difference,'seal_hashes_pass':True,'test_release_authorized':False,
 'replay_wall_seconds':time.perf_counter()-start,'replay_peak_working_set_bytes':mem.PeakWorkingSetSize if ok else None,
 'memory_scope':'one representative fixed-noise replay plus verification, not measured whole batch peak',
 'tests':{'command':'.venv/Scripts/python -m unittest discover -s tests -v','passed':75},
 'note':'historical batch initial_factors.tau preceded fixed substitution; factor_trace[0] and every computed ELBO use constant. Final subclass fixes initial metadata only; exact replay verified.'})
dump(out/'decision.json',{'successful_rbmf_revision':None,'independent_test_recommended_now':False,
 'reasons':['No deployable RBMF single-mechanism candidate meets preregistered improvement and budget thresholds',
            'Paired test LF/ice, version baseline audit and final method freeze incomplete'],
 'retained_numerical_variant':'factor_stop; stricter convergence reporting, no accuracy claim',
 'retained_prior_candidate':'ard_b1; negative practical-effect result; not promoted',
 'no_combined_model_search':True})
with zipfile.ZipFile(out/'source.zip','w',zipfile.ZIP_DEFLATED) as z:
 for folder in ['src','scripts','configs','docs','tests']:
  for p in (ROOT/folder).rglob('*'):
   if p.suffix in ['.py','.md','.json'] and '__pycache__' not in p.parts:z.write(p,p.relative_to(ROOT))
dump(out/'manifest.json',{'source_sha256':digest(out/'source.zip'),
    'protocols':{str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'docs').glob('p0*.md')},
    'original_model_sha256':digest(ROOT/'src/rbmf_cs/model.py'),
    'training_sha256':digest(ROOT/'outputs/comparison_validation_v1/inputs/training_artifacts.npz'),
    'selection_sha256':digest(ROOT/'outputs/comparison_validation_v1/selection.json')})
print((out/'verification.json').read_text())
