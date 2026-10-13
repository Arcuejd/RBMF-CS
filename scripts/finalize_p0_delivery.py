"""Package immutable P0 evidence; no fitting or holdout field loading."""
import json,shutil,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from run_p0_diagnosis import digest,dump
out=ROOT/'outputs/p0_delivery_v1';out.mkdir(exist_ok=False)
scores=json.loads((ROOT/'outputs/comparison_validation_v1/scores.json').read_text())
selected=json.loads((ROOT/'outputs/comparison_validation_v1/reconstruction_selection.json').read_text())
audit=[]
for method in selected:
    rr=[r for r in scores if r['method']==method];valid=[r for r in rr if r['valid'] and r['score'] is not None]
    best=min(valid,key=lambda r:(r['score'],r['index']))
    audit.append({'method':method,'candidates':len(rr),'invalid':len(rr)-len(valid),'selected_index':best['index'],
        'selected_score':best['score'],'matches_archived_selection':best['parameters']==selected[method],
        'total_historical_seconds':sum(r['seconds'] for r in rr)})
assert all(r['matches_archived_selection'] for r in audit)
dump(out/'historical_selection_audit.json',audit)
for name,path in {'original_config.json':'outputs/comparison_formal_v1/config.json',
 'original_selection.json':'outputs/comparison_validation_v1/selection.json',
 'original_scores.json':'outputs/comparison_validation_v1/scores.json',
 'historical_requirements_freeze.txt':'outputs/comparison_validation_v1/requirements-freeze.txt',
 'tests_final.log':'outputs/p0_release_v1/tests_final.log',
 'report.md':'docs/p0_research_report_v1.md','holdout_protocol.md':'docs/p0_holdout_protocol_v1.md'}.items():
    shutil.copy2(ROOT/path,out/name)
with zipfile.ZipFile(out/'source.zip','w',zipfile.ZIP_DEFLATED) as z:
 for folder in ['src','scripts','configs','docs','tests']:
  for p in (ROOT/folder).rglob('*'):
   if p.suffix in ['.py','.md','.json'] and '__pycache__' not in p.parts:z.write(p,p.relative_to(ROOT))
files=[]
for folder in [p for p in (ROOT/'outputs').glob('p0_*') if p.is_dir()]:
 for p in sorted(folder.rglob('*')):
  if p.is_file():files.append({'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':digest(p)})
dump(out/'manifest.json',{'status':'research_complete_negative_rbmf_result_test_preparation_incomplete',
 'independent_test_performance_run':False,'promoted_rbmf_revision':None,'tests_passed':76,
 'performance_thresholds':{'relative_rmse_improvement_pct':5,'months_improved':8,'gain_4_to_12_pct':2},
 'files':files,'limitations':['No qualified paired independent test yet','Whole batch peak memory not measured',
 'Two initial development batches predate additional I/O guard; exact-date schema isolation applied throughout'],
 'source_independence':'unverified, reference products use in-situ observations',
 'visual_review':{'budget_curves':'readable, consistent axes, no smoothing','monthly_curves':'readable, all months retained'}})
print('Packaged',len(files),'files; historical candidates',sum(r['candidates'] for r in audit))
