"""Metadata-only ancillary time/region inventory, no prediction imports."""
import csv,json,zipfile,re,sys
from datetime import datetime,timedelta
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'outputs'/('p0_ancillary_v2' if len(sys.argv)<2 else sys.argv[1]);out.mkdir(exist_ok=False)
base=Path('D:/Dataset/RBMF-CS_polar_sample')
profiles=[];errors=[];skipped=[]
for p in sorted((base/'itp').rglob('*.zip')):
 with zipfile.ZipFile(p) as z:
  for n in z.namelist():
   if not re.fullmatch(r'itp\d+grd\d+\.dat',n):
    skipped.append({'archive':str(p),'member':n,'reason':'not gridded temperature profile'});continue
   try:
    with z.open(n) as f:
     header=f.readline().decode('ascii');fields=f.readline().decode('ascii').split()
    year,day,lon,lat=int(fields[0]),float(fields[1]),float(fields[2]),float(fields[3])
    date=datetime(year,1,1)+timedelta(days=day-1);signed=(lon+180)%360-180
    profiles.append({'archive':str(p),'member':n,'month':int(date.strftime('%Y%m')),'latitude':lat,'longitude':signed,
      'within_current_region':70<=lat<=80 and -20<=signed<=20,'exposure':'header_quality_only_this_run',
      'depth_50m_qc_audited':False,'assimilation_independence':'unknown'})
   except Exception as exc:errors.append({'archive':str(p),'member':n,'error':repr(exc)})
inventory=json.loads((ROOT/'outputs/p0_holdout_v1/ancillary_inventory.json').read_text())
rows=[]
for year in range(2015,2027):
 for month in range(1,5 if year==2026 else 13):
  date=year*100+month
  def found(kind,prefix):return [r['path'] for r in inventory if r['category']==kind and prefix in Path(r['path']).name and str(date) in Path(r['path']).name]
  rr=[r for r in profiles if r['month']==date]
  rows.append({'month':date,'iap_temperature_quality_candidate':True,
    'oras_temperature_files':len(found('oras5','votemper')),'oras_salinity_files':len(found('oras5','vosaline')),
    'seaice_files':len(found('ice','sic_')),'itp_header_profiles_all_regions':len(rr),
    'itp_header_profiles_current_region':sum(r['within_current_region'] for r in rr),
    'argo_candidate_years':'not downloaded in existing 2014-selected inventory',
    'eligible':False})
with (out/'monthly_gaps.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(out/'itp_headers.json').write_text(json.dumps(profiles,indent=2))
summary={'parsed_itp_headers':len(profiles),'parse_errors':errors,'excluded_nonprofile_members':skipped,
 'all_year_counts':dict(sorted(Counter(r['month']//100 for r in profiles).items())),
 'candidate_year_counts':dict(sorted(Counter(r['month']//100 for r in profiles if r['month']>=201501).items())),
 'candidate_current_region_profiles':sum(r['month']>=201501 and r['within_current_region'] for r in profiles),
 'note':'headers only; 50m QC not established; no performance evaluation or assimilation independence claim'}
(out/'summary.json').write_text(json.dumps(summary,indent=2));print(summary)
