"""Read-only ZIP audit, no model imports or performance metrics."""
import hashlib,json,re,shutil,sys,time,zipfile
from pathlib import Path
import numpy as np
from netCDF4 import Dataset
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from audit_local_dataset import metadata
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while b:=f.read(1024*1024): h.update(b)
    return h.hexdigest()

def dump(p,v):
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2,default=lambda x: x.tolist() if isinstance(x,np.ndarray) else x.item()),encoding='utf-8')

def run(out):
    out.mkdir(parents=True,exist_ok=False); (out/'sealed').mkdir()
    started=time.perf_counter()
    source=Path('D:/Dataset')
    with np.load(ROOT/'outputs/comparison_validation_v1/inputs/aligned_fields.npz') as z:
        ids=z['original_grid_flat_ids'];coords=z['coordinates_latlon']
    archives=[];members=[];arrays=[];dates=[];meta={}
    products={'temperature':source/'Temperature_Anomaly_IAPv4.2_gridded_data_2000_2026',
              'salinity':source/'Salinity_IAPdata_v2.2_2000_2026'}
    for product,folder in products.items():
        for path in sorted(folder.glob('*.zip')):
            print('Audit',path.name,flush=True)
            archive_hash=digest(path)
            with zipfile.ZipFile(path) as z:
                entries=[v for v in z.infolist() if v.filename.endswith('.nc')]
                archives.append({'product':product,'path':str(path),'bytes':path.stat().st_size,
                    'sha256':archive_hash,'uncompressed_bytes':sum(v.file_size for v in entries),'months':len(entries)})
                for v in entries:
                    match=re.search(r'year_(\d{4})_month_(\d{2})',v.filename)
                    if not match: raise ValueError(v.filename)
                    date=int(''.join(match.groups()))
                    exposure='training' if date<201301 else 'development_validation' if date<201401 else 'exposed_exploratory' if date<201501 else 'quality_metadata_only_user_attested_no_performance'
                    record={'product':product,'date':date,'archive':str(path),'member':v.filename,
                        'crc32':v.CRC,'bytes':v.file_size,'exposure':exposure,'quality_audited_this_run':False}
                    # Salinity is inventory only; one metadata specimen per archive.
                    read=(product=='temperature' and date>=201501) or v==entries[0]
                    if read:
                        content=z.read(v.filename);record['member_sha256']=hashlib.sha256(content).hexdigest()
                        with Dataset('memory.nc',memory=content) as ds:
                            if v==entries[0]: meta[path.name]=metadata(ds)
                            if product=='temperature' and date>=201501:
                                depth=np.asarray(ds['depth_std'][:]); k=np.flatnonzero(depth==50.)
                                if len(k)!=1: raise ValueError('no exact 50m')
                                if int(ds['time'][0])!=date: raise ValueError('time mismatch')
                                lat=np.asarray(ds['lat'][:]);lon=np.asarray(ds['lon'][:]); yy,xx=np.meshgrid(lat,lon,indexing='ij')
                                expected=np.column_stack((yy.ravel()[ids],xx.ravel()[ids]))
                                # Existing coordinates may use signed longitudes.
                                grid_ok=np.allclose(expected[:,0],coords[:,0]) and np.allclose((expected[:,1]-coords[:,1])%360,0)
                                if not grid_ok: raise ValueError('frozen grid mismatch')
                                var=ds['temp']; unit=str(var.units)
                                if 'celsius' not in unit.lower(): raise ValueError('unit')
                                a=np.ma.asarray(var[:,:,int(k[0])]).astype(float).filled(np.nan).ravel()[ids]
                                for key in ['_FillValue','missing_value']:
                                    if key in var.ncattrs(): a[a==float(var.getncattr(key))]=np.nan
                                a[a==999]=np.nan
                                valid=np.isfinite(a)
                                record.update({'quality_audited_this_run':True,'unit':unit,'depth_m':50.,
                                    'grid_matches_frozen_domain':bool(grid_ok),'finite_nodes':int(valid.sum()),
                                    'domain_nodes':len(ids),'quality_min_anomaly_c':float(np.nanmin(a)),
                                    'quality_max_anomaly_c':float(np.nanmax(a)),
                                    'qc_variables':[n for n in ds.variables if 'qc' in n.lower() or 'flag' in n.lower()],
                                    'paired_lf_ice_qualified':False,'absolute_conversion_qualified':False})
                                arrays.append(a);dates.append(date)
                    members.append(record)
    order=np.argsort(dates)
    np.savez_compressed(out/'sealed/temperature_anomaly_50m_candidates.npz',dates=np.asarray(dates)[order],
        anomaly_c=np.asarray(arrays)[order],original_grid_flat_ids=ids,coordinates_latlon=coords)
    dump(out/'archive_metadata.json',meta)
    # Metadata/file inventories only, never read candidate prediction arrays.
    ancillary=[]
    for folder in [source/'RBMF-CS_polar_sample',source/'RBMF-CS_polar_validation_2013_20261011']:
        for category in ['oras5','ice','argo','itp']:
            for p in (folder/category).rglob('*'):
                if p.is_file(): ancillary.append({'category':category,'path':str(p),'bytes':p.stat().st_size})
    dump(out/'ancillary_inventory.json',ancillary)
    years=[]
    for year in range(2000,2027):
        mm=[r for r in members if r['product']=='temperature' and r['date']//100==year]
        years.append({'year':year,'months':len(mm),'complete_year':len(mm)==12,
            'exposure':mm[0]['exposure'],'eligible_for_test':False,
            'status':'historically_exposed' if year<2015 else 'HF_quality_candidate_pending_LF_ice_baseline_version',
            'finite_all_frozen_nodes':all(r.get('finite_nodes')==len(ids) for r in mm) if year>=2015 else None})
    sealed=out/'sealed/temperature_anomaly_50m_candidates.npz'
    manifest={'status':'sealed_quality_candidates_not_test_ready','performance_evaluation_run':False,
        'archives':archives,'members':members,'years':years,
        'sealed_artifact':{'path':str(sealed.resolve()),'sha256':digest(sealed)},
        'exposure_evidence':['User explicitly attests no performance evaluation of new archive years',
            'docs/local_dataset_assessment.md: 2000-2026 quality/missingness audit only',
            'docs/comparison_protocol.md and formal_report: 2000-2012 train, 2013 development, 2014 exposed'],
        'source_independence':'not established; IAP and ORAS5 incorporate in-situ observations',
        'blockers':['candidate ORAS5 temperature/salinity and sea-ice not yet paired locally',
            'v4.2 anomaly climatology/version continuity not certified by local file metadata',
            'Argo/ITP identity-level assimilation cross-check absent',
            'final method and practical threshold not frozen'],
        'storage':{'free_before_or_after_bytes':shutil.disk_usage(out).free,
                   'max_single_temperature_member_bytes':max(r['bytes'] for r in members if r['product']=='temperature'),
                   'full_extraction_performed':False},'wall_seconds':time.perf_counter()-started}
    dump(out/'manifest.json',manifest)
    dump(out/'seal.json',{'manifest_sha256':digest(out/'manifest.json'),'sealed_sha256':digest(sealed),
        'release_authorized':False,'required':'final method hash, protocol hash, complete pairing and explicit evaluation release'})
    print('sealed',len(dates),'months',flush=True)

if __name__=='__main__':run(Path(sys.argv[1]))
