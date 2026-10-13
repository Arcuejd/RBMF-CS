"""Download and audit 2013 LF/ice, reusing frozen 2000-2012 preprocessing.

This script never selects parameters or reads test field arrays. All new inputs,
aligned arrays and audits are saved in a separate project directory.
"""
import argparse
import calendar
import concurrent.futures
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import zipfile

import numpy as np
from netCDF4 import Dataset, num2date

import prepare_polar_sample as fetch
from align_polar_sample import digest, read_float, layer, oras_fields, ice_fields

ROOT=Path(__file__).resolve().parents[1]
MONTHS=list(range(201301,201313))
DEFAULT_OUTPUT=Path('D:/Dataset/RBMF-CS_polar_validation_2013_20261011')
ORAS='https://icdc.cen.uni-hamburg.de/thredds/'
ICE='https://noaadata.apps.nsidc.org/NOAA/G02202_V6/'


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')


def download(output):
    fetch.OUT=output
    catalogs=[(ORAS+f'catalog/ftpthredds/EASYInit/oras5/r1x1/{v}/opa0/catalog.xml',f'catalogs/oras_{v}.xml') for v in ['votemper','vosaline']]
    catalogs += [(ORAS+'catalog/ftpthredds/EASYInit/oras5/r1x1/LSM_r1x1/catalog.xml','catalogs/oras_masks.xml'),
                 (ICE+'north/monthly/','catalogs/ice_monthly.html'),
                 (ICE+'ancillary/','catalogs/ice_ancillary.html'),
                 (ORAS+'fileServer/ftpthredds/EASYInit/oras5/r1x1/Correction_to_ORAS5_r1x1_files.pdf','catalogs/oras_mask_correction.pdf')]
    records=[fetch.download(url,path) for url,path in catalogs]
    jobs=[]
    for variable in ['votemper','vosaline']:
        catalog=(output/f'catalogs/oras_{variable}.xml').read_text()
        for month in MONTHS:
            name=f'{variable}_ORAS5_1m_{month}_r1x1.nc'
            if name not in catalog: raise ValueError(f'Official catalog missing {name}')
            jobs.append((ORAS+f'fileServer/ftpthredds/EASYInit/oras5/r1x1/{variable}/opa0/{name}',f'oras5/{name}'))
    ice_catalog=(output/'catalogs/ice_monthly.html').read_text()
    for month in MONTHS:
        name=f'sic_psn25_{month}_F17_v06r00.nc'
        if name not in ice_catalog: raise ValueError(f'Official catalog missing {name}')
        jobs.append((ICE+'north/monthly/'+name,'ice/'+name))
    name='tmask_r1x1.nc'
    if name not in (output/'catalogs/oras_masks.xml').read_text(): raise ValueError('Missing ORAS mask')
    jobs.append((ORAS+'fileServer/ftpthredds/EASYInit/oras5/r1x1/LSM_r1x1/'+name,'oras5/'+name))
    name='G02202-ancillary-psn25-v06r00.nc'
    if name not in (output/'catalogs/ice_ancillary.html').read_text(): raise ValueError('Missing ice ancillary')
    jobs.append((ICE+'ancillary/'+name,'ice/'+name))
    failures=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futures={pool.submit(fetch.download,url,path):url for url,path in jobs}
        for future in concurrent.futures.as_completed(futures):
            try: records.append(future.result())
            except Exception as exc: failures.append({'url':futures[future],'error':str(exc)})
    manifest={'retrieval_checked_utc':datetime.now(timezone.utc).isoformat(),
              'months':MONTHS,'records':records,'failures':failures,
              'oras_product':'ORAS5 r1x1, opa0; regridded 1 degree single member',
              'ice_product':'NOAA/NSIDC G02202 V6 north monthly 25 km F17',
              'transport':'HTTPS, certificate verification enabled; no login required'}
    write_json(output/'downloads.json',manifest)
    if failures: raise RuntimeError(f'{len(failures)} unresolved downloads; rerun to resume completed-file cache')


def audit_sources(output):
    manifest=json.loads((output/'downloads.json').read_text())
    if manifest['failures']: raise ValueError('Unresolved download failures')
    for record in manifest['records']:
        p=Path(record['path'])
        if p.stat().st_size!=record['bytes'] or digest(p)!=record['sha256']: raise ValueError(f'Source hash mismatch: {p}')
    audits=[]
    with Dataset(output/'ice/G02202-ancillary-psn25-v06r00.nc') as ancillary:
        ix,iy=np.asarray(ancillary['x'][:]),np.asarray(ancillary['y'][:])
        projection=str(ancillary['crs'].proj4text)
        if str(ancillary.geospatial_bounds_crs)!='EPSG:3411': raise ValueError('Ancillary CRS changed')
        if ancillary['latitude'].shape!=(448,304): raise ValueError('Ancillary grid shape changed')
        if ancillary['latitude'].units!='degrees_north' or ancillary['longitude'].units!='degrees_east': raise ValueError('Ice geographic units changed')
        if 'F17_polemask' not in ancillary['polehole_bitmask'].flag_meanings: raise ValueError('Unknown F17 pole mask')
    reference=None
    for month in MONTHS:
        for variable in ['votemper','vosaline']:
            p=output/f'oras5/{variable}_ORAS5_1m_{month}_r1x1.nc'
            with Dataset(p) as ds:
                v=ds[variable]
                unit='C' if variable=='votemper' else 'psu'
                if str(v.units).strip().lower()!=unit.lower() or v.dimensions!=('time_counter','deptht','lat','lon'): raise ValueError(f'Variable contract changed: {p}')
                if len(ds['time_counter'])!=1: raise ValueError('Expected one monthly ORAS snapshot')
                if ds['lat'].units!='degrees_north' or ds['lon'].units!='degrees_east' or ds['deptht'].units!='m': raise ValueError('ORAS coordinate units changed')
                coords=tuple(np.asarray(ds[x][:]) for x in ['lat','lon','deptht'])
                if reference is None: reference=coords
                if not all(np.array_equal(a,b) for a,b in zip(coords,reference)): raise ValueError('ORAS grids differ across time/products')
                stamp=num2date(ds['time_counter'][0],ds['time_counter'].units,calendar=getattr(ds['time_counter'],'calendar','standard'))
                if int(stamp.strftime('%Y%m'))!=month: raise ValueError('ORAS CF time / filename mismatch')
                if 'monthly mean' not in str(ds.title).lower() or 'ORAS5' not in str(ds.source): raise ValueError('Unrecognized ORAS provenance')
                audits.append({'file':str(p.relative_to(output)),'variable':variable,'source':str(ds.source),'title':str(ds.title),
                    'unit':unit,'source_unit_string':str(v.units),'definition':'potential temperature' if variable=='votemper' else 'practical salinity',
                    'shape':list(v.shape),'cf_time':stamp.isoformat(),'calendar':str(getattr(ds['time_counter'],'calendar','standard')),
                    'month':month,'monthly_interval_policy':'calendar-month mean declared by product title; timestamp is representative, not exact equality with ice timestamp',
                    'time_bounds_present':any(n in ds.variables for n in ['time_bounds','time_bnds'])})
        p=output/f'ice/sic_psn25_{month}_F17_v06r00.nc'
        with Dataset(p) as ds:
            if len(ds['time'])!=1: raise ValueError('Expected one monthly ice snapshot')
            stamp=num2date(ds['time'][0],ds['time'].units,calendar=getattr(ds['time'],'calendar','standard'))
            days=calendar.monthrange(2013,month%100)[1]
            start=f'2013-{month%100:02d}-01T00:00:00Z'
            end=f'2013-{month%100:02d}-{days:02d}T23:59:59Z'
            if int(stamp.strftime('%Y%m'))!=month or ds.time_coverage_start!=start or ds.time_coverage_end!=end: raise ValueError('Ice monthly time coverage mismatch')
            if 'F17' not in ds.platform or 'Version 6' not in ds.title: raise ValueError('Ice product/sensor mismatch')
            if ds.geospatial_bounds_crs!='EPSG:3411' or ds['crs'].proj4text!=projection: raise ValueError('Ice CRS mismatch')
            if not np.array_equal(ds['x'][:],ix) or not np.array_equal(ds['y'][:],iy): raise ValueError('Ice ancillary x/y mismatch')
            v=ds['cdr_seaice_conc_monthly']
            if v.units!='1' or v.shape!=(1,448,304) or not np.isclose(v.scale_factor,.01): raise ValueError('Ice units, packing, or grid mismatch')
            value=read_float(v,0); good=np.isfinite(value)
            if not np.all((value[good]>=0)&(value[good]<=1)): raise ValueError('Decoded ice outside [0,1]')
            audits.append({'file':str(p.relative_to(output)),'variable':'cdr_seaice_conc_monthly','title':str(ds.title),
                'unit':'1, decoded fraction','scale_factor':float(v.scale_factor),'packed_fill_value':int(v._FillValue),
                'shape':list(v.shape),'cf_time':stamp.isoformat(),'calendar':str(getattr(ds['time'],'calendar','standard')),
                'month':month,'coverage_start':start,'coverage_end':end,'crs':'EPSG:3411',
                'sensor':str(ds.sensor),'platform':str(ds.platform),'valid_native_cells':int(good.sum())})
    with Dataset(output/'oras5/tmask_r1x1.nc') as ds:
        if ds['tmask'].shape!=(1,75,180,360): raise ValueError('ORAS mask grid shape mismatch')
        if not np.array_equal(ds['lat'][:],reference[0]) or not np.array_equal(ds['lon'][:],reference[1]): raise ValueError('ORAS mask horizontal mismatch')
    return audits


def align(output,previous):
    output.mkdir(parents=True,exist_ok=True)
    audits=audit_sources(output)
    with np.load(previous/'recovered_iap_offsets_50m.npz') as z: lat,lon=z['latitude'],z['longitude']
    # Independently reread only the twelve 2013 IAP members; verify their hashes.
    source_records=json.loads((previous/'iap_sources.json').read_text())
    hf_by_date={}; hf_audits=[]
    with np.load(previous/'recovered_iap_offsets_50m.npz') as z: offsets=z['offsets_c']
    for source in source_records:
        match=re.search(r'year_(2013)_month_(\d{2})',source['member'])
        if not match: continue
        month=int(''.join(match.groups()))
        with zipfile.ZipFile(source['archive']) as archive: content=archive.read(source['member'])
        import hashlib
        if hashlib.sha256(content).hexdigest()!=source['sha256']: raise ValueError('IAP source member hash mismatch')
        with Dataset('iap_validation.nc',memory=content) as ds:
            if int(ds['time'][0])!=month or not np.array_equal(ds['lat'][:],lat) or not np.array_equal(ds['lon'][:],lon): raise ValueError('IAP validation time/grid mismatch')
            hf_by_date[month]=layer(ds,'temp')+offsets[month%100-1]
            hf_audits.append(dict(source,month=month,input_unit=str(ds['temp'].units),depth_m=50,
                                 raw_definition='IAP temperature anomaly; restored using frozen recovered monthly offsets'))
    if sorted(hf_by_date)!=MONTHS: raise ValueError('Incomplete 2013 IAP source members')
    raw_hf=np.stack([hf_by_date[m] for m in MONTHS])
    lf,static,indices,weights,conversion=oras_fields(output,lat,lon,MONTHS)
    report={'status':'completed','validation_months':MONTHS,'source_audits':audits,'hf_source_audits':hf_audits,'oras_conversion':conversion,
            'training_period':[200001,201212],'test_year_reserved':2014,'no_parameter_selection_performed':True,
            'months_used_for_alignment':MONTHS,'alignment':'same frozen training/static domains, coordinates and node order as prior artifacts',
            'regions':{},'frozen_input_hashes':{},'limitations':[
                'Monthly in-situ ORAS temperature converted from monthly mean potential temperature and salinity is a nonlinear-average approximation.',
                'Retrospective products and in-situ assimilation independence remain unverified.',
                'Ice pole hole, coast, land and missing stay unknown; not zero ice.',
                '2013 field validation supports grid-field parameter validation, not independent ITP/Argo field validation.',
                '2014 has already been inspected in an exploratory deployment run; it is not a never-seen confirmatory test set.']}
    for region in ['atlantic','central_arctic']:
        folder=output/'aligned_50m'/region; folder.mkdir(parents=True,exist_ok=True)
        src=previous/region
        report['frozen_input_hashes'][str(src/'training_artifacts.npz')]=digest(src/'training_artifacts.npz')
        report['frozen_input_hashes'][str(src/'aligned_fields.npz')]=digest(src/'aligned_fields.npz')
        with np.load(src/'training_artifacts.npz') as z: artifact={k:z[k] for k in z.files}
        if np.any(artifact['train_dates']>201212): raise ValueError('Frozen training contains validation or test dates')
        with np.load(src/'aligned_fields.npz') as z:
            dates=z['validation_dates']; mask=z['valid_domain_mask']; coords=z['coordinates_latlon']
            ids=z['original_grid_flat_ids']; hf=z['validation_hf_insitu_c']
            old_anomaly=z['validation_hf_anomaly_c']; old_model=z['validation_hf_model']
            expected_lf_indices=z['lf_source_flat_ids']; expected_lf_weights=z['lf_source_weights']
        if dates.tolist()!=MONTHS or np.any(mask&~static): raise ValueError('Frozen domain or dates incompatible')
        np.testing.assert_allclose(hf,raw_hf[:,mask],atol=1e-12)
        if not np.isfinite(hf).all() or not np.isfinite(lf[:,mask]).all(): raise ValueError('Missing validation field on frozen domain; domain must not be shrunk')
        np.testing.assert_array_equal(indices[mask.ravel()],expected_lf_indices)
        np.testing.assert_allclose(weights[mask.ravel()],expected_lf_weights,atol=0,rtol=0)
        season=artifact['monthly_climatology_c'][dates%100-1]
        hf_anomaly=hf-season; lf_anomaly=lf[:,mask]-season
        scale=float(artifact['scale_c']); mean=artifact['mean_c']
        hf_model=(hf_anomaly-mean)/scale; lf_model=(lf_anomaly-mean)/scale
        np.testing.assert_allclose(hf_anomaly,old_anomaly,atol=1e-12)
        np.testing.assert_allclose(hf_model,old_model,atol=1e-12)
        ice,qa,ice_ids,ice_km=ice_fields(output,coords,MONTHS)
        shutil.copyfile(src/'training_artifacts.npz',folder/'training_artifacts.npz')
        if digest(folder/'training_artifacts.npz')!=digest(src/'training_artifacts.npz'): raise ValueError('Training artifact changed')
        np.savez_compressed(folder/'validation_fields.npz',validation_dates=dates,
            coordinates_latlon=coords,original_grid_flat_ids=ids,valid_domain_mask=mask,
            validation_hf_insitu_c=hf,validation_lf_insitu_c=lf[:,mask],
            validation_hf_anomaly_c=hf_anomaly,validation_lf_anomaly_c=lf_anomaly,
            validation_hf_model=hf_model,validation_lf_model=lf_model,
            ice_fraction=ice,ice_qa_flags=qa,ice_source_flat_ids=ice_ids,ice_distance_km=ice_km,
            lf_source_flat_ids=indices[mask.ravel()],lf_source_weights=weights[mask.ravel()])
        info={'shape':list(hf.shape),'field_unit':'degree_Celsius','temperature_definition':'in_situ',
              'depth_m':50,'time_resolution':'calendar-month mean','months':MONTHS,
              'baseline':'frozen 2000-2012 HF monthly climatology','ice_unit':'fraction [0,1]',
              'ice_valid_nodes_per_month':np.isfinite(ice).sum(axis=1).tolist(),
              'ice_unknown_nodes_per_month':(~np.isfinite(ice)).sum(axis=1).tolist(),
              'max_ice_mapping_distance_km':float(ice_km.max()),
              'hf_lf_complete_on_frozen_domain':True,'training_artifact_byte_identical':True,
              'node_order_unchanged':True,'domain_derived_from_validation':False}
        write_json(folder/'metadata.json',info); report['regions'][region]=info
        print(region,json.dumps(info),flush=True)
    report['script_sha256']=digest(__file__)
    report['shared_alignment_module_sha256']=digest(ROOT/'scripts/align_polar_sample.py')
    report['output_hashes']={str(p.relative_to(output)):digest(p) for p in (output/'aligned_50m').rglob('*') if p.is_file()}
    write_json(output/'audit_report.json',report)
    lines=['# 2013年极地验证数据补齐与审计','',
        '用途：只用于2013年验证、选择参数；训练固定2000–2012，2014禁止调参。原2014结果属于探索性测试，已查看过；严格确认性评价需另保留未查看时段。','',
        '已下载24个ORAS5月文件（12温度+12配套实用盐度），12个G02202 V6 F17北极月海冰文件，另含tmask和海冰辅助坐标/掩膜。URL、下载时间、字节和SHA256见downloads.json，逐文件变量/时间/空间检查见audit_report.json。','',
        'ORAS5采用ICDC r1x1/opa0，并非原生0.25°网格或完整集合；位温℃、实用盐度psu，先掩膜再做TEOS-10原位温度转换，插值至50 m，周期双线性映射至固定IAP格点。海冰EPSG:3411、25 km，CF自动解包0.01缩放得到0–1浓度，按辅助经纬度球面最近邻映射，最大40 km；F17极区观测空洞、陆地、海岸及缺测保持NaN。原QA保留。','',
        '时间按日历月份配对，不要求ORAS代表日与海冰时间戳相等；海冰起止覆盖逐月核验，ORAS的月平均产品定义与CF代表时间共同核验。无ORAS原生时间界限变量时，不把推定日历月起止声称为文件内实测时间界限。','',
        '异常、中心化、尺度、POD及协方差全部复用原训练产物，副本字节一致；未重新训练，也未根据验证或测试修改共同域。输出仅包含2013验证字段和2000–2012训练产物，不导出2014场。','',
        '| 区域 | 验证HF/LF矩阵（月份×格点） | 每月海冰已知格点 |','|---|---|---|']
    for region,info in report['regions'].items(): lines.append(f"| {region} | {info['shape']} | {info['ice_valid_nodes_per_month']} |")
    lines+=['','文件：aligned_50m/各区域/validation_fields.npz保存HF、LF、海冰、异常、模型数组、固定坐标和源网格权重；training_artifacts.npz为冻结训练副本。','',
        '尚未进行参数搜索或部署训练。盐度用于物理转换，不是新增的盐度重构目标。此包未补齐2013独立ITP/Argo观测。','',
        '来源：[ICDC ORAS5目录](https://icdc.cen.uni-hamburg.de/thredds/catalog/ftpthredds/EASYInit/oras5/catalog.html)、[ORAS5掩膜修正说明](https://icdc.cen.uni-hamburg.de/thredds/fileServer/ftpthredds/EASYInit/oras5/r1x1/Correction_to_ORAS5_r1x1_files.pdf)、[NSIDC G02202 V6](https://nsidc.org/data/g02202/versions/6)。','']
    (output/'README_zh.md').write_text('\n'.join(lines),encoding='utf-8')
    verify(output)


def verify(output):
    audit_sources(output)
    report=json.loads((output/'audit_report.json').read_text())
    for relative,expected in report['output_hashes'].items():
        if digest(output/relative)!=expected: raise ValueError(f'Output hash mismatch: {relative}')
    results={}
    for region in report['regions']:
        folder=output/'aligned_50m'/region
        with np.load(folder/'training_artifacts.npz') as t,np.load(folder/'validation_fields.npz') as f:
            np.testing.assert_array_equal(f['validation_dates'],MONTHS)
            if np.any(t['train_dates']>201212): raise ValueError('Training period violation')
            if any('test' in key for key in f.files): raise ValueError('Test field exported to validation')
            for fidelity in ['hf','lf']:
                absolute=f['validation_'+fidelity+'_insitu_c']
                anomaly=absolute-t['monthly_climatology_c'][f['validation_dates']%100-1]
                if not np.isfinite(absolute).all(): raise ValueError('Incomplete matrix')
                np.testing.assert_allclose(f['validation_'+fidelity+'_anomaly_c'],anomaly,atol=1e-12)
                np.testing.assert_allclose(f['validation_'+fidelity+'_model'],(anomaly-t['mean_c'])/t['scale_c'],atol=1e-12)
            np.testing.assert_allclose(f['lf_source_weights'].sum(axis=1),1,atol=1e-12)
            np.testing.assert_allclose(t['basis'].T@t['basis'],np.eye(t['basis'].shape[1]),atol=1e-10)
            ice=f['ice_fraction']; good=np.isfinite(ice)
            if not np.all((ice[good]>=0)&(ice[good]<=1)): raise ValueError('Invalid ice fractions')
            results[region]={'passed':True,'shape':list(f['validation_hf_model'].shape),'months':MONTHS}
    write_json(output/'verification.json',{'passed':True,'regions':results,'source_and_alignment_audit_passed':True})
    print('Validation package verified:',output,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--previous',type=Path,default=Path('D:/Dataset/RBMF-CS_polar_sample/aligned_50m'))
    parser.add_argument('--stage',choices=['all','download','align','verify'],default='all')
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    if args.stage in ['all','download']: download(args.output)
    if args.stage in ['all','align']: align(args.output,args.previous)
    if args.stage=='verify': verify(args.output)
