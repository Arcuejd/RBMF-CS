"""Align the 50 m polar sample with train-only seasonal/POD artifacts.

Output is a new directory. IAP absolute-minus-anomaly offsets are recovered only
from 2000 and verified using 2001; no 2014 value enters any fitted artifact.
"""
import argparse
from collections import Counter
from datetime import datetime, timedelta
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import zipfile

import gsw
import numpy as np
from netCDF4 import Dataset, num2date
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from rbmf_cs.alignment import vertical_bracket, periodic_stencil, sample_stencil, training_artifacts

TARGETS = [201401, 201404, 201407, 201410]
DEPTH = 50.


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk := f.read(1024*1024):
            h.update(chunk)
    return h.hexdigest()


def read_float(v, key=slice(None)):
    a = np.ma.asarray(v[key]).astype(float).filled(np.nan)
    for n in ['_FillValue', 'missing_value']:
        if n in v.ncattrs():
            a[a == float(v.getncattr(n))] = np.nan
    a[~np.isfinite(a)] = np.nan
    return a


def layer(ds, var):
    d = np.asarray(ds['depth_std'][:])
    k = np.flatnonzero(d == DEPTH)
    if len(k) != 1:
        raise ValueError('IAP must contain exact 50 m layer')
    if 'celsius' not in str(ds[var].units).lower():
        raise ValueError('Unexpected IAP temperature unit')
    return read_float(ds[var], (slice(None), slice(None), int(k[0])))


def load_iap(data_root, output):
    cache = output/'iap_50m_source_cache.npz'
    if cache.exists():
        z = np.load(cache)
        return z['dates'], z['anomaly'], z['lat'], z['lon']
    dates, values, sources = [], [], []
    lat = lon = None
    for p in sorted(data_root.glob('Temperature*/Temperature*20[01]0_20[01]9.zip')):
        with zipfile.ZipFile(p) as z:
            for name in sorted(z.namelist()):
                match = re.search(r'year_(\d{4})_month_(\d{2})', name)
                if not match:
                    continue
                date = int(''.join(match.groups()))
                if not (200001 <= date <= 201312 or date in TARGETS):
                    continue
                content = z.read(name)
                with Dataset('memory.nc', memory=content) as ds:
                    this_lat, this_lon = np.asarray(ds['lat'][:]), np.asarray(ds['lon'][:])
                    if lat is None:
                        lat, lon = this_lat, this_lon
                    if not (np.array_equal(lat, this_lat) and np.array_equal(lon, this_lon)):
                        raise ValueError('IAP grid changed')
                    if int(ds['time'][0]) != date:
                        raise ValueError('IAP filename/time mismatch')
                    values.append(layer(ds, 'temp'))
                dates.append(date)
                sources.append({'archive':str(p), 'member':name, 'sha256':hashlib.sha256(content).hexdigest()})
                if len(dates) % 24 == 0:
                    print('IAP snapshots read', len(dates), date, flush=True)
    expected = [y*100+m for y in range(2000,2014) for m in range(1,13)] + TARGETS
    if dates != expected:
        raise ValueError('Missing, duplicate, or unexpected IAP dates')
    array = np.stack(values)
    np.savez_compressed(cache, dates=dates, anomaly=array, lat=lat, lon=lon)
    (output/'iap_sources.json').write_text(json.dumps(sources, indent=2), encoding='utf-8')
    return np.asarray(dates), array, lat, lon


def recover_offsets(base, dates, anomaly, lat, lon):
    offsets, checks = [], []
    for m in range(1,13):
        pair = []
        for year in [2000,2001]:
            p = base/'iap_baseline'/f'IAPv4_Temp_monthly_1_6000m_year_{year}_month_{m:02d}.nc'
            with Dataset(p) as ds:
                if not (np.array_equal(ds['lat'][:], lat) and np.array_equal(ds['lon'][:], lon)):
                    raise ValueError('Baseline grid mismatch')
                pair.append(layer(ds,'temp')-anomaly[np.flatnonzero(dates == year*100+m)[0]])
        valid = np.isfinite(pair[0]) & np.isfinite(pair[1])
        error = float(np.max(np.abs(pair[0][valid]-pair[1][valid]))) if valid.any() else float('inf')
        if not valid.any() or error > 1e-4:
            raise ValueError(f'IAP anomaly offset is not fixed across training years: month={m}, max_error={error}')
        candidate = base/'iap_baseline'/f'IAP_temperature_2024v_climatology_1_6000m_month_{m:02d}.nc'
        with Dataset(candidate) as ds:
            published = layer(ds, 'temperature')
        good = valid & np.isfinite(published)
        checks.append({'month':m, 'verification_max_error_c':error,
                       'verification_valid_cells':int(valid.sum()),
                       'unmatched_2024_climatology_max_error_c':float(np.max(np.abs(pair[0][good]-published[good])))})
        offsets.append(pair[0])
    return np.stack(offsets), checks


def oras_fields(base, lat, lon, months=None):
    months = TARGETS if months is None else list(months)
    if not months or len(set(months)) != len(months):
        raise ValueError('Explicit nonempty unique monthly selection required')
    sample = base/f'oras5/votemper_ORAS5_1m_{months[0]}_r1x1.nc'
    with Dataset(sample) as ds:
        olat, olon, depth = [np.asarray(ds[v][:]) for v in ['lat','lon','deptht']]
    lo, hi, fraction = vertical_bracket(depth, DEPTH)
    with Dataset(base/'oras5/tmask_r1x1.nc') as ds:
        mask = np.ma.asarray(ds['tmask'][0]).filled(0) == 1
    yy, xx = np.meshgrid(lat, lon, indexing='ij')
    points = np.column_stack((yy.ravel(), xx.ravel()))
    idx, weights = periodic_stencil(olat, olon, points)
    static_native = mask[lo] & mask[hi]
    static_target = sample_stencil(np.where(static_native,1.,np.nan).ravel(),idx,weights).reshape(yy.shape)
    native_lat, native_lon = np.meshgrid(olat,olon,indexing='ij')
    result, corrections = [], []
    for month in months:
        with Dataset(base/f'oras5/votemper_ORAS5_1m_{month}_r1x1.nc') as td, Dataset(base/f'oras5/vosaline_ORAS5_1m_{month}_r1x1.nc') as sd:
            for ds in [td,sd]:
                if not (np.array_equal(ds['lat'][:],olat) and np.array_equal(ds['lon'][:],olon) and np.array_equal(ds['deptht'][:],depth)):
                    raise ValueError('ORAS5 coordinate mismatch')
                stamp = num2date(ds['time_counter'][0], ds['time_counter'].units)
                if int(stamp.strftime('%Y%m')) != month:
                    raise ValueError('ORAS5 time does not match target month')
            if str(td['votemper'].units) != 'C' or str(sd['vosaline'].units).lower() != 'psu':
                raise ValueError('Unexpected ORAS5 physical units')
            converted = []
            for k in [lo,hi]:
                pt = read_float(td['votemper'],(0,k,slice(None),slice(None)))
                sp = read_float(sd['vosaline'],(0,k,slice(None),slice(None)))
                good = mask[k] & np.isfinite(pt) & np.isfinite(sp) & (sp >= 0)
                pressure = gsw.p_from_z(-float(depth[k]),native_lat)
                sa = gsw.SA_from_SP(np.where(good,sp,np.nan),pressure,native_lon,native_lat)
                insitu = gsw.t_from_CT(sa,gsw.CT_from_pt(sa,np.where(good,pt,np.nan)),pressure)
                corrections.append(float(np.nanmax(np.abs(insitu-pt))))
                converted.append(insitu)
        at_depth = converted[0]*(1-fraction)+converted[1]*fraction
        result.append(sample_stencil(at_depth.ravel(),idx,weights).reshape(yy.shape))
    return np.stack(result), np.isfinite(static_target), idx, weights, {
        'source_depths_m':[float(depth[lo]),float(depth[hi])], 'upper_weight':fraction,
        'max_native_pt_to_insitu_c':max(corrections), 'method':'TEOS-10 at native levels, linear vertical, periodic bilinear horizontal'}


def unit_vectors(lat, lon):
    la,lo = np.deg2rad(lat),np.deg2rad(lon)
    return np.column_stack((np.cos(la)*np.cos(lo),np.cos(la)*np.sin(lo),np.sin(la)))


def ice_fields(base, points, months=None):
    months = TARGETS if months is None else list(months)
    with Dataset(base/'ice/G02202-ancillary-psn25-v06r00.nc') as ds:
        ilat, ilon = read_float(ds['latitude']).ravel(),read_float(ds['longitude']).ravel()
        surface = np.asarray(ds['surface_type'][:]).ravel()
        polehole = np.asarray(ds['polehole_bitmask'][:]).ravel()
    source_valid = np.isfinite(ilat) & np.isfinite(ilon)
    original_idx = np.flatnonzero(source_valid)
    distance, row = cKDTree(unit_vectors(ilat[source_valid],ilon[source_valid])).query(unit_vectors(points[:,0],points[:,1]))
    source_idx = original_idx[row]
    km = 2*6371.0088*np.arcsin(np.clip(distance/2,0,1))
    # All selected months are F17. Reject that sensor's polar observation hole.
    static_good = (surface[source_idx] == 50) & ((polehole[source_idx]&16) == 0) & (km <= 40.)
    out, flags = [], []
    for month in months:
        p = base/f'ice/sic_psn25_{month}_F17_v06r00.nc'
        with Dataset(p) as ds:
            value = read_float(ds['cdr_seaice_conc_monthly'],0).ravel()[source_idx]
            flag = np.ma.asarray(ds['cdr_seaice_conc_monthly_qa_flag'][0]).filled(255).ravel()[source_idx]
        good = static_good & np.isfinite(value) & (value >= 0) & (value <= 1)
        out.append(np.where(good,value,np.nan))
        flags.append(flag)
    return np.stack(out),np.stack(flags),source_idx,km


def observed_50m(base, record):
    if record['source'] == 'ITP_Level3':
        archive, member = record['file'].split('::')
        with zipfile.ZipFile(archive) as z:
            lines = z.read(member).decode('ascii').splitlines()
        a = np.loadtxt(io.StringIO('\n'.join(lines[3:])), comments='%', usecols=(0,1),ndmin=2)
        segments = [(a[:,0],a[:,1],dict(record))]
    else:
        segments = []
        with Dataset(record['file']) as ds:
            for j in range(len(ds.dimensions['N_PROF'])):
                mode = np.asarray(ds['DATA_MODE'][:]).astype('U1')[j]
                pq = np.asarray(ds['POSITION_QC'][:]).astype('U1')[j]
                tq = np.asarray(ds['JULD_QC'][:]).astype('U1')[j]
                if mode != 'D' or pq not in ['1','2'] or tq not in ['1','2']:
                    continue
                q = (np.asarray(ds['TEMP_ADJUSTED_QC'][j]).astype('U1') == '1') & (np.asarray(ds['PRES_ADJUSTED_QC'][j]).astype('U1') == '1')
                pressure = read_float(ds['PRES_ADJUSTED'],j)
                temp = read_float(ds['TEMP_ADJUSTED'],j)
                stamp = num2date(ds['JULD'][j],ds['JULD'].units)
                position = dict(record, profile=j, lat=float(ds['LATITUDE'][j]),
                                lon=(float(ds['LONGITUDE'][j])+180)%360-180,
                                month=stamp.strftime('%Y%m'),day=stamp.strftime('%Y%m%d'))
                if int(position['month']) not in TARGETS or not np.isfinite([position['lat'],position['lon']]).all():
                    continue
                segments.append((np.where(q,pressure,np.nan),np.where(q,temp,np.nan),position))
    for pressure,temp,position in segments:
        good = np.isfinite(pressure)&np.isfinite(temp)&(pressure>=0)&(np.abs(temp)<100)
        z = -gsw.z_from_p(pressure[good],position['lat'])
        t = temp[good]
        order = np.argsort(z)
        z,t = z[order],t[order]
        z, unique = np.unique(z,return_index=True)
        t = t[unique]
        if len(z)<2 or not z[0] <= DEPTH <= z[-1]:
            continue
        lo,hi,w = vertical_bracket(z,DEPTH)
        if z[hi]-z[lo] <= 10:
            return float((1-w)*t[lo]+w*t[hi]),float(z[hi]-z[lo]),position
    return None,None,record


def observation_rows(base, inventory, lat, lon, mask, artifact, name):
    global_to_node = np.full(mask.size,-1,int)
    global_to_node[np.flatnonzero(mask.ravel())] = np.arange(mask.sum())
    rows, rejected = [],Counter()
    for r in inventory:
        value,gap,r = observed_50m(base,r)
        la,lo = r['lat'],r['lon']
        in_region = (70<=la<80 and -20<=lo<20) if name == 'atlantic' else (80<=la<=90)
        if not in_region:
            continue
        try:
            idx,w = periodic_stencil(lat,lon,[[la,lo]])
        except ValueError:
            rejected['outside_grid_centers'] += 1
            continue
        use = w[0] > 1e-12
        nodes = global_to_node[idx[0,use]]
        if np.any(nodes<0):
            rejected['stencil_touches_land_or_outside_fixed_domain'] += 1
            continue
        if value is None:
            rejected['no_50m_bracket_or_strict_qc'] += 1
            continue
        weights = w[0,use]
        month = int(r['month'])%100
        baseline = float(weights@artifact['monthly_climatology_c'][month-1,nodes])
        mean = float(weights@artifact['mean_c'][nodes])
        rows.append({'source':r['source'],'platform':r['platform'],'profile':r['profile'],
                     'source_file':r['file'],'month':int(r['month']),'day':r['day'],
                     'latitude':la,'longitude':lo,'depth_m':DEPTH,'vertical_bracket_gap_m':gap,
                     'node_ids':nodes.tolist(),'weights':weights.tolist(),
                     'temperature_insitu_c':value,'anomaly_c':value-baseline,
                     'model_value':(value-baseline-mean)/float(artifact['scale_c']),
                     'role':'unassigned_collocated_profile','monthly_mean_observation':False,
                     'assimilation_independence':'not_verified'})
    return rows,dict(rejected)


def main(data_root, base, output):
    output.mkdir(parents=True,exist_ok=True)
    dates,anomaly,lat,lon = load_iap(data_root,output)
    offsets,baseline_checks = recover_offsets(base,dates,anomaly,lat,lon)
    absolute = anomaly+offsets[dates%100-1]
    train = dates <= 201212
    validation = (dates >= 201301)&(dates <= 201312)
    test = np.isin(dates,TARGETS)
    if np.any(dates[train]//100 == 2014) or np.any(dates[train]>=201301):
        raise ValueError('Data leakage in training split')
    lf,static_ocean,lf_indices,lf_weights,conversion = oras_fields(base,lat,lon)
    yy,xx = np.meshgrid(lat,lon,indexing='ij')
    common = np.isfinite(absolute[train]).all(axis=0)&np.isfinite(offsets).all(axis=0)&static_ocean
    inventory = json.loads((base/'reports/profile_inventory.json').read_text())
    report = {'target':'50 m in-situ seawater temperature, degrees Celsius',
              'train_period':[200001,201212],'validation_period':[201301,201312],
              'test_months':TARGETS,'entire_test_year_excluded':2014,
              'baseline_method':'2000 absolute-minus-anomaly monthly offsets verified independently with 2001; rebased to 2000-2012 HF monthly means',
              'published_climatology_2024_used':False,'baseline_checks':baseline_checks,
              'oras_conversion':conversion,'regions':{},
              'limitations':['Published IAP/ORAS5 are retrospective analysis products; their original assimilation is not made causal by this split.',
                             'Argo/ITP rows are instantaneous collocated profiles, not verified monthly means or independent validation.',
                             'Only four LF months were downloaded; train/validation LF calibration is not included.',
                             'Ice pole hole, coast/land, missing data remain NaN and must be excluded by deployment constraints.']}
    for name,region in [('atlantic',(yy>=70)&(yy<80)&(((xx+180)%360-180)>=-20)&(((xx+180)%360-180)<20)),
                        ('central_arctic',yy>=80)]:
        mask = common&region
        n = int(mask.sum())
        if n<2:
            raise ValueError('Insufficient static/train-only domain')
        artifact = training_artifacts(absolute[train][:,mask],dates[train])
        regional = output/name
        regional.mkdir(exist_ok=True)
        np.savez_compressed(regional/'training_artifacts.npz',**artifact)
        coordinates = np.column_stack((yy[mask],(xx[mask]+180)%360-180))
        season = artifact['monthly_climatology_c']
        def normalized(values,months):
            anom = values-season[np.asarray(months)%100-1]
            return anom,(anom-artifact['mean_c'])/float(artifact['scale_c'])
        test_hf = absolute[test][:,mask]
        test_lf = lf[:,mask]
        if not (np.isfinite(test_hf).all() and np.isfinite(test_lf).all()):
            raise ValueError('Future field missing on frozen training/static domain; no test-driven domain shrink allowed')
        hf_anom,hf_model = normalized(test_hf,dates[test])
        lf_anom,lf_model = normalized(test_lf,TARGETS)
        val_anom,val_model = normalized(absolute[validation][:,mask],dates[validation])
        ice,qa,ice_indices,ice_km = ice_fields(base,coordinates)
        np.savez_compressed(regional/'aligned_fields.npz',coordinates_latlon=coordinates,
            node_ids=np.arange(n),original_grid_flat_ids=np.flatnonzero(mask.ravel()),valid_domain_mask=mask,
            train_dates=dates[train],validation_dates=dates[validation],test_dates=dates[test],
            train_hf_insitu_c=absolute[train][:,mask],train_hf_anomaly_c=artifact['train_anomaly_c'],
            train_hf_model=artifact['train_model'],validation_hf_insitu_c=absolute[validation][:,mask],
            validation_hf_anomaly_c=val_anom,validation_hf_model=val_model,
            test_hf_insitu_c=test_hf,test_lf_insitu_c=test_lf,test_hf_anomaly_c=hf_anom,
            test_lf_anomaly_c=lf_anom,test_hf_model=hf_model,test_lf_model=lf_model,
            ice_fraction=ice,ice_qa_flags=qa,ice_source_flat_ids=ice_indices,ice_distance_km=ice_km,
            lf_source_flat_ids=lf_indices[mask.ravel()],lf_source_weights=lf_weights[mask.ravel()])
        rows,rejected = observation_rows(base,inventory,lat,lon,mask,artifact,name)
        (regional/'aligned_observations.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
        # Metadata prevents accidental mixing of the new scientific baseline with old demos.
        info = {'nodes':n,'train_shape':list(artifact['train_model'].shape),'test_shape':list(hf_model.shape),
                'pod_modes':artifact['basis'].shape[1],'pod_training_energy':float(artifact['pod_energy_fraction']),
                'scale_c':float(artifact['scale_c']),'training_monthly_mean_max_abs_c':float(max(np.max(np.abs(artifact['train_anomaly_c'][dates[train]%100==m].mean(axis=0))) for m in range(1,13))),
                'observations_kept':len(rows),'observation_rejections':rejected,
                'observations_by_month':dict(Counter(str(r['month']) for r in rows)),
                'ice_valid_nodes_per_month':np.isfinite(ice).sum(axis=1).tolist(),
                'max_ice_mapping_distance_km':float(ice_km.max()),
                'field_unit':'degree_Celsius','temperature_definition':'in_situ','depth_m':DEPTH,
                'baseline':'HF training-only monthly climatology 2000-2012',
                'node_order':'original IAP latitude ascending then longitude 0.5..359.5, filtered; coordinate longitudes wrapped without reordering',
                'observations_need_monthly_representativeness_policy':True}
        report['regions'][name] = info
        (regional/'metadata.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
        print(name,json.dumps(info),flush=True)
    np.savez_compressed(output/'recovered_iap_offsets_50m.npz',offsets_c=offsets,latitude=lat,longitude=lon,
                        anchor_year=2000,verification_year=2001)
    report['script_sha256']=digest(__file__)
    report['alignment_module_sha256']=digest(ROOT/'src/rbmf_cs/alignment.py')
    source_files = set((base/'oras5').glob('*.nc')) | set((base/'ice').glob('*.nc')) | set((base/'iap_baseline').glob('*.nc'))
    source_files.update(Path(r['file'].split('::')[0]) for r in inventory)
    report['source_file_sha256']={str(p):digest(p) for p in sorted(source_files)}
    lines = ['# 真实极地数据对齐结果', '',
        '重构目标：50 m 原位海水温度（℃），IAP/ORAS5为月平均场。',
        '训练：2000–2012（156个月）；验证：2013（12个月）；测试：2014年1、4、7、10月。2014全年不参与POD、气候态、归一化或协方差拟合。', '',
        '## 物理与异常基准', '',
        'IAP绝对温度由2000年绝对场减去本地异常恢复的逐月固定偏移还原，并用2001年12个月独立核对。下载的2024版气候态与本地异常不匹配，未采用；不推断原始偏移的气候平均期。',
        '统一异常基准为2000–2012年HF逐日历月均值。HF、LF和点观测均减去同一基准，再使用同一训练均值与标量尺度归一化。',
        'ORAS5先应用tmask（1为有效海水），按盐度、压力、经纬度用TEOS-10将位温转换为原位温度，再垂向插值至50 m，周期双线性插值到IAP格点。由月均位温/盐度转换是月均原位温度的近似，不代表对瞬时场逐时转换后平均。', '',
        '## 固定共同有效域', '',
        '格点顺序固定为IAP原始纬度升序、经度升序筛选；共同域仅使用训练HF有效性、固定偏移有效性与ORAS5静态海洋掩膜，不根据测试温度缩小区域。矩阵为时间×格点；basis为格点×模态。', '',
        '| 区域 | 有效格点 | 训练矩阵 | POD模态 | 保留能量 | 对齐观测 |',
        '|---|---:|---|---:|---:|---:|']
    for name,info in report['regions'].items():
        label = '北大西洋侧70–80°N、20°W–20°E' if name == 'atlantic' else '中央北冰洋80°N以北'
        lines.append(f"| {label} | {info['nodes']} | {info['train_shape']} | {info['pod_modes']} | {info['pod_training_energy']:.4%} | {info['observations_kept']} |")
    lines += ['', '## 文件与使用边界', '',
        '每个区域的aligned_fields.npz保存物理场、统一异常、模型数组、坐标、掩膜、海冰与源网格权重；training_artifacts.npz保存训练气候态、均值、尺度、POD与协方差；aligned_observations.json保存严格质控后的逐剖面值及稀疏观测算子。alignment_report.json提供完整参数、计数及SHA256。',
        '海冰为0–1浓度，球面最近邻映射到目标格点；陆地、海岸、F17极区观测空洞与缺测保持NaN，不能按无冰处理。保留原QA标记，未把所有非零QA误删。',
        '点观测依然是瞬时剖面，尚需月平均代表性误差方案和训练/同化/验证用途划分。IAP是格点HF候选，ORAS5是LF候选；二者精度等级以及点观测同化独立性尚未证实。',
        '本次已完成数据对齐和训练期POD拟合，未执行重构性能实验。仅四个月LF，尚无完整训练/验证LF校准数据。原始回溯产品内部同化使用的数据，不能由本项目的时间划分消除。',
        '工程成本、可达性、设备能力和部署约束需另行提供。', '']
    (output/'README_zh.md').write_text('\n'.join(lines),encoding='utf-8')
    report['output_files']={str(p.relative_to(output)):digest(p) for p in sorted(output.rglob('*')) if p.is_file() and p.name != 'alignment_report.json'}
    (output/'alignment_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    manifest_path=base/'manifest.json'
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
        manifest['physical_alignment_status']='completed_50m_train_2000_2012_test_2014'
        manifest['alignment_report']=str(output/'alignment_report.json')
        manifest['alignment_readme']=str(output/'README_zh.md')
        manifest_path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('Completed alignment:',output,flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-root',type=Path,default=Path('D:/Dataset'))
    parser.add_argument('--base',type=Path,default=Path('D:/Dataset/RBMF-CS_polar_sample'))
    parser.add_argument('--output',type=Path,default=Path('D:/Dataset/RBMF-CS_polar_sample/aligned_50m'))
    args=parser.parse_args()
    main(args.data_root,args.base,args.output)
