"""Audit downloaded fields and actual profile coverage; no reconstruction fit."""
import argparse
import collections
import csv
import io
import json
import re
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import gsw
import numpy as np
from netCDF4 import Dataset, num2date

ROOT = Path(__file__).resolve().parents[1]
MONTHS = ['201401', '201404', '201407', '201410']
TARGETS = [1, 10, 50, 100]
GAPS = {1: 2., 10: 5., 50: 10., 100: 10.}


def regions(lat, lon):
    return [n for n, yes in {
        'north70': lat >= 70,
        'original_window_70_80N_20_60E': 70 <= lat < 80 and 20 <= lon < 60,
        'atlantic_70_80N_20W_20E': 70 <= lat < 80 and -20 <= lon < 20,
        'canadian_sector_70N_170W_100W': lat >= 70 and -170 <= lon < -100,
        'central_north80': lat >= 80,
    }.items() if yes]


def coverage(pressure, temperature, latitude):
    good = np.isfinite(pressure) & np.isfinite(temperature) & (pressure >= 0) & (np.abs(temperature) < 100)
    depth = -gsw.z_from_p(pressure[good], latitude)
    depth = np.unique(depth[np.isfinite(depth)])
    result = {}
    for d in TARGETS:
        below, above = depth[depth <= d], depth[depth >= d]
        result[str(d)] = bool(len(below) and len(above) and above[0]-below[-1] <= GAPS[d])
    return result, (float(depth.min()), float(depth.max())) if len(depth) else (None, None)


def qc(v):
    return np.asarray(v[:]).astype('U1')


def profiles(base):
    records, errors = [], []
    for p in sorted((base / 'argo/profiles').rglob('*.nc')):
        try:
            with Dataset(p) as ds:
                for k in range(len(ds.dimensions['N_PROF'])):
                    date = num2date(ds['JULD'][k], ds['JULD'].units)
                    month = date.strftime('%Y%m')
                    lat, lon = float(ds['LATITUDE'][k]), float(ds['LONGITUDE'][k])
                    lon = (lon+180) % 360-180
                    if month not in MONTHS or not np.isfinite(lat) or lat < 70:
                        continue
                    mode = qc(ds['DATA_MODE'])[k]
                    suffix = '_ADJUSTED' if mode in ['D', 'A'] else ''
                    pres = np.ma.asarray(ds['PRES'+suffix][k]).filled(np.nan)
                    temp = np.ma.asarray(ds['TEMP'+suffix][k]).filled(np.nan)
                    mask = ((qc(ds['PRES'+suffix+'_QC'])[k] == '1') & (qc(ds['TEMP'+suffix+'_QC'])[k] == '1'))
                    cov, bounds = coverage(np.where(mask, pres, np.nan), np.where(mask, temp, np.nan), lat)
                    pq = qc(ds['POSITION_QC'])[k]
                    tq = qc(ds['JULD_QC'])[k]
                    records.append({'source': 'Argo', 'file': str(p), 'profile': k,
                        'platform': p.parent.parent.name, 'month': month, 'day': date.strftime('%Y%m%d'),
                        'lat': lat, 'lon': lon, 'mode': mode, 'position_qc': pq, 'time_qc': tq,
                        'strict_delayed_position_time': bool(mode == 'D' and pq in ['1', '2'] and tq in ['1', '2']),
                        'coverage': cov, 'min_depth_m': bounds[0], 'max_depth_m': bounds[1]})
        except Exception as exc:
            errors.append({'file': str(p), 'error': str(exc)})
    for p in sorted((base / 'itp/final').glob('*.zip')):
        with zipfile.ZipFile(p) as z:
            for name in z.namelist():
                if not re.search(r'itp\d+grd\d+\.dat$', name):
                    continue
                try:
                    text = z.read(name).decode('ascii', errors='replace')
                    lines = text.splitlines()
                    if len(lines) < 4 or not lines[0].startswith('%ITP'):
                        continue
                    meta = [float(x) for x in lines[1].split()]
                    year, day, lon, lat = meta[:4]
                    date = datetime(int(year), 1, 1) + timedelta(days=day-1)
                    month = date.strftime('%Y%m')
                    lon = (lon+180) % 360-180
                    if month not in MONTHS or not np.isfinite(lat) or not 70 <= lat <= 90:
                        continue
                    a = np.loadtxt(io.StringIO('\n'.join(lines[3:])), comments='%', usecols=(0, 1), ndmin=2)
                    cov, bounds = coverage(a[:, 0], a[:, 1], lat)
                    records.append({'source': 'ITP_Level3', 'file': str(p)+'::'+name, 'profile': name,
                        'platform': p.stem.replace('final', ''), 'month': month, 'day': date.strftime('%Y%m%d'),
                        'lat': lat, 'lon': lon, 'mode': 'Level3', 'coverage': cov,
                        'min_depth_m': bounds[0], 'max_depth_m': bounds[1]})
                except Exception as exc:
                    errors.append({'file': str(p)+'::'+name, 'error': str(exc)})
        print('Processed', p.name, flush=True)
    # N_PROF can contain multiple vertical sampling schemes for one float cycle.
    # Count each GDAC profile file once; union only already-QC'd depth support.
    argo = {}
    for r in records:
        if r['source'] != 'Argo':
            continue
        key = r['file']
        strict = {str(d):r['strict_delayed_position_time'] and r['coverage'][str(d)] for d in TARGETS}
        if key not in argo:
            r['sampling_segments'] = 1
            r['strict_depth_coverage'] = strict
            argo[key] = r
        else:
            old = argo[key]
            old['sampling_segments'] += 1
            old['coverage'] = {str(d):old['coverage'][str(d)] or r['coverage'][str(d)] for d in TARGETS}
            old['strict_depth_coverage'] = {str(d):old['strict_depth_coverage'][str(d)] or strict[str(d)] for d in TARGETS}
            bounds = [v for v in [old['min_depth_m'], r['min_depth_m']] if v is not None]
            old['min_depth_m'] = min(bounds) if bounds else None
            bounds = [v for v in [old['max_depth_m'], r['max_depth_m']] if v is not None]
            old['max_depth_m'] = max(bounds) if bounds else None
    return list(argo.values()) + [r for r in records if r['source'] != 'Argo'], errors


def summarize(records):
    summary = {}
    names = ['north70', 'original_window_70_80N_20_60E', 'atlantic_70_80N_20W_20E',
             'canadian_sector_70N_170W_100W', 'central_north80']
    for source in ['Argo', 'ITP_Level3']:
        summary[source] = {}
        for name in names:
            summary[source][name] = {}
            for month in MONTHS:
                rs = [r for r in records if r['source'] == source and r['month'] == month and name in regions(r['lat'], r['lon'])]
                summary[source][name][month] = {
                    'profiles': len(rs), 'platforms': len({r['platform'] for r in rs}),
                    'days': len({r['day'] for r in rs}),
                    'one_degree_cells': len({(int(np.floor(r['lat'])), int(np.floor(r['lon']))) for r in rs}),
                    'depth_bracket_profiles': {str(d): sum(r['coverage'][str(d)] for r in rs) for d in TARGETS},
                    'depth_bracket_platforms': {str(d): len({r['platform'] for r in rs if r['coverage'][str(d)]}) for d in TARGETS},
                    'strict_argo_depth_profiles': {str(d): sum(r['strict_depth_coverage'][str(d)] for r in rs) for d in TARGETS} if source == 'Argo' else None,
                    'min_depth_m': min((r['min_depth_m'] for r in rs if r['min_depth_m'] is not None), default=None),
                    'max_depth_m': max((r['max_depth_m'] for r in rs if r['max_depth_m'] is not None), default=None),
                    'latitude_range': [min(r['lat'] for r in rs), max(r['lat'] for r in rs)] if rs else None,
                    'longitude_range': [min(r['lon'] for r in rs), max(r['lon'] for r in rs)] if rs else None,
                }
    return summary


def fields(base):
    output = []
    with Dataset(base / 'oras5/tmask_r1x1.nc') as mask_ds:
        ocean_mask = np.ma.asarray(mask_ds['tmask'][:]).filled(0) == 1
    for p in sorted((base / 'oras5').glob('*.nc')) + sorted((base / 'ice').glob('*.nc')):
        with Dataset(p) as ds:
            entry = {'file': str(p), 'dimensions': {n:len(v) for n,v in ds.dimensions.items()},
                     'variables': {n:{'shape':list(v.shape), 'units':str(getattr(v, 'units', ''))} for n,v in ds.variables.items()}}
            for n in ['time_counter', 'time']:
                if n in ds.variables:
                    v = ds[n]
                    if 'since' in getattr(v, 'units', ''):
                        entry['decoded_time'] = [str(x) for x in np.atleast_1d(num2date(v[:], v.units))]
                    else:
                        entry['undecoded_time'] = {'units':str(getattr(v, 'units', '')), 'values':np.asarray(v[:]).tolist()}
            if 'deptht' in ds.variables:
                entry['depth_m'] = ds['deptht'][:].tolist()
            for n in ['votemper', 'vosaline', 'cdr_seaice_conc_monthly']:
                if n not in ds.variables:
                    continue
                a = np.ma.asarray(ds[n][:]).filled(np.nan)
                valid = np.isfinite(a)
                if n in ['votemper', 'vosaline']:
                    if a.shape != ocean_mask.shape:
                        raise ValueError('ORAS5 field/mask shape mismatch')
                    valid &= ocean_mask
                    entry['mask_applied'] = 'tmask_r1x1.nc:tmask'
                if n.startswith('cdr'):
                    valid &= (a >= 0) & (a <= 1)
                entry[n] = {'valid_values':int(valid.sum()), 'min':float(a[valid].min()), 'max':float(a[valid].max())}
            output.append(entry)
    return output


def main(base, output):
    output.mkdir(parents=True, exist_ok=True)
    records, errors = profiles(base)
    report = {'sample_months':MONTHS, 'source_root':str(base), 'depth_bracket_max_gap_m':GAPS,
              'pressure_to_depth':'TEOS-10 gsw.z_from_p, zero sea surface height',
              'coverage':summarize(records), 'errors':errors, 'fields':fields(base),
              'argo_counting':'one GDAC file/float cycle, union of QC-valid vertical sampling segments',
              'itp_selected':json.loads((base/'itp_selected.json').read_text()),
              'limitations':['ITP counts include only available Level III archives, not all systems with trajectories',
                            'Argo depth counts require temperature and pressure QC=1; strict counts also require delayed mode and position/time QC 1 or 2',
                            'Vertical bracketing is not proof of monthly representativeness, independent validation or deployment feasibility']}
    (output / 'coverage.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (output / 'profile_inventory.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    print(json.dumps(report['coverage'], indent=2), flush=True)
    print('ERRORS', len(errors), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=Path('D:/Dataset/RBMF-CS_polar_sample'))
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/polar_sample_audit')
    args = parser.parse_args()
    main(args.input, args.output)
