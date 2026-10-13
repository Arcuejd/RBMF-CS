"""Read-only inspection of local IAP files; reports are written in the project."""
import argparse
import json
import re
import zipfile
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

ROOT = Path(__file__).resolve().parents[1]


def plain(value):
    if isinstance(value, bytes):
        return value.decode('utf-8', errors='replace')
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def metadata(ds):
    out = {'dimensions': {k: len(v) for k, v in ds.dimensions.items()},
           'attributes': {k: plain(ds.getncattr(k)) for k in ds.ncattrs()}, 'variables': {}}
    for name, v in ds.variables.items():
        item = {'dimensions': v.dimensions, 'shape': v.shape, 'dtype': str(v.dtype),
                'attributes': {k: plain(v.getncattr(k)) for k in v.ncattrs()}}
        if v.ndim <= 1:
            a = np.ma.asarray(v[:])
            item['values'] = plain(a.filled(np.nan)) if np.issubdtype(a.dtype, np.floating) else plain(a.data)
        out['variables'][name] = item
    return out


def open_member(z, name):
    return Dataset('in_memory.nc', memory=z.read(name))


def inspect(root, output):
    result = {'root': str(root), 'merged': {}, 'archives': []}
    for p in sorted(root.rglob('*.nc')):
        with Dataset(p) as ds:
            m = metadata(ds)
            for name, v in ds.variables.items():
                if v.ndim >= 2:
                    a = np.ma.asarray(v[:])
                    valid = ~np.ma.getmaskarray(a) & np.isfinite(a.data)
                    m['variables'][name]['valid_count'] = int(valid.sum())
                    m['variables'][name]['valid_min'] = float(a.data[valid].min()) if valid.any() else None
                    m['variables'][name]['valid_max'] = float(a.data[valid].max()) if valid.any() else None
            result['merged'][str(p)] = m
    for p in sorted(root.rglob('*.zip')):
        with zipfile.ZipFile(p) as z:
            names = sorted(n for n in z.namelist() if n.endswith('.nc'))
            months = [re.search(r'year_(\d{4})_month_(\d{2})', n).groups() for n in names]
            item = {'path': str(p), 'count': len(names), 'months': ['-'.join(m) for m in months]}
            with open_member(z, names[0]) as ds:
                item['first_metadata'] = metadata(ds)
            result['archives'].append(item)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'metadata.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def scan(root, output):
    products = {}
    depth_targets = [1., 10., 50., 100.]
    regions = {
        'global': (-90., 90.), 'north_ge60': (60., 90.),
        'north_ge66_5': (66.5, 90.), 'north_ge70': (70., 90.),
        'north_ge80': (80., 90.), 'south_le60': (-90., -60.),
        'south_le66_5': (-90., -66.5), 'south_le70': (-90., -70.),
        'south_60_70': (-70., -60.),
    }
    merged_checks = {}
    for product, var in [('salinity', 'salinity'), ('temperature_anomaly', 'temp')]:
        paths = sorted(root.rglob('Salinity*.zip' if product == 'salinity' else 'Temperature*.zip'))
        snapshots, dates = [], []
        merged_path = root / 'IAP_merged' / (product + '_surface_2000_2009.nc')
        with Dataset(merged_path) as ds:
            merged = np.ma.asarray(ds[var][:]).filled(np.nan)
            merged_dates = ds['time'][:].astype(int)
            merged_lat, merged_lon = np.asarray(ds['lat'][:]), np.asarray(ds['lon'][:])
        checks = {'checked_months': 0, 'mismatched_months': [], 'grid_equal': True}
        latitude = longitude = depths = None
        for p in paths:
            with zipfile.ZipFile(p) as z:
                for name in sorted(n for n in z.namelist() if n.endswith('.nc')):
                    date_parts = re.search(r'year_(\d{4})_month_(\d{2})', name).groups()
                    date = int(''.join(date_parts))
                    with open_member(z, name) as ds:
                        lat, lon = np.asarray(ds['lat'][:]), np.asarray(ds['lon'][:])
                        dep = np.asarray(ds['depth_std'][:])
                        if latitude is None:
                            latitude, longitude, depths = lat, lon, dep
                        if not (np.array_equal(latitude, lat) and np.array_equal(longitude, lon)
                                and np.array_equal(depths, dep)):
                            raise ValueError('Grid or depth changed: ' + name)
                        idx = [int(np.flatnonzero(dep == d)[0]) for d in depth_targets]
                        v = ds[var]
                        a = np.ma.asarray(v[:, :, idx])
                        values = np.asarray(a.filled(np.nan), dtype=float)
                        # Salinity stores its fill value as the string "999".
                        for attr in ('_FillValue', 'missing_value'):
                            if attr in v.ncattrs():
                                fill = float(v.getncattr(attr))
                                values[values == fill] = np.nan
                        values[~np.isfinite(values)] = np.nan
                        if int(ds['time'][0]) != date:
                            raise ValueError('Date disagrees with filename: ' + name)
                    dates.append(date)
                    snapshots.append(values)
                    found = np.flatnonzero(merged_dates == date)
                    if len(found):
                        checks['checked_months'] += 1
                        if not np.array_equal(values[:, :, 0], merged[found[0]], equal_nan=True):
                            checks['mismatched_months'].append(date)
                        checks['grid_equal'] &= bool(np.array_equal(lat, merged_lat) and np.array_equal(lon, merged_lon))
                    if len(dates) % 40 == 0:
                        print(f'{product}: scanned {len(dates)} months, latest {date}', flush=True)
        dates = np.asarray(dates)
        data = np.stack(snapshots)
        finite = np.isfinite(data)
        train = dates < 202001
        common_train = finite[train].all(axis=0)
        common_all = finite.all(axis=0)
        expected = [y * 100 + m for y in range(2000, 2027) for m in range(1, 13) if y * 100 + m <= dates[-1]]
        summary = {'variable': var, 'count': len(dates), 'first': int(dates[0]), 'last': int(dates[-1]),
                   'missing_months': sorted(set(expected) - set(dates.tolist())),
                   'duplicate_months': len(dates) - len(set(dates.tolist())),
                   'train_months_before_2020': int(train.sum()), 'regions': {}, 'valid_by_latitude': {}}
        for region, (low, high) in regions.items():
            mask = (latitude >= low) & (latitude <= high)
            summary['regions'][region] = {}
            for k, depth in enumerate(depth_targets):
                f = finite[..., k][:, mask, :]
                valid_counts = f.sum(axis=(1, 2))
                ntrain = int(common_train[mask, :, k].sum())
                test_valid = f[~train][:, common_train[mask, :, k]]
                summary['regions'][region][str(int(depth))] = {
                    'grid_cells': int(mask.sum() * len(longitude)),
                    'min_valid_per_month': int(valid_counts.min()), 'max_valid_per_month': int(valid_counts.max()),
                    'common_train_nodes': ntrain, 'common_all_nodes': int(common_all[mask, :, k].sum()),
                    'train_matrix_shape': [int(train.sum()), ntrain],
                    'test_missing_on_train_domain': int((~test_valid).sum()),
                    'constant_train_nodes': int((np.ptp(data[train][..., k][:, mask, :][:, common_train[mask, :, k]], axis=0) == 0).sum()),
                    'range_on_common_all': [float(np.min(data[..., k][:, mask, :][:, common_all[mask, :, k]])),
                                            float(np.max(data[..., k][:, mask, :][:, common_all[mask, :, k]]))] if common_all[mask, :, k].any() else None,
                    'months_with_zero_valid_cells': dates[valid_counts == 0].tolist(),
                }
        for k, depth in enumerate(depth_targets):
            summary['valid_by_latitude'][str(int(depth))] = [
                {'latitude': float(lat), 'common_all_nodes': int(common_all[j, :, k].sum()),
                 'common_train_nodes': int(common_train[j, :, k].sum())}
                for j, lat in enumerate(latitude) if abs(lat) >= 60]
        products[product] = (summary, common_train, common_all, dates)
        merged_checks[product] = checks
        print(f'{product}: complete; merged comparison {checks}', flush=True)
    report = {'depths_checked_m': depth_targets, 'train_period_for_audit': '2000-01 through 2019-12',
              'products': {p: v[0] for p, v in products.items()}, 'merged_checks': merged_checks,
              'cross_variable_common_nodes': {}}
    for region, (low, high) in regions.items():
        mask = (latitude >= low) & (latitude <= high)
        common = products['salinity'][2] & products['temperature_anomaly'][2]
        report['cross_variable_common_nodes'][region] = {
            str(int(d)): int(common[mask, :, k].sum()) for k, d in enumerate(depth_targets)}
    report['product_dates_equal'] = bool(np.array_equal(products['salinity'][3], products['temperature_anomaly'][3]))
    output.mkdir(parents=True, exist_ok=True)
    (output / 'coverage.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    np.savez_compressed(output / 'valid_domain_masks.npz', latitude=latitude, longitude=longitude,
                        depths_m=depth_targets, dates=products['salinity'][3],
                        salinity_common_train=products['salinity'][1], salinity_common_all=products['salinity'][2],
                        temperature_common_train=products['temperature_anomaly'][1],
                        temperature_common_all=products['temperature_anomaly'][2])
    print(f'Coverage report saved to {output / "coverage.json"}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('D:/Dataset'))
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs' / 'dataset_audit')
    parser.add_argument('--scan', action='store_true')
    args = parser.parse_args()
    if args.scan:
        scan(args.root, args.output)
    else:
        inspect(args.root, args.output)
