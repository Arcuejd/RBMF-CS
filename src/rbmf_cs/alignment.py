"""Explicit spatial operators and training-only seasonal preprocessing."""
import numpy as np
from .basis import TrainingTransform, PODBasis, TrainingCovariance


def vertical_bracket(depths, target):
    z = np.asarray(depths, float)
    if z.ndim != 1 or not np.isfinite(z).all() or np.any(np.diff(z) <= 0):
        raise ValueError('depths must be finite and strictly increasing')
    if not z[0] <= target <= z[-1]:
        raise ValueError('vertical extrapolation is prohibited')
    upper = int(np.searchsorted(z, target, side='left'))
    if z[upper] == target:
        return upper, upper, 0.
    lower = upper-1
    return lower, upper, float((target-z[lower])/(z[upper]-z[lower]))


def periodic_stencil(latitudes, longitudes, points):
    """Bilinear regular global lon-grid sampling; returns flat C-order indices."""
    lat, lon, p = map(lambda a: np.asarray(a, float), (latitudes, longitudes, points))
    if lat.ndim != 1 or lon.ndim != 1 or min(len(lat), len(lon)) < 2 or p.ndim != 2 or p.shape[1] != 2:
        raise ValueError('expected 1D grids and (P,2) latitude/longitude points')
    dl = np.diff(lon)
    if not np.allclose(dl, dl[0]) or not np.isclose(dl[0]*len(lon), 360) or np.any(np.diff(lat) <= 0):
        raise ValueError('requires ascending latitudes and regular periodic longitudes')
    if not np.isfinite(p).all() or np.any((p[:,0] < lat[0]) | (p[:,0] > lat[-1])):
        raise ValueError('spatial extrapolation is prohibited')
    y = np.clip(np.searchsorted(lat, p[:,0], side='right')-1, 0, len(lat)-2)
    fy = (p[:,0]-lat[y])/(lat[y+1]-lat[y])
    xf = ((p[:,1]-lon[0]) % 360)/dl[0]
    x = np.floor(xf).astype(int) % len(lon)
    fx = xf-np.floor(xf)
    nx = (x+1) % len(lon)
    indices = np.column_stack((y*len(lon)+x, y*len(lon)+nx, (y+1)*len(lon)+x, (y+1)*len(lon)+nx))
    weights = np.column_stack(((1-fy)*(1-fx), (1-fy)*fx, fy*(1-fx), fy*fx))
    return indices, weights


def sample_stencil(flat_values, indices, weights):
    """Reject a stencil if any positive-weight source is missing (no filling)."""
    values = np.asarray(flat_values, float)[..., indices]
    needed = weights > 1e-12
    valid = np.all(np.isfinite(values) | ~needed, axis=-1)
    total = np.sum(np.where(needed, values, 0.)*weights, axis=-1)
    return np.where(valid, total, np.nan)


def training_artifacts(absolute_hf, dates, *, train_last=201212, energy_fraction=.99):
    """Fit a 12-month climatology and fixed POD from an explicitly bounded set."""
    dates = np.asarray(dates, int)
    x = np.asarray(absolute_hf, float)
    if x.ndim != 2 or x.shape[0] != len(dates) or not np.isfinite(x).all():
        raise ValueError('training HF must be a complete (T,N) matrix')
    if len(set(dates.tolist())) != len(dates) or np.any(dates > train_last):
        raise ValueError('training dates overlap reserved validation/test period')
    month = dates % 100
    if set(month.tolist()) != set(range(1,13)) or not 0 < energy_fraction <= 1:
        raise ValueError('all twelve calendar months and valid energy fraction required')
    climatology = np.stack([x[month == m].mean(axis=0) for m in range(1,13)])
    anomaly = x-climatology[month-1]
    transform = TrainingTransform.fit(anomaly)
    normalized = transform.transform(anomaly)
    full = PODBasis.fit(normalized, min(normalized.shape))
    cumulative = np.cumsum(full.singular_values**2)/np.sum(full.singular_values**2)
    k = min(int(np.searchsorted(cumulative, energy_fraction))+1, len(cumulative))
    return {'monthly_climatology_c': climatology, 'mean_c':transform.mean, 'scale_c':np.array(transform.scale),
            'basis':full.matrix[:,:k], 'singular_values':full.singular_values[:k],
            'pod_energy_fraction':np.array(cumulative[k-1]),
            'covariance_factor':TrainingCovariance.fit(normalized).factor,
            'train_dates':dates, 'train_anomaly_c':anomaly, 'train_model':normalized}
