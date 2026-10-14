"""Normalized calibrated-LF Gaussian residual model; independent of RBMF."""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize_scalar
from scipy.stats import norm

LEVELS = np.array([.5, .8, .9, .95])


def physical(mean, variance, baseline, scale):
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('positive physical scale required')
    return baseline + scale * np.asarray(mean), scale**2 * np.asarray(variance)


def check_dates(dates, allowed):
    dates = np.asarray(dates)
    if dates.ndim != 1 or dates.dtype.kind not in 'iu' or len(set(dates.tolist())) != len(dates):
        raise ValueError('unique one-dimensional months required')
    if not set(dates).issubset(set(allowed)) or np.any(dates // 100 != 2013):
        raise ValueError('calibration may only access explicitly permitted 2013 months')


@dataclass
class ResidualPrior:
    basis: np.ndarray
    intercept: float
    slope: float
    coefficient_mean: np.ndarray
    coefficient_variance: float
    discrepancy_variance: float
    training_dates: list
    calibration_rank: int

    def metadata(self):
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in vars(self).items() if k != 'basis'}


def learn_prior(hf, lf, dates, allowed_dates, basis, weights):
    check_dates(dates, allowed_dates)
    x, l, b, w = map(lambda a: np.asarray(a, float), (hf, lf, basis, weights))
    if x.ndim != 2 or x.shape != l.shape or len(x) != len(dates) or len(x) < 3:
        raise ValueError('at least three paired training months required')
    if b.shape[0] != x.shape[1] or w.shape != (x.shape[1],):
        raise ValueError('spatial shape mismatch')
    if not all(np.isfinite(v).all() for v in (x, l, b, w)) or np.any(w <= 0):
        raise ValueError('finite training arrays and positive weights required')
    if not np.allclose(b.T @ b, np.eye(b.shape[1]), atol=1e-10, rtol=0):
        raise ValueError('fixed Euclidean orthonormal basis required')
    w = w / w.sum()
    design = np.column_stack([np.ones(x.size), l.ravel()])
    sw = np.sqrt(np.tile(w / len(x), len(x)))
    theta, _, rank, _ = np.linalg.lstsq(sw[:, None] * design, sw * x.ravel(), rcond=1e-10)
    if rank < 2:
        theta = np.array([np.mean(x @ w), 0.])
    residual = x - theta[0] - theta[1] * l
    coefficients = residual @ b
    m = coefficients.mean(axis=0)
    c = max(float(np.sum((coefficients-m)**2) / ((len(x)-1)*b.shape[1])), 1e-8)
    d = max(float(np.mean((residual-coefficients@b.T)**2 @ w)), 1e-8)
    return ResidualPrior(b.copy(), float(theta[0]), float(theta[1]), m, c, d,
                         list(map(int, dates)), int(rank))


class CLBR:
    def __init__(self, prior, multiplier=1., scale_c=1., noise_std_bounds_c=(.001, 1.)):
        self.prior = prior
        self.g = float(multiplier)
        self.scale = float(scale_c)
        lo, hi = noise_std_bounds_c
        if not (np.isfinite([self.g, self.scale, lo, hi]).all() and self.g > 0 and self.scale > 0 and 0 < lo < hi):
            raise ValueError('invalid scale, prior multiplier or noise bounds')
        self.bounds = (np.asarray([lo, hi]) / self.scale)**2

    def fit(self, low, observations, indices, *, remove_lf=False, fixed_noise_variance=None):
        p = self.prior
        b = p.basis
        n, k = b.shape
        low = np.asarray(low, float)
        y = np.asarray(observations, float)
        ids = np.asarray(indices)
        if ids.size == 0:
            ids = ids.astype(int)
        if low.shape != (n,) or not np.isfinite(low).all():
            raise ValueError('finite aligned LF required, including ablation')
        if y.ndim != 1 or ids.shape != y.shape or ids.dtype.kind not in 'iu':
            raise ValueError('one-dimensional observations and integer indices required')
        if np.any(ids < 0) or np.any(ids >= n) or np.isinf(y).any():
            raise ValueError('invalid index or infinite observation')
        keep = ~np.isnan(y)
        y, ids = y[keep], ids[keep]
        self.ids_, self.y_ = ids, y
        self.lf_mean_ = np.zeros(n) if remove_lf else p.intercept + p.slope * low
        u = self.lf_mean_ + b @ p.coefficient_mean
        c = self.g * p.coefficient_variance
        d = p.discrepancy_variance
        if c <= 0 or d <= 0 or not np.isfinite([c, d]).all():
            raise ValueError('strictly positive finite prior variances required')
        h = b[ids]
        match = ids[:, None] == ids[None, :]
        latent_sensor = c * (h @ h.T) + d * match
        r = y - u[ids]

        def nll(log_variance):
            s = latent_sensor + np.exp(log_variance) * np.eye(len(ids))
            chol = cho_factor(s, lower=True)
            return .5 * (len(ids)*np.log(2*np.pi) + 2*np.log(np.diag(chol[0])).sum() + r@cho_solve(chol, r))

        if fixed_noise_variance is not None:
            if not np.isfinite(fixed_noise_variance) or fixed_noise_variance <= 0:
                raise ValueError('conditional noise variance must be positive')
            noise = float(fixed_noise_variance)
            noise_status = 'fixed_conditional_reference'
        elif not len(ids):
            noise = float(self.bounds[0])
            noise_status = 'unidentified_no_observations'
        else:
            grid = np.linspace(*np.log(self.bounds), 13)
            values = [float(nll(v)) for v in grid]
            j = int(np.argmin(values))
            opt = minimize_scalar(nll, bounds=(grid[max(0, j-1)], grid[min(12, j+1)]), method='bounded',
                                  options={'xatol': 1e-7, 'maxiter': 100})
            if not opt.success:
                raise RuntimeError('online noise marginal-likelihood optimizer failed')
            options = [(values[t], grid[t]) for t in range(13)] + [(float(opt.fun), float(opt.x))]
            _, best = min(options)
            noise = float(np.exp(best))
            noise_status = 'lower_bound' if np.isclose(noise, self.bounds[0], rtol=1e-6, atol=0) else (
                'upper_bound' if np.isclose(noise, self.bounds[1], rtol=1e-6, atol=0) else 'interior')
        self.noise_variance_ = noise
        # Full cross covariance includes persistent discrepancy at repeated locations.
        cross = c * (b @ h.T) + d * (np.arange(n)[:, None] == ids[None, :])
        s = latent_sensor + noise * np.eye(len(ids))
        solved = np.linalg.solve(s, r)
        gain = np.linalg.solve(s, cross.T).T
        self.mean_ = u + cross @ solved
        self.variance_ = np.maximum(c*np.sum(b*b, axis=1) + d - np.sum(gain*cross, axis=1), 0.)
        self.coefficient_mean_ = p.coefficient_mean + c * h.T @ solved
        self.coefficient_covariance_ = c*np.eye(k) - c*c * h.T @ np.linalg.solve(s, h)
        self.gain_ = gain
        self.diagnostics_ = {'converged': True, 'noise_status': noise_status,
            'noise_std_c': float(np.sqrt(noise)*self.scale), 'observations': len(ids),
            'dropped_nan': int((~keep).sum()), 'effective_dof_frozen': float(np.trace(np.linalg.solve(s, latent_sensor))),
            'conditional_log_evidence': -float(nll(np.log(noise))) if len(ids) else 0.}
        return self

    def deterministic_prediction(self):
        """Independent augmented least-squares MAP; matches Bayes on unobserved nodes."""
        p = self.prior
        h = p.basis[self.ids_]
        c = self.g*p.coefficient_variance
        r = p.discrepancy_variance*(self.ids_[:, None] == self.ids_[None, :]) + self.noise_variance_*np.eye(len(h))
        chol = np.linalg.cholesky(r)
        design = np.vstack([np.linalg.solve(chol, h), np.eye(h.shape[1])/np.sqrt(c)])
        rhs = np.r_[np.linalg.solve(chol, self.y_-self.lf_mean_[self.ids_]), p.coefficient_mean/np.sqrt(c)]
        a = np.linalg.lstsq(design, rhs, rcond=None)[0]
        return self.lf_mean_ + p.basis @ a

    def intervals(self, future=False, factors=None):
        factors = norm.ppf((1+LEVELS)/2) if factors is None else np.asarray(factors)
        variance = self.variance_ + (self.noise_variance_ if future else 0)
        half = np.sqrt(variance)[:, None] * factors
        return self.mean_[:, None]-half, self.mean_[:, None]+half


def weighted_quantiles(values, weights, quantiles=LEVELS):
    x, w = np.asarray(values, float).ravel(), np.asarray(weights, float).ravel()
    if x.shape != w.shape or not len(x) or not np.isfinite(x).all() or not np.isfinite(w).all() or np.any(w <= 0):
        raise ValueError('finite scores with positive weights required')
    order = np.argsort(x, kind='stable')
    cdf = np.cumsum(w[order])/w.sum()
    return x[order][np.minimum(np.searchsorted(cdf, quantiles, side='left'), len(x)-1)]


def interval_metrics(truth, mean, lower, upper, weights):
    truth, mean, w = np.asarray(truth), np.asarray(mean), np.asarray(weights)
    lower, upper = np.asarray(lower), np.asarray(upper)
    if lower.shape != (len(truth), 4) or upper.shape != lower.shape or np.any(upper < lower):
        raise ValueError('ordered four-level intervals required')
    alpha = 1-LEVELS
    width = upper-lower
    covered = (truth[:, None] >= lower) & (truth[:, None] <= upper)
    score = width + 2/alpha*(np.maximum(lower-truth[:, None], 0)+np.maximum(truth[:, None]-upper, 0))
    wis = (.5*abs(truth-mean) + np.sum(alpha/2*score, axis=1))/4.5
    return {'coverage': np.average(covered, axis=0, weights=w), 'coverage_unweighted': covered.mean(axis=0),
            'width': np.average(width, axis=0, weights=w), 'interval_score': np.average(score, axis=0, weights=w),
            'wis': float(np.average(wis, weights=w))}
