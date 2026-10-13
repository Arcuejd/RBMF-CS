"""Explicit comparison estimators and truth-free, constraint-adapted planners.

These are adaptations, not claims of strict reproductions of published codes.
"""
from dataclasses import replace
from time import perf_counter
import numpy as np
from .model import RBMFCS, VBConfig
from .planning import (MCEIR, structural_entropy_gain, uncertainty_reduction,
                       redundancy, normalize_scores)
from .constraints import distances_km

RECONSTRUCTION = ('lf_only', 'pod_ridge', 'pod_lasso', 'gaussian_sbl_vb',
                  'deterministic_mf', 'rbmf', 'no_robust', 'no_mf')
DEPLOYMENT = ('random', 'maximin', 'pod_qr', 'entropy_qr', 'conditional_entropy',
              'posterior_variance', 'variance_reduction', 'mc_eir',
              'no_entropy', 'no_uncertainty', 'no_redundancy', 'no_cost')


def entropy_weights(training, direction='negative'):
    x = np.asarray(training, float)
    if x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError('finite training T x N with T>=2 required')
    if direction not in ('negative', 'positive'):
        raise ValueError('explicit indicator direction required')
    span = np.ptp(x, axis=0)
    z = (x.max(axis=0)-x) if direction == 'negative' else (x-x.min(axis=0))
    z = np.divide(z, span, out=np.zeros_like(z), where=span > 0)
    total = z.sum(axis=0)
    p = np.divide(z, total, out=np.zeros_like(z), where=total > 0)
    logp = np.log(p, out=np.zeros_like(p), where=p > 0)
    e = -np.sum(p*logp, axis=0)/np.log(len(x))
    e[span == 0] = 1.
    information = np.clip(1-e, 0, 1)
    if information.sum() <= 0:
        raise ValueError('entropy weights undefined for all-constant training')
    return information/information.sum()


def qr_residual_norm(basis, selected, candidates):
    """Column-pivoted QR residual norm, with forced existing sensor columns."""
    h = np.asarray(basis)[candidates]
    if selected:
        _, s, vh = np.linalg.svd(np.asarray(basis)[selected], full_matrices=False)
        rank = int(np.sum(s > max(np.asarray(basis)[selected].shape)*np.finfo(float).eps*s[0])) if len(s) else 0
        if rank == np.asarray(basis).shape[1]:
            return np.zeros(len(candidates))
        q = vh[:rank]
        h = h-(h@q.T)@q
    return np.linalg.norm(h, axis=1)


class ComparisonPlanner(MCEIR):
    def __init__(self, *args, method, coordinates, priorities, entropy_weight, **kwargs):
        super().__init__(*args, **kwargs)
        if method not in DEPLOYMENT:
            raise ValueError(method)
        self.method = method
        self.coordinates = np.asarray(coordinates)
        self.priorities = np.asarray(priorities)
        self.entropy_weight = np.asarray(entropy_weight)
        self.distance = distances_km(self.coordinates, 'latlon_degrees') if method == 'maximin' else None

    def score_candidates(self, model, selected, feasible_candidates):
        ids = sorted(feasible_candidates)
        cfg = self.config
        size = len(ids)
        dh, du, red = np.zeros((3, size))
        method = self.method
        combined = method in ('mc_eir', 'no_entropy', 'no_uncertainty', 'no_redundancy', 'no_cost')
        if method == 'conditional_entropy' or (combined and method != 'no_entropy'):
            dh = structural_entropy_gain(self.training_covariance, selected, ids, cfg)
        if method == 'variance_reduction' or (combined and method != 'no_uncertainty'):
            du = uncertainty_reduction(model.basis, model.cov_alpha_, model.basis[ids],
                                       1/(float(model.tau_.mean)*self.reliability[ids]))
        if combined and method != 'no_redundancy':
            red = redundancy(self.training_covariance, selected, ids, cfg.epsilon_r)
        nh, nu, nr, scales = normalize_scores(dh, du, red, cfg)
        cost = self.base_cost[ids]+cfg.maintenance_weight*self.maintenance_cost[ids]+cfg.failure_weight*self.failure_risk[ids]+cfg.epsilon_cost
        if combined:
            raw = dh+cfg.beta_raw*du-cfg.eta_raw*red
            score = (nh+cfg.beta*nu-cfg.eta*nr)/(1 if method == 'no_cost' else cost)
        elif method == 'random':
            score = self.priorities[ids]+1.
        elif method == 'maximin':
            score = self.distance[np.ix_(ids, selected)].min(axis=1) if size else np.array([])
        elif method in ('pod_qr', 'entropy_qr'):
            score = qr_residual_norm(model.basis, selected, ids)
            if method == 'entropy_qr':
                score *= self.entropy_weight[ids]
        elif method == 'conditional_entropy':
            score = dh
        elif method == 'posterior_variance':
            score = model.posterior_variance(ids)
        else:
            score = du
        if not combined:
            raw = score
        return {'candidates': [{'node': int(j), 'eir': float(score[i]),
                    'effective_information_raw': float(raw[i]), 'effective_cost': float(cost[i]),
                    'delta_h_raw': float(dh[i]), 'delta_u_raw': float(du[i]),
                    'redundancy_raw': float(red[i]), 'selection_rule': method}
                    for i, j in enumerate(ids)], 'normalization': scales,
                'posterior_fit_count': model.fit_count_}


def model_config(method, config):
    if method == 'gaussian_sbl_vb':
        return replace(config, robust_likelihood=False, multifidelity=False)
    if method == 'no_robust':
        return replace(config, robust_likelihood=False)
    if method == 'no_mf':
        return replace(config, multifidelity=False)
    return config


def ridge(h, y, regularization):
    if not np.isfinite(regularization) or regularization <= 0:
        raise ValueError('strictly positive ridge penalty required')
    return np.linalg.solve(h.T@h+regularization*np.eye(h.shape[1]), h.T@y)


def lasso(h, y, regularization, max_iter=10000, tolerance=1e-8):
    if not np.isfinite(regularization) or regularization <= 0:
        raise ValueError('positive L1 penalty required')
    alpha = np.zeros(h.shape[1])
    norms = np.sum(h*h, axis=0)
    residual = y.copy()
    history = []
    for iteration in range(1, max_iter+1):
        for j in range(len(alpha)):
            if norms[j] == 0:
                continue
            old = alpha[j]
            rho = h[:, j]@residual+norms[j]*old
            alpha[j] = np.sign(rho)*max(abs(rho)-regularization, 0)/norms[j]
            residual -= h[:, j]*(alpha[j]-old)
        gradient = -h.T@residual
        violation = np.where(alpha != 0, abs(gradient+regularization*np.sign(alpha)),
                             np.maximum(abs(gradient)-regularization, 0))
        gap = float(violation.max(initial=0))
        history.append({'iteration': iteration, 'kkt_violation': gap})
        if gap < tolerance:
            break
    return alpha, {'converged': gap < tolerance, 'iterations': iteration, 'history': history}


def reconstruct(method, basis, low, ids, observations, config=None, regularization=.1):
    if method not in RECONSTRUCTION:
        raise ValueError(method)
    start = perf_counter()
    ids, y = np.asarray(ids, int), np.asarray(observations, float)
    keep = np.isfinite(y)
    if np.isinf(y).any():
        raise ValueError('infinite observation')
    h, y = basis[ids[keep]], y[keep]
    diagnostics = {'converged': True, 'iterations': 1}
    model = None
    if method in ('rbmf', 'gaussian_sbl_vb', 'no_robust', 'no_mf'):
        model = RBMFCS(basis, model_config(method, config or VBConfig()))
        model.fit(low, observations, indices=ids)
        prediction = model.predict()
        diagnostics = model.diagnostics_
    elif method == 'lf_only':
        prediction = low.copy()
    elif method == 'pod_lasso':
        alpha, diagnostics = lasso(h, y, regularization)
        prediction = basis@alpha
    else:
        target = y-low[ids[keep]] if method == 'deterministic_mf' else y
        prediction = basis@ridge(h, target, regularization)
        if method == 'deterministic_mf':
            prediction += low
    elapsed = perf_counter()-start
    return prediction, model, diagnostics, elapsed


def perturbation(shape, seed, date, scenario, sigma=.02, stream=0):
    """Independent streams; node/time realizations shared across algorithms."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, date, stream]))
    noise = np.zeros(shape)
    missing = np.zeros(shape, bool)
    if scenario in ('gaussian', 'outlier', 'missing'):
        noise = rng.normal(0, sigma, shape)
    elif scenario == 'student':
        noise = rng.standard_t(3, shape)*sigma/np.sqrt(3)
    elif scenario not in ('clean', 'lf_bias'):
        raise ValueError(scenario)
    if scenario == 'outlier':
        noise += (rng.random(shape) < .05)*rng.choice([-0.5, .5], shape)
    if scenario == 'missing':
        missing = rng.random(shape) < .2
    return noise, missing


class SelectedObserver:
    """Acquisition-only API. Never passed to a planner."""
    def __init__(self, values, candidates):
        self.__values = np.array(values, copy=True)
        self.__allowed = frozenset(map(int, candidates))
        self.log = []

    def acquire(self, node):
        if node not in self.__allowed or any(r['node'] == node for r in self.log):
            raise ValueError('evaluation, duplicate or out-of-domain acquisition')
        value = float(self.__values[node])
        self.log.append({'node': int(node), 'value': value if np.isfinite(value) else None})
        return value


def fixed_domain(original_ids):
    ids = np.asarray(original_ids)
    evaluation = np.flatnonzero(ids % 5 == 0)
    candidates = np.flatnonzero(ids % 5 != 0)
    if not len(evaluation) or len(candidates) < 4:
        raise ValueError('insufficient fixed evaluation/candidate domain')
    return evaluation, candidates


def metrics(prediction, truth, evaluation, weights, denominator):
    error = np.asarray(prediction)[evaluation]-np.asarray(truth)[evaluation]
    w = np.asarray(weights)[evaluation]
    rmse = float(np.sqrt(np.average(error**2, weights=w)))
    if denominator <= 0 or not np.isfinite(denominator):
        raise ValueError('NRMSE requires positive fixed training denominator')
    return {'rmse_c': rmse, 'mae_c': float(np.average(abs(error), weights=w)),
            'nrmse': rmse/denominator, 'nrmse_denominator_c': float(denominator)}


def future_interval(model, seed, samples=4096):
    """Marginal 95% future-observation intervals, no truth/calibration input."""
    rng = np.random.default_rng(seed)
    tau = rng.gamma(float(model.tau_.shape), 1/float(model.tau_.rate), samples)
    lam = rng.gamma(model.config.nu/2, 2/model.config.nu, samples) if model.config.robust_likelihood else np.ones(samples)
    # Independent marginal draws suffice for marginal intervals, not joint coverage.
    latent = model.predict()+rng.normal(size=(samples, model.n))*np.sqrt(model.posterior_variance())
    future = latent+rng.normal(size=latent.shape)/np.sqrt((tau*lam)[:, None])
    return np.quantile(future, [.025, .975], axis=0)
