"""MC-EIR (5-1--21): fixed-hyperparameter Gaussian planning, never refitting.

Only historical covariance, current posterior, known costs and historical
reliability enter scoring. There is deliberately no observation callback here.
"""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from .constraints import node_ids


@dataclass(frozen=True)
class PlanningConfig:
    entropy_reference_variance: float = 0.01
    epsilon_k: float = 1e-8
    epsilon_r: float = 1e-8
    epsilon_h: float = 1e-12
    epsilon_u: float = 1e-12
    epsilon_cost: float = 1e-12
    beta: float = 1.
    eta: float = 0.2
    maintenance_weight: float = 1.
    failure_weight: float = 1.
    beta_raw: float = 1.
    eta_raw: float = 0.2
    eir_threshold: float = 0.

    def __post_init__(self):
        for name in ('entropy_reference_variance', 'epsilon_k', 'epsilon_r',
                     'epsilon_h', 'epsilon_u', 'epsilon_cost'):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be finite and positive')
        for name in ('beta', 'eta', 'maintenance_weight', 'failure_weight',
                     'beta_raw', 'eta_raw', 'eir_threshold'):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f'{name} must be finite and nonnegative')


def _nonnegative(values, scale=1.):
    values = np.asarray(values, float)
    if not np.isfinite(values).all() or np.any(values < -1e-10*max(1., float(scale))):
        raise FloatingPointError('invalid negative or nonfinite covariance-derived gain')
    return np.maximum(values, 0.)


def structural_entropy_gain(training_covariance, selected, candidates, config):
    """5-2/3: conditional training variance, Cholesky solve and fixed reference."""
    if not len(candidates):
        return np.empty(0)
    diagonal = training_covariance.diagonal()[candidates]
    residual = diagonal.copy()
    if len(selected):
        cross = training_covariance.block(candidates, selected)
        block = training_covariance.block(selected, selected) + config.epsilon_k*np.eye(len(selected))
        residual -= np.sum(cross*cho_solve(cho_factor(block, lower=True), cross.T).T, axis=1)
    residual = _nonnegative(residual, np.max(diagonal, initial=0.))
    return .5*np.log1p(residual/config.entropy_reference_variance)


def uncertainty_reduction(basis, covariance, candidate_rows, planning_variance):
    """5-11 for arbitrary (not necessarily orthonormal) fixed bases."""
    psi, sigma, h = map(lambda x: np.asarray(x, float), (basis, covariance, candidate_rows))
    r = np.asarray(planning_variance, float)
    if psi.ndim != 2 or h.ndim != 2 or sigma.shape != (psi.shape[1], psi.shape[1]) or h.shape[1] != psi.shape[1]:
        raise ValueError('incompatible basis, covariance or candidate dimensions')
    if not all(np.isfinite(x).all() for x in (psi, sigma, h, r)) or np.any(r <= 0):
        raise ValueError('finite inputs and positive planning variances required')
    if r.ndim > 1 or (r.ndim == 1 and r.shape != (len(h),)):
        raise ValueError('planning variance must be scalar or one per candidate')
    if not np.allclose(sigma, sigma.T, atol=1e-12, rtol=1e-10) or np.linalg.eigvalsh(sigma).min() < -1e-12:
        raise ValueError('coefficient covariance must be symmetric PSD')
    v = h@sigma
    numerator = np.einsum('ij,jk,ik->i', v, psi.T@psi, v)
    denominator = r+np.sum(v*h, axis=1)
    return _nonnegative(numerator/denominator)


def rank_one_covariance(covariance, row, planning_variance):
    """5-5, diagnostic/planning utility; never substitutes for robust update."""
    if not np.isfinite(planning_variance) or planning_variance <= 0:
        raise ValueError('positive finite planning variance required')
    v = covariance@row
    return covariance-np.outer(v, v)/(planning_variance+row@v)


def redundancy(training_covariance, selected, candidates, epsilon_r):
    if not len(selected) or not len(candidates):
        return np.zeros(len(candidates))
    diagonal = training_covariance.diagonal()
    correlation = abs(training_covariance.block(candidates, selected))
    correlation /= np.sqrt((diagonal[candidates, None]+epsilon_r)*(diagonal[selected][None, :]+epsilon_r))
    if np.any(correlation > 1+1e-10):
        raise FloatingPointError('invalid training correlations')
    return np.clip(correlation, 0., 1.).max(axis=1)


def normalize_scores(delta_h, delta_u, red, config):
    """5-14--16, exactly max+epsilon on the already feasible set."""
    h, u, red = (np.asarray(x, float) for x in (delta_h, delta_u, red))
    if h.ndim != 1 or u.shape != h.shape or red.shape != h.shape:
        raise ValueError('score arrays must have matching vector shapes')
    if not all(np.isfinite(x).all() for x in (h, u, red)) or np.any(h < 0) or np.any(u < 0) or np.any((red < 0) | (red > 1)):
        raise ValueError('invalid raw scores')
    hmax, umax = float(np.max(h, initial=0.)), float(np.max(u, initial=0.))
    scales = {'max_delta_h': hmax, 'max_delta_u': umax,
              'delta_h_denominator': hmax+config.epsilon_h,
              'delta_u_denominator': umax+config.epsilon_u,
              'redundancy_scale': 1., 'feasible_count': len(h)}
    return h/scales['delta_h_denominator'], u/scales['delta_u_denominator'], red.copy(), scales


class MCEIR:
    def __init__(self, training_covariance, base_cost, maintenance_cost, failure_risk,
                 config=None, *, historical_reliability=None):
        self.config = config or PlanningConfig()
        self.training_covariance = training_covariance
        factor = np.asarray(training_covariance.factor)
        if factor.ndim != 2 or not np.isfinite(factor).all():
            raise ValueError('finite empirical covariance factor required')
        self.n = len(factor)
        self.base_cost, self.maintenance_cost, self.failure_risk = [np.array(x, float, copy=True) for x in (base_cost, maintenance_cost, failure_risk)]
        for values in (self.base_cost, self.maintenance_cost, self.failure_risk):
            if values.shape != (self.n,) or not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError('nonnegative finite grid-aligned costs and risks required')
        self.reliability = np.ones(self.n) if historical_reliability is None else np.array(historical_reliability, float, copy=True)
        if self.reliability.shape != (self.n,) or not np.isfinite(self.reliability).all() or np.any(self.reliability <= 0):
            raise ValueError('historical reliability must be positive and grid-aligned')
        self.reliability.setflags(write=False)

    def score_candidates(self, model, selected, feasible_candidates):
        model._require_fit()
        if model.n != self.n:
            raise ValueError('model and training covariance grid mismatch')
        selected = node_ids(selected, self.n)
        candidates = sorted(node_ids(feasible_candidates, self.n))
        if set(selected).intersection(candidates):
            raise ValueError('selected nodes cannot be candidates')
        cfg = self.config
        tau = float(model.tau_.mean)
        r = 1/(tau*self.reliability[candidates])
        dh = structural_entropy_gain(self.training_covariance, selected, candidates, cfg)
        du = uncertainty_reduction(model.basis, model.cov_alpha_, model.basis[candidates], r)
        red = redundancy(self.training_covariance, selected, candidates, cfg.epsilon_r)
        nh, nu, nr, scales = normalize_scores(dh, du, red, cfg)
        rows = []
        for i, j in enumerate(candidates):
            soft = cfg.maintenance_weight*self.maintenance_cost[j]+cfg.failure_weight*self.failure_risk[j]
            cost = self.base_cost[j]+soft+cfg.epsilon_cost
            gain = nh[i]+cfg.beta*nu[i]-cfg.eta*nr[i]
            rows.append({'node': j, 'delta_h_raw': float(dh[i]), 'delta_u_raw': float(du[i]),
                         'redundancy_raw': float(red[i]), 'delta_h_normalized': float(nh[i]),
                         'delta_u_normalized': float(nu[i]), 'redundancy_normalized': float(nr[i]),
                         'effective_information': float(gain),
                         'effective_information_raw': float(dh[i]+cfg.beta_raw*du[i]-cfg.eta_raw*red[i]),
                         'base_cost': float(self.base_cost[j]), 'maintenance_cost': float(self.maintenance_cost[j]),
                         'failure_risk': float(self.failure_risk[j]), 'soft_penalty': float(soft),
                         'effective_cost': float(cost), 'eir': float(gain/cost),
                         'planning_variance': float(r[i]), 'historical_reliability': float(self.reliability[j])})
        return {'candidates': rows, 'normalization': scales, 'posterior_fit_count': model.fit_count_,
                'tau_mean': tau, 'planning_convention': 'reciprocal_effective_precision',
                'entropy_reference_variance': cfg.entropy_reference_variance}


def select_next(scores, threshold):
    """5-28: strict exceedance; exact ties go to the smallest stable node ID."""
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError('threshold must be finite and nonnegative')
    if not scores['candidates']:
        return None, 'no_feasible_candidate'
    best = min(scores['candidates'], key=lambda x: (-x['eir'], x['node']))
    if not np.isfinite(best['eir']):
        raise FloatingPointError('nonfinite EIR')
    if best['eir'] <= threshold:
        return None, 'insufficient_marginal_value'
    return best, None
