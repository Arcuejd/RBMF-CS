"""Section 4 / Algorithm 1, single-snapshot mean-field coordinate ascent."""
from dataclasses import dataclass, asdict
from time import perf_counter
import numpy as np
from scipy.special import digamma, gammaln
from .numerics import solve_gaussian, residual_energy, discrepancy_energy, local_precision_parameters


@dataclass(frozen=True)
class VBConfig:
    robust_likelihood: bool = True
    multifidelity: bool = True
    a_gamma: float = 1e-2
    b_gamma: float = 1e-2
    a_tau: float = 1e-2
    b_tau: float = 1e-2
    a_delta: float = 2.
    b_delta: float = 2.
    rho0: float = 1.
    kappa_rho0: float = 1.
    nu: float = 4.
    initial_gamma: float = 1.
    initial_tau: float = 25.
    initial_tau_delta: float = 1.
    max_iter: int = 2000
    min_iter: int = 3
    tolerance: float = 1e-7
    convergence: str = 'elbo'
    jitter_relative: float = 1e-12
    jitter_attempts: int = 6
    max_full_covariance_size: int = 2000

    def __post_init__(self):
        if not isinstance(self.robust_likelihood, bool) or not isinstance(self.multifidelity, bool):
            raise ValueError('likelihood and multifidelity switches must be boolean')
        positive = ('a_gamma','b_gamma','a_tau','b_tau','a_delta','b_delta',
                    'kappa_rho0','nu','initial_gamma','initial_tau','initial_tau_delta',
                    'tolerance','jitter_relative')
        for name in positive:
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if not np.isfinite(self.rho0):
            raise ValueError('rho0 must be finite')
        for name in ('max_iter','min_iter','jitter_attempts','max_full_covariance_size'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.min_iter > self.max_iter or self.convergence not in ('elbo','factors'):
            raise ValueError('invalid stopping settings')


@dataclass
class GammaFactor:
    shape: np.ndarray
    rate: np.ndarray

    @property
    def mean(self):
        return self.shape / self.rate

    @property
    def expected_log(self):
        return digamma(self.shape) - np.log(self.rate)

    def entropy(self):
        a, b = self.shape, self.rate
        return np.sum(a - np.log(b) + gammaln(a) + (1-a)*digamma(a))

    def prior_expected_log(self, a, b):
        return np.sum(a*np.log(b)-gammaln(a)+(a-1)*self.expected_log-b*self.mean)

    def to_dict(self):
        return {'shape': np.asarray(self.shape).tolist(), 'rate': np.asarray(self.rate).tolist(),
                'mean': np.asarray(self.mean).tolist()}


class RBMFCS:
    """Fixed-basis latent field posterior. All inputs are in common model units.

    fit(low_fidelity, observations, indices=...) starts a fresh fit.
    update(observations, indices=...) appends independent observations, preserves
    old factors, initializes new lambda from its prior, then refits ALL factors.
    Alternatively supply sampling_matrix with shape (p,N) to either method.
    NaN observations and corresponding design rows are removed; inf is invalid.
    Methods return self; diagnostics_ explicitly reports convergence/failure to
    converge. No deployment rule or automatic observation acquisition is hidden
    in this estimator.
    """
    def __init__(self, basis, config=None):
        self.config = config or VBConfig()
        self.basis = np.array(basis, dtype=float, copy=True)
        if self.basis.ndim != 2 or min(self.basis.shape) == 0 or not np.isfinite(self.basis).all():
            raise ValueError('basis must be a finite nonempty (N,K) matrix')
        if self.basis.shape[1] > self.basis.shape[0]:
            raise ValueError('K must be <= N')
        self.basis.setflags(write=False)
        self.n, self.k = self.basis.shape
        self.gram = self.basis.T @ self.basis
        self.fit_count_ = 0

    def _observations(self, observations, indices, sampling_matrix):
        y = np.asarray(observations, float)
        if y.ndim != 1 or np.isinf(y).any():
            raise ValueError('observations must be a vector; NaN is allowed, infinity is not')
        if (indices is None) == (sampling_matrix is None):
            raise ValueError('supply exactly one of indices or sampling_matrix')
        if indices is not None:
            idx = np.asarray(indices)
            if idx.size == 0: idx = idx.astype(int)
            if idx.shape != y.shape or idx.dtype.kind not in 'iu' or np.any(idx < 0) or np.any(idx >= self.n):
                raise ValueError('indices must be valid integer grid locations matching y')
            h = self.basis[idx]
        else:
            c = np.asarray(sampling_matrix, float)
            if c.shape != (len(y), self.n) or not np.isfinite(c).all():
                raise ValueError('sampling_matrix must be finite (p,N)')
            h = c @ self.basis
        keep = ~np.isnan(y)
        return y[keep].copy(), h[keep].copy(), int((~keep).sum())

    def fit(self, low_fidelity, observations, *, indices=None, sampling_matrix=None):
        low = np.asarray(low_fidelity, float)
        if low.shape != (self.n,) or not np.isfinite(low).all():
            raise ValueError('LF must be a finite aligned N-vector; mask domain before fitting')
        y, h, missing = self._observations(observations, indices, sampling_matrix)
        self.low_ = low.copy()
        self.low_projection_ = self.basis.T @ low
        self.low_norm_ = float(low @ low)
        self.y_, self.h_ = y, h
        self.missing_removed_ = missing
        c = self.config
        self.gamma_ = GammaFactor(np.full(self.k,c.a_gamma), np.full(self.k,c.a_gamma/c.initial_gamma))
        self.local_ = GammaFactor(np.full(len(y),c.nu/2), np.full(len(y),c.nu/2))
        self.tau_ = GammaFactor(np.asarray(c.a_tau), np.asarray(c.a_tau/c.initial_tau))
        self.delta_ = GammaFactor(np.asarray(c.a_delta), np.asarray(c.a_delta/c.initial_tau_delta))
        self.rho_mean_, self.rho_variance_ = c.rho0, 1/c.kappa_rho0
        self.mean_alpha_ = np.zeros(self.k)
        self.cov_alpha_ = np.eye(self.k)/c.initial_gamma
        return self._run(warm_start=False, new_observations=len(y))

    def update(self, observations, *, indices=None, sampling_matrix=None):
        self._require_fit()
        y, h, missing = self._observations(observations, indices, sampling_matrix)
        self.y_ = np.concatenate((self.y_, y))
        self.h_ = np.vstack((self.h_, h))
        self.missing_removed_ += missing
        self.local_ = GammaFactor(np.concatenate((self.local_.shape, np.full(len(y),self.config.nu/2))),
                                  np.concatenate((self.local_.rate, np.full(len(y),self.config.nu/2))))
        return self._run(warm_start=True, new_observations=len(y))

    def _parameters(self):
        values = [self.mean_alpha_, self.cov_alpha_.ravel(), np.array([self.rho_mean_,self.rho_variance_])]
        for factor in (self.gamma_,self.local_,self.tau_,self.delta_):
            values.extend([np.atleast_1d(factor.shape),np.atleast_1d(factor.rate)])
        return np.concatenate(values)

    def _run(self, warm_start, new_observations):
        start = perf_counter()
        c = self.config
        initial = self.factor_summary()
        initial_elbo = self._elbo()
        previous_elbo = initial_elbo
        history = []
        converged = False
        for iteration in range(1,c.max_iter+1):
            before = self._parameters()
            tau = float(self.tau_.mean)
            delta = float(self.delta_.mean) if c.multifidelity else 0.
            weights = self.local_.mean
            precision = np.diag(self.gamma_.mean) + delta*self.gram + tau*(self.h_.T @ (weights[:,None]*self.h_))
            rhs = delta*self.rho_mean_*self.low_projection_ + tau*(self.h_.T @ (weights*self.y_))
            self.mean_alpha_,self.cov_alpha_,jitter = solve_gaussian(precision,rhs,c.jitter_relative,c.jitter_attempts)
            second = self.mean_alpha_**2 + np.diag(self.cov_alpha_)
            self.gamma_ = GammaFactor(np.full(self.k,c.a_gamma+.5), c.b_gamma+.5*second)
            xi = residual_energy(self.y_,self.h_,self.mean_alpha_,self.cov_alpha_)
            if c.robust_likelihood:
                self.local_ = GammaFactor(*local_precision_parameters(c.nu,tau,xi))
            self.tau_ = GammaFactor(np.asarray(c.a_tau+len(self.y_)/2), np.asarray(c.b_tau+.5*(self.local_.mean@xi)))
            if c.multifidelity:
                self.rho_variance_ = 1/(c.kappa_rho0+delta*self.low_norm_)
                self.rho_mean_ = self.rho_variance_*(c.kappa_rho0*c.rho0+delta*(self.low_projection_@self.mean_alpha_))
                xi_delta = discrepancy_energy(self.basis,self.low_,self.mean_alpha_,self.cov_alpha_,self.rho_mean_,self.rho_variance_,self.gram)
                self.delta_ = GammaFactor(np.asarray(c.a_delta+self.n/2), np.asarray(c.b_delta+.5*xi_delta))
            elbo = self._elbo()
            after = self._parameters()
            if not np.isfinite(after).all() or not np.isfinite(elbo):
                raise FloatingPointError('nonfinite VB state; check scaling/priors')
            relative_elbo = abs(elbo-previous_elbo)/max(1.,abs(previous_elbo))
            relative_factors = float(np.max(np.abs(after-before)/np.maximum(1.,np.abs(before))))
            history.append({'iteration':iteration,'elbo':elbo,'relative_elbo_change':relative_elbo,
                            'relative_factor_change':relative_factors,'precision_jitter':jitter})
            criterion = relative_elbo if c.convergence == 'elbo' else relative_factors
            if iteration >= c.min_iter and criterion < c.tolerance:
                converged = True
                break
            previous_elbo = elbo
        self.fit_count_ += 1
        elbos = np.array([initial_elbo]+[entry['elbo'] for entry in history])
        self.diagnostics_ = {'converged':converged,'stop_reason':'tolerance' if converged else 'max_iter',
                             'criterion':c.convergence,'iterations':iteration,'fit_count':self.fit_count_,
                             'warm_start':warm_start,'new_observations':new_observations,'n_observations':len(self.y_),
                             'missing_removed_total':self.missing_removed_,'initial_factors':initial,
                             'initial_elbo':initial_elbo,'history':history,
                             'elbo_decreases':int(np.sum(np.diff(elbos)<-1e-9*np.maximum(1.,np.abs(elbos[:-1])))),
                             'max_precision_jitter':max(x['precision_jitter'] for x in history),
                             'runtime_seconds':perf_counter()-start}
        return self

    def _elbo(self):
        """4-55, including normalized auxiliary likelihood and all entropies."""
        c = self.config
        log2pi = np.log(2*np.pi)
        xi = residual_energy(self.y_,self.h_,self.mean_alpha_,self.cov_alpha_)
        xd = discrepancy_energy(self.basis,self.low_,self.mean_alpha_,self.cov_alpha_,self.rho_mean_,self.rho_variance_,self.gram)
        local_log = self.local_.expected_log if c.robust_likelihood else 0.
        value = .5*np.sum(self.tau_.expected_log+local_log-log2pi-self.tau_.mean*self.local_.mean*xi)
        if c.multifidelity:
            value += .5*(self.n*(self.delta_.expected_log-log2pi)-self.delta_.mean*xd)
        second = self.mean_alpha_**2+np.diag(self.cov_alpha_)
        value += .5*np.sum(self.gamma_.expected_log-log2pi-self.gamma_.mean*second)
        if c.multifidelity:
            value += .5*(np.log(c.kappa_rho0)-log2pi-c.kappa_rho0*((self.rho_mean_-c.rho0)**2+self.rho_variance_))
        factors = [(self.gamma_,c.a_gamma,c.b_gamma),(self.tau_,c.a_tau,c.b_tau)]
        if c.robust_likelihood:
            factors.append((self.local_,c.nu/2,c.nu/2))
        if c.multifidelity:
            factors.append((self.delta_,c.a_delta,c.b_delta))
        for factor,a,b in factors:
            value += factor.prior_expected_log(a,b)+factor.entropy()
        sign,logdet = np.linalg.slogdet(self.cov_alpha_)
        if sign <= 0: raise FloatingPointError('coefficient covariance not positive definite')
        value += .5*(self.k*(1+log2pi)+logdet)
        if c.multifidelity:
            value += .5*(1+log2pi+np.log(self.rho_variance_))
        return float(value)

    def _require_fit(self):
        if self.fit_count_ == 0:
            raise RuntimeError('call fit first')

    def predict(self, indices=None):
        """4-51: latent field mean in model units, with no future noise."""
        self._require_fit()
        psi = self.basis if indices is None else self.basis[indices]
        return psi @ self.mean_alpha_

    def posterior_variance(self, indices=None):
        """4-53/54: latent uncertainty map; O(N K^2) work, O(N K) memory."""
        self._require_fit()
        psi = self.basis if indices is None else self.basis[indices]
        return np.maximum(0.,np.einsum('...i,ij,...j->...',psi,self.cov_alpha_,psi))

    def covariance_product(self, vector):
        """Sigma_x @ vector (or matrix), without materializing Sigma_x."""
        self._require_fit()
        return self.basis @ (self.cov_alpha_ @ (self.basis.T @ np.asarray(vector)))

    def posterior_covariance(self, *, allow_large=False):
        """Explicit opt-in full Sigma_x; guarded against accidental large grids."""
        self._require_fit()
        if self.n > self.config.max_full_covariance_size and not allow_large:
            raise ValueError('full covariance exceeds limit; use variance or covariance_product')
        return self.basis @ self.cov_alpha_ @ self.basis.T

    def total_uncertainty(self):
        self._require_fit()
        return float(np.sum(self.gram*self.cov_alpha_.T))

    def marginal_noise_variance(self):
        """Integrates q(tau) and a NEW lambda prior, NOT reciprocal mean tau.

        Finite only if nu>2 and q(tau).shape>1. This is not field uncertainty.
        """
        self._require_fit()
        if self.tau_.shape <= 1:
            return float('inf')
        if not self.config.robust_likelihood:
            return float(self.tau_.rate/(self.tau_.shape-1))
        if self.config.nu <= 2:
            return float('inf')
        return float(self.config.nu/(self.config.nu-2)*self.tau_.rate/(self.tau_.shape-1))

    def factor_summary(self):
        return {**{name:factor.to_dict() for name,factor in
                   (('gamma',self.gamma_),('lambda',self.local_),('tau',self.tau_),('tau_delta',self.delta_))},
                'rho':{'mean':float(self.rho_mean_),'variance':float(self.rho_variance_),
                       'second_moment':float(self.rho_mean_**2+self.rho_variance_)}}

    def configuration(self):
        return asdict(self.config)
