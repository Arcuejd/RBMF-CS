"""P0-only instrumentation and conditional known-noise control.

The original estimator is untouched. FixedPrecision is a constant, not a
Gamma posterior: its prior and entropy terms are absent from the ELBO.
"""
import numpy as np
from .model import RBMFCS


class FixedPrecision:
    def __init__(self, value):
        if not np.isfinite(value) or value <= 0:
            raise ValueError('known precision must be finite and positive')
        self.mean = np.asarray(value)
        self.expected_log = np.log(self.mean)
        # Constant bookkeeping for the inherited convergence vector only.
        self.shape, self.rate = np.asarray(1.), np.asarray(1/value)

    def entropy(self): return 0.
    def prior_expected_log(self, a, b): return 0.
    def to_dict(self): return {'constant_precision': float(self.mean)}


class DiagnosticRBMF(RBMFCS):
    def __init__(self, basis, config=None, fixed_tau=None):
        super().__init__(basis, config)
        self.fixed_tau = fixed_tau
        if fixed_tau is not None: FixedPrecision(fixed_tau)
        self.factor_trace = []

    def _run(self, warm_start, new_observations):
        if self.fixed_tau is not None:
            self.tau_ = FixedPrecision(self.fixed_tau)
        return super()._run(warm_start,new_observations)

    def _elbo(self):
        # Called at initialization and after every complete coordinate sweep.
        # tau's temporary free update has no downstream use before this call.
        if self.fixed_tau is not None:
            self.tau_ = FixedPrecision(self.fixed_tau)
        value = super()._elbo()
        self.factor_trace.append({'factors': self.factor_summary(), 'elbo': value,
                                  'alpha': self.mean_alpha_.tolist(),
                                  'covariance_diagonal': np.diag(self.cov_alpha_).tolist()})
        return value

    def marginal_noise_variance(self):
        if self.fixed_tau is None: return super().marginal_noise_variance()
        if not self.config.robust_likelihood: return 1/self.fixed_tau
        if self.config.nu <= 2: return float('inf')
        return self.config.nu/(self.config.nu-2)/self.fixed_tau


def precision_diagnostics(model, scale, evaluation, weights):
    h, psi, cov = model.h_, model.basis, model.cov_alpha_
    obs = float(model.tau_.mean)*(h.T@(model.local_.mean[:, None]*h))
    ard = np.diag(model.gamma_.mean)
    lf = float(model.delta_.mean)*model.gram if model.config.multifidelity else np.zeros_like(ard)
    ro = float(model.tau_.mean)*(h.T@(model.local_.mean*model.y_))
    rl = float(model.delta_.mean)*model.rho_mean_*model.low_projection_ if model.config.multifidelity else np.zeros(model.k)
    # Reconstruct a synchronized conditional posterior; stored alpha was
    # updated before the other factors in the final coordinate sweep.
    sync = np.linalg.inv(obs+ard+lf)
    eo, el = psi@(sync@ro)*scale, psi@(sync@rl)*scale
    def rms(v): return float(np.sqrt(np.average(v[evaluation]**2, weights=weights[evaluation])))
    directions = []
    for v in h:
        norm = float(v@v)
        directions.append({name: float(v@p@v/norm) if norm else 0.
                           for name,p in [('observation',obs),('ard',ard),('lf',lf)]})
    return {'effective_dof_frozen': float(np.trace(sync@obs)),
            'effective_dof_stored': float(np.trace(cov@obs)),
            'precision_spectra': {name: np.linalg.eigvalsh(p).tolist() for name,p in [('obs',obs),('ard',ard),('lf',lf)]},
            'observed_direction_rayleigh': directions, 'rhs_observation': ro.tolist(), 'rhs_lf': rl.tolist(),
            'rhs_cosine': float(ro@rl/(np.linalg.norm(ro)*np.linalg.norm(rl))) if np.linalg.norm(ro)*np.linalg.norm(rl) else None,
            'observation_contribution_rms_c': rms(eo), 'lf_contribution_rms_c': rms(el),
            'sweep_mean_residual': float(np.linalg.norm(model.mean_alpha_-sync@(ro+rl))),
            'prediction_vs_zero_rms_c': rms(psi@model.mean_alpha_*scale)}
