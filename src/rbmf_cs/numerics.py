"""Small dense coefficient algebra; no N by N field matrices by default."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve


def gaussian_posterior(basis, low_fidelity, design, observations, gamma,
                       local_precision, tau, tau_delta, rho,
                       jitter_relative=1e-12, jitter_attempts=6):
    """Equations 4-22/23 at fixed expectations. Return mean, covariance, jitter.

    Jitter is attempted only after Cholesky fails and is reported in precision
    units. Solving against I constructs the necessary K by K covariance, not
    an explicit inverse operation or a full field covariance.
    """
    psi, h = np.asarray(basis, float), np.asarray(design, float)
    precision = np.diag(gamma) + tau_delta * (psi.T @ psi)
    precision += tau * (h.T @ (np.asarray(local_precision)[:, None] * h))
    rhs = tau_delta * rho * (psi.T @ low_fidelity)
    rhs += tau * (h.T @ (local_precision * observations))
    return solve_gaussian(precision, rhs, jitter_relative, jitter_attempts)


def solve_gaussian(precision, rhs, jitter_relative=1e-12, jitter_attempts=6):
    precision = (precision + precision.T) * 0.5
    scale = max(1., float(np.max(np.abs(np.diag(precision)))))
    jitter = 0.
    for attempt in range(jitter_attempts + 1):
        try:
            factor = cho_factor(precision + jitter * np.eye(len(rhs)), lower=True)
            mean = cho_solve(factor, rhs)
            cov = cho_solve(factor, np.eye(len(rhs)))
            return mean, (cov + cov.T) * 0.5, jitter
        except np.linalg.LinAlgError:
            if attempt == jitter_attempts:
                raise
            jitter = jitter_relative * scale * 10. ** attempt


def residual_energy(y, h, mean, cov):
    """4-29/30, including coefficient posterior uncertainty."""
    return (y - h @ mean) ** 2 + np.einsum('ij,jk,ik->i', h, cov, h)


def discrepancy_energy(psi, low, mean, cov, rho_mean, rho_variance, gram=None):
    """4-46: stable sum of three nonnegative energies."""
    gram = psi.T @ psi if gram is None else gram
    residual = psi @ mean - rho_mean * low
    return float(np.sum(gram * cov.T) + residual @ residual
                 + rho_variance * (low @ low))


def local_precision_parameters(nu, tau, energy):
    """4-31--34, Gamma shape/rate (not shape/scale)."""
    return np.full(np.shape(energy), (nu + 1.) / 2.), (nu + tau * energy) / 2.
