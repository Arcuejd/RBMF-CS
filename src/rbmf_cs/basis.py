"""Training-only normalization, POD, and factored empirical spatial covariance."""
from dataclasses import dataclass
import numpy as np


def _ensemble(x):
    x=np.asarray(x,float)
    if x.ndim != 2 or min(x.shape)<2 or not np.isfinite(x).all():
        raise ValueError('training ensemble must be finite (T,N), T,N>=2')
    return x


@dataclass(frozen=True)
class TrainingTransform:
    """Shared HF training spatial mean and ONE scalar RMS for both fidelities.

    Model fields are anomalies in these units. The multifidelity relation is
    assumed in anomaly space; an arbitrary raw-space intercept is not fitted.
    A scalar scale preserves POD orthonormality and uniform physical weighting.
    """
    mean: np.ndarray
    scale: float

    @classmethod
    def fit(cls, high_fidelity_training, scale_floor=1e-12):
        x=_ensemble(high_fidelity_training)
        if not np.isfinite(scale_floor) or scale_floor<=0: raise ValueError('invalid scale_floor')
        mean=x.mean(axis=0)
        scale=float(np.sqrt(np.mean((x-mean)**2)))
        if scale <= scale_floor: raise ValueError('degenerate training variance')
        mean.setflags(write=False)
        return cls(mean,scale)

    def transform(self, x):
        return (np.asarray(x)-self.mean)/self.scale

    def inverse(self, x):
        return np.asarray(x)*self.scale+self.mean

    def inverse_variance(self, variance):
        return np.asarray(variance)*self.scale**2


@dataclass(frozen=True)
class PODBasis:
    matrix: np.ndarray
    singular_values: np.ndarray
    energy_fraction: float

    @classmethod
    def fit(cls, transformed_training, n_modes):
        x=_ensemble(transformed_training)
        if isinstance(n_modes,bool) or not isinstance(n_modes,int) or not 1<=n_modes<=min(x.shape):
            raise ValueError('n_modes must lie in [1,min(T,N)]')
        _,s,vh=np.linalg.svd(x,full_matrices=False)
        if s[0]<=0: raise ValueError('zero training ensemble')
        basis=vh[:n_modes].T.copy()
        # Deterministic signs remove the otherwise arbitrary SVD sign ambiguity.
        pivots=np.argmax(abs(basis),axis=0)
        basis*=np.sign(basis[pivots,np.arange(n_modes)])
        basis.setflags(write=False)
        return cls(basis,s[:n_modes].copy(),float(np.sum(s[:n_modes]**2)/np.sum(s**2)))


@dataclass(frozen=True)
class TrainingCovariance:
    """5-1 exactly as a factor: K=F F^T; no mandatory N by N allocation."""
    factor: np.ndarray

    @classmethod
    def fit(cls, training):
        x=_ensemble(training)
        factor=(x-x.mean(axis=0)).T/np.sqrt(len(x)-1)
        factor.setflags(write=False)
        return cls(factor)

    def block(self, rows, columns):
        return self.factor[np.atleast_1d(rows)] @ self.factor[np.atleast_1d(columns)].T

    def diagonal(self):
        return np.sum(self.factor**2,axis=1)

    def product(self, vector):
        return self.factor @ (self.factor.T @ vector)
