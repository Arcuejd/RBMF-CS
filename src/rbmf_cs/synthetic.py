"""Synthetic demo only: correlated fields from a common latent process.

LF is generated first from exogenous latent variables. HF uses the shared
process plus smooth discrepancy. No held-out HF field is used to form LF.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SyntheticConfig:
    seed: int = 20261008
    nx: int = 12
    ny: int = 10
    n_train: int = 100
    n_test: int = 8
    latent_modes: int = 6
    length_scale_km: float = 28.
    domain_x_km: float = 110.
    domain_y_km: float = 90.
    temporal_ar: float = 0.6
    rho: float = 1.15
    discrepancy_scale: float = 0.12
    lf_bias_scale: float = 0.04

    def __post_init__(self):
        for name in ('seed','nx','ny','n_train','n_test','latent_modes'):
            v=getattr(self,name)
            if isinstance(v,bool) or not isinstance(v,int) or v<0: raise ValueError(f'invalid {name}')
        if min(self.nx,self.ny,self.n_train)<2 or self.n_test<1 or self.latent_modes<1:
            raise ValueError('grid and training dimensions too small')
        for name in ('length_scale_km','domain_x_km','domain_y_km'):
            if not np.isfinite(getattr(self,name)) or getattr(self,name)<=0: raise ValueError(name)
        if not np.isfinite(self.temporal_ar) or abs(self.temporal_ar)>=1: raise ValueError('abs(temporal_ar)<1')
        if not np.isfinite(self.rho) or self.rho<=0: raise ValueError('rho must be positive')
        for name in ('discrepancy_scale','lf_bias_scale'):
            if not np.isfinite(getattr(self,name)) or getattr(self,name)<0: raise ValueError(name)


def make_synthetic(config=None):
    c=config or SyntheticConfig()
    rng=np.random.default_rng(c.seed)
    xx,yy=np.meshgrid(np.linspace(0,c.domain_x_km,c.nx),np.linspace(0,c.domain_y_km,c.ny))
    coordinates=np.column_stack((xx.ravel(),yy.ravel()))
    centers=rng.uniform([0,0],[c.domain_x_km,c.domain_y_km],(c.latent_modes,2))
    modes=np.exp(-np.sum((coordinates[:,None,:]-centers[None,:,:])**2,axis=2)/(2*c.length_scale_km**2))
    modes/=np.sqrt(np.mean(modes**2,axis=0))
    count=c.n_train+c.n_test
    latent=rng.normal(size=(count,c.latent_modes))
    for t in range(1,count): latent[t]=c.temporal_ar*latent[t-1]+np.sqrt(1-c.temporal_ar**2)*latent[t]
    shared=latent @ modes.T / np.sqrt(c.latent_modes)
    background=2.+0.15*np.cos(coordinates[:,0]/c.domain_x_km*np.pi)
    bias=c.lf_bias_scale*np.sin(coordinates[:,1]/c.domain_y_km*np.pi)
    lf=background+shared+bias
    discrepancy_modes=np.column_stack((np.sin(xx.ravel()/c.domain_x_km*2*np.pi),
                                       np.cos(yy.ravel()/c.domain_y_km*2*np.pi)))
    discrepancy=rng.normal(size=(count,2)) @ discrepancy_modes.T / np.sqrt(2)
    hf=background+c.rho*shared+c.discrepancy_scale*discrepancy
    ice=np.clip(.3+.3*yy.ravel()/c.domain_y_km+.1*np.sin(xx.ravel()/c.domain_x_km*np.pi),0,1)
    return {'train_lf':lf[:c.n_train], 'train_hf':hf[:c.n_train],
            'test_lf':lf[c.n_train:], 'test_hf':hf[c.n_train:],
            'coordinates_km':coordinates,'ice_fraction':ice,
            'train_time':np.arange(c.n_train),'test_time':np.arange(c.n_train,count)}
