"""Same RBMF augmented probability model, joint Gaussian q(alpha,rho).

Only the variational family changes. Original model.py remains untouched.
Known positive tau is supported exclusively for conditional reference problems.
"""
from time import perf_counter
import numpy as np
from .model import RBMFCS, GammaFactor
from .p0_diagnostics import FixedPrecision
from .numerics import solve_gaussian, residual_energy, local_precision_parameters


class StructuredRBMF(RBMFCS):
    def __init__(self,basis,config=None,fixed_tau=None):
        super().__init__(basis,config)
        if not self.config.multifidelity:raise ValueError('joint experiment requires LF factor')
        if fixed_tau is not None:FixedPrecision(fixed_tau)
        self.fixed_tau=fixed_tau
        self.cross_covariance_=np.zeros(self.k)

    def joint_precision_rhs(self):
        t=float(self.tau_.mean);d=float(self.delta_.mean);v=self.low_projection_
        p=np.zeros((self.k+1,self.k+1))
        p[:-1,:-1]=np.diag(self.gamma_.mean)+d*self.gram+t*(self.h_.T@(self.local_.mean[:,None]*self.h_))
        p[:-1,-1]=p[-1,:-1]=-d*v
        p[-1,-1]=self.config.kappa_rho0+d*self.low_norm_
        rhs=np.r_[t*(self.h_.T@(self.local_.mean*self.y_)),self.config.kappa_rho0*self.config.rho0]
        return p,rhs

    def _parameters(self):
        return np.r_[super()._parameters(),self.cross_covariance_]

    def joint_covariance(self):
        return np.block([[self.cov_alpha_,self.cross_covariance_[:,None]],
                         [self.cross_covariance_[None,:],np.array([[self.rho_variance_]])]])

    def discrepancy_expectation(self):
        mean_residual=self.basis@self.mean_alpha_-self.rho_mean_*self.low_
        variance=float(np.sum(self.gram*self.cov_alpha_.T)+self.low_norm_*self.rho_variance_
                       -2*self.low_projection_@self.cross_covariance_)
        if variance < -1e-9:raise FloatingPointError('negative joint residual variance')
        return float(mean_residual@mean_residual+max(0.,variance))

    def _elbo(self):
        base=super()._elbo()
        sign,logjoint=np.linalg.slogdet(self.joint_covariance())
        if sign<=0:raise FloatingPointError('joint covariance not positive definite')
        # Replace product entropy by joint entropy; correct E(alpha*rho).
        entropy_correction=.5*(logjoint-np.linalg.slogdet(self.cov_alpha_)[1]-np.log(self.rho_variance_))
        return float(base+self.delta_.mean*(self.low_projection_@self.cross_covariance_)+entropy_correction)

    def _run(self,warm_start,new_observations):
        started=perf_counter();c=self.config
        if not warm_start:self.cross_covariance_=np.zeros(self.k)
        if self.fixed_tau is not None:self.tau_=FixedPrecision(self.fixed_tau)
        initial=self.factor_summary();initial_elbo=self._elbo();previous=initial_elbo
        history=[];converged=False
        for iteration in range(1,c.max_iter+1):
            before=self._parameters();tau=float(self.tau_.mean)
            precision,rhs=self.joint_precision_rhs()
            mean,cov,jitter=solve_gaussian(precision,rhs,c.jitter_relative,c.jitter_attempts)
            self.mean_alpha_=mean[:-1];self.rho_mean_=float(mean[-1])
            self.cov_alpha_=cov[:-1,:-1];self.rho_variance_=float(cov[-1,-1]);self.cross_covariance_=cov[:-1,-1]
            second=self.mean_alpha_**2+np.diag(self.cov_alpha_)
            self.gamma_=GammaFactor(np.full(self.k,c.a_gamma+.5),c.b_gamma+.5*second)
            xi=residual_energy(self.y_,self.h_,self.mean_alpha_,self.cov_alpha_)
            if c.robust_likelihood:self.local_=GammaFactor(*local_precision_parameters(c.nu,tau,xi))
            if self.fixed_tau is None:
                self.tau_=GammaFactor(np.asarray(c.a_tau+len(self.y_)/2),np.asarray(c.b_tau+.5*(self.local_.mean@xi)))
            self.delta_=GammaFactor(np.asarray(c.a_delta+self.n/2),np.asarray(c.b_delta+.5*self.discrepancy_expectation()))
            elbo=self._elbo();after=self._parameters()
            if not np.isfinite(after).all() or not np.isfinite(elbo):raise FloatingPointError('nonfinite structured state')
            rel=abs(elbo-previous)/max(1.,abs(previous));change=float(np.max(abs(after-before)/np.maximum(1.,abs(before))))
            history.append({'iteration':iteration,'elbo':elbo,'relative_elbo_change':rel,
              'relative_factor_change':change,'precision_jitter':jitter,'cross_covariance_norm':float(np.linalg.norm(self.cross_covariance_)),
              'factors':self.factor_summary()})
            if iteration>=c.min_iter and (change if c.convergence=='factors' else rel)<c.tolerance:
                converged=True;break
            previous=elbo
        self.fit_count_+=1
        elbos=np.array([initial_elbo]+[h['elbo'] for h in history])
        self.diagnostics_={'converged':converged,'stop_reason':'tolerance' if converged else 'max_iter',
          'criterion':c.convergence,'iterations':iteration,'fit_count':self.fit_count_,
          'warm_start':warm_start,'new_observations':new_observations,'n_observations':len(self.y_),
          'missing_removed_total':self.missing_removed_,'initial_factors':initial,'initial_elbo':initial_elbo,
          'history':history,'elbo_decreases':int(np.sum(np.diff(elbos)<-1e-9*np.maximum(1.,abs(elbos[:-1])))),
          'max_precision_jitter':max(h['precision_jitter'] for h in history),'runtime_seconds':perf_counter()-started}
        return self

    def factor_summary(self):
        return {**super().factor_summary(),'alpha_rho_covariance':self.cross_covariance_.tolist()}

    def marginal_noise_variance(self):
        if self.fixed_tau is None:return super().marginal_noise_variance()
        if not self.config.robust_likelihood:return 1/self.fixed_tau
        return self.config.nu/(self.config.nu-2)/self.fixed_tau if self.config.nu>2 else float('inf')


def integrated_no_observation(basis,low,gamma,delta,kappa=1.,rho0=1.):
    """Exact alpha-integrated rho posterior conditional on fixed precisions.

    Applies to original auxiliary-observation target, without online data.
    Q is nonnegative; it remains positive even with LF inside POD if ARD>0.
    """
    b=np.asarray(basis);l=np.asarray(low);g=np.broadcast_to(gamma,(b.shape[1],))
    if np.any(g<=0) or delta<=0 or kappa<=0:raise ValueError('positive precisions required')
    v=b.T@l;a=np.diag(g)+delta*(b.T@b)
    q=float(delta*(l@l)-delta**2*v@np.linalg.solve(a,v))
    return {'rho_mean':float(kappa*rho0/(kappa+q)),'rho_variance':float(1/(kappa+q)),
            'extra_rho_precision':q,'orthogonal_lf_energy':float(np.linalg.norm(l-b@np.linalg.lstsq(b,l,rcond=None)[0])**2)}
