import unittest
from dataclasses import replace
import numpy as np
from numpy.testing import assert_allclose
from scipy import stats
from rbmf_cs.model import VBConfig
from rbmf_cs.p0_diagnostics import DiagnosticRBMF
from rbmf_cs.structured_rbmf import StructuredRBMF,integrated_no_observation


class StructuredTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(245);self.b=np.linalg.qr(rng.normal(size=(8,3)))[0]
        self.low=self.b@np.array([2.,-.3,.5])+rng.normal(0,.2,8)
        self.ids=np.array([0,2,4,7]);self.y=self.low[self.ids]+.02
        self.c=VBConfig(a_gamma=2,b_gamma=1,a_tau=2,b_tau=1,a_delta=2,b_delta=1,
                        max_iter=3000,tolerance=1e-8,convergence='factors')

    def test_joint_mean_covariance_augmented_svd(self):
        m=StructuredRBMF(self.b,self.c).fit(self.low,self.y,indices=self.ids)
        k=m.k;d=float(m.delta_.mean);t=float(m.tau_.mean)
        aug=np.vstack([np.c_[np.sqrt(t*m.local_.mean)[:,None]*m.h_,np.zeros(len(self.y))],
            np.c_[np.diag(np.sqrt(m.gamma_.mean)),np.zeros(k)],
            np.sqrt(d)*np.c_[self.b,-self.low],np.r_[np.zeros(k),np.sqrt(self.c.kappa_rho0)][None,:]])
        rhs=np.r_[np.sqrt(t*m.local_.mean)*self.y,np.zeros(k+m.n),np.sqrt(self.c.kappa_rho0)*self.c.rho0]
        u,s,v=np.linalg.svd(aug,full_matrices=False)
        assert_allclose(np.r_[m.mean_alpha_,m.rho_mean_],v.T@((u.T@rhs)/s),atol=2e-7)
        assert_allclose(m.joint_covariance(),(v.T/s**2)@v,atol=2e-7)
        self.assertTrue(m.diagnostics_['converged']);self.assertEqual(m.diagnostics_['elbo_decreases'],0)

    def test_joint_elbo_independent_normalized_density(self):
        m=StructuredRBMF(self.b,self.c).fit(self.low,self.y,indices=self.ids);c=self.c
        rng=np.random.default_rng(148);n=80000
        z=rng.multivariate_normal(np.r_[m.mean_alpha_,m.rho_mean_],m.joint_covariance(),n);alpha=z[:,:-1];rho=z[:,-1]
        q=stats.multivariate_normal.logpdf(z,np.r_[m.mean_alpha_,m.rho_mean_],m.joint_covariance())
        p=stats.norm.logpdf(rho,c.rho0,1/np.sqrt(c.kappa_rho0));samples={}
        for name,f,a,b in [('gamma',m.gamma_,c.a_gamma,c.b_gamma),('tau',m.tau_,c.a_tau,c.b_tau),('delta',m.delta_,c.a_delta,c.b_delta),('local',m.local_,c.nu/2,c.nu/2)]:
            shape=np.atleast_1d(f.shape);rate=np.atleast_1d(f.rate);x=rng.gamma(shape,1/rate,size=(n,len(shape)));samples[name]=x
            q+=stats.gamma.logpdf(x,shape,scale=1/rate).sum(1);p+=stats.gamma.logpdf(x,a,scale=1/b).sum(1)
        p+=stats.norm.logpdf(alpha,scale=1/np.sqrt(samples['gamma'])).sum(1)
        p+=stats.norm.logpdf(self.y,loc=alpha@m.h_.T,scale=1/np.sqrt(samples['tau']*samples['local'])).sum(1)
        residual=alpha@self.b.T-rho[:,None]*self.low
        p+=stats.norm.logpdf(0,loc=residual,scale=1/np.sqrt(samples['delta'])).sum(1)
        values=p-q;self.assertLess(abs(m._elbo()-values.mean()),6*values.std()/np.sqrt(n))
        energy=np.sum(residual**2,axis=1)
        self.assertLess(abs(m.discrepancy_expectation()-energy.mean()),6*energy.std()/np.sqrt(n))

    def test_zero_lf_joint_reduces_to_factorized(self):
        mf=DiagnosticRBMF(self.b,self.c).fit(np.zeros(8),self.y,indices=self.ids)
        joint=StructuredRBMF(self.b,self.c).fit(np.zeros(8),self.y,indices=self.ids)
        assert_allclose(joint.mean_alpha_,mf.mean_alpha_,atol=1e-12)
        assert_allclose(joint.cross_covariance_,0,atol=0)
        self.assertAlmostEqual(joint._elbo(),mf._elbo(),places=11)

    def test_exact_no_observation_reference(self):
        gamma=np.array([.4,2.,.1]);d=3.;c=self.c
        result=integrated_no_observation(self.b,self.low,gamma,d,c.kappa_rho0,c.rho0)
        aug=np.vstack([np.c_[np.diag(np.sqrt(gamma)),np.zeros(3)],np.sqrt(d)*np.c_[self.b,-self.low],
                       np.r_[np.zeros(3),np.sqrt(c.kappa_rho0)][None,:]])
        rhs=np.r_[np.zeros(11),np.sqrt(c.kappa_rho0)*c.rho0]
        mean=np.linalg.lstsq(aug,rhs,rcond=None)[0]
        self.assertAlmostEqual(result['rho_mean'],mean[-1],places=12)
        self.assertGreater(result['extra_rho_precision'],0)

    def test_conditioned_noise_and_update(self):
        c=replace(self.c,robust_likelihood=False)
        m=StructuredRBMF(self.b,c,fixed_tau=25.).fit(self.low,self.y[:2],indices=self.ids[:2])
        m.update(self.y[2:],indices=self.ids[2:])
        fresh=StructuredRBMF(self.b,c,fixed_tau=25.).fit(self.low,self.y,indices=self.ids)
        assert_allclose(m.predict(),fresh.predict(),atol=1e-7)
        self.assertAlmostEqual(m.marginal_noise_variance(),.04)
        self.assertEqual(m.diagnostics_['elbo_decreases'],0)

if __name__=='__main__':unittest.main()
