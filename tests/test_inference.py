import unittest
from dataclasses import replace
import numpy as np
from numpy.testing import assert_allclose
from scipy.integrate import quad
from scipy import stats
from rbmf_cs import RBMFCS, VBConfig
from rbmf_cs.numerics import gaussian_posterior, discrepancy_energy, local_precision_parameters, solve_gaussian


class NumericalTests(unittest.TestCase):
    def test_fixed_gaussian_matches_independent_augmented_svd(self):
        rng=np.random.default_rng(17)
        psi=rng.normal(size=(7,3)); low=rng.normal(size=7)
        h=psi[[1,3,5]]; y=np.array([.2,1.,-.4])
        gamma=np.array([.4,2.,1.]); weights=np.array([.3,1.,2.])
        tau,delta,rho=3.,.8,1.2
        mean,cov,jitter=gaussian_posterior(psi,low,h,y,gamma,weights,tau,delta,rho)
        # Independent Bayesian regression as whitened augmented least squares.
        a=np.vstack((np.diag(np.sqrt(gamma)),np.sqrt(delta)*psi,np.sqrt(tau*weights)[:,None]*h))
        b=np.concatenate((np.zeros(3),np.sqrt(delta)*rho*low,np.sqrt(tau*weights)*y))
        reference=np.linalg.lstsq(a,b,rcond=None)[0]
        _,s,vt=np.linalg.svd(a,full_matrices=False)
        reference_cov=(vt.T/s**2)@vt
        assert_allclose(mean,reference,rtol=1e-12,atol=1e-12)
        assert_allclose(cov,reference_cov,rtol=1e-12,atol=1e-12)
        self.assertEqual(jitter,0)

    def test_scalar_gaussian_matches_numerical_integration(self):
        psi=np.array([[1.],[2.]])
        low=np.array([.4,.9]); h=psi[[0]]; y=np.array([1.1])
        mean,cov,_=gaussian_posterior(psi,low,h,y,np.array([.8]),np.array([.6]),3.,2.,1.2)
        def density(a):
            return stats.norm.pdf(a,scale=1/np.sqrt(.8))*np.prod(stats.norm.pdf(psi[:,0]*a-1.2*low,scale=1/np.sqrt(2.)))*stats.norm.pdf(y[0],loc=a,scale=1/np.sqrt(1.8))
        z=quad(density,-10,10,epsabs=1e-12)[0]
        m=quad(lambda a:a*density(a),-10,10,epsabs=1e-12)[0]/z
        v=quad(lambda a:(a-m)**2*density(a),-10,10,epsabs=1e-12)[0]/z
        assert_allclose(mean,[m],atol=1e-10); assert_allclose(cov,[[v]],atol=1e-10)

    def test_student_t_outlier_precision_and_integral(self):
        a,b=local_precision_parameters(4.,2.,np.array([.2,100.]))
        self.assertLess(a[1]/b[1],a[0]/b[0]/20)
        for i,energy in enumerate([.2,100.]):
            def density(lam):
                return stats.gamma.pdf(lam,2.,scale=.5)*np.sqrt(lam)*np.exp(-energy*lam)
            z=quad(density,0,np.inf,epsabs=1e-12)[0]
            expected=quad(lambda v:v*density(v),0,np.inf,epsabs=1e-12)[0]/z
            self.assertAlmostEqual(a[i]/b[i],expected,places=8)

    def test_discrepancy_energy_matches_independent_monte_carlo(self):
        rng=np.random.default_rng(801)
        psi=rng.normal(size=(5,2)); low=rng.normal(size=5)
        mean=np.array([.6,-.3]); cov=np.array([[.7,.2],[.2,.4]])
        rho,rv=.9,.6
        actual=discrepancy_energy(psi,low,mean,cov,rho,rv)
        alpha=rng.multivariate_normal(mean,cov,150000)
        rs=rng.normal(rho,np.sqrt(rv),150000)
        energy=np.sum((alpha@psi.T-rs[:,None]*low)**2,axis=1)
        self.assertLess(abs(actual-energy.mean()),5*energy.std()/np.sqrt(len(energy)))
        # Either omitted variance term would be far larger than MC tolerance.
        self.assertGreater(np.trace(psi@cov@psi.T),.5)
        self.assertGreater(rv*(low@low),.5)

    def test_fixed_hyperparameter_assimilation_contracts_covariance(self):
        psi=np.eye(3); low=np.ones(3); gamma=np.ones(3)
        _,before,_=gaussian_posterior(psi,low,psi[:1],np.ones(1),gamma,np.ones(1),2.,1.,1.)
        _,after,_=gaussian_posterior(psi,low,psi[:2],np.ones(2),gamma,np.ones(2),2.,1.,1.)
        self.assertGreaterEqual(np.linalg.eigvalsh(before-after).min(),-1e-12)

    def test_cholesky_regularization_is_reported(self):
        _,cov,jitter=solve_gaussian(np.array([[1.,1.],[1.,1.]]),np.ones(2))
        self.assertGreater(jitter,0); self.assertTrue(np.isfinite(cov).all())


class ModelTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(3)
        self.psi=np.linalg.qr(rng.normal(size=(15,3)))[0]
        self.low=self.psi@np.array([1.,-.4,.8])+.05*rng.normal(size=15)
        self.idx=np.array([0,2,4,6,8,10,12])
        self.y=(self.psi@np.array([1.2,-.3,.5]))[self.idx]+.02*rng.normal(size=7)
        self.config=VBConfig(tolerance=1e-9,max_iter=2500)

    def fit(self,config=None):
        return RBMFCS(self.psi,config or self.config).fit(self.low,self.y,indices=self.idx)

    def test_convergence_elbo_and_positive_semidefinite_covariances(self):
        m=self.fit()
        self.assertTrue(m.diagnostics_['converged'])
        self.assertEqual(m.diagnostics_['elbo_decreases'],0)
        for cov in (m.cov_alpha_,m.posterior_covariance()):
            assert_allclose(cov,cov.T,atol=1e-12)
            self.assertGreaterEqual(np.linalg.eigvalsh(cov).min(),-1e-12)
        assert_allclose(m.posterior_variance(),np.diag(m.posterior_covariance()),atol=1e-12)
        self.assertAlmostEqual(m.total_uncertainty(),np.trace(m.posterior_covariance()),places=12)
        vec=np.arange(15.)
        assert_allclose(m.covariance_product(vec),m.posterior_covariance()@vec,atol=1e-12)

    def test_gamma_retains_coefficient_variance(self):
        m=self.fit()
        assert_allclose(m.gamma_.rate,self.config.b_gamma+.5*(m.mean_alpha_**2+np.diag(m.cov_alpha_)))
        self.assertTrue(np.all(m.gamma_.rate>self.config.b_gamma+.5*m.mean_alpha_**2))

    def test_all_six_factors_update_from_initial_values(self):
        m=self.fit(replace(self.config,max_iter=1,min_iter=1))
        for name in ('gamma','lambda','tau','tau_delta'):
            actual=m.factor_summary()[name]['mean']; initial=m.diagnostics_['initial_factors'][name]['mean']
            self.assertFalse(np.allclose(actual,initial),name)
        self.assertNotEqual(m.rho_mean_,m.config.rho0)
        self.assertGreater(np.linalg.norm(m.mean_alpha_),0)
        self.assertAlmostEqual(float(m.delta_.shape),self.config.a_delta+15/2)
        self.assertAlmostEqual(float(m.tau_.shape),self.config.a_tau+7/2)

    def test_missing_rows_removed_not_zero_filled(self):
        y=self.y.copy(); y[[1,4]]=np.nan
        keep=~np.isnan(y)
        m=RBMFCS(self.psi,self.config).fit(self.low,y,indices=self.idx)
        reference=RBMFCS(self.psi,self.config).fit(self.low,y[keep],indices=self.idx[keep])
        assert_allclose(m.mean_alpha_,reference.mean_alpha_,atol=0,rtol=0)
        assert_allclose(m.cov_alpha_,reference.cov_alpha_,atol=0,rtol=0)
        self.assertEqual(m.diagnostics_['n_observations'],5)
        self.assertEqual(m.diagnostics_['missing_removed_total'],2)
        zero=RBMFCS(self.psi,self.config).fit(self.low,np.nan_to_num(y),indices=self.idx)
        self.assertGreater(np.linalg.norm(m.mean_alpha_-zero.mean_alpha_),1e-4)

    def test_sampling_matrix_equals_index_interface(self):
        m=self.fit()
        c=np.eye(15)[self.idx]
        ref=RBMFCS(self.psi,self.config).fit(self.low,self.y,sampling_matrix=c)
        assert_allclose(m.mean_alpha_,ref.mean_alpha_,atol=1e-12)

    def test_warm_start_refits_all_factors_and_retains_initial_state(self):
        m=self.fit()
        old=m.factor_summary(); old_mean=m.mean_alpha_.copy(); old_cov=m.cov_alpha_.copy()
        m.update([1.8],indices=[14])
        self.assertEqual(m.fit_count_,2)
        self.assertTrue(m.diagnostics_['warm_start'])
        self.assertEqual(m.diagnostics_['initial_factors']['gamma'],old['gamma'])
        self.assertEqual(m.diagnostics_['initial_factors']['rho'],old['rho'])
        self.assertEqual(m.diagnostics_['initial_factors']['lambda']['mean'][-1],1.)
        self.assertEqual(m.diagnostics_['n_observations'],8)
        self.assertGreater(np.linalg.norm(m.mean_alpha_-old_mean),1e-5)
        self.assertGreater(np.linalg.norm(m.cov_alpha_-old_cov),1e-7)
        for name in ('gamma','tau','tau_delta'):
            self.assertFalse(np.allclose(m.factor_summary()[name]['mean'],old[name]['mean']))
        self.assertTrue(m.diagnostics_['converged'])
        # No assertion of monotone full-VB uncertainty or reconstruction error.

    def test_max_iteration_status(self):
        m=self.fit(replace(self.config,max_iter=1,min_iter=1,tolerance=1e-15))
        self.assertFalse(m.diagnostics_['converged'])
        self.assertEqual(m.diagnostics_['stop_reason'],'max_iter')

    def test_all_factor_stopping_option(self):
        m=self.fit(replace(self.config,convergence='factors',tolerance=1e-6))
        self.assertTrue(m.diagnostics_['converged'])
        self.assertLess(m.diagnostics_['history'][-1]['relative_factor_change'],1e-6)

    def test_zero_lf_retains_rho_prior_and_empty_data_is_valid(self):
        m=RBMFCS(self.psi,self.config).fit(np.zeros(15),[],indices=[])
        self.assertAlmostEqual(m.rho_mean_,self.config.rho0)
        self.assertAlmostEqual(m.rho_variance_,1/self.config.kappa_rho0)
        self.assertAlmostEqual(float(m.tau_.mean),self.config.a_tau/self.config.b_tau)
        self.assertEqual(len(m.local_.mean),0)
        self.assertTrue(np.isfinite(m.posterior_variance()).all())

    def test_full_covariance_guard(self):
        m=self.fit(replace(self.config,max_full_covariance_size=10))
        with self.assertRaises(ValueError): m.posterior_covariance()
        self.assertEqual(m.posterior_covariance(allow_large=True).shape,(15,15))
        self.assertEqual(m.posterior_variance().shape,(15,))

    def test_noise_scale_is_not_marginal_variance(self):
        m=self.fit()
        v=m.marginal_noise_variance()
        self.assertGreater(v,1/float(m.tau_.mean))
        m=self.fit(replace(self.config,nu=2.))
        self.assertEqual(m.marginal_noise_variance(),float('inf'))

    def test_validation_and_immutable_basis(self):
        m=RBMFCS(self.psi,self.config)
        with self.assertRaises(RuntimeError): m.predict()
        with self.assertRaises(ValueError): m.basis[0,0]=3
        for kwargs in ({'nu':0},{'a_delta':0},{'tolerance':float('nan')},{'min_iter':3000}):
            with self.assertRaises(ValueError): VBConfig(**kwargs)
        with self.assertRaises(ValueError): m.fit(self.low,[np.inf],indices=[0])
        with self.assertRaises(ValueError): m.fit(self.low,[1.],indices=[15])
        with self.assertRaises(ValueError): m.fit(self.low,[1.],indices=[.5])
        with self.assertRaises(ValueError): m.fit(np.full(15,np.nan),self.y,indices=self.idx)

    def test_elbo_matches_independent_log_joint_monte_carlo(self):
        # Integrate normalized densities with scipy.stats, independent of the
        # analytical entropy/digamma ELBO implementation. Proper, stable priors.
        c=replace(self.config,a_gamma=2.,b_gamma=1.,a_tau=2.,b_tau=1.,a_delta=2.,b_delta=1.)
        m=self.fit(c); rng=np.random.default_rng(38); size=60000
        alpha=rng.multivariate_normal(m.mean_alpha_,m.cov_alpha_,size)
        rho=rng.normal(m.rho_mean_,np.sqrt(m.rho_variance_),size)
        qlog=stats.multivariate_normal.logpdf(alpha,m.mean_alpha_,m.cov_alpha_)+stats.norm.logpdf(rho,m.rho_mean_,np.sqrt(m.rho_variance_))
        prior=stats.norm.logpdf(rho,c.rho0,1/np.sqrt(c.kappa_rho0))
        samples={}
        for name,factor,a,b in (('gamma',m.gamma_,c.a_gamma,c.b_gamma),('lambda',m.local_,c.nu/2,c.nu/2),('tau',m.tau_,c.a_tau,c.b_tau),('delta',m.delta_,c.a_delta,c.b_delta)):
            shape=np.atleast_1d(factor.shape); rate=np.atleast_1d(factor.rate)
            values=rng.gamma(shape,1/rate,size=(size,len(shape))); samples[name]=values
            qlog+=stats.gamma.logpdf(values,shape,scale=1/rate).sum(axis=1)
            prior+=stats.gamma.logpdf(values,a,scale=1/b).sum(axis=1)
        prior+=stats.norm.logpdf(alpha,scale=1/np.sqrt(samples['gamma'])).sum(axis=1)
        observation=stats.norm.logpdf(m.y_[None,:],loc=alpha@m.h_.T,scale=1/np.sqrt(samples['tau']*samples['lambda'])).sum(axis=1)
        auxiliary=stats.norm.logpdf(0,loc=alpha@m.basis.T-rho[:,None]*m.low_,scale=1/np.sqrt(samples['delta'])).sum(axis=1)
        integrand=prior+observation+auxiliary-qlog
        self.assertLess(abs(m._elbo()-integrand.mean()),6*integrand.std()/np.sqrt(size))


if __name__=='__main__': unittest.main()
